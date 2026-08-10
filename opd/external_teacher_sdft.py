"""SDFT with an optional frozen external teacher instead of the student backbone."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from trl.experimental.sdft import SDFTTrainer
from trl.models import prepare_deepspeed, prepare_fsdp
from trl.trainer.utils import create_model_from_path


class ExternalTeacherSDFTTrainer(SDFTTrainer):
    def __init__(self, *, teacher_model_name_or_path: str | None = None, **kwargs):
        self.teacher_model_name_or_path = teacher_model_name_or_path
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._student_gpu = 0
        self._teacher_uses_multi_gpu = gpu_count >= 3
        if teacher_model_name_or_path and gpu_count >= 2:
            args = kwargs.get("args")
            if args is not None:
                model_init_kwargs = dict(args.model_init_kwargs or {})
                model_init_kwargs["device_map"] = {"": self._student_gpu}
                args.model_init_kwargs = model_init_kwargs
        super().__init__(**kwargs)
        if self.teacher_model_name_or_path:
            # Keep student training on a single GPU; teacher lives on the other device(s).
            self.args._n_gpu = 1

    def _setup_teacher_model(self) -> None:
        if not self.teacher_model_name_or_path:
            return super()._setup_teacher_model()

        model_init_kwargs = dict(self.args.model_init_kwargs or {})
        model_init_kwargs.setdefault("trust_remote_code", self.args.trust_remote_code)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_device_map = gpu_count >= 2
        if self._teacher_uses_multi_gpu:
            # Student stays on GPU 0; shard teacher across GPUs 1..N-1.
            # e.g. 4 GPUs: cuda:0 = Qwen3-8B student, cuda:1-3 = R1-Distill-Qwen-14B teacher
            teacher_gpus = list(range(1, gpu_count))
            max_memory = {self._student_gpu: "0GiB"}
            for idx in teacher_gpus:
                max_memory[idx] = "40GiB"
            model_init_kwargs["device_map"] = "auto"
            model_init_kwargs["max_memory"] = max_memory
        elif use_device_map:
            model_init_kwargs["device_map"] = {"": 1}
        elif self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            model_init_kwargs["device_map"] = None

        self.teacher_model = create_model_from_path(self.teacher_model_name_or_path, **model_init_kwargs)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        if use_device_map:
            return
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

    def _get_teacher_context_for_self_distillation(self):
        if self.teacher_model_name_or_path:
            return nullcontext()
        return super()._get_teacher_context_for_self_distillation()

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        if self.teacher_model_name_or_path and model is self.teacher_model:
            teacher_device = next(model.parameters()).device
            input_ids = input_ids.to(teacher_device)
            attention_mask = attention_mask.to(teacher_device)
            logits = super()._forward_logits(model, input_ids, attention_mask, logits_to_keep)
            return logits.to(self.accelerator.device)
        return super()._forward_logits(model, input_ids, attention_mask, logits_to_keep)
