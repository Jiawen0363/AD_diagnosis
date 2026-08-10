"""DistillationTrainer extension: external teacher + privileged context in teacher prompt only."""

from __future__ import annotations

from typing import Any

import torch
from trl.experimental.distillation.distillation_trainer import DistillationTrainer, _DistillationCollator
from trl.trainer.utils import pad


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return str(content)


def _stringify_privileged_context(privileged_context: Any) -> str:
    if privileged_context is None:
        raise ValueError("privileged_context must not be None for PI distillation.")
    if isinstance(privileged_context, str):
        return privileged_context
    return str(privileged_context)


def compose_teacher_messages(
    prompt: list[dict[str, Any]],
    privileged_context: Any,
    *,
    teacher_prompt_template: str,
) -> list[dict[str, Any]]:
    privileged_text = _stringify_privileged_context(privileged_context)
    prompt_text = _extract_last_user_text(prompt)
    teacher_text = teacher_prompt_template.format(
        prompt=prompt_text,
        privileged_context=privileged_text,
    )
    if len(prompt) > 1:
        return prompt[:-1] + [{"role": "user", "content": teacher_text}]
    return [{"role": "user", "content": teacher_text}]


class PrivilegedDistillationCollator(_DistillationCollator):
    """Tokenize student prompt and PI-augmented teacher prompt separately."""

    def __init__(
        self,
        tokenizer,
        max_length: int,
        max_prompt_length: int,
        *,
        max_teacher_prompt_length: int | None = None,
        teacher_prompt_template: str = "{prompt}\n\n{privileged_context}",
        messages_key: str = "prompt",
        privileged_context_key: str = "privileged_context",
        ignore_index: int = -100,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            messages_key=messages_key,
            ignore_index=ignore_index,
        )
        self.max_teacher_prompt_length = max_teacher_prompt_length or max_prompt_length
        self.teacher_prompt_template = teacher_prompt_template
        self.privileged_context_key = privileged_context_key

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        teacher_prompt_ids_list: list[list[int]] = []
        for example in examples:
            teacher_messages = compose_teacher_messages(
                example[self.messages_key],
                example[self.privileged_context_key],
                teacher_prompt_template=self.teacher_prompt_template,
            )
            formatted_teacher_prompt = self.tokenizer.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            teacher_prompt_ids = self.tokenizer(
                formatted_teacher_prompt,
                truncation=True,
                max_length=self.max_teacher_prompt_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            teacher_prompt_ids_list.append(list(teacher_prompt_ids))

        pad_id = self.tokenizer.pad_token_id
        teacher_prompts_t = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in teacher_prompt_ids_list],
            padding_side="left",
            padding_value=pad_id,
        )
        teacher_prompt_mask_t = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in teacher_prompt_ids_list],
            padding_side="left",
            padding_value=0,
        )
        batch["teacher_prompts"] = teacher_prompts_t
        batch["teacher_prompt_attention_mask"] = teacher_prompt_mask_t
        return batch


class PrivilegedDistillationTrainer(DistillationTrainer):
    """External teacher distillation where only the teacher prompt receives PI."""

    def __init__(
        self,
        *args,
        teacher_prompt_template: str = "{prompt}\n\n{privileged_context}",
        max_teacher_prompt_length: int | None = None,
        **kwargs,
    ):
        self.teacher_prompt_template = teacher_prompt_template
        self.max_teacher_prompt_length = max_teacher_prompt_length
        if kwargs.get("data_collator") is None:
            processing_class = kwargs.get("processing_class")
            config = kwargs.get("args")
            kwargs["data_collator"] = PrivilegedDistillationCollator(
                tokenizer=processing_class,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
                max_teacher_prompt_length=max_teacher_prompt_length,
                teacher_prompt_template=teacher_prompt_template,
            )
        super().__init__(*args, **kwargs)
        self._local_teacher_tokenizer_matches_student = True

        if getattr(self.args, "gradient_checkpointing", False):
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "enable_input_require_grads"):
                unwrapped.enable_input_require_grads()
            elif getattr(unwrapped, "base_model", None) is not None and hasattr(
                unwrapped.base_model, "enable_input_require_grads"
            ):
                unwrapped.base_model.enable_input_require_grads()

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        extra_columns = ["prompt", "privileged_context", "teacher_prompts", "teacher_prompt_attention_mask"]
        if self._signature_columns is None:
            self._signature_columns = extra_columns
        else:
            for col in extra_columns:
                if col not in self._signature_columns:
                    self._signature_columns.append(col)

    def _build_teacher_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        student_prompt_length = self._compute_prompt_length(inputs)
        completion_ids = inputs["input_ids"][:, student_prompt_length:]
        completion_mask = inputs["attention_mask"][:, student_prompt_length:]

        teacher_prompt_ids = inputs["teacher_prompts"]
        teacher_prompt_mask = inputs["teacher_prompt_attention_mask"]
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        teacher_prompt_length = teacher_prompt_ids.shape[1]
        return teacher_input_ids, teacher_attention_mask, student_prompt_length, teacher_prompt_length

    def _get_teacher_logits(self, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        if self.teacher_model is None:
            return super()._get_teacher_logits(inputs)

        teacher_input_ids, teacher_attention_mask, _, _ = self._build_teacher_inputs(inputs)
        self.teacher_model.eval()
        with torch.no_grad():
            return self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            ).logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.teacher_model is None or self.use_teacher_server:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if self.use_liger_loss:
            raise NotImplementedError("PrivilegedDistillationTrainer does not support Liger loss yet.")

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        _, _, student_prompt_length, teacher_prompt_length = self._build_teacher_inputs(inputs)
        labels = inputs["labels"][:, student_prompt_length:]
        completion_tokens = inputs["input_ids"][:, student_prompt_length:]

        teacher_logits = self._get_teacher_logits(inputs)
        student_logits = student_outputs.logits[:, student_prompt_length - 1 : -1, :]
        teacher_logits = teacher_logits[:, teacher_prompt_length - 1 : -1, :]

        comp_len = min(student_logits.size(1), teacher_logits.size(1), labels.size(1))
        student_logits = student_logits[:, :comp_len, :]
        teacher_logits = teacher_logits[:, :comp_len, :]
        labels = labels[:, :comp_len]
        completion_tokens = completion_tokens[:, :comp_len]

        if (labels != -100).sum() == 0:
            loss = student_outputs.logits.sum() * 0.0
            return (loss, student_outputs) if return_outputs else loss

        if self.beta > 0 and self.loss_top_k == 1:
            loss = self._compute_local_sparse_top_1_divergence_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                completion_tokens=completion_tokens,
                labels=labels,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.loss_top_k,
                add_tail=self.loss_add_tail,
                num_items_in_batch=num_items_in_batch,
            )

        return (loss, student_outputs) if return_outputs else loss
