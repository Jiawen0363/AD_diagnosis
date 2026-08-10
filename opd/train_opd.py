#!/usr/bin/env python3
"""Train a tutor model with on-policy distillation using TRL.

Privileged-information distillation setup:

    s_t: ordinary tutor observation (task prompt + dialogue history)
    z:   teacher-only privileged context (reference solution, student level,
         prior/missing knowledge, ...)
    pi_teacher(a | s_t, z): frozen base teacher on privileged prompt
    pi_theta(a | s_t):      student tutor trained with LoRA; z is never used
                            at deployment time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

try:
    from opd.data import load_message_dataset, load_sdft_dataset
except ImportError:  # Allows `python train_opd.py` from inside opd/.
    from data import load_message_dataset, load_sdft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPD/GKD training for coding tutor models.")

    parser.add_argument("--trainer", choices=["auto", "sdft", "distillation", "gkd"], default="auto")
    parser.add_argument("--model_name_or_path", required=True, help="Student base model path or HF id.")
    parser.add_argument("--teacher_model_name_or_path", default=None, help="Local teacher path/HF id for GKD.")
    parser.add_argument("--teacher_model_server_url", default=None, help="External vLLM teacher server URL.")
    parser.add_argument("--use_teacher_server", action="store_true")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--eval_data_path", default=None, help="Optional SDFT val JSONL for eval during training.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument(
        "--max_teacher_prompt_length",
        type=int,
        default=None,
        help="Max tokens for PI-augmented teacher prompt (defaults to max_prompt_length + 384).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lmbda", type=float, default=1.0, help="1.0 means fully on-policy OPD.")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--loss_top_k", type=int, default=100)

    parser.add_argument("--teacher_model_kind", default="base", choices=["base", "live", "ema"])
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument(
        "--teacher_prompt_template",
        default="{prompt}\n\n{privileged_context}",
        help="Template used to inject privileged context into the teacher prompt.",
    )
    parser.add_argument("--distillation_mode", default="topk_logits", choices=["sampled_token", "full_logits", "topk_logits"])
    parser.add_argument("--distillation_topk", type=int, default=100)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument(
        "--chat_template",
        default=None,
        help="Optional path to a Jinja chat template (e.g. scripts/run/qwen3_nonthinking.jinja for Qwen3).",
    )

    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])

    return parser.parse_args()


def resolve_trainer_name(args: argparse.Namespace) -> str:
    if args.trainer != "auto":
        return args.trainer
    if args.use_teacher_server or args.teacher_model_server_url:
        return "distillation"
    if args.teacher_model_name_or_path:
        return "distillation"
    return "sdft"


def common_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none" if args.report_to == "none" else args.report_to.split(","),
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
    }
    return kwargs


def build_peft_config(args: argparse.Namespace):
    if not args.use_lora:
        return None

    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    if args.chat_template:
        template_path = Path(args.chat_template)
        if not template_path.is_file():
            raise FileNotFoundError(f"Chat template not found: {template_path}")
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        print(f"Loaded chat template from {template_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_model_init_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.bf16:
        kwargs["torch_dtype"] = "bfloat16"
    elif args.fp16:
        kwargs["torch_dtype"] = "float16"
    return kwargs


def resolve_max_teacher_prompt_length(args: argparse.Namespace) -> int:
    if args.max_teacher_prompt_length is not None:
        return args.max_teacher_prompt_length
    return min(args.max_length - 64, args.max_prompt_length + 384)


def build_distillation_config(args: argparse.Namespace):
    from trl.experimental.distillation import DistillationConfig

    kwargs = common_training_kwargs(args)
    kwargs.update(
        {
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_new_tokens,
            "lmbda": args.lmbda,
            "beta": args.beta,
            "temperature": args.temperature,
            "loss_top_k": args.loss_top_k,
            "use_teacher_server": bool(args.use_teacher_server or args.teacher_model_server_url),
            "teacher_model_server_url": args.teacher_model_server_url,
            "model_init_kwargs": build_model_init_kwargs(args),
            "teacher_model_init_kwargs": build_model_init_kwargs(args),
        }
    )
    return DistillationConfig(**kwargs)


def build_gkd_config(args: argparse.Namespace):
    from trl.experimental.gkd import GKDConfig

    kwargs = common_training_kwargs(args)
    kwargs.update(
        {
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "lmbda": args.lmbda,
            "beta": args.beta,
            "temperature": args.temperature,
            "teacher_model_init_kwargs": build_model_init_kwargs(args),
        }
    )
    return GKDConfig(**kwargs)


def build_sdft_config(args: argparse.Namespace):
    from trl.experimental.sdft import SDFTConfig

    kwargs = common_training_kwargs(args)
    kwargs.update(
        {
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_new_tokens,
            "temperature": args.temperature,
            "distillation_alpha": args.beta,
            "distillation_mode": args.distillation_mode,
            "distillation_topk": args.distillation_topk,
            "teacher_model_kind": args.teacher_model_kind,
            "num_generations": args.num_generations,
            "teacher_prompt_template": args.teacher_prompt_template,
            "model_init_kwargs": build_model_init_kwargs(args),
        }
    )
    return SDFTConfig(**kwargs)


def train_with_distillation(args: argparse.Namespace, train_dataset, eval_dataset, tokenizer, peft_config) -> None:
    config = build_distillation_config(args)
    use_privileged_teacher = bool(args.teacher_model_name_or_path) and not config.use_teacher_server
    if use_privileged_teacher:
        try:
            from opd.privileged_distillation import PrivilegedDistillationTrainer as TrainerCls
        except ImportError:
            from privileged_distillation import PrivilegedDistillationTrainer as TrainerCls
    else:
        from trl.experimental.distillation import DistillationTrainer as TrainerCls

    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "args": config,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if use_privileged_teacher:
        trainer_kwargs.update(
            {
                "teacher_prompt_template": args.teacher_prompt_template,
                "max_teacher_prompt_length": resolve_max_teacher_prompt_length(args),
            }
        )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if not config.use_teacher_server:
        if not args.teacher_model_name_or_path:
            raise ValueError("--teacher_model_name_or_path is required when not using a teacher server.")
        trainer_kwargs["teacher_model"] = args.teacher_model_name_or_path

    trainer = TrainerCls(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train_with_gkd(args: argparse.Namespace, train_dataset, tokenizer, peft_config) -> None:
    if not args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for GKDTrainer.")

    from trl.experimental.gkd import GKDTrainer

    config = build_gkd_config(args)
    trainer = GKDTrainer(
        model=args.model_name_or_path,
        teacher_model=args.teacher_model_name_or_path,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train_with_sdft(args: argparse.Namespace, train_dataset, eval_dataset, tokenizer, peft_config) -> None:
    config = build_sdft_config(args)
    if args.teacher_model_name_or_path:
        try:
            from opd.external_teacher_sdft import ExternalTeacherSDFTTrainer as TrainerCls
        except ImportError:
            from external_teacher_sdft import ExternalTeacherSDFTTrainer as TrainerCls
    else:
        from trl.experimental.sdft import SDFTTrainer as TrainerCls

    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "args": config,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if args.teacher_model_name_or_path:
        trainer_kwargs["teacher_model_name_or_path"] = args.teacher_model_name_or_path
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = TrainerCls(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def uses_sdft_dataset(args: argparse.Namespace, trainer_name: str) -> bool:
    if trainer_name == "sdft":
        return True
    if trainer_name == "distillation" and args.teacher_model_name_or_path:
        return True
    return False


def load_datasets(args: argparse.Namespace, trainer_name: str, *, drop_last_assistant: bool):
    if uses_sdft_dataset(args, trainer_name):
        train_dataset = load_sdft_dataset(
            args.data_path,
            drop_last_assistant=drop_last_assistant,
            max_samples=args.max_train_samples,
        )
        eval_dataset = None
        if args.eval_data_path:
            eval_dataset = load_sdft_dataset(
                args.eval_data_path,
                drop_last_assistant=drop_last_assistant,
                max_samples=None,
            )
        return train_dataset, eval_dataset

    train_dataset = load_message_dataset(
        args.data_path,
        drop_last_assistant=drop_last_assistant,
        max_samples=args.max_train_samples,
    )
    return train_dataset, None


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer_name = resolve_trainer_name(args)
    print(f"Using OPD trainer path: {trainer_name}")
    print(f"Student model: {args.model_name_or_path}")
    if trainer_name == "distillation" and (args.use_teacher_server or args.teacher_model_server_url):
        print(f"Teacher server: {args.teacher_model_server_url}")
    elif trainer_name == "distillation":
        print(f"Teacher model: {args.teacher_model_name_or_path or args.teacher_model_server_url}")
        if args.teacher_model_name_or_path and not (args.use_teacher_server or args.teacher_model_server_url):
            print(f"Teacher PI template: {args.teacher_prompt_template!r}")
            print(f"Max teacher prompt length: {resolve_max_teacher_prompt_length(args)}")
    elif trainer_name == "sdft":
        if args.teacher_model_name_or_path:
            print(f"Teacher model: {args.teacher_model_name_or_path} (external, frozen)")
        else:
            print(f"Teacher model kind: {args.teacher_model_kind} (same backbone as student)")

    tokenizer = load_tokenizer(args)
    drop_last_assistant = args.lmbda >= 1.0
    train_dataset, eval_dataset = load_datasets(
        args,
        trainer_name,
        drop_last_assistant=drop_last_assistant,
    )
    if eval_dataset is not None:
        print(f"Eval dataset: {args.eval_data_path}")
    peft_config = build_peft_config(args)

    if trainer_name == "distillation":
        train_with_distillation(args, train_dataset, eval_dataset, tokenizer, peft_config)
    elif trainer_name == "gkd":
        train_with_gkd(args, train_dataset, tokenizer, peft_config)
    else:
        train_with_sdft(args, train_dataset, eval_dataset, tokenizer, peft_config)


if __name__ == "__main__":
    main()
