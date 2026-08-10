#!/usr/bin/env python3
"""Smoke test: load DeepSeek-R1-Distill-Qwen-32B and run a short generation."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Qwen-32B distill model.")
    parser.add_argument(
        "--model-path",
        default="/data/models/DeepSeek-R1-Distill-Qwen-32B",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--load-mode",
        choices=["bf16_auto", "bf16_single", "4bit_single"],
        default="bf16_auto",
        help="bf16_auto: device_map=auto across visible GPUs; others for ablation.",
    )
    return parser.parse_args()


def print_gpu_mem(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: no CUDA")
        return
    print(f"\n=== {label} ===")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(
            f"  cuda:{i} free={free/1e9:.1f}GB total={total/1e9:.1f}GB "
            f"alloc={alloc/1e9:.1f}GB reserved={reserved/1e9:.1f}GB"
        )


def build_model_kwargs(args: argparse.Namespace) -> dict:
    common = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.load_mode == "bf16_auto":
        return {**common, "torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_mode == "bf16_single":
        return {**common, "torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_mode == "4bit_single":
        from transformers import BitsAndBytesConfig

        return {
            **common,
            "device_map": {"": 0},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    raise ValueError(f"Unsupported load mode: {args.load_mode}")


def main() -> None:
    args = parse_args()
    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print("visible GPU count:", torch.cuda.device_count())
    print("model:", args.model_path)
    print("load_mode:", args.load_mode)

    print_gpu_mem("before load")

    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("[2/3] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **build_model_kwargs(args),
    )
    model.eval()
    print("device_map:", getattr(model, "hf_device_map", "unknown"))
    print_gpu_mem("after load")

    print("[3/3] Short generation...")
    prompt = "Briefly explain what a Cookie Theft picture description task is."
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("OUTPUT:", text[:500])
    print_gpu_mem("after generate")
    print("\nOK: smoke test passed")


if __name__ == "__main__":
    main()
