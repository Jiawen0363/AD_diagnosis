#!/usr/bin/env python3
"""Generate language rationales with a SFT LoRA adapter (FastChat vicuna template)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTCHAT_DIR = PROJECT_ROOT / "fastchat"
sys.path.insert(0, str(FASTCHAT_DIR))

from conversation import get_conv_template  # noqa: E402

DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompt" / "rationale_prompt.txt"
DEFAULT_INPUT_FILES = [
    PROJECT_ROOT / "data" / "ad_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "control_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "test_s2t_wav2vec.csv",
]
SUBJECT_ID_PATTERN = re.compile(r"(adrso\d+|adrsdt\d+)")
THINK_OPEN_TAG = "<" + "think" + ">"
THINK_CLOSE_TAG = "</" + "think" + ">"


def load_instruction(prompt_path: Path) -> str:
    template = prompt_path.read_text(encoding="utf-8").strip()
    marker = "Transcript:\n{transcript}"
    if marker not in template:
        raise ValueError(f"Expected marker in {prompt_path}")
    return template.split(marker, 1)[0].strip()


def build_prompt(instruction: str, transcript: str, template_name: str) -> str:
    conv = get_conv_template(template_name)
    conv.set_system_message(instruction)
    conv.append_message(conv.roles[0], f"Transcript:\n{transcript}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_user_prompt(prompt_template: str, transcript: str) -> str:
    return prompt_template.replace("{transcript}", transcript)


def build_chat_template_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_qwen_thinking_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=True,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def strip_thinking_output(text: str) -> tuple[str, str]:
    raw = text.strip()
    if not raw:
        return "", ""

    if THINK_CLOSE_TAG in raw:
        thinking_part, final_part = raw.split(THINK_CLOSE_TAG, 1)
        thinking = thinking_part.replace(THINK_OPEN_TAG, "").strip()
        final = final_part.strip()
        if final:
            return final, thinking

    cleaned = re.sub(
        rf"{re.escape(THINK_OPEN_TAG)}.*?{re.escape(THINK_CLOSE_TAG)}",
        "",
        raw,
        flags=re.DOTALL,
    ).strip()
    if cleaned:
        return cleaned, ""
    return raw, ""


def build_record_key(row: dict[str, str], row_number: int) -> str:
    index_value = (row.get("") or "").strip()
    file_value = (row.get("file") or "").strip()
    if index_value:
        return f"idx:{index_value}"
    if file_value:
        return f"file:{file_value}"
    return f"row:{row_number}"


def make_record(
    row: dict[str, str],
    transcript: str,
    rationale: str,
    *,
    raw_output: str,
    thinking: str = "",
    generator: str,
) -> dict[str, Any]:
    return {
        "index": row.get(""),
        "file": row.get("file"),
        "label": row.get("label"),
        "mmse": row.get("mmse"),
        "speech": transcript,
        "rationale": rationale,
        "raw_output": raw_output,
        "thinking": thinking,
        "generator": generator,
    }


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_records(output_json: Path, records: list[dict[str, Any]]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_existing_records(output_json: Path) -> dict[str, dict[str, Any]]:
    if not output_json.exists():
        return {}
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    existing: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(payload, start=1):
        key = build_record_key(item, i)
        existing[key] = item
    return existing


def collect_ordered_records(
    rows: list[dict[str, str]], records_by_key: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        records_by_key[build_record_key(row, idx)]
        for idx, row in enumerate(rows, start=1)
        if build_record_key(row, idx) in records_by_key
    ]


def build_output_files(output_suffix: str, output_dir: Path) -> list[Path]:
    normalized = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"
    return [
        output_dir / f"ad_s2t_wav2vec_rationale{normalized}.json",
        output_dir / f"control_s2t_wav2vec_rationale{normalized}.json",
        output_dir / f"test_s2t_wav2vec_rationale{normalized}.json",
    ]


def clean_response(text: str, eos_token: str) -> str:
    cleaned = text.split(eos_token)[0].strip()
    for marker in ("ASSISTANT:", "USER:", "</s>"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()
    return cleaned


def generate_file(
    *,
    rows: list[dict[str, str]],
    output_json: Path,
    instruction: str,
    prompt_template: str,
    model,
    tokenizer,
    generation_config: GenerationConfig,
    template_name: str,
    enable_thinking: bool,
    use_chat_template: bool,
    batch_size: int,
    max_input_len: int,
    generator: str,
    force: bool,
    save_every: int,
) -> None:
    existing = {} if force else load_existing_records(output_json)
    records_by_key = dict(existing)
    pending: list[tuple[int, str, dict[str, str], str]] = []
    resumed = 0
    generated = 0

    print(f"\nProcessing {output_json.name} ({len(rows)} samples)")
    for idx, row in enumerate(rows, start=1):
        key = build_record_key(row, idx)
        if key in records_by_key and str(records_by_key[key].get("rationale", "")).strip():
            resumed += 1
            continue
        transcript = (row.get("Speech") or "").strip()
        pending.append((idx, key, row, transcript))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if enable_thinking:
            prompts = [
                build_qwen_thinking_prompt(
                    tokenizer,
                    build_user_prompt(prompt_template, transcript),
                )
                if transcript
                else ""
                for _, _, _, transcript in batch
            ]
        elif use_chat_template:
            prompts = [
                build_chat_template_prompt(
                    tokenizer,
                    build_user_prompt(prompt_template, transcript),
                )
                if transcript
                else ""
                for _, _, _, transcript in batch
            ]
        else:
            prompts = [
                build_prompt(instruction, transcript, template_name)
                if transcript
                else ""
                for _, _, _, transcript in batch
            ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                generation_config=generation_config,
            )

        for (idx, key, row, transcript), output_ids in zip(batch, outputs):
            if not transcript:
                record = make_record(row, transcript, "", raw_output="", generator=generator)
            else:
                new_tokens = output_ids[input_len:]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
                if enable_thinking:
                    rationale, thinking = strip_thinking_output(raw_output)
                else:
                    rationale = clean_response(raw_output, tokenizer.eos_token)
                    thinking = ""
                record = make_record(
                    row,
                    transcript,
                    rationale,
                    raw_output=raw_output,
                    thinking=thinking,
                    generator=generator,
                )
            records_by_key[key] = record
            generated += 1
            print(f"  [{idx:>3}/{len(rows)}] done", flush=True)

            if generated % save_every == 0:
                save_records(output_json, collect_ordered_records(rows, records_by_key))

    final_records = collect_ordered_records(rows, records_by_key)
    save_records(output_json, final_records)
    print(
        f"Saved: {output_json} (resumed {resumed}, newly generated {generated})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rationales with SFT LoRA adapter.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/sft-llama31-8b-rationale/checkpoint-45",
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-suffix", default="sft_llama31")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "rationale")
    parser.add_argument("--template-name", default="vicuna_v1.1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=None,
        help="Override base model path from adapter_config (e.g. when /data_old is unavailable).",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Format prompts with tokenizer.apply_chat_template (for Qwen non-thinking OPD/SFT).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Use Qwen3 chat template with thinking enabled (strips think blocks).",
    )
    parser.add_argument("--generator", default="llama31-8b-sft-lora-rationale")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()
    instruction = load_instruction(args.prompt_file)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 4096 if args.enable_thinking else 512
    temperature = 0.6 if args.enable_thinking else args.temperature
    top_p = 0.95 if args.enable_thinking else args.top_p
    adapter_config = json.loads(
        (args.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    base_model_path = args.base_model_path or adapter_config["base_model_name_or_path"]
    if args.base_model_path:
        print(f"Using overridden base model: {base_model_path}")
    else:
        print(f"Loading base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    chat_template_path = args.adapter_path / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        print(f"Loaded chat template from {chat_template_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    generation_config = GenerationConfig(
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_files = build_output_files(args.output_suffix.strip(), args.output_dir)
    for input_csv, output_json in zip(DEFAULT_INPUT_FILES, output_files):
        rows = read_csv_rows(input_csv)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        generate_file(
            rows=rows,
            output_json=output_json,
            instruction=instruction,
            prompt_template=prompt_template,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_name=args.template_name,
            enable_thinking=args.enable_thinking,
            use_chat_template=args.use_chat_template,
            batch_size=max(1, args.batch_size),
            max_input_len=args.max_input_len,
            generator=args.generator,
            force=args.force,
            save_every=max(1, args.save_every),
        )


if __name__ == "__main__":
    main()
