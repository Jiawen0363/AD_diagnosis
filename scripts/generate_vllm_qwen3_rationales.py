#!/usr/bin/env python3
"""Generate language rationales with local Qwen3-8B via vLLM (thinking stripped)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path("/data/models/Qwen3-8B")
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompt" / "rationale_prompt.txt"
DEFAULT_INPUT_FILES = [
    PROJECT_ROOT / "data" / "ad_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "control_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "test_s2t_wav2vec.csv",
]
THINK_OPEN_TAG = "<" + "think" + ">"
THINK_CLOSE_TAG = "</" + "think" + ">"


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_prompt(template: str, transcript: str) -> str:
    return template.replace("{transcript}", transcript)


def build_record_key(row: dict[str, str], row_number: int) -> str:
    index_value = (row.get("") or "").strip()
    file_value = (row.get("file") or "").strip()
    if index_value:
        return f"idx:{index_value}"
    if file_value:
        return f"file:{file_value}"
    return f"row:{row_number}"


def save_records(output_json: Path, records: list[dict[str, Any]]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_existing_records(output_json: Path) -> dict[str, dict[str, Any]]:
    if not output_json.exists():
        return {}
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, list):
        return {}

    existing: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        index_value = str(item.get("index", "")).strip()
        file_value = str(item.get("file", "")).strip()
        if index_value:
            key = f"idx:{index_value}"
        elif file_value:
            key = f"file:{file_value}"
        else:
            key = f"row:{i}"
        existing[key] = item
    return existing


def make_record(
    row: dict[str, str],
    transcript: str,
    rationale: str,
    *,
    raw_output: str = "",
    thinking: str = "",
    generator: str = "qwen3-8b-vllm-thinking-stripped",
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


def collect_ordered_records(
    rows: list[dict[str, str]], records_by_key: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        row_key = build_record_key(row, idx)
        record = records_by_key.get(row_key)
        if record is not None:
            ordered.append(record)
    return ordered


def build_output_files(output_suffix: str, output_dir: Path) -> list[Path]:
    normalized = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"
    return [
        output_dir / f"ad_s2t_wav2vec_rationale{normalized}.json",
        output_dir / f"control_s2t_wav2vec_rationale{normalized}.json",
        output_dir / f"test_s2t_wav2vec_rationale{normalized}.json",
    ]


def strip_thinking_output(text: str) -> tuple[str, str]:
    """Return (final_text, thinking_text)."""
    raw = text.strip()
    if not raw:
        return "", ""

    if THINK_CLOSE_TAG in raw:
        thinking_part, final_part = raw.split(THINK_CLOSE_TAG, 1)
        thinking = thinking_part.replace(THINK_OPEN_TAG, "").strip()
        final = final_part.strip()
        if final:
            return final, thinking

    # Fallback: remove any remaining think blocks.
    cleaned = re.sub(
        rf"{re.escape(THINK_OPEN_TAG)}.*?{re.escape(THINK_CLOSE_TAG)}",
        "",
        raw,
        flags=re.DOTALL,
    ).strip()
    if cleaned:
        return cleaned, ""
    return raw, ""


def build_chat_prompts(
    tokenizer,
    prompts: list[str],
    *,
    enable_thinking: bool,
) -> list[str]:
    formatted: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            formatted_prompt = tokenizer.apply_chat_template(messages, **kwargs)
        formatted.append(formatted_prompt)
    return formatted


def generate_file_rationales(
    *,
    input_csv: Path,
    output_json: Path,
    prompt_template: str,
    llm,
    tokenizer,
    sampling_params,
    batch_size: int,
    enable_thinking: bool,
    save_every: int,
    max_samples: int | None,
    force: bool,
    generator: str,
) -> None:
    rows = read_csv_rows(input_csv)
    if max_samples is not None:
        rows = rows[:max_samples]

    existing_records = {} if force else load_existing_records(output_json)
    records_by_key: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, str, dict[str, str], str]] = []
    resumed_count = 0
    newly_generated = 0

    print(f"\nProcessing {input_csv.name} ({len(rows)} samples)")
    for idx, row in enumerate(rows, start=1):
        row_key = build_record_key(row, idx)
        existing = existing_records.get(row_key)
        if existing and str(existing.get("rationale", "")).strip():
            records_by_key[row_key] = existing
            resumed_count += 1
            print(f"  [{idx:>3}/{len(rows)}] resume-skip", flush=True)
            continue
        transcript = (row.get("Speech") or "").strip()
        pending.append((idx, row_key, row, transcript))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [
            build_prompt(prompt_template, transcript) if transcript else ""
            for _, _, _, transcript in batch
        ]
        formatted_prompts = build_chat_prompts(
            tokenizer,
            prompts,
            enable_thinking=enable_thinking,
        )

        outputs = llm.generate(formatted_prompts, sampling_params)
        for (idx, row_key, row, transcript), output in zip(batch, outputs):
            if not transcript:
                record = make_record(row, transcript, "", generator=generator)
            else:
                raw_text = output.outputs[0].text
                rationale, thinking = strip_thinking_output(raw_text)
                record = make_record(
                    row,
                    transcript,
                    rationale,
                    raw_output=raw_text,
                    thinking=thinking,
                    generator=generator,
                )
            records_by_key[row_key] = record
            newly_generated += 1
            print(f"  [{idx:>3}/{len(rows)}] done", flush=True)

            if newly_generated % save_every == 0:
                checkpoint_records = collect_ordered_records(rows, records_by_key)
                save_records(output_json, checkpoint_records)
                print(
                    f"    checkpoint saved ({len(checkpoint_records)} records): "
                    f"{output_json}",
                    flush=True,
                )

    final_records = collect_ordered_records(rows, records_by_key)
    save_records(output_json, final_records)
    print(
        f"Saved: {output_json} "
        f"(resumed {resumed_count}, newly generated {newly_generated})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rationale JSONs with local Qwen3-8B via vLLM."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-suffix", default="qwen3")
    parser.add_argument(
        "--generator",
        default="qwen3-8b-vllm-thinking-stripped",
        help="Value stored in output JSON 'generator' field.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "rationale",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu", type=int, default=4, help="CUDA device id.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of free GPU memory for vLLM.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen3 thinking mode in chat template.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print(
            "SLURM job detected; using scheduler-assigned "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()
    output_files = build_output_files(args.output_suffix.strip(), args.output_dir)

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )

    print(f"Loading vLLM model from {args.model_path} on GPU {args.gpu}")
    llm = LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for input_csv, output_json in zip(DEFAULT_INPUT_FILES, output_files):
        generate_file_rationales(
            input_csv=input_csv,
            output_json=output_json,
            prompt_template=prompt_template,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            batch_size=max(1, args.batch_size),
            enable_thinking=not args.disable_thinking,
            save_every=max(1, args.save_every),
            max_samples=args.max_samples,
            force=args.force,
            generator=args.generator,
        )


if __name__ == "__main__":
    main()
