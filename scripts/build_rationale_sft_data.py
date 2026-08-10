#!/usr/bin/env python3
"""Build FastChat SFT data for rationale generation.

Input:  instruction (task prompt) + transcript (human turn)
Output: DeepSeek-generated language rationale (assistant turn)

Only train samples (ad + control, 166) are included in the SFT split by default,
but the builder can include test transcripts too since the SFT task does not
expose diagnosis/MMSE labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompt" / "rationale_prompt.txt"
RATIONALE_DIR = PROJECT_ROOT / "data" / "rationale"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft"

SUBJECT_ID_PATTERN = re.compile(r"(adrso\d+|adrsdt\d+)")


def extract_subject_id(file_path: str) -> str:
    match = SUBJECT_ID_PATTERN.search(str(file_path))
    if not match:
        raise ValueError(f"Could not extract subject id from: {file_path}")
    return match.group(1)


def load_instruction_template(prompt_path: Path) -> tuple[str, str]:
    """Return (instruction_text, transcript_prefix)."""
    template = prompt_path.read_text(encoding="utf-8").strip()
    marker = "Transcript:\n{transcript}"
    if marker not in template:
        raise ValueError(f"Expected marker '{marker}' in {prompt_path}")
    instruction = template.split(marker, 1)[0].strip()
    return instruction, "Transcript:\n"


def load_rationale_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {path}")
    return records


def build_sample(
    record: dict[str, Any],
    *,
    instruction: str,
    transcript_prefix: str,
    split: str,
) -> dict[str, Any] | None:
    transcript = str(record.get("speech", "")).strip()
    rationale = str(record.get("rationale", "")).strip()
    if not transcript or not rationale:
        return None

    subject_id = extract_subject_id(record["file"])
    label = str(record.get("label", "")).strip()
    label_name = "ad" if label.startswith("1") else "control"

    return {
        "id": subject_id,
        "subject_id": subject_id,
        "split": split,
        "label": label_name,
        "mmse": record.get("mmse"),
        "instruction": instruction,
        "conversations": [
            {
                "from": "human",
                "value": f"{transcript_prefix}{transcript}",
            },
            {
                "from": "gpt",
                "value": rationale,
            },
        ],
        "metadata": {
            "source_file": record.get("file"),
            "rationale_source": "deepseek",
            "task": "cookie_theft_language_rationale",
        },
    }


def build_dataset(
    *,
    prompt_path: Path,
    rationale_dir: Path,
    splits: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    instruction, transcript_prefix = load_instruction_template(prompt_path)
    samples: list[dict[str, Any]] = []

    for split_name, filename in splits:
        records = load_rationale_records(rationale_dir / filename)
        for record in records:
            sample = build_sample(
                record,
                instruction=instruction,
                transcript_prefix=transcript_prefix,
                split=split_name,
            )
            if sample is not None:
                samples.append(sample)

    samples.sort(key=lambda x: x["subject_id"])
    return samples


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_json_object(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rationale_lens = [
        len(s["conversations"][1]["value"].split()) for s in samples
    ]
    transcript_lens = [
        len(s["conversations"][0]["value"].split()) for s in samples
    ]
    return {
        "num_samples": len(samples),
        "ad": sum(1 for s in samples if s["label"] == "ad"),
        "control": sum(1 for s in samples if s["label"] == "control"),
        "rationale_words_mean": round(sum(rationale_lens) / len(rationale_lens), 1),
        "rationale_words_min": min(rationale_lens),
        "rationale_words_max": max(rationale_lens),
        "transcript_words_mean": round(sum(transcript_lens) / len(transcript_lens), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FastChat SFT data for rationale generation."
    )
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_PATH)
    parser.add_argument("--rationale-dir", type=Path, default=RATIONALE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_splits = [
        ("train", "ad_s2t_wav2vec_rationale.json"),
        ("train", "control_s2t_wav2vec_rationale.json"),
        ("test", "test_s2t_wav2vec_rationale.json"),
    ]

    sft_samples = build_dataset(
        prompt_path=args.prompt_file,
        rationale_dir=args.rationale_dir,
        splits=sft_splits,
    )

    out_dir = args.output_dir
    write_json(out_dir / "rationale_sft_train.json", sft_samples)
    write_jsonl(out_dir / "rationale_sft_train.jsonl", sft_samples)

    manifest = {
        "task": "transcript + instruction -> deepseek language rationale",
        "format": "fastchat",
        "fields": {
            "instruction": "rationale task prompt (same as prompt/rationale_prompt.txt without transcript)",
            "conversations[0]": "human turn with transcript",
            "conversations[1]": "gpt turn with deepseek rationale",
        },
        "sft_file": "rationale_sft_train.json",
        "sft_samples": summarize(sft_samples),
        "notes": [
            "SFT uses all 237 samples (train+test): task exposes no diagnosis/MMSE labels.",
            "Qwen/Llama share the same JSON; choose template_name at finetune time.",
        ],
    }
    write_json_object(out_dir / "manifest.json", manifest)

    print("Built rationale SFT data")
    print(f"  sft: {out_dir / 'rationale_sft_train.json'} ({len(sft_samples)} samples)")
    print(f"  stats: {json.dumps(manifest['sft_samples'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
