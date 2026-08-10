#!/usr/bin/env python3
"""Data helpers for OPD/GKD/SDFT training.

The existing imitation data is stored as Alpaca-like records with a long
`instruction` and an incremental `conversations` list.

- GKD / Distillation trainers expect conversational records under `messages`.
- SDFT expects a student `prompt` plus teacher-only `privileged_context`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_MAP = {
    "student": "user",
    "human": "user",
    "user": "user",
    "tutor": "assistant",
    "assistant": "assistant",
    "gpt": "assistant",
}

try:
    from opd.privileged import build_privileged_from_record, student_observation
except ImportError:
    from privileged import build_privileged_from_record, student_observation


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file."""
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as f:
        if data_path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {data_path}")
    return data


def _turn_to_message(turn: dict[str, Any]) -> dict[str, str]:
    if "role" in turn and "content" in turn:
        role = ROLE_MAP.get(str(turn["role"]).lower(), str(turn["role"]).lower())
        return {"role": role, "content": str(turn["content"])}

    if "from" in turn and "value" in turn:
        role = ROLE_MAP.get(str(turn["from"]).lower())
        if role is None:
            raise ValueError(f"Unsupported conversation role: {turn['from']}")
        return {"role": role, "content": str(turn["value"])}

    if len(turn) == 1:
        source, value = next(iter(turn.items()))
        role = ROLE_MAP.get(str(source).lower())
        if role is not None:
            return {"role": role, "content": str(value)}

    raise ValueError(f"Unsupported conversation turn format: {turn}")


def split_instruction_privileged(instruction: str) -> tuple[str, str | None]:
    """Backward-compatible wrapper around `opd.privileged.split_instruction_privileged`."""
    from opd.privileged import split_instruction_privileged as _split

    return _split(instruction)


def to_messages(record: dict[str, Any], *, drop_last_assistant: bool = True) -> list[dict[str, str]]:
    """Convert one repo record to TRL conversational `messages`.

    For fully on-policy training (`lmbda=1.0`), the assistant completion can be
    omitted. Dropping the final tutor turn keeps prior dialogue context while
    asking the student model to generate the next tutor response itself.
    """
    instruction = record.get("instruction") or record.get("prompt")
    if not instruction:
        raise ValueError("Record must contain `instruction` or `prompt`.")

    messages = [{"role": "user", "content": str(instruction)}]
    for turn in record.get("conversations", []):
        messages.append(_turn_to_message(turn))

    if drop_last_assistant and messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    return messages


def to_sdft_record(record: dict[str, Any], *, drop_last_assistant: bool = True) -> dict[str, Any]:
    """Convert one repo record to TRL SDFT format.

    Student prompt encodes the ordinary state s_t (task + dialogue history).
    Teacher-only privileged_context encodes z (reference solution, student level,
    prior/missing knowledge, etc.).
    """
    student_instruction = student_observation(record)
    privileged_context = build_privileged_from_record(record)

    messages = [{"role": "user", "content": student_instruction}]
    for turn in record.get("conversations", []):
        messages.append(_turn_to_message(turn))

    if drop_last_assistant and messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    return {
        "prompt": messages,
        "privileged_context": privileged_context,
    }


def build_message_records(
    records: list[dict[str, Any]],
    *,
    drop_last_assistant: bool = True,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if max_samples is not None:
        records = records[:max_samples]

    converted = []
    for idx, record in enumerate(records):
        messages = to_messages(record, drop_last_assistant=drop_last_assistant)
        converted.append(
            {
                "messages": messages,
                "source_index": record.get("id", idx),
            }
        )
    return converted


def is_prebuilt_sdft_record(record: dict[str, Any]) -> bool:
    """Detect records already stored as prompt + privileged_context (e.g. OPD fold JSONL)."""
    prompt = record.get("prompt")
    return isinstance(prompt, list) and record.get("privileged_context") is not None


def build_sdft_records(
    records: list[dict[str, Any]],
    *,
    drop_last_assistant: bool = True,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if max_samples is not None:
        records = records[:max_samples]

    converted = []
    for idx, record in enumerate(records):
        if is_prebuilt_sdft_record(record):
            messages = list(record["prompt"])
            if drop_last_assistant and messages and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            item = {
                "prompt": messages,
                "privileged_context": str(record["privileged_context"]).strip(),
            }
        else:
            item = to_sdft_record(record, drop_last_assistant=drop_last_assistant)
        item["source_index"] = record.get("id", record.get("subject_id", idx))
        converted.append(item)
    return converted


def load_message_dataset(
    path: str | Path,
    *,
    drop_last_assistant: bool = True,
    max_samples: int | None = None,
):
    """Load records as a Hugging Face Dataset.

    Imported lazily so this module can still be used for conversion checks in
    environments where `datasets` is not installed.
    """
    from datasets import Dataset

    records = read_records(path)
    return Dataset.from_list(
        build_message_records(
            records,
            drop_last_assistant=drop_last_assistant,
            max_samples=max_samples,
        )
    )


def load_sdft_dataset(
    path: str | Path,
    *,
    drop_last_assistant: bool = True,
    max_samples: int | None = None,
):
    """Load records as a Hugging Face Dataset for SDFT training."""
    from datasets import Dataset

    records = read_records(path)
    return Dataset.from_list(
        build_sdft_records(
            records,
            drop_last_assistant=drop_last_assistant,
            max_samples=max_samples,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tutoring data to TRL messages format.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL data path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--keep-last-assistant", action="store_true", help="Keep the final tutor response.")
    parser.add_argument(
        "--format",
        choices=["messages", "sdft"],
        default="messages",
        help="Output format: TRL messages for GKD, or prompt/privileged_context for SDFT.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    records = read_records(args.input)
    if args.format == "sdft":
        converted = build_sdft_records(
            records,
            drop_last_assistant=not args.keep_last_assistant,
            max_samples=args.max_samples,
        )
    else:
        converted = build_message_records(
            records,
            drop_last_assistant=not args.keep_last_assistant,
            max_samples=args.max_samples,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(converted)} records to {output_path}")


if __name__ == "__main__":
    main()
