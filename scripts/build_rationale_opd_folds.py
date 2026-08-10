#!/usr/bin/env python3
"""Build 5-fold OPD datasets from existing CV folds.

Uses `results/folds.json` classification splits on the 166-sample train set.
For each fold:
  - train.jsonl: fold-train subjects, privileged context includes label + MMSE severity
  - val.jsonl:   fold-val subjects, privileged context uses unknown (no label/MMSE leakage)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from opd.rationale_privileged import build_privileged_context_from_record  # noqa: E402

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_sdft_record(
    record: dict[str, Any],
    *,
    use_privileged_labels: bool,
    fold: int,
    split_role: str,
) -> dict[str, Any]:
    instruction = str(record["instruction"]).strip()
    messages = [{"role": "user", "content": instruction}]
    for turn in record.get("conversations", []):
        role = ROLE_MAP.get(str(turn.get("from", "")).lower())
        if role is None:
            raise ValueError(f"Unsupported role in record {record.get('id')}: {turn}")
        messages.append({"role": role, "content": str(turn["value"]).strip()})

    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    return {
        "prompt": messages,
        "privileged_context": build_privileged_context_from_record(
            record,
            use_labels=use_privileged_labels,
        ),
        "subject_id": record.get("subject_id") or record.get("id"),
        "label": record.get("label"),
        "mmse": record.get("mmse"),
        "fold": fold,
        "split_role": split_role,
        "use_privileged_labels": use_privileged_labels,
    }


def load_train_rationale_records(sft_path: Path) -> list[dict[str, Any]]:
    records = read_json(sft_path)
    train_records = [r for r in records if str(r.get("split", "")).lower() == "train"]
    if not train_records:
        raise ValueError(f"No train-split records found in {sft_path}")
    by_subject = {str(r["subject_id"]): r for r in train_records}
    return list(by_subject.values())


def build_fold_datasets(
    *,
    records_by_subject: dict[str, dict[str, Any]],
    folds_payload: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "task": "ad_rationale_opd",
        "fold_source": "results/folds.json classification splits",
        "privileged_info": {
            "train_split": "label + mmse_severity from groundtruth",
            "val_split": "diagnosis=unknown, mmse_severity=unknown",
        },
        "folds": [],
    }

    for fold_info in folds_payload["classification"]:
        fold = int(fold_info["fold"])
        train_subjects = list(fold_info["train_subjects"])
        val_subjects = list(fold_info["val_subjects"])

        missing = [
            sid
            for sid in train_subjects + val_subjects
            if sid not in records_by_subject
        ]
        if missing:
            raise ValueError(f"Fold {fold} references missing subjects: {missing[:5]}")

        train_rows = [
            to_sdft_record(
                records_by_subject[sid],
                use_privileged_labels=True,
                fold=fold,
                split_role="train",
            )
            for sid in sorted(train_subjects)
        ]
        val_rows = [
            to_sdft_record(
                records_by_subject[sid],
                use_privileged_labels=False,
                fold=fold,
                split_role="val",
            )
            for sid in sorted(val_subjects)
        ]

        fold_dir = PROJECT_ROOT / "data" / "opd" / f"fold{fold}"
        write_jsonl(fold_dir / "train.jsonl", train_rows)
        write_jsonl(fold_dir / "val.jsonl", val_rows)

        manifest["folds"].append(
            {
                "fold": fold,
                "train_n": len(train_rows),
                "val_n": len(val_rows),
                "train_dir": str(fold_dir),
                "train_file": str(fold_dir / "train.jsonl"),
                "val_file": str(fold_dir / "val.jsonl"),
            }
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5-fold OPD datasets for rationale task.")
    parser.add_argument(
        "--sft-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "rationale_sft_train.json",
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "folds.json",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "opd" / "manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_train_rationale_records(args.sft_path)
    records_by_subject = {str(r["subject_id"]): r for r in records}
    folds_payload = read_json(args.folds_path)

    manifest = build_fold_datasets(
        records_by_subject=records_by_subject,
        folds_payload=folds_payload,
    )
    manifest["num_train_subjects_total"] = len(records_by_subject)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Built OPD fold datasets under {PROJECT_ROOT / 'data' / 'opd'}")
    for fold in manifest["folds"]:
        print(
            f"  fold {fold['fold']}: train={fold['train_n']} val={fold['val_n']}"
        )
    print(f"Manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()
