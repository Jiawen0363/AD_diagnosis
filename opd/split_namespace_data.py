#!/usr/bin/env python3
"""Split OPD/SDFT data by namespace.

This prevents tutor turns from the same programming task leaking across train
and test splits. By default, sample 20 namespaces for test and 80 namespaces
for OPD training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly split OPD data by namespace.")
    parser.add_argument("--input", required=True, help="Input merged JSON file.")
    parser.add_argument("--output-dir", default=None, help="Directory for split files. Defaults to input parent.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-namespaces", type=int, default=20)
    parser.add_argument("--train-namespaces", type=int, default=80)
    parser.add_argument("--prefix", default=None, help="Output prefix. Defaults to input file stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    prefix = args.prefix or input_path.stem

    records = read_records(input_path)
    missing_namespace = [idx for idx, record in enumerate(records) if not record.get("namespace")]
    if missing_namespace:
        raise ValueError(
            f"{len(missing_namespace)} records are missing `namespace`; "
            "regenerate the data with the updated SFT converter first."
        )

    namespaces = sorted({str(record["namespace"]) for record in records})
    required = args.test_namespaces + args.train_namespaces
    if len(namespaces) < required:
        raise ValueError(
            f"Need at least {required} unique namespaces, found {len(namespaces)} in {input_path}."
        )

    rng = random.Random(args.seed)
    shuffled = namespaces[:]
    rng.shuffle(shuffled)

    test_namespaces = sorted(shuffled[: args.test_namespaces])
    train_namespaces = sorted(shuffled[args.test_namespaces : required])
    unused_namespaces = sorted(shuffled[required:])

    test_set = set(test_namespaces)
    train_set = set(train_namespaces)
    train_records = [record for record in records if record["namespace"] in train_set]
    test_records = [record for record in records if record["namespace"] in test_set]

    train_path = output_dir / f"{prefix}.train{args.train_namespaces}_namespaces.json"
    test_path = output_dir / f"{prefix}.test{args.test_namespaces}_namespaces.json"
    split_path = output_dir / f"{prefix}.namespace_split_seed{args.seed}.json"

    counts = Counter(str(record["namespace"]) for record in records)
    split_manifest = {
        "seed": args.seed,
        "input": str(input_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "num_total_records": len(records),
        "num_total_namespaces": len(namespaces),
        "num_train_records": len(train_records),
        "num_test_records": len(test_records),
        "train_namespaces": train_namespaces,
        "test_namespaces": test_namespaces,
        "unused_namespaces": unused_namespaces,
        "records_per_namespace": dict(sorted(counts.items())),
    }

    write_json(train_path, train_records)
    write_json(test_path, test_records)
    write_json(split_path, split_manifest)

    print(f"Total namespaces: {len(namespaces)} ({len(records)} records)")
    print(f"Train: {len(train_namespaces)} namespaces, {len(train_records)} records -> {train_path}")
    print(f"Test: {len(test_namespaces)} namespaces, {len(test_records)} records -> {test_path}")
    if unused_namespaces:
        print(f"Unused: {len(unused_namespaces)} namespaces")
    print(f"Split manifest -> {split_path}")


if __name__ == "__main__":
    main()
