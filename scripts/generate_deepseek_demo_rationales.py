#!/usr/bin/env python3
"""Generate demographic-context rationales via DeepSeek for all samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

from data import attach_demographics, extract_subject_id, load_train_data
from demographics import format_education_years

DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompt" / "rationale_demo_prompt.txt"
DEFAULT_INPUT_FILES = [
    PROJECT_ROOT / "data" / "ad_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "control_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "test_s2t_wav2vec.csv",
]


def save_records(output_json: Path, records: list[dict[str, Any]]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def call_deepseek_chat(
    api_key: str,
    prompt_text: str,
    model: str,
    api_base: str,
    timeout_s: int,
    max_retries: int,
    retry_sleep_s: float,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        req = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt < max_retries and should_retry:
                sleep_s = retry_sleep_s * (2**attempt)
                print(
                    f"HTTP {exc.code}; retrying in {sleep_s:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"DeepSeek API HTTPError {exc.code}: {body}") from exc
        except error.URLError as exc:
            if attempt < max_retries:
                sleep_s = retry_sleep_s * (2**attempt)
                print(
                    f"URLError; retrying in {sleep_s:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"DeepSeek API URLError: {exc}") from exc

    raise RuntimeError("DeepSeek API call failed after retries.")


def build_prompt(template: str, age: str, gender: str, educ: str) -> str:
    return (
        template.replace("{age}", age)
        .replace("{gender}", gender)
        .replace("{educ}", educ)
    )


def build_record_key(row: dict[str, Any], row_number: int) -> str:
    subject_id = str(row.get("subject_id", "")).strip()
    if subject_id:
        return f"sid:{subject_id}"
    file_value = str(row.get("file", "")).strip()
    if file_value:
        return f"file:{file_value}"
    return f"row:{row_number}"


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
        subject_id = str(item.get("subject_id", "")).strip()
        file_value = str(item.get("file", "")).strip()
        if subject_id:
            key = f"sid:{subject_id}"
        elif file_value:
            key = f"file:{file_value}"
        else:
            key = f"row:{i}"
        existing[key] = item
    return existing


def make_record(row: dict[str, Any], demo_rationale: str) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "file": row.get("file"),
        "subject_id": row.get("subject_id"),
        "label": row.get("label"),
        "mmse": row.get("mmse"),
        "age": row.get("age"),
        "gender": row.get("gender"),
        "educ": row.get("educ"),
        "demographics_imputed": bool(row.get("demographics_imputed", False)),
        "demo_rationale": demo_rationale,
    }


def collect_ordered_records(
    rows: list[dict[str, Any]], records_by_key: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        row_key = build_record_key(row, idx)
        record = records_by_key.get(row_key)
        if record is not None:
            ordered.append(record)
    return ordered


def build_output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "ad_s2t_wav2vec_demo_rationale.json",
        output_dir / "control_s2t_wav2vec_demo_rationale.json",
        output_dir / "test_s2t_wav2vec_demo_rationale.json",
    ]


def compute_imputation_stats(train_df) -> dict[str, Any]:
    import pandas as pd

    age_median = float(train_df["age"].median())
    educ_median = float(train_df["educ"].median())
    gender_mode = (
        train_df["gender"].astype(str).str.strip().str.lower().mode().iloc[0]
        if train_df["gender"].notna().any()
        else "female"
    )
    return {
        "age_median": age_median,
        "educ_median": educ_median,
        "gender_mode": gender_mode,
    }


def prepare_rows(input_csv: Path, impute_stats: dict[str, Any]) -> list[dict[str, Any]]:
    import pandas as pd

    raw_rows = read_csv_rows(input_csv)
    frame = pd.DataFrame(raw_rows)
    frame["subject_id"] = frame["file"].map(extract_subject_id)
    frame = attach_demographics(frame)

    prepared: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        age = row["age"]
        gender = row["gender"]
        educ = row["educ"]
        imputed = False
        if pd.isna(age):
            age = impute_stats["age_median"]
            imputed = True
        if pd.isna(educ):
            educ = impute_stats["educ_median"]
            imputed = True
        if pd.isna(gender) or not str(gender).strip():
            gender = impute_stats["gender_mode"]
            imputed = True

        prepared.append(
            {
                "index": raw_rows[idx].get(""),
                "file": row["file"],
                "subject_id": row["subject_id"],
                "label": row.get("label"),
                "mmse": row.get("mmse"),
                "age": float(age),
                "gender": str(gender).strip().lower(),
                "educ": None
                if educ is None or (isinstance(educ, float) and pd.isna(educ))
                else float(educ),
                "demographics_imputed": imputed
                or not bool(row.get("has_demographics", False)),
            }
        )
    return prepared


def generate_file_demo_rationales(
    input_csv: Path,
    output_json: Path,
    prompt_template: str,
    impute_stats: dict[str, Any],
    api_key: str,
    model: str,
    api_base: str,
    timeout_s: int,
    max_retries: int,
    retry_sleep_s: float,
    save_every: int,
    concurrency: int,
    force: bool = False,
) -> None:
    rows = prepare_rows(input_csv, impute_stats)
    existing_records = {} if force else load_existing_records(output_json)
    records_by_key: dict[str, dict[str, Any]] = {}
    pending_items: list[tuple[int, str, dict[str, Any]]] = []
    newly_generated = 0
    resumed_count = 0

    print(f"\nProcessing {input_csv.name} ({len(rows)} samples)")
    for idx, row in enumerate(rows, start=1):
        row_key = build_record_key(row, idx)
        existing = existing_records.get(row_key)
        if existing and str(existing.get("demo_rationale", "")).strip():
            records_by_key[row_key] = existing
            resumed_count += 1
            print(f"  [{idx:>3}/{len(rows)}] resume-skip", flush=True)
            continue
        pending_items.append((idx, row_key, row))

    def process_one(
        idx: int, row_key: str, row: dict[str, Any]
    ) -> tuple[int, str, dict[str, Any]]:
        age_text = str(int(float(row["age"])))
        gender_text = str(row["gender"])
        educ_text = format_education_years(row.get("educ"))
        prompt = build_prompt(prompt_template, age_text, gender_text, educ_text)
        demo_rationale = call_deepseek_chat(
            api_key=api_key,
            prompt_text=prompt,
            model=model,
            api_base=api_base,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_sleep_s=retry_sleep_s,
        )
        return idx, row_key, make_record(row, demo_rationale)

    if pending_items:
        max_workers = max(1, min(concurrency, len(pending_items)))
        print(f"  using concurrency={max_workers}", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_one, idx, row_key, row): idx
                for idx, row_key, row in pending_items
            }
            for future in as_completed(future_map):
                idx, row_key, record = future.result()
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
        description="Generate demographic-context rationale JSONs for all samples."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--api-base",
        default=(
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/v1"
        ),
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all demo rationales even if output JSON already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "rationale",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found in environment or .env file.")

    prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()
    train_df = attach_demographics(load_train_data())
    impute_stats = compute_imputation_stats(train_df)
    output_files = build_output_files(args.output_dir)

    for input_csv, output_json in zip(DEFAULT_INPUT_FILES, output_files):
        generate_file_demo_rationales(
            input_csv=input_csv,
            output_json=output_json,
            prompt_template=prompt_template,
            impute_stats=impute_stats,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep,
            save_every=max(1, args.save_every),
            concurrency=max(1, args.concurrency),
            force=args.force,
        )


if __name__ == "__main__":
    main()
