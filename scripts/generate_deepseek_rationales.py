#!/usr/bin/env python3
"""Generate cognitive-linguistic rationales for all AD samples via DeepSeek API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompt" / "cognitive-linguistic_deficits.txt"
DEFAULT_INPUT_FILES = [
    PROJECT_ROOT / "data" / "ad_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "control_s2t_wav2vec.csv",
    PROJECT_ROOT / "data" / "test_s2t_wav2vec.csv",
]
DEFAULT_OUTPUT_FILES = [
    PROJECT_ROOT / "data" / "ad_s2t_wav2vec_rationale.json",
    PROJECT_ROOT / "data" / "control_s2t_wav2vec_rationale.json",
    PROJECT_ROOT / "data" / "test_s2t_wav2vec_rationale.json",
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
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


def make_record(row: dict[str, str], transcript: str, rationale: str) -> dict[str, Any]:
    return {
        "index": row.get(""),
        "file": row.get("file"),
        "label": row.get("label"),
        "mmse": row.get("mmse"),
        "speech": transcript,
        "rationale": rationale,
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


def build_output_files(output_suffix: str, output_dir: Path | None = None) -> list[Path]:
    base_dir = output_dir or (PROJECT_ROOT / "data")
    if not output_suffix:
        return [
            base_dir / "ad_s2t_wav2vec_rationale.json",
            base_dir / "control_s2t_wav2vec_rationale.json",
            base_dir / "test_s2t_wav2vec_rationale.json",
        ]

    normalized = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"
    return [
        base_dir / f"ad_s2t_wav2vec_rationale{normalized}.json",
        base_dir / f"control_s2t_wav2vec_rationale{normalized}.json",
        base_dir / f"test_s2t_wav2vec_rationale{normalized}.json",
    ]


def generate_file_rationales(
    input_csv: Path,
    output_json: Path,
    prompt_template: str,
    api_key: str,
    model: str,
    api_base: str,
    timeout_s: int,
    max_retries: int,
    retry_sleep_s: float,
    save_every: int,
    concurrency: int,
) -> None:
    rows = read_csv_rows(input_csv)
    existing_records = load_existing_records(output_json)
    records_by_key: dict[str, dict[str, Any]] = {}
    pending_items: list[tuple[int, str, dict[str, str], str]] = []
    newly_generated = 0
    resumed_count = 0

    print(f"\nProcessing {input_csv.name} ({len(rows)} samples)")
    for idx, row in enumerate(rows, start=1):
        row_key = build_record_key(row, idx)
        existing = existing_records.get(row_key)
        if existing and str(existing.get("rationale", "")).strip():
            records_by_key[row_key] = existing
            resumed_count += 1
            print(
                f"  [{idx:>3}/{len(rows)}] resume-skip",
                flush=True,
            )
            continue

        transcript = (row.get("Speech") or "").strip()
        pending_items.append((idx, row_key, row, transcript))

    def process_one(
        idx: int, row_key: str, row: dict[str, str], transcript: str
    ) -> tuple[int, str, dict[str, Any]]:
        if not transcript:
            rationale = ""
        else:
            prompt = build_prompt(prompt_template, transcript)
            rationale = call_deepseek_chat(
                api_key=api_key,
                prompt_text=prompt,
                model=model,
                api_base=api_base,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_sleep_s=retry_sleep_s,
            )
        return idx, row_key, make_record(row, transcript, rationale)

    if pending_items:
        max_workers = max(1, min(concurrency, len(pending_items)))
        print(f"  using concurrency={max_workers}", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_one, idx, row_key, row, transcript): idx
                for idx, row_key, row, transcript in pending_items
            }
            for future in as_completed(future_map):
                idx, row_key, record = future.result()
                records_by_key[row_key] = record
                newly_generated += 1
                print(f"  [{idx:>3}/{len(rows)}] done", flush=True)

                # Real-time checkpointing: persist frequently so progress is visible and resumable.
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
        description="Generate rationale JSONs for ad/control/test wav2vec transcripts."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to prompt template text file containing {transcript}.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--api-base",
        default=(
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/v1"
        ),
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per sample.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Base retry backoff seconds.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N newly generated samples.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent API requests per input file.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output files, e.g. 'compressed' -> *_rationale_compressed.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory for output rationale JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found in environment or .env file.")

    prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()

    output_files = build_output_files(args.output_suffix.strip(), args.output_dir)

    for input_csv, output_json in zip(DEFAULT_INPUT_FILES, output_files):
        generate_file_rationales(
            input_csv=input_csv,
            output_json=output_json,
            prompt_template=prompt_template,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep,
            save_every=max(1, args.save_every),
            concurrency=max(1, args.concurrency),
        )


if __name__ == "__main__":
    main()
