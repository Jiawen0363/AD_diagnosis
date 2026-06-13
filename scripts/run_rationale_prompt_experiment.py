#!/usr/bin/env python3
"""Run traceable rationale-prompt bias experiments on AD/control samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AD_CSV = PROJECT_ROOT / "data" / "ad_s2t_wav2vec.csv"
DEFAULT_CONTROL_CSV = PROJECT_ROOT / "data" / "control_s2t_wav2vec.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "test_s2t_wav2vec.csv"
DEFAULT_BASE_PROMPT = PROJECT_ROOT / "prompt" / "compressed_rationale_prompt.txt"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "rationale_prompt_iter"


NEUTRAL_PROMPT_V1 = """You are analyzing a transcript from the Cookie Theft picture description task.

Write an objective, conservative summary of observable language behavior in the transcript.

Important constraints:
- Do not infer a medical diagnosis, cognitive status, or disease group.
- Treat the transcript as ASR output that may contain recognition noise. If errors may reflect transcription noise, describe them cautiously as "unclear transcription/possible sound-level distortion" rather than as impairment.
- Do not use strongly pathological wording such as "severe", "profound", "aphasia", "jargon", "neologistic", "impairment", or "breakdown" unless the transcript is largely unintelligible and the claim is directly supported.
- If the speaker conveys three or more key Cookie Theft scene elements (for example: mother/woman, sink/water overflow, boy/stool, cookie jar, girl/sister, dishes/window), explicitly state that the overall scene gist is preserved.
- Mention both preserved abilities and observable difficulties. Avoid making normal disfluencies sound clinically abnormal.
- Do not mention AD, Control, dementia, diagnosis, MMSE, or labels.

Write one concise clinical-style paragraph of 80-130 words. Focus on fluency, repetitions/self-corrections, word-finding or vague wording, grammar, organization, completeness of scene details, and semantic clarity.

Transcript:
{transcript}
"""


CLASSIFICATION_PROMPT = """You are given only a rationale summarizing speech from a Cookie Theft picture description task.

Predict whether the original participant is more likely from the AD group or the Control group.

Use only the rationale. Do not assume that disfluencies or ASR noise automatically indicate AD. Respond as JSON only:
{
  "prediction": "AD" or "Control",
  "confidence": "low" or "medium" or "high",
  "reason": "one short sentence"
}

Rationale:
{rationale}
"""


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
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def call_deepseek_chat(
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_s: int,
    max_retries: int,
    retry_sleep_s: float,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
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
            if not should_retry or attempt >= max_retries:
                raise RuntimeError(f"DeepSeek API HTTPError {exc.code}: {body}") from exc
        except (error.URLError, TimeoutError, IncompleteRead, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"DeepSeek API call failed: {exc}") from exc

        sleep_s = retry_sleep_s * (2**attempt)
        print(f"Retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(sleep_s)

    raise RuntimeError("DeepSeek API call failed after retries.")


def make_sample_record(row: dict[str, str], group: str) -> dict[str, Any]:
    return {
        "source_group": group,
        "index": row.get(""),
        "file": row.get("file"),
        "label": row.get("label"),
        "mmse": row.get("mmse"),
        "speech": (row.get("Speech") or "").strip(),
    }


def sample_balanced(
    ad_csv: Path, control_csv: Path, sample_size_per_group: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ad_rows = read_csv_rows(ad_csv)
    control_rows = read_csv_rows(control_csv)
    if sample_size_per_group > len(ad_rows) or sample_size_per_group > len(control_rows):
        raise ValueError("sample_size_per_group exceeds available AD/control rows.")

    samples = [
        make_sample_record(row, "AD")
        for row in rng.sample(ad_rows, sample_size_per_group)
    ] + [
        make_sample_record(row, "Control")
        for row in rng.sample(control_rows, sample_size_per_group)
    ]
    rng.shuffle(samples)
    return samples


def sample_balanced_test(
    test_csv: Path, sample_size_per_group: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = read_csv_rows(test_csv)
    ad_rows = [row for row in rows if row.get("label") == "1.0"]
    control_rows = [row for row in rows if row.get("label") == "0.0"]
    if sample_size_per_group > len(ad_rows) or sample_size_per_group > len(control_rows):
        raise ValueError("sample_size_per_group exceeds available AD/control test rows.")

    samples = [
        make_sample_record(row, "AD")
        for row in rng.sample(ad_rows, sample_size_per_group)
    ] + [
        make_sample_record(row, "Control")
        for row in rng.sample(control_rows, sample_size_per_group)
    ]
    rng.shuffle(samples)
    return samples


def generate_rationales(
    samples: list[dict[str, Any]],
    prompt_template: str,
    args: argparse.Namespace,
    api_key: str,
) -> list[dict[str, Any]]:
    def worker(sample: dict[str, Any]) -> dict[str, Any]:
        prompt = prompt_template.replace("{transcript}", sample["speech"])
        rationale = call_deepseek_chat(
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep,
        )
        return {**sample, "rationale": rationale}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, sample) for sample in samples]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"Generated rationale {i}/{len(samples)}: {result['source_group']} {result['index']}")
    return sorted(results, key=lambda x: (x["source_group"], int(x["index"] or 0)))


def parse_prediction(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
        else:
            parsed = {}

    prediction = str(parsed.get("prediction", "")).strip()
    if prediction not in {"AD", "Control"}:
        lowered = raw.lower()
        prediction = "Control" if "control" in lowered and "ad" not in lowered else "AD"
    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    reason = str(parsed.get("reason", raw)).strip()
    return {"prediction": prediction, "confidence": confidence, "reason": reason}


def classify_rationales(
    rationales: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> list[dict[str, Any]]:
    def worker(record: dict[str, Any]) -> dict[str, Any]:
        prompt = CLASSIFICATION_PROMPT.replace("{rationale}", record["rationale"])
        raw = call_deepseek_chat(
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            prompt=prompt,
            temperature=0.0,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep,
        )
        parsed = parse_prediction(raw)
        return {
            "source_group": record["source_group"],
            "index": record["index"],
            "file": record["file"],
            "true_label": record["source_group"],
            "prediction": parsed["prediction"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "raw_response": raw,
            "rationale": record["rationale"],
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, record) for record in rationales]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"Classified {i}/{len(rationales)}: "
                f"{result['true_label']} -> {result['prediction']}"
            )
    return sorted(results, key=lambda x: (x["source_group"], int(x["index"] or 0)))


def compute_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(p["prediction"] == p["true_label"] for p in predictions)
    controls = [p for p in predictions if p["true_label"] == "Control"]
    ads = [p for p in predictions if p["true_label"] == "AD"]
    control_as_ad = sum(p["prediction"] == "AD" for p in controls)
    ad_as_control = sum(p["prediction"] == "Control" for p in ads)
    ad_prediction_rate = sum(p["prediction"] == "AD" for p in predictions) / total if total else 0

    return {
        "n": total,
        "accuracy": correct / total if total else 0,
        "ad_prediction_rate": ad_prediction_rate,
        "control_false_positive_rate": control_as_ad / len(controls) if controls else 0,
        "ad_false_negative_rate": ad_as_control / len(ads) if ads else 0,
        "counts": {
            "AD": len(ads),
            "Control": len(controls),
            "correct": correct,
            "control_predicted_ad": control_as_ad,
            "ad_predicted_control": ad_as_control,
        },
    }


def write_notes(iter_dir: Path, summary: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    control_fp = [p for p in predictions if p["true_label"] == "Control" and p["prediction"] == "AD"]
    ad_fn = [p for p in predictions if p["true_label"] == "AD" and p["prediction"] == "Control"]
    if summary["control_false_positive_rate"] > 0.35:
        bias = "AD-leaning: many Control rationales are classified as AD."
    elif summary["ad_false_negative_rate"] > 0.35:
        bias = "Control-leaning: many AD rationales are classified as Control."
    else:
        bias = "No strong directional bias in this small sample."

    lines = [
        "# Iteration Summary",
        "",
        f"- Accuracy: {summary['accuracy']:.3f}",
        f"- AD prediction rate: {summary['ad_prediction_rate']:.3f}",
        f"- Control false-positive rate: {summary['control_false_positive_rate']:.3f}",
        f"- AD false-negative rate: {summary['ad_false_negative_rate']:.3f}",
        f"- Bias interpretation: {bias}",
        "",
        "## Control False Positives",
    ]
    if control_fp:
        for item in control_fp[:5]:
            lines.append(f"- index {item['index']}: {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(["", "## AD False Negatives"])
    if ad_fn:
        for item in ad_fn[:5]:
            lines.append(f"- index {item['index']}: {item['reason']}")
    else:
        lines.append("- None")

    (iter_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a rationale prompt bias experiment.")
    parser.add_argument("--iteration", default="iter_001")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--api-base",
        default=(
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/v1"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--ad-csv", type=Path, default=DEFAULT_AD_CSV)
    parser.add_argument("--control-csv", type=Path, default=DEFAULT_CONTROL_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--sample-source",
        choices=["train", "test"],
        default="train",
        help="Use train AD/control CSVs or sample AD/control rows from the labeled test CSV.",
    )
    parser.add_argument("--base-prompt", type=Path, default=DEFAULT_BASE_PROMPT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found in environment or .env.")

    iter_dir = args.experiment_root / args.iteration
    iter_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = iter_dir / "rationale_prompt.txt"
    if not prompt_path.exists():
        prompt_path.write_text(NEUTRAL_PROMPT_V1, encoding="utf-8")
    prompt_template = prompt_path.read_text(encoding="utf-8")
    if "{transcript}" not in prompt_template:
        raise RuntimeError(f"{prompt_path} must contain {{transcript}}.")

    if args.sample_source == "test":
        samples = sample_balanced_test(args.test_csv, args.sample_size, args.seed)
    else:
        samples = sample_balanced(args.ad_csv, args.control_csv, args.sample_size, args.seed)
    save_json(
        iter_dir / "sample_manifest.json",
        {
            "iteration": args.iteration,
            "sample_source": args.sample_source,
            "sample_size_per_group": args.sample_size,
            "seed": args.seed,
            "ad_csv": str(args.ad_csv),
            "control_csv": str(args.control_csv),
            "test_csv": str(args.test_csv),
            "samples": samples,
        },
    )
    save_json(
        iter_dir / "config.json",
        {
            "model": args.model,
            "api_base": args.api_base,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "base_prompt": str(args.base_prompt),
            "sample_source": args.sample_source,
        },
    )

    rationales_path = iter_dir / "rationales.json"
    if rationales_path.exists():
        rationales = json.loads(rationales_path.read_text(encoding="utf-8"))
        print(f"Loaded existing rationales: {len(rationales)}")
    else:
        rationales = generate_rationales(samples, prompt_template, args, api_key)
        save_json(rationales_path, rationales)
        save_json(
            iter_dir / "ad_rationales.json",
            [r for r in rationales if r["source_group"] == "AD"],
        )
        save_json(
            iter_dir / "control_rationales.json",
            [r for r in rationales if r["source_group"] == "Control"],
        )

    predictions_path = iter_dir / "rationale_only_predictions.json"
    if predictions_path.exists():
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        print(f"Loaded existing predictions: {len(predictions)}")
    else:
        predictions = classify_rationales(rationales, args, api_key)
        save_json(predictions_path, predictions)

    summary = compute_summary(predictions)
    save_json(iter_dir / "summary.json", summary)
    write_notes(iter_dir, summary, predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
