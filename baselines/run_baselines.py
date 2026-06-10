#!/usr/bin/env python3
"""Run AD_diagnosis transcript baselines with 5-fold cross-validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from cv import load_config, load_or_create_folds
from data import PROJECT_ROOT, get_xy, load_train_data
from features import get_text_embeddings
from train import run_fold, run_vector_fold


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def aggregate_fold_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metadata_cols = {"baseline", "embedding_model", "task", "model", "fold"}
    metric_cols = [c for c in df.columns if c not in metadata_cols]

    summary_rows = []
    for (baseline, task, model), group in df.groupby(["baseline", "task", "model"]):
        row = {"baseline": baseline, "task": task, "model": model}
        if "embedding_model" in group.columns:
            row["embedding_model"] = group["embedding_model"].iloc[0]
        for metric in metric_cols:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== CV Summary ===")
    for task in summary_df["task"].unique():
        print(f"\n[{task}]")
        task_df = summary_df[summary_df["task"] == task]
        for _, row in task_df.iterrows():
            if task == "classification":
                print(
                    f"  {row['baseline']}/{row['model']}: "
                    f"acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}, "
                    f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}"
                )
            else:
                print(
                    f"  {row['baseline']}/{row['model']}: "
                    f"rmse={row['rmse_mean']:.3f}±{row['rmse_std']:.3f}, "
                    f"pearson_r={row['pearson_r_mean']:.3f}±{row['pearson_r_std']:.3f}"
                )


def run_tfidf_baseline(config: dict) -> tuple[list[dict], pd.DataFrame]:
    train_df = load_train_data()
    texts, y_cls, y_reg, _ = get_xy(train_df)
    cls_folds, reg_folds, _ = load_or_create_folds()

    tfidf_config = config["tfidf"]
    rows: list[dict] = []

    for model_name in config["models"]["classification"]:
        for fold_idx, (train_idx, val_idx) in enumerate(cls_folds):
            metrics = run_fold(
                texts=texts,
                y=y_cls,
                train_idx=train_idx,
                val_idx=val_idx,
                task="classification",
                model_name=model_name,
                tfidf_config=tfidf_config,
            )
            rows.append(
                {
                    "baseline": "tfidf",
                    "task": "classification",
                    "model": model_name,
                    "fold": fold_idx,
                    **metrics,
                }
            )

    for model_name in config["models"]["regression"]:
        for fold_idx, (train_idx, val_idx) in enumerate(reg_folds):
            metrics = run_fold(
                texts=texts,
                y=y_reg,
                train_idx=train_idx,
                val_idx=val_idx,
                task="regression",
                model_name=model_name,
                tfidf_config=tfidf_config,
            )
            rows.append(
                {
                    "baseline": "tfidf",
                    "task": "regression",
                    "model": model_name,
                    "fold": fold_idx,
                    **metrics,
                }
            )

    return rows, aggregate_fold_results(rows)


def run_embedding_baseline(config: dict) -> tuple[list[dict], pd.DataFrame]:
    train_df = load_train_data()
    texts, y_cls, y_reg, subject_ids = get_xy(train_df)
    cls_folds, reg_folds, _ = load_or_create_folds()
    provider = config["embedding"].get("provider", "sentence_transformers")
    baseline_name = (
        f"embedding_{safe_name(provider)}_{safe_name(config['embedding']['model_name'])}"
    )
    model_config = config.get("embedding_models", config["models"])

    embeddings = get_text_embeddings(
        texts=texts,
        subject_ids=subject_ids,
        embedding_config=config["embedding"],
        project_root=PROJECT_ROOT,
    )

    rows: list[dict] = []

    for model_name in model_config["classification"]:
        for fold_idx, (train_idx, val_idx) in enumerate(cls_folds):
            metrics = run_vector_fold(
                features=embeddings,
                y=y_cls,
                train_idx=train_idx,
                val_idx=val_idx,
                task="classification",
                model_name=model_name,
            )
            rows.append(
                {
                    "baseline": "embedding",
                    "embedding_model": config["embedding"]["model_name"],
                    "task": "classification",
                    "model": model_name,
                    "fold": fold_idx,
                    **metrics,
                }
            )

    for model_name in model_config["regression"]:
        for fold_idx, (train_idx, val_idx) in enumerate(reg_folds):
            metrics = run_vector_fold(
                features=embeddings,
                y=y_reg,
                train_idx=train_idx,
                val_idx=val_idx,
                task="regression",
                model_name=model_name,
            )
            rows.append(
                {
                    "baseline": "embedding",
                    "embedding_model": config["embedding"]["model_name"],
                    "task": "regression",
                    "model": model_name,
                    "fold": fold_idx,
                    **metrics,
                }
            )

    for row in rows:
        row["baseline"] = baseline_name

    return rows, aggregate_fold_results(rows)


def save_results(
    rows: list[dict], summary_df: pd.DataFrame, results_dir: Path, baseline: str
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = results_dir / f"cv_per_fold_{baseline}.csv"
    summary_path = results_dir / f"cv_summary_{baseline}.csv"

    pd.DataFrame(rows).to_csv(per_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(results_dir / f"cv_results_{baseline}.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved per-fold results to {per_fold_path}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AD_diagnosis baselines")
    parser.add_argument(
        "--baseline",
        choices=["tfidf", "embedding", "teacher"],
        default="tfidf",
        help="Which baseline to run",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override config.yaml embedding.model_name for --baseline embedding",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence_transformers", "openai"],
        default=None,
        help="Override config.yaml embedding.provider for --baseline embedding",
    )
    args = parser.parse_args()

    config = load_config()
    results_dir = PROJECT_ROOT / config["results_dir"]

    if args.baseline == "tfidf":
        print("Running Baseline 1: TF-IDF + LR/SVC/Ridge")
        rows, summary_df = run_tfidf_baseline(config)
    elif args.baseline == "embedding":
        if args.embedding_provider:
            config["embedding"]["provider"] = args.embedding_provider
        if args.embedding_model:
            config["embedding"]["model_name"] = args.embedding_model
        print(
            "Running Baseline 2: "
            f"{config['embedding']['model_name']} embedding + classifier/regressor"
        )
        rows, summary_df = run_embedding_baseline(config)
    else:
        raise NotImplementedError("Teacher baseline is not implemented yet.")

    output_name = rows[0]["baseline"] if rows else args.baseline
    save_results(rows, summary_df, results_dir, output_name)
    print_summary(summary_df)


if __name__ == "__main__":
    main()
