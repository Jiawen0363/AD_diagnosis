#!/usr/bin/env python3
"""Train final SVC/Ridge on full train set and evaluate on test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from data import (
    PROJECT_ROOT,
    get_xy,
    get_xy_with_rationale,
    load_test_data,
    load_test_data_with_rationale,
    load_train_data,
    load_train_data_with_rationale,
)
from features import build_tfidf_pipeline, get_concat_text_embeddings, get_text_embeddings
from metrics import classification_metrics, regression_metrics
from train import fit_and_predict, fit_and_predict_tfidf


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_embedding_config(config: dict) -> dict:
    return {
        **config["embedding"],
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    }


def evaluate_feature_set(
    feature_set: str,
    x_train,
    x_test,
    y_train_cls: np.ndarray,
    y_test_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_test_reg: np.ndarray,
    test_subject_ids: list[str],
    *,
    feature_type: str = "embedding",
    embedding_model: str | None = "text-embedding-3-small",
    tfidf_config: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    if feature_type == "tfidf":
        if tfidf_config is None:
            raise ValueError("tfidf_config is required for TF-IDF evaluation.")
        cls_pred = fit_and_predict_tfidf(
            x_train=x_train,
            y_train=y_train_cls,
            x_test=x_test,
            task="classification",
            model_name="svc",
            tfidf_config=tfidf_config,
        )
        reg_pred = fit_and_predict_tfidf(
            x_train=x_train,
            y_train=y_train_reg,
            x_test=x_test,
            task="regression",
            model_name="ridge",
            tfidf_config=tfidf_config,
        )
    else:
        cls_pred = fit_and_predict(
            x_train=x_train,
            y_train=y_train_cls,
            x_test=x_test,
            task="classification",
            model_name="svc",
        )
        reg_pred = fit_and_predict(
            x_train=x_train,
            y_train=y_train_reg,
            x_test=x_test,
            task="regression",
            model_name="ridge",
        )

    cls_metrics = classification_metrics(y_test_cls, cls_pred)
    reg_metrics = regression_metrics(y_test_reg, reg_pred)

    predictions = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label": cls_pred.astype(int),
            "true_mmse": y_test_reg,
            "pred_mmse": reg_pred,
        }
    )

    summary = {
        "feature_set": feature_set,
        "feature_type": feature_type,
        "classifier": "svc",
        "regressor": "ridge",
        "embedding_model": embedding_model,
        "train_size": int(len(y_train_cls)),
        "test_size": int(len(y_test_cls)),
        "classification": cls_metrics,
        "regression": reg_metrics,
    }
    return summary, predictions


def print_summary(summary: dict) -> None:
    cls = summary["classification"]
    reg = summary["regression"]
    print(f"\n[{summary['feature_set']}]")
    print(
        f"  classification/svc: "
        f"acc={cls['accuracy']:.3f}, f1={cls['f1']:.3f}"
    )
    print(
        f"  regression/ridge: "
        f"rmse={reg['rmse']:.3f}, mae={reg['mae']:.3f}, "
        f"pearson_r={reg['pearson_r']:.3f}"
    )


def save_outputs(
    summaries: list[dict],
    predictions: dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "test_summary_all_features.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    for feature_set, pred_df in predictions.items():
        pred_path = results_dir / f"test_predictions_{feature_set}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate transcript-only and concat models on test set"
    )
    parser.add_argument(
        "--feature-set",
        choices=["transcript", "concat", "tfidf", "both", "all"],
        default="all",
        help="Which feature set(s) to evaluate",
    )
    args = parser.parse_args()

    config = load_config()
    embedding_config = build_embedding_config(config)
    results_dir = PROJECT_ROOT / config["results_dir"]

    train_df = load_train_data()
    test_df = load_test_data()
    train_transcripts, y_train_cls, y_train_reg, train_subject_ids = get_xy(train_df)
    test_transcripts, y_test_cls, y_test_reg, test_subject_ids = get_xy(test_df)

    summaries: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}

    if args.feature_set in {"transcript", "both", "all"}:
        print("Embedding train/test transcripts...")
        x_train = get_text_embeddings(
            texts=train_transcripts,
            subject_ids=train_subject_ids,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="train",
        )
        x_test = get_text_embeddings(
            texts=test_transcripts,
            subject_ids=test_subject_ids,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="test",
        )
        summary, pred_df = evaluate_feature_set(
            feature_set="transcript",
            x_train=x_train,
            x_test=x_test,
            y_train_cls=y_train_cls,
            y_test_cls=y_test_cls,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            test_subject_ids=test_subject_ids,
            feature_type="embedding",
        )
        summaries.append(summary)
        predictions["transcript"] = pred_df
        print_summary(summary)

    if args.feature_set in {"concat", "both", "all"}:
        train_df = load_train_data_with_rationale()
        test_df = load_test_data_with_rationale()
        (
            train_transcripts,
            train_rationales,
            y_train_cls,
            y_train_reg,
            train_subject_ids,
        ) = get_xy_with_rationale(train_df)
        (
            test_transcripts,
            test_rationales,
            y_test_cls,
            y_test_reg,
            test_subject_ids,
        ) = get_xy_with_rationale(test_df)

        print("Embedding train/test transcript+rationale concat features...")
        x_train = get_concat_text_embeddings(
            transcripts=train_transcripts,
            rationales=train_rationales,
            subject_ids=train_subject_ids,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="train_transcript_rationale_concat",
        )
        x_test = get_concat_text_embeddings(
            transcripts=test_transcripts,
            rationales=test_rationales,
            subject_ids=test_subject_ids,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="test_transcript_rationale_concat",
        )
        summary, pred_df = evaluate_feature_set(
            feature_set="concat",
            x_train=x_train,
            x_test=x_test,
            y_train_cls=y_train_cls,
            y_test_cls=y_test_cls,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            test_subject_ids=test_subject_ids,
            feature_type="embedding",
        )
        summaries.append(summary)
        predictions["concat"] = pred_df
        print_summary(summary)

    if args.feature_set in {"tfidf", "all"}:
        print("Fitting TF-IDF on train transcripts and evaluating on test...")
        summary, pred_df = evaluate_feature_set(
            feature_set="tfidf",
            x_train=train_transcripts,
            x_test=test_transcripts,
            y_train_cls=y_train_cls,
            y_test_cls=y_test_cls,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            test_subject_ids=test_subject_ids,
            feature_type="tfidf",
            embedding_model=None,
            tfidf_config=config["tfidf"],
        )
        summaries.append(summary)
        predictions["tfidf"] = pred_df
        print_summary(summary)

    save_outputs(summaries, predictions, results_dir)


if __name__ == "__main__":
    main()
