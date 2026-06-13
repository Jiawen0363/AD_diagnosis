#!/usr/bin/env python3
"""Scheme C: late fusion of TF-IDF(transcript) and rationale embedding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from data import (
    PROJECT_ROOT,
    get_xy_with_rationale,
    load_test_data_with_rationale,
    load_train_data_with_rationale,
)
from features import build_tfidf_pipeline, get_text_embeddings
from metrics import classification_metrics, regression_metrics


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def fit_tfidf_svc_scores(
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
    tfidf_config: dict,
) -> np.ndarray:
    pipeline = build_tfidf_pipeline(
        SVC(kernel="linear", probability=True), tfidf_config
    )
    pipeline.fit(train_texts, y_train)
    return pipeline.predict_proba(eval_texts)[:, 1]


def fit_tfidf_ridge_preds(
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
    tfidf_config: dict,
) -> np.ndarray:
    pipeline = build_tfidf_pipeline(Ridge(), tfidf_config)
    pipeline.fit(train_texts, y_train)
    return pipeline.predict(eval_texts)


def fit_embedding_svc_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray
) -> np.ndarray:
    clf = SVC(kernel="linear", probability=True)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_eval)[:, 1]


def fit_embedding_ridge_preds(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray
) -> np.ndarray:
    reg = Ridge()
    reg.fit(x_train, y_train)
    return reg.predict(x_eval)


def tune_weight_classification(
    transcript_scores: np.ndarray,
    rationale_scores: np.ndarray,
    y_true: np.ndarray,
) -> float:
    best_w = 0.5
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        fused = w * transcript_scores + (1.0 - w) * rationale_scores
        pred = (fused >= 0.5).astype(int)
        metrics = classification_metrics(y_true, pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_w = float(w)
    return best_w


def tune_weight_regression(
    transcript_preds: np.ndarray,
    rationale_preds: np.ndarray,
    y_true: np.ndarray,
) -> float:
    best_w = 0.5
    best_rmse = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        fused = w * transcript_preds + (1.0 - w) * rationale_preds
        metrics = regression_metrics(y_true, fused)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_w = float(w)
    return best_w


def collect_oof_tfidf_rationale(
    train_transcripts: list[str],
    x_train_rationale: np.ndarray,
    y: np.ndarray,
    task: str,
    tfidf_config: dict,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    oof_tfidf = np.zeros(n, dtype=np.float64)
    oof_rationale = np.zeros(n, dtype=np.float64)

    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for train_idx, val_idx in splitter.split(train_transcripts, y):
            train_texts = [train_transcripts[i] for i in train_idx]
            val_texts = [train_transcripts[i] for i in val_idx]
            oof_tfidf[val_idx] = fit_tfidf_svc_scores(
                train_texts, y[train_idx], val_texts, tfidf_config
            )
            oof_rationale[val_idx] = fit_embedding_svc_scores(
                x_train_rationale[train_idx], y[train_idx], x_train_rationale[val_idx]
            )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in splitter.split(train_transcripts, y):
            train_texts = [train_transcripts[i] for i in train_idx]
            val_texts = [train_transcripts[i] for i in val_idx]
            oof_tfidf[val_idx] = fit_tfidf_ridge_preds(
                train_texts, y[train_idx], val_texts, tfidf_config
            )
            oof_rationale[val_idx] = fit_embedding_ridge_preds(
                x_train_rationale[train_idx], y[train_idx], x_train_rationale[val_idx]
            )

    return oof_tfidf, oof_rationale


def fuse(scores_t, scores_r, w: float, task: str):
    fused = w * scores_t + (1.0 - w) * scores_r
    if task == "classification":
        return (fused >= 0.5).astype(int)
    return fused


def main() -> None:
    config = load_config()
    cv_cfg = config["cv"]
    tfidf_config = config["tfidf"]
    results_dir = PROJECT_ROOT / config["results_dir"]
    embedding_config = {
        **config["embedding"],
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    }

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

    print("Loading rationale embeddings (text-embedding-3-small)...")
    x_train_r = get_text_embeddings(
        train_rationales,
        train_subject_ids,
        embedding_config,
        PROJECT_ROOT,
        cache_suffix="train_transcript_rationale_concat_rationale",
    )
    x_test_r = get_text_embeddings(
        test_rationales,
        test_subject_ids,
        embedding_config,
        PROJECT_ROOT,
        cache_suffix="test_transcript_rationale_concat_rationale",
    )

    print("Collecting OOF predictions on train for weight tuning...")
    oof_cls_t, oof_cls_r = collect_oof_tfidf_rationale(
        train_transcripts,
        x_train_r,
        y_train_cls,
        task="classification",
        tfidf_config=tfidf_config,
        n_splits=cv_cfg["n_splits"],
        random_state=cv_cfg["random_state"],
    )
    oof_reg_t, oof_reg_r = collect_oof_tfidf_rationale(
        train_transcripts,
        x_train_r,
        y_train_reg,
        task="regression",
        tfidf_config=tfidf_config,
        n_splits=cv_cfg["n_splits"],
        random_state=cv_cfg["random_state"],
    )

    w_cls = tune_weight_classification(oof_cls_t, oof_cls_r, y_train_cls)
    w_reg = tune_weight_regression(oof_reg_t, oof_reg_r, y_train_reg)

    print("Fitting final branch models on full train...")
    test_cls_t = fit_tfidf_svc_scores(
        train_transcripts, y_train_cls, test_transcripts, tfidf_config
    )
    test_cls_r = fit_embedding_svc_scores(x_train_r, y_train_cls, x_test_r)
    test_reg_t = fit_tfidf_ridge_preds(
        train_transcripts, y_train_reg, test_transcripts, tfidf_config
    )
    test_reg_r = fit_embedding_ridge_preds(x_train_r, y_train_reg, x_test_r)

    results = [
        {
            "method": "tfidf_transcript_only",
            "classification": classification_metrics(
                y_test_cls, (test_cls_t >= 0.5).astype(int)
            ),
            "regression": regression_metrics(y_test_reg, test_reg_t),
            "fusion_weight_transcript": 1.0,
        },
        {
            "method": "rationale_embedding_only",
            "classification": classification_metrics(
                y_test_cls, (test_cls_r >= 0.5).astype(int)
            ),
            "regression": regression_metrics(y_test_reg, test_reg_r),
            "fusion_weight_transcript": 0.0,
        },
        {
            "method": "late_fusion_equal",
            "classification": classification_metrics(
                y_test_cls,
                fuse(test_cls_t, test_cls_r, 0.5, "classification"),
            ),
            "regression": regression_metrics(
                y_test_reg, fuse(test_reg_t, test_reg_r, 0.5, "regression")
            ),
            "fusion_weight_transcript_cls": 0.5,
            "fusion_weight_transcript_reg": 0.5,
        },
        {
            "method": "late_fusion_tuned",
            "classification": classification_metrics(
                y_test_cls,
                fuse(test_cls_t, test_cls_r, w_cls, "classification"),
            ),
            "regression": regression_metrics(
                y_test_reg, fuse(test_reg_t, test_reg_r, w_reg, "regression")
            ),
            "fusion_weight_transcript_cls": w_cls,
            "fusion_weight_transcript_reg": w_reg,
        },
    ]

    out_path = results_dir / "test_late_fusion_tfidf_rationale.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label_tuned": fuse(test_cls_t, test_cls_r, w_cls, "classification"),
            "true_mmse": y_test_reg,
            "pred_mmse_tuned": fuse(test_reg_t, test_reg_r, w_reg, "regression"),
            "tfidf_score": test_cls_t,
            "rationale_score": test_cls_r,
        }
    )
    pred_df.to_csv(
        results_dir / "test_predictions_late_fusion_tfidf_rationale_tuned.csv",
        index=False,
    )

    print(
        f"\nTuned fusion weights: classification w_tfidf={w_cls:.2f}, "
        f"regression w_tfidf={w_reg:.2f}"
    )
    print(f"Saved {out_path}")
    print("\n=== Test results (Scheme C) ===")
    for row in results:
        cls = row["classification"]
        reg = row["regression"]
        print(
            f"{row['method']}: "
            f"F1={cls['f1']:.3f}, Acc={cls['accuracy']:.3f}, "
            f"RMSE={reg['rmse']:.3f}, r={reg['pearson_r']:.3f}"
        )


if __name__ == "__main__":
    main()
