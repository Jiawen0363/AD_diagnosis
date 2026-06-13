#!/usr/bin/env python3
"""Diagnostic: TF-IDF on rationale text only, evaluated on test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from data import (
    PROJECT_ROOT,
    get_xy_with_rationale,
    load_test_data_with_rationale,
    load_train_data_with_rationale,
)
from metrics import classification_metrics, regression_metrics
from train import fit_and_predict_tfidf


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    results_dir = PROJECT_ROOT / config["results_dir"]
    tfidf_config = config["tfidf"]

    train_df = load_train_data_with_rationale()
    test_df = load_test_data_with_rationale()
    (
        _train_transcripts,
        train_rationales,
        y_train_cls,
        y_train_reg,
        train_subject_ids,
    ) = get_xy_with_rationale(train_df)
    (
        _test_transcripts,
        test_rationales,
        y_test_cls,
        y_test_reg,
        test_subject_ids,
    ) = get_xy_with_rationale(test_df)

    results = []
    for classifier in ("lr", "svc"):
        cls_pred = fit_and_predict_tfidf(
            train_rationales,
            y_train_cls,
            test_rationales,
            "classification",
            classifier,
            tfidf_config,
        )
        cls_metrics = classification_metrics(y_test_cls, cls_pred)
        results.append(
            {
                "method": "TF-IDF (rationale only)",
                "classifier": classifier,
                "task": "classification",
                "metrics": cls_metrics,
            }
        )

    reg_pred = fit_and_predict_tfidf(
        train_rationales,
        y_train_reg,
        test_rationales,
        "regression",
        "ridge",
        tfidf_config,
    )
    reg_metrics = regression_metrics(y_test_reg, reg_pred)
    results.append(
        {
            "method": "TF-IDF (rationale only)",
            "regressor": "ridge",
            "task": "regression",
            "metrics": reg_metrics,
        }
    )

    svc_pred = fit_and_predict_tfidf(
        train_rationales,
        y_train_cls,
        test_rationales,
        "classification",
        "svc",
        tfidf_config,
    )
    pred_df = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label_svc": svc_pred.astype(int),
            "true_mmse": y_test_reg,
            "pred_mmse_ridge": reg_pred,
        }
    )

    out_json = results_dir / "test_rationale_tfidf_diagnostic.json"
    out_csv = results_dir / "test_predictions_rationale_tfidf.csv"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    pred_df.to_csv(out_csv, index=False)

    print("=== TF-IDF (rationale only) on test set ===")
    for row in results:
        if row["task"] == "classification":
            m = row["metrics"]
            print(
                f"classification/{row['classifier']}: "
                f"acc={m['accuracy']:.3f}, f1={m['f1']:.3f}"
            )
        else:
            m = row["metrics"]
            print(
                f"regression/ridge: "
                f"rmse={m['rmse']:.3f}, mae={m['mae']:.3f}, r={m['pearson_r']:.3f}"
            )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
