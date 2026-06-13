#!/usr/bin/env python3
"""Run all test-set experiments needed for the results tables."""

from __future__ import annotations

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
from features import get_concat_text_embeddings, get_text_embeddings
from metrics import classification_metrics, regression_metrics
from train import fit_and_predict, fit_and_predict_tfidf


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def evaluate(
    x_train,
    x_test,
    y_train_cls: np.ndarray,
    y_test_cls: np.ndarray,
    y_train_reg: np.ndarray,
    y_test_reg: np.ndarray,
    *,
    method: str,
    embedding_model: str | None,
    classifier: str,
    regressor: str,
    feature_type: str,
    tfidf_config: dict | None = None,
) -> dict:
    if feature_type == "tfidf":
        cls_pred = fit_and_predict_tfidf(
            x_train, y_train_cls, x_test, "classification", classifier, tfidf_config
        )
        reg_pred = fit_and_predict_tfidf(
            x_train, y_train_reg, x_test, "regression", regressor, tfidf_config
        )
    else:
        cls_pred = fit_and_predict(
            x_train, y_train_cls, x_test, "classification", classifier
        )
        reg_pred = fit_and_predict(
            x_train, y_train_reg, x_test, "regression", regressor
        )

    return {
        "method": method,
        "embedding_model": embedding_model,
        "classifier": classifier,
        "regressor": regressor,
        "feature_type": feature_type,
        "classification": classification_metrics(y_test_cls, cls_pred),
        "regression": regression_metrics(y_test_reg, reg_pred),
    }


def embedding_config_for(model_name: str, provider: str, config: dict) -> dict:
    cfg = {**config["embedding"], "model_name": model_name, "provider": provider}
    if provider == "sentence_transformers":
        cfg["batch_size"] = 8
    return cfg


def main() -> None:
    config = load_config()
    results_dir = PROJECT_ROOT / config["results_dir"]
    tfidf_config = config["tfidf"]

    train_df = load_train_data()
    test_df = load_test_data()
    train_transcripts, y_train_cls, y_train_reg, train_subject_ids = get_xy(train_df)
    test_transcripts, y_test_cls, y_test_reg, test_subject_ids = get_xy(test_df)

    train_df_r = load_train_data_with_rationale()
    test_df_r = load_test_data_with_rationale()
    (
        train_transcripts_r,
        train_rationales,
        y_train_cls_r,
        y_train_reg_r,
        train_subject_ids_r,
    ) = get_xy_with_rationale(train_df_r)
    (
        test_transcripts_r,
        test_rationales,
        y_test_cls_r,
        y_test_reg_r,
        test_subject_ids_r,
    ) = get_xy_with_rationale(test_df_r)

    rows: list[dict] = []

    print("TF-IDF + LR/SVC ...")
    rows.append(
        evaluate(
            train_transcripts,
            test_transcripts,
            y_train_cls,
            y_test_cls,
            y_train_reg,
            y_test_reg,
            method="TF-IDF (transcript)",
            embedding_model=None,
            classifier="lr",
            regressor="ridge",
            feature_type="tfidf",
            tfidf_config=tfidf_config,
        )
    )
    rows.append(
        evaluate(
            train_transcripts,
            test_transcripts,
            y_train_cls,
            y_test_cls,
            y_train_reg,
            y_test_reg,
            method="TF-IDF (transcript)",
            embedding_model=None,
            classifier="svc",
            regressor="ridge",
            feature_type="tfidf",
            tfidf_config=tfidf_config,
        )
    )

    embedding_specs = [
        ("BAAI/bge-m3", "sentence_transformers", "transcript"),
        ("text-embedding-3-small", "openai", "transcript"),
        ("text-embedding-3-large", "openai", "transcript"),
        ("text-embedding-3-small", "openai", "concat"),
        ("text-embedding-3-large", "openai", "concat"),
    ]

    for model_name, provider, feature_kind in embedding_specs:
        print(f"{model_name} ({feature_kind}) + SVC/Ridge ...")
        emb_cfg = embedding_config_for(model_name, provider, config)

        if feature_kind == "transcript":
            x_train = get_text_embeddings(
                train_transcripts, train_subject_ids, emb_cfg, PROJECT_ROOT, "train"
            )
            x_test = get_text_embeddings(
                test_transcripts, test_subject_ids, emb_cfg, PROJECT_ROOT, "test"
            )
            method = "Embedding (transcript)"
        else:
            cache_tag = "train_transcript_rationale_concat"
            cache_tag_test = "test_transcript_rationale_concat"
            x_train = get_concat_text_embeddings(
                train_transcripts_r,
                train_rationales,
                train_subject_ids_r,
                emb_cfg,
                PROJECT_ROOT,
                cache_tag,
            )
            x_test = get_concat_text_embeddings(
                test_transcripts_r,
                test_rationales,
                test_subject_ids_r,
                emb_cfg,
                PROJECT_ROOT,
                cache_tag_test,
            )
            method = "Embedding (transcript+rationale)"

        rows.append(
            evaluate(
                x_train,
                x_test,
                y_train_cls,
                y_test_cls,
                y_train_reg,
                y_test_reg,
                method=method,
                embedding_model=model_name,
                classifier="svc",
                regressor="ridge",
                feature_type="embedding",
            )
        )

    out_path = results_dir / "test_table_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    cls_rows = []
    reg_rows = []
    for row in rows:
        cls = row["classification"]
        reg = row["regression"]
        emb = row["embedding_model"] or "-"
        cls_rows.append(
            {
                "Method": row["method"],
                "Embedding model": emb,
                "Classification Model": row["classifier"].upper()
                if row["classifier"] == "lr"
                else row["classifier"].upper(),
                "F1": round(cls["f1"], 3),
                "Acc": round(cls["accuracy"], 3),
            }
        )
        reg_rows.append(
            {
                "Method": row["method"],
                "Embedding model": emb,
                "Regression Model": row["regressor"].capitalize(),
                "RMSE": round(reg["rmse"], 3),
                "Pearson r": round(reg["pearson_r"], 3),
            }
        )

    pd.DataFrame(cls_rows).to_csv(results_dir / "test_table_classification.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(results_dir / "test_table_regression.csv", index=False)

    print(f"\nSaved {out_path}")
    print("\n=== Classification (test) ===")
    for item in cls_rows:
        print(
            f"{item['Method']} | {item['Embedding model']} | "
            f"{item['Classification Model']} | F1={item['F1']} | Acc={item['Acc']}"
        )
    print("\n=== Regression (test) ===")
    for item in reg_rows:
        print(
            f"{item['Method']} | {item['Embedding model']} | "
            f"{item['Regression Model']} | RMSE={item['RMSE']} | r={item['Pearson r']}"
        )


if __name__ == "__main__":
    main()
