#!/usr/bin/env python3
"""Evaluate triple concat with LLM demo_rationale on train CV and test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from cv import load_or_create_folds
from data import load_test_data_with_rationale, load_train_data_with_rationale
from demographics import DemographicFeatureEncoder
from eval_disagreement_routing_v3 import build_embedding_config
from features import get_concat_text_embeddings, get_triple_concat_text_embeddings
from metrics import classification_metrics, regression_metrics
from train import fit_and_predict, run_vector_fold


def stack_demo(x: np.ndarray, demo: np.ndarray | None) -> np.ndarray:
    if demo is None:
        return x
    return np.hstack([x, demo])


def cv_evaluate(
    x: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    cls_folds,
    reg_folds,
) -> tuple[dict, dict]:
    cls_rows = [
        run_vector_fold(x, y_cls, tr, va, "classification", "svc")
        for tr, va in cls_folds
    ]
    reg_rows = [
        run_vector_fold(x, y_reg, tr, va, "regression", "ridge")
        for tr, va in reg_folds
    ]
    cls_metrics = {k: float(np.mean([r[k] for r in cls_rows])) for k in cls_rows[0]}
    reg_metrics = {k: float(np.mean([r[k] for r in reg_rows])) for k in reg_rows[0]}
    return cls_metrics, reg_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM triple concat features.")
    parser.add_argument(
        "--use-tabular-demo",
        action="store_true",
        help="Append tabular demographics to feature vectors.",
    )
    args = parser.parse_args()

    with open(SCRIPT_DIR.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    embedding_config = build_embedding_config(config)
    results_dir = SCRIPT_DIR.parent / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, reg_folds, _ = load_or_create_folds()

    train_df = load_train_data_with_rationale(
        with_demographics=True,
        with_demo_rationale=True,
    )
    test_df = load_test_data_with_rationale(
        with_demographics=True,
        with_demo_rationale=True,
    )

    y_train_cls = train_df["label"].astype(int).to_numpy()
    y_train_reg = train_df["mmse"].astype(float).to_numpy()
    y_test_cls = test_df["label"].astype(int).to_numpy()
    y_test_reg = test_df["mmse"].astype(float).to_numpy()

    demo_encoder = DemographicFeatureEncoder.fit(train_df)
    demo_train = demo_encoder.transform(train_df)
    demo_test = demo_encoder.transform(test_df)

    x_train_triple = get_triple_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        demo_rationales=train_df["demo_rationale_text"].tolist(),
        subject_ids=train_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_demo_llm_struct_concat",
    )
    x_test_triple = get_triple_concat_text_embeddings(
        transcripts=test_df["text"].tolist(),
        rationales=test_df["rationale_text"].tolist(),
        demo_rationales=test_df["demo_rationale_text"].tolist(),
        subject_ids=test_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="test_transcript_rationale_demo_llm_struct_concat",
    )

    x_train_concat = get_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        subject_ids=train_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_concat",
    )
    x_test_concat = get_concat_text_embeddings(
        transcripts=test_df["text"].tolist(),
        rationales=test_df["rationale_text"].tolist(),
        subject_ids=test_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="test_transcript_rationale_concat",
    )

    if args.use_tabular_demo:
        x_train_triple = stack_demo(x_train_triple, demo_train)
        x_test_triple = stack_demo(x_test_triple, demo_test)
        x_train_concat = stack_demo(x_train_concat, demo_train)
        x_test_concat = stack_demo(x_test_concat, demo_test)

    results = []

    for name, x_tr, x_te in [
        ("concat_language", x_train_concat, x_test_concat),
        ("triple_concat_llm_demo_rationale", x_train_triple, x_test_triple),
    ]:
        cv_cls, cv_reg = cv_evaluate(x_tr, y_train_cls, y_train_reg, cls_folds, reg_folds)
        test_cls = classification_metrics(
            y_test_cls,
            fit_and_predict(x_tr, y_train_cls, x_te, "classification", "svc").astype(int),
        )
        test_reg = regression_metrics(
            y_test_reg,
            fit_and_predict(x_tr, y_train_reg, x_te, "regression", "ridge"),
        )
        results.append(
            {
                "model": name,
                "use_tabular_demo": args.use_tabular_demo,
                "cv_classification": cv_cls,
                "cv_regression": cv_reg,
                "test_classification": test_cls,
                "test_regression": test_reg,
            }
        )

    suffix = "_with_tabular" if args.use_tabular_demo else ""
    out_path = results_dir / f"triple_concat_llm_demo_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Triple concat with LLM demo_rationale ===")
    for row in results:
        cv_cls = row["cv_classification"]
        cv_reg = row["cv_regression"]
        te_cls = row["test_classification"]
        te_reg = row["test_regression"]
        print(f"\n[{row['model']}]")
        print(
            f"  CV:      cls f1={cv_cls['f1']:.3f}, "
            f"reg rmse={cv_reg['rmse']:.3f}, r={cv_reg['pearson_r']:.3f}"
        )
        print(
            f"  Test:    cls f1={te_cls['f1']:.3f}, "
            f"reg rmse={te_reg['rmse']:.3f}, r={te_reg['pearson_r']:.3f}"
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
