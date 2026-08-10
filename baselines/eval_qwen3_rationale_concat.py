#!/usr/bin/env python3
"""Evaluate concat / triple concat using Qwen3-generated rationales."""

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
from demographics import build_demo_rationales
from eval_disagreement_routing_v3 import build_embedding_config
from features import get_concat_text_embeddings, get_triple_concat_text_embeddings
from metrics import classification_metrics, regression_metrics
from train import fit_and_predict, run_vector_fold


def cv_evaluate(x, y_cls, y_reg, cls_folds, reg_folds):
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
    parser = argparse.ArgumentParser(
        description="Evaluate concat/triple concat with Qwen3 rationales."
    )
    parser.add_argument("--rationale-suffix", default="qwen3")
    parser.add_argument(
        "--cache-tag",
        default="qwen3",
        help="Embedding cache suffix tag for rationale variant.",
    )
    parser.add_argument(
        "--rationale-generator",
        default="qwen3-8b-vllm-thinking-stripped",
        help="Generator label stored in results JSON.",
    )
    parser.add_argument(
        "--results-prefix",
        default="qwen3",
        help="Prefix for result model names and output filename.",
    )
    args = parser.parse_args()

    with open(SCRIPT_DIR.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    embedding_config = build_embedding_config(config)
    results_dir = SCRIPT_DIR.parent / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, reg_folds, _ = load_or_create_folds()
    train_df = load_train_data_with_rationale(
        rationale_suffix=args.rationale_suffix,
        with_demographics=True,
        with_demo_rationale=True,
    )
    test_df = load_test_data_with_rationale(
        rationale_suffix=args.rationale_suffix,
        with_demographics=True,
        with_demo_rationale=True,
    )

    y_train_cls = train_df["label"].astype(int).to_numpy()
    y_train_reg = train_df["mmse"].astype(float).to_numpy()
    y_test_cls = test_df["label"].astype(int).to_numpy()
    y_test_reg = test_df["mmse"].astype(float).to_numpy()

    demo_texts = build_demo_rationales(train_df)
    demo_texts_test = build_demo_rationales(test_df)
    cache_tag = args.cache_tag

    x_train_concat = get_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        subject_ids=train_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix=f"train_transcript_rationale_{cache_tag}_concat",
    )
    x_test_concat = get_concat_text_embeddings(
        transcripts=test_df["text"].tolist(),
        rationales=test_df["rationale_text"].tolist(),
        subject_ids=test_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix=f"test_transcript_rationale_{cache_tag}_concat",
    )
    x_train_triple = get_triple_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        demo_rationales=demo_texts,
        subject_ids=train_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix=f"train_transcript_rationale_demo_{cache_tag}_concat",
    )
    x_test_triple = get_triple_concat_text_embeddings(
        transcripts=test_df["text"].tolist(),
        rationales=test_df["rationale_text"].tolist(),
        demo_rationales=demo_texts_test,
        subject_ids=test_df["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix=f"test_transcript_rationale_demo_{cache_tag}_concat",
    )

    prefix = args.results_prefix
    results = []
    for name, x_tr, x_te in [
        (f"concat_{prefix}_rationale", x_train_concat, x_test_concat),
        (f"triple_concat_{prefix}_rationale", x_train_triple, x_test_triple),
    ]:
        cv_cls, cv_reg = cv_evaluate(
            x_tr, y_train_cls, y_train_reg, cls_folds, reg_folds
        )
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
                "rationale_suffix": args.rationale_suffix,
                "rationale_generator": args.rationale_generator,
                "demo_rationale_source": "template",
                "cv_classification": cv_cls,
                "cv_regression": cv_reg,
                "test_classification": test_cls,
                "test_regression": test_reg,
            }
        )

    out_path = results_dir / f"{prefix}_rationale_concat_results_{cache_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== {prefix} rationale concat evaluation ({cache_tag}) ===")
    for row in results:
        cv_cls = row["cv_classification"]
        cv_reg = row["cv_regression"]
        te_cls = row["test_classification"]
        te_reg = row["test_regression"]
        print(f"\n[{row['model']}]")
        print(
            f"  CV:   cls f1={cv_cls['f1']:.3f}, "
            f"reg rmse={cv_reg['rmse']:.3f}, r={cv_reg['pearson_r']:.3f}"
        )
        print(
            f"  Test: cls f1={te_cls['f1']:.3f}, "
            f"reg rmse={te_reg['rmse']:.3f}, r={te_reg['pearson_r']:.3f}"
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
