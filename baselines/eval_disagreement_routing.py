#!/usr/bin/env python3
"""Disagreement routing: use rationale-assisted model when embedding != TF-IDF."""

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
from features import get_concat_text_embeddings, get_text_embeddings
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


def route_classification(
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_assist: np.ndarray,
    *,
    default_source: str = "tfidf",
) -> tuple[np.ndarray, np.ndarray]:
    disagree = pred_emb != pred_tfidf
    if default_source == "tfidf":
        default_pred = pred_tfidf
    elif default_source == "embedding":
        default_pred = pred_emb
    else:
        raise ValueError(f"Unknown default_source: {default_source}")

    routed = np.where(disagree, pred_assist, default_pred)
    return routed.astype(int), disagree


def route_regression(
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_assist: np.ndarray,
    disagree: np.ndarray,
    *,
    default_source: str = "tfidf",
) -> np.ndarray:
    if default_source == "tfidf":
        default_pred = pred_tfidf
    elif default_source == "embedding":
        default_pred = pred_emb
    else:
        raise ValueError(f"Unknown default_source: {default_source}")

    return np.where(disagree, pred_assist, default_pred)


def analyze_routing(
    y_true_cls: np.ndarray,
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_assist: np.ndarray,
    pred_routed: np.ndarray,
    disagree: np.ndarray,
) -> dict:
    ok_emb = pred_emb == y_true_cls
    ok_tfidf = pred_tfidf == y_true_cls
    ok_assist = pred_assist == y_true_cls
    ok_routed = pred_routed == y_true_cls

    def count_rescues(base_ok: np.ndarray, routed_ok: np.ndarray, mask: np.ndarray) -> int:
        return int((~base_ok & routed_ok & mask).sum())

    return {
        "n_total": int(len(y_true_cls)),
        "n_agree": int((~disagree).sum()),
        "n_disagree": int(disagree.sum()),
        "accuracy_on_agree_subset": float((pred_routed[~disagree] == y_true_cls[~disagree]).mean())
        if (~disagree).any()
        else None,
        "accuracy_on_disagree_subset": float((pred_routed[disagree] == y_true_cls[disagree]).mean())
        if disagree.any()
        else None,
        "rescues_vs_tfidf_on_disagree": count_rescues(ok_tfidf, ok_routed, disagree),
        "rescues_vs_embedding_on_disagree": count_rescues(ok_emb, ok_routed, disagree),
        "harmed_vs_tfidf_on_disagree": count_rescues(ok_routed, ok_tfidf, disagree),
        "assist_correct_on_disagree": int((ok_assist & disagree).sum()),
        "tfidf_correct_on_disagree": int((ok_tfidf & disagree).sum()),
        "embedding_correct_on_disagree": int((ok_emb & disagree).sum()),
        "both_wrong_on_disagree": int((~ok_tfidf & ~ok_emb & disagree).sum()),
    }


def print_block(title: str, cls_metrics: dict, reg_metrics: dict, extra: dict | None = None) -> None:
    print(f"\n[{title}]")
    print(
        f"  classification: acc={cls_metrics['accuracy']:.3f}, f1={cls_metrics['f1']:.3f}"
    )
    print(
        f"  regression: rmse={reg_metrics['rmse']:.3f}, "
        f"mae={reg_metrics['mae']:.3f}, r={reg_metrics['pearson_r']:.3f}"
    )
    if extra:
        print(
            f"  routing: agree={extra['n_agree']}, disagree={extra['n_disagree']}, "
            f"rescues_vs_tfidf={extra['rescues_vs_tfidf_on_disagree']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Route to rationale-assisted predictions when transcript embedding "
            "and TF-IDF disagree."
        )
    )
    parser.add_argument(
        "--assist-source",
        choices=["concat"],
        default="concat",
        help="Rationale-assisted model used on disagreement cases",
    )
    parser.add_argument(
        "--default-source",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="Model used when embedding and TF-IDF agree",
    )
    parser.add_argument(
        "--route-regression",
        action="store_true",
        help="Also route MMSE on disagreement; default keeps TF-IDF regression",
    )
    args = parser.parse_args()

    config = load_config()
    embedding_config = build_embedding_config(config)
    results_dir = PROJECT_ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_data()
    test_df = load_test_data()
    train_transcripts, y_train_cls, y_train_reg, train_subject_ids = get_xy(train_df)
    test_transcripts, y_test_cls, y_test_reg, test_subject_ids = get_xy(test_df)

    print("Loading transcript embeddings...")
    x_train_emb = get_text_embeddings(
        texts=train_transcripts,
        subject_ids=train_subject_ids,
        embedding_config=embedding_config,
        project_root=PROJECT_ROOT,
        cache_suffix="train",
    )
    x_test_emb = get_text_embeddings(
        texts=test_transcripts,
        subject_ids=test_subject_ids,
        embedding_config=embedding_config,
        project_root=PROJECT_ROOT,
        cache_suffix="test",
    )

    print("Fitting transcript embedding models...")
    pred_emb_cls = fit_and_predict(
        x_train_emb, y_train_cls, x_test_emb, "classification", "svc"
    ).astype(int)
    pred_emb_reg = fit_and_predict(
        x_train_emb, y_train_reg, x_test_emb, "regression", "ridge"
    )

    print("Fitting transcript TF-IDF models...")
    pred_tfidf_cls = fit_and_predict_tfidf(
        train_transcripts,
        y_train_cls,
        test_transcripts,
        "classification",
        "svc",
        config["tfidf"],
    ).astype(int)
    pred_tfidf_reg = fit_and_predict_tfidf(
        train_transcripts,
        y_train_reg,
        test_transcripts,
        "regression",
        "ridge",
        config["tfidf"],
    )

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

    if args.assist_source == "concat":
        print("Loading transcript+rationale concat embeddings for assist model...")
        x_train_assist = get_concat_text_embeddings(
            transcripts=train_transcripts_r,
            rationales=train_rationales,
            subject_ids=train_subject_ids_r,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="train_transcript_rationale_concat",
        )
        x_test_assist = get_concat_text_embeddings(
            transcripts=test_transcripts_r,
            rationales=test_rationales,
            subject_ids=test_subject_ids_r,
            embedding_config=embedding_config,
            project_root=PROJECT_ROOT,
            cache_suffix="test_transcript_rationale_concat",
        )
        pred_assist_cls = fit_and_predict(
            x_train_assist, y_train_cls_r, x_test_assist, "classification", "svc"
        ).astype(int)
        pred_assist_reg = fit_and_predict(
            x_train_assist, y_train_reg_r, x_test_assist, "regression", "ridge"
        )

    pred_routed_cls, disagree = route_classification(
        pred_emb_cls,
        pred_tfidf_cls,
        pred_assist_cls,
        default_source=args.default_source,
    )
    if args.route_regression:
        pred_routed_reg = route_regression(
            pred_emb_reg,
            pred_tfidf_reg,
            pred_assist_reg,
            disagree,
            default_source=args.default_source,
        )
    else:
        pred_routed_reg = pred_tfidf_reg

    cls_metrics = classification_metrics(y_test_cls, pred_routed_cls)
    reg_metrics = regression_metrics(y_test_reg, pred_routed_reg)
    routing_stats = analyze_routing(
        y_test_cls,
        pred_emb_cls,
        pred_tfidf_cls,
        pred_assist_cls,
        pred_routed_cls,
        disagree,
    )

    baseline_cls = {
        "transcript_embedding": classification_metrics(y_test_cls, pred_emb_cls),
        "transcript_tfidf": classification_metrics(y_test_cls, pred_tfidf_cls),
        "transcript_rationale_concat": classification_metrics(y_test_cls, pred_assist_cls),
    }
    baseline_reg = {
        "transcript_embedding": regression_metrics(y_test_reg, pred_emb_reg),
        "transcript_tfidf": regression_metrics(y_test_reg, pred_tfidf_reg),
        "transcript_rationale_concat": regression_metrics(y_test_reg, pred_assist_reg),
    }

    pred_df = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label_embedding": pred_emb_cls,
            "pred_label_tfidf": pred_tfidf_cls,
            "pred_label_assist": pred_assist_cls,
            "pred_label_routed": pred_routed_cls,
            "disagree": disagree.astype(int),
            "true_mmse": y_test_reg,
            "pred_mmse_embedding": pred_emb_reg,
            "pred_mmse_tfidf": pred_tfidf_reg,
            "pred_mmse_assist": pred_assist_reg,
            "pred_mmse_routed": pred_routed_reg,
        }
    )

    summary = {
        "method": "disagreement_routing",
        "rule": (
            "if embedding_cls != tfidf_cls -> use assist model "
            f"({args.assist_source}); else -> use {args.default_source}"
        ),
        "route_regression": args.route_regression,
        "assist_source": args.assist_source,
        "default_source": args.default_source,
        "embedding_model": embedding_config["model_name"],
        "classifier": "svc",
        "regressor": "ridge",
        "train_size": int(len(y_train_cls)),
        "test_size": int(len(y_test_cls)),
        "classification": cls_metrics,
        "regression": reg_metrics,
        "routing": routing_stats,
        "baselines": {
            "classification": baseline_cls,
            "regression": baseline_reg,
        },
    }

    out_json = results_dir / "test_disagreement_routing.json"
    out_csv = results_dir / "test_predictions_disagreement_routing.csv"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    pred_df.to_csv(out_csv, index=False)

    print("=== Disagreement Routing (test set) ===")
    print_block("routed", cls_metrics, reg_metrics, routing_stats)
    print_block(
        "baseline/embedding",
        baseline_cls["transcript_embedding"],
        baseline_reg["transcript_embedding"],
    )
    print_block(
        "baseline/tfidf",
        baseline_cls["transcript_tfidf"],
        baseline_reg["transcript_tfidf"],
    )
    print_block(
        "baseline/assist_concat",
        baseline_cls["transcript_rationale_concat"],
        baseline_reg["transcript_rationale_concat"],
    )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
