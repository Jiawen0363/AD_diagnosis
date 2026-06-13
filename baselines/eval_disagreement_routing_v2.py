#!/usr/bin/env python3
"""Disagreement routing v2: on disagreement, use OOF-best of tfidf/embedding/concat."""

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

from cv import load_or_create_folds
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


MODEL_NAMES = ("embedding", "tfidf", "concat")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_embedding_config(config: dict) -> dict:
    return {
        **config["embedding"],
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    }


def collect_oof_classification_embedding(
    features: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.int64)
    for train_idx, val_idx in folds:
        pred = fit_and_predict(
            features[train_idx],
            y[train_idx],
            features[val_idx],
            "classification",
            "svc",
        )
        oof[val_idx] = pred.astype(int)
    return oof


def collect_oof_regression_embedding(
    features: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in folds:
        pred = fit_and_predict(
            features[train_idx],
            y[train_idx],
            features[val_idx],
            "regression",
            "ridge",
        )
        oof[val_idx] = pred
    return oof


def collect_oof_classification_tfidf(
    texts: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tfidf_config: dict,
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.int64)
    for train_idx, val_idx in folds:
        pred = fit_and_predict_tfidf(
            [texts[i] for i in train_idx],
            y[train_idx],
            [texts[i] for i in val_idx],
            "classification",
            "svc",
            tfidf_config,
        )
        oof[val_idx] = pred.astype(int)
    return oof


def collect_oof_regression_tfidf(
    texts: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tfidf_config: dict,
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in folds:
        pred = fit_and_predict_tfidf(
            [texts[i] for i in train_idx],
            y[train_idx],
            [texts[i] for i in val_idx],
            "regression",
            "ridge",
            tfidf_config,
        )
        oof[val_idx] = pred
    return oof


def pick_best_classification_model(
    oof_preds: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    tiebreaker: str = "tfidf",
) -> tuple[str, dict[str, dict[str, float]]]:
    scores = {
        name: classification_metrics(y, pred)
        for name, pred in oof_preds.items()
    }
    best_f1 = max(scores[name]["f1"] for name in MODEL_NAMES)
    candidates = [name for name in MODEL_NAMES if scores[name]["f1"] == best_f1]
    best = tiebreaker if tiebreaker in candidates else candidates[0]
    return best, scores


def pick_best_regression_model(
    oof_preds: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    tiebreaker: str = "tfidf",
) -> tuple[str, dict[str, dict[str, float]]]:
    scores = {
        name: regression_metrics(y, pred)
        for name, pred in oof_preds.items()
    }
    best_rmse = min(scores[name]["rmse"] for name in MODEL_NAMES)
    candidates = [name for name in MODEL_NAMES if scores[name]["rmse"] == best_rmse]
    best = tiebreaker if tiebreaker in candidates else candidates[0]
    return best, scores


def route_with_oof_winner(
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_concat: np.ndarray,
    disagree: np.ndarray,
    winner: str,
    *,
    default_source: str = "tfidf",
) -> tuple[np.ndarray, np.ndarray]:
    pred_by_model = {
        "embedding": pred_emb,
        "tfidf": pred_tfidf,
        "concat": pred_concat,
    }
    default_pred = pred_by_model[default_source]
    winner_pred = pred_by_model[winner]
    routed = np.where(disagree, winner_pred, default_pred)
    chosen = np.where(disagree, winner, default_source)
    return routed, chosen


def analyze_routing_v2(
    y_true_cls: np.ndarray,
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_concat: np.ndarray,
    pred_routed: np.ndarray,
    disagree: np.ndarray,
    chosen_model: np.ndarray,
    winner_cls: str,
) -> dict:
    ok_emb = pred_emb == y_true_cls
    ok_tfidf = pred_tfidf == y_true_cls
    ok_concat = pred_concat == y_true_cls
    ok_routed = pred_routed == y_true_cls

    def count_rescues(base_ok: np.ndarray, routed_ok: np.ndarray, mask: np.ndarray) -> int:
        return int((~base_ok & routed_ok & mask).sum())

    disagree_choices = chosen_model[disagree]
    choice_counts = {
        name: int((disagree_choices == name).sum()) for name in MODEL_NAMES
    }

    return {
        "n_total": int(len(y_true_cls)),
        "n_agree": int((~disagree).sum()),
        "n_disagree": int(disagree.sum()),
        "oof_winner_cls": winner_cls,
        "disagree_model_choice_counts": choice_counts,
        "accuracy_on_agree_subset": float((pred_routed[~disagree] == y_true_cls[~disagree]).mean())
        if (~disagree).any()
        else None,
        "accuracy_on_disagree_subset": float((pred_routed[disagree] == y_true_cls[disagree]).mean())
        if disagree.any()
        else None,
        "rescues_vs_tfidf_on_disagree": count_rescues(ok_tfidf, ok_routed, disagree),
        "rescues_vs_embedding_on_disagree": count_rescues(ok_emb, ok_routed, disagree),
        "harmed_vs_tfidf_on_disagree": count_rescues(ok_routed, ok_tfidf, disagree),
        "concat_correct_on_disagree": int((ok_concat & disagree).sum()),
        "tfidf_correct_on_disagree": int((ok_tfidf & disagree).sum()),
        "embedding_correct_on_disagree": int((ok_emb & disagree).sum()),
    }


def print_block(title: str, cls_metrics: dict, reg_metrics: dict, extra: str | None = None) -> None:
    print(f"\n[{title}]")
    print(
        f"  classification: acc={cls_metrics['accuracy']:.3f}, f1={cls_metrics['f1']:.3f}"
    )
    print(
        f"  regression: rmse={reg_metrics['rmse']:.3f}, "
        f"mae={reg_metrics['mae']:.3f}, r={reg_metrics['pearson_r']:.3f}"
    )
    if extra:
        print(f"  {extra}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Disagreement routing v2: when embedding != TF-IDF, use the model "
            "with best train OOF among embedding/tfidf/concat."
        )
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

    cls_folds, reg_folds, _ = load_or_create_folds()

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
        _y_test_cls_r,
        _y_test_reg_r,
        test_subject_ids_r,
    ) = get_xy_with_rationale(test_df_r)

    print("Loading embeddings...")
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
    x_train_concat = get_concat_text_embeddings(
        transcripts=train_transcripts_r,
        rationales=train_rationales,
        subject_ids=train_subject_ids_r,
        embedding_config=embedding_config,
        project_root=PROJECT_ROOT,
        cache_suffix="train_transcript_rationale_concat",
    )
    x_test_concat = get_concat_text_embeddings(
        transcripts=test_transcripts_r,
        rationales=test_rationales,
        subject_ids=test_subject_ids_r,
        embedding_config=embedding_config,
        project_root=PROJECT_ROOT,
        cache_suffix="test_transcript_rationale_concat",
    )

    print("Collecting train OOF predictions...")
    oof_cls = {
        "embedding": collect_oof_classification_embedding(
            x_train_emb, y_train_cls, cls_folds
        ),
        "tfidf": collect_oof_classification_tfidf(
            train_transcripts, y_train_cls, cls_folds, config["tfidf"]
        ),
        "concat": collect_oof_classification_embedding(
            x_train_concat, y_train_cls_r, cls_folds
        ),
    }
    oof_reg = {
        "embedding": collect_oof_regression_embedding(
            x_train_emb, y_train_reg, reg_folds
        ),
        "tfidf": collect_oof_regression_tfidf(
            train_transcripts, y_train_reg, reg_folds, config["tfidf"]
        ),
        "concat": collect_oof_regression_embedding(
            x_train_concat, y_train_reg_r, reg_folds
        ),
    }

    winner_cls, oof_cls_scores = pick_best_classification_model(
        oof_cls, y_train_cls, tiebreaker=args.default_source
    )
    winner_reg, oof_reg_scores = pick_best_regression_model(
        oof_reg, y_train_reg, tiebreaker=args.default_source
    )

    print("Fitting full train models for test prediction...")
    pred_emb_cls = fit_and_predict(
        x_train_emb, y_train_cls, x_test_emb, "classification", "svc"
    ).astype(int)
    pred_tfidf_cls = fit_and_predict_tfidf(
        train_transcripts,
        y_train_cls,
        test_transcripts,
        "classification",
        "svc",
        config["tfidf"],
    ).astype(int)
    pred_concat_cls = fit_and_predict(
        x_train_concat, y_train_cls_r, x_test_concat, "classification", "svc"
    ).astype(int)

    pred_emb_reg = fit_and_predict(
        x_train_emb, y_train_reg, x_test_emb, "regression", "ridge"
    )
    pred_tfidf_reg = fit_and_predict_tfidf(
        train_transcripts,
        y_train_reg,
        test_transcripts,
        "regression",
        "ridge",
        config["tfidf"],
    )
    pred_concat_reg = fit_and_predict(
        x_train_concat, y_train_reg_r, x_test_concat, "regression", "ridge"
    )

    disagree = pred_emb_cls != pred_tfidf_cls
    pred_routed_cls, chosen_cls = route_with_oof_winner(
        pred_emb_cls,
        pred_tfidf_cls,
        pred_concat_cls,
        disagree,
        winner_cls,
        default_source=args.default_source,
    )
    pred_routed_cls = pred_routed_cls.astype(int)

    if args.route_regression:
        pred_routed_reg, chosen_reg = route_with_oof_winner(
            pred_emb_reg,
            pred_tfidf_reg,
            pred_concat_reg,
            disagree,
            winner_reg,
            default_source=args.default_source,
        )
    else:
        pred_routed_reg = pred_tfidf_reg
        chosen_reg = np.full(len(y_test_cls), args.default_source, dtype=object)

    cls_metrics = classification_metrics(y_test_cls, pred_routed_cls)
    reg_metrics = regression_metrics(y_test_reg, pred_routed_reg)
    routing_stats = analyze_routing_v2(
        y_test_cls,
        pred_emb_cls,
        pred_tfidf_cls,
        pred_concat_cls,
        pred_routed_cls,
        disagree,
        chosen_cls,
        winner_cls,
    )

    baseline_cls = {
        name: classification_metrics(y_test_cls, pred)
        for name, pred in {
            "embedding": pred_emb_cls,
            "tfidf": pred_tfidf_cls,
            "concat": pred_concat_cls,
        }.items()
    }
    baseline_reg = {
        name: regression_metrics(y_test_reg, pred)
        for name, pred in {
            "embedding": pred_emb_reg,
            "tfidf": pred_tfidf_reg,
            "concat": pred_concat_reg,
        }.items()
    }

    pred_df = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label_embedding": pred_emb_cls,
            "pred_label_tfidf": pred_tfidf_cls,
            "pred_label_concat": pred_concat_cls,
            "pred_label_routed": pred_routed_cls,
            "chosen_model_cls": chosen_cls,
            "disagree": disagree.astype(int),
            "true_mmse": y_test_reg,
            "pred_mmse_embedding": pred_emb_reg,
            "pred_mmse_tfidf": pred_tfidf_reg,
            "pred_mmse_concat": pred_concat_reg,
            "pred_mmse_routed": pred_routed_reg,
            "chosen_model_reg": chosen_reg,
        }
    )

    summary = {
        "method": "disagreement_routing_v2",
        "rule": (
            "if embedding_cls != tfidf_cls -> use train-OOF-best model "
            f"({winner_cls} for cls"
            + (f", {winner_reg} for reg" if args.route_regression else "")
            + f"); else -> use {args.default_source}"
        ),
        "route_regression": args.route_regression,
        "default_source": args.default_source,
        "oof_winner_cls": winner_cls,
        "oof_winner_reg": winner_reg,
        "oof_train_classification": oof_cls_scores,
        "oof_train_regression": oof_reg_scores,
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

    out_json = results_dir / "test_disagreement_routing_v2.json"
    out_csv = results_dir / "test_predictions_disagreement_routing_v2.csv"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    pred_df.to_csv(out_csv, index=False)

    print("=== Disagreement Routing v2 (test set) ===")
    print(f"Train OOF winner (cls): {winner_cls}")
    print(f"Train OOF winner (reg): {winner_reg}")
    for name in MODEL_NAMES:
        s = oof_cls_scores[name]
        print(f"  OOF cls/{name}: f1={s['f1']:.3f}, acc={s['accuracy']:.3f}")
    print_block(
        "routed",
        cls_metrics,
        reg_metrics,
        (
            f"routing: agree={routing_stats['n_agree']}, "
            f"disagree={routing_stats['n_disagree']}, "
            f"rescues_vs_tfidf={routing_stats['rescues_vs_tfidf_on_disagree']}"
        ),
    )
    for name in MODEL_NAMES:
        print_block(
            f"baseline/{name}",
            baseline_cls[name],
            baseline_reg[name],
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
