#!/usr/bin/env python3
"""Disagreement routing v3 with optional demographic features in meta-learner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
from demographics import DemographicFeatureEncoder, demographics_available_mask
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


def collect_oof_probabilities_embedding(
    features: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in folds:
        model = SVC(kernel="linear", probability=True)
        model.fit(features[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(features[val_idx])[:, 1]
    return oof


def collect_oof_probabilities_tfidf(
    texts: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tfidf_config: dict,
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in folds:
        pipeline = build_tfidf_pipeline(
            SVC(kernel="linear", probability=True),
            tfidf_config,
        )
        pipeline.fit([texts[i] for i in train_idx], y[train_idx])
        oof[val_idx] = pipeline.predict_proba([texts[i] for i in val_idx])[:, 1]
    return oof


def _stack_features(
    language_features: np.ndarray,
    demo_features: np.ndarray | None,
) -> np.ndarray:
    if demo_features is None:
        return language_features
    return np.hstack([language_features, demo_features])


def build_meta_matrix(
    emb_prob: np.ndarray,
    tfidf_prob: np.ndarray,
    concat_prob: np.ndarray,
    demo_features: np.ndarray | None,
) -> np.ndarray:
    blocks = [emb_prob, tfidf_prob, concat_prob]
    if demo_features is not None:
        blocks.append(demo_features)
    return np.column_stack(blocks)


def fit_probability_embedding(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = SVC(kernel="linear", probability=True)
    model.fit(x_train, y_train)
    pred = model.predict(x_eval).astype(int)
    prob = model.predict_proba(x_eval)[:, 1]
    return pred, prob


def fit_probability_tfidf(
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
    tfidf_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    pipeline = build_tfidf_pipeline(
        SVC(kernel="linear", probability=True),
        tfidf_config,
    )
    pipeline.fit(train_texts, y_train)
    pred = pipeline.predict(eval_texts).astype(int)
    prob = pipeline.predict_proba(eval_texts)[:, 1]
    return pred, prob


def route_with_meta_learner(
    pred_tfidf: np.ndarray,
    meta_pred: np.ndarray,
    disagree: np.ndarray,
) -> np.ndarray:
    routed = pred_tfidf.copy()
    routed[disagree] = meta_pred[disagree]
    return routed.astype(int)


def meta_coefficient_summary(
    meta_model: LogisticRegression,
    demo_encoder: DemographicFeatureEncoder | None,
) -> dict:
    names = ["embedding", "tfidf", "concat"]
    if demo_encoder is not None:
        names.extend(demo_encoder.feature_names())
    coefs = {
        name: float(meta_model.coef_[0][idx])
        for idx, name in enumerate(names)
    }
    coefs["intercept"] = float(meta_model.intercept_[0])
    return coefs


def analyze_routing_v3(
    y_true_cls: np.ndarray,
    pred_emb: np.ndarray,
    pred_tfidf: np.ndarray,
    pred_concat: np.ndarray,
    pred_meta: np.ndarray,
    pred_routed: np.ndarray,
    disagree: np.ndarray,
) -> dict:
    ok_emb = pred_emb == y_true_cls
    ok_tfidf = pred_tfidf == y_true_cls
    ok_concat = pred_concat == y_true_cls
    ok_meta = pred_meta == y_true_cls
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
        "meta_correct_on_disagree": int((ok_meta & disagree).sum()),
        "concat_correct_on_disagree": int((ok_concat & disagree).sum()),
        "tfidf_correct_on_disagree": int((ok_tfidf & disagree).sum()),
        "embedding_correct_on_disagree": int((ok_emb & disagree).sum()),
        "rescues_vs_tfidf_on_disagree": count_rescues(ok_tfidf, ok_routed, disagree),
        "rescues_vs_concat_on_disagree": count_rescues(ok_concat, ok_routed, disagree),
        "harmed_vs_tfidf_on_disagree": count_rescues(ok_routed, ok_tfidf, disagree),
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
            "Disagreement routing v3: when embedding != TF-IDF, arbitrate "
            "with a logistic meta-learner trained on OOF probabilities."
        )
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="Disable demographic features in meta-learner and regression",
    )
    parser.add_argument(
        "--route-regression",
        action="store_true",
        help="Use concat+demographics Ridge for MMSE instead of TF-IDF only",
    )
    args = parser.parse_args()
    use_demographics = not args.no_demographics

    config = load_config()
    embedding_config = build_embedding_config(config)
    results_dir = PROJECT_ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, reg_folds, _ = load_or_create_folds()

    train_df = load_train_data(with_demographics=use_demographics)
    test_df = load_test_data(with_demographics=use_demographics)
    train_transcripts, y_train_cls, y_train_reg, train_subject_ids = get_xy(train_df)
    test_transcripts, y_test_cls, y_test_reg, test_subject_ids = get_xy(test_df)

    train_df_r = load_train_data_with_rationale(with_demographics=use_demographics)
    test_df_r = load_test_data_with_rationale(with_demographics=use_demographics)
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
        y_test_reg_r,
        test_subject_ids_r,
    ) = get_xy_with_rationale(test_df_r)

    demo_encoder = None
    demo_train = demo_test = None
    if use_demographics:
        demo_encoder = DemographicFeatureEncoder.fit(train_df)
        demo_train = demo_encoder.transform(train_df)
        demo_test = demo_encoder.transform(test_df)
        train_demo_cov = int(demographics_available_mask(train_df).sum())
        test_demo_cov = int(demographics_available_mask(test_df).sum())
        print(
            "Demographics coverage: "
            f"train={train_demo_cov}/{len(train_df)}, "
            f"test={test_demo_cov}/{len(test_df)} "
            f"(missing test rows imputed from train)"
        )

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

    print("Collecting train OOF probabilities for meta-learner...")
    oof_emb_prob = collect_oof_probabilities_embedding(
        x_train_emb, y_train_cls, cls_folds
    )
    oof_tfidf_prob = collect_oof_probabilities_tfidf(
        train_transcripts, y_train_cls, cls_folds, config["tfidf"]
    )
    oof_concat_prob = collect_oof_probabilities_embedding(
        x_train_concat, y_train_cls_r, cls_folds
    )
    x_meta_train = build_meta_matrix(
        oof_emb_prob, oof_tfidf_prob, oof_concat_prob, demo_train
    )
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(x_meta_train, y_train_cls)
    meta_train_metrics = classification_metrics(
        y_train_cls, meta_model.predict(x_meta_train)
    )

    print("Fitting full train models for test prediction...")
    pred_emb_cls, test_emb_prob = fit_probability_embedding(
        x_train_emb, y_train_cls, x_test_emb
    )
    pred_tfidf_cls, test_tfidf_prob = fit_probability_tfidf(
        train_transcripts,
        y_train_cls,
        test_transcripts,
        config["tfidf"],
    )
    pred_concat_cls, test_concat_prob = fit_probability_embedding(
        x_train_concat, y_train_cls_r, x_test_concat
    )

    x_meta_test = build_meta_matrix(
        test_emb_prob, test_tfidf_prob, test_concat_prob, demo_test
    )
    pred_meta_cls = meta_model.predict(x_meta_test).astype(int)
    pred_meta_prob = meta_model.predict_proba(x_meta_test)[:, 1]

    disagree = pred_emb_cls != pred_tfidf_cls
    pred_routed_cls = route_with_meta_learner(
        pred_tfidf_cls,
        pred_meta_cls,
        disagree,
    )

    pred_tfidf_reg = fit_and_predict_tfidf(
        train_transcripts,
        y_train_reg,
        test_transcripts,
        "regression",
        "ridge",
        config["tfidf"],
    )
    x_train_reg = _stack_features(x_train_concat, demo_train if use_demographics else None)
    x_test_reg = _stack_features(x_test_concat, demo_test if use_demographics else None)
    pred_concat_reg = fit_and_predict(
        x_train_reg, y_train_reg_r, x_test_reg, "regression", "ridge"
    )

    if args.route_regression and use_demographics:
        pred_routed_reg = pred_concat_reg
    else:
        pred_routed_reg = pred_tfidf_reg

    cls_metrics = classification_metrics(y_test_cls, pred_routed_cls)
    reg_metrics = regression_metrics(y_test_reg, pred_routed_reg)
    routing_stats = analyze_routing_v3(
        y_test_cls,
        pred_emb_cls,
        pred_tfidf_cls,
        pred_concat_cls,
        pred_meta_cls,
        pred_routed_cls,
        disagree,
    )

    baseline_cls = {
        "embedding": classification_metrics(y_test_cls, pred_emb_cls),
        "tfidf": classification_metrics(y_test_cls, pred_tfidf_cls),
        "concat": classification_metrics(y_test_cls, pred_concat_cls),
        "meta_all_samples": classification_metrics(y_test_cls, pred_meta_cls),
    }
    baseline_reg = {
        "tfidf": regression_metrics(y_test_reg, pred_tfidf_reg),
        "concat_plus_demographics": regression_metrics(y_test_reg, pred_concat_reg),
    }

    pred_df = pd.DataFrame(
        {
            "subject_id": test_subject_ids,
            "true_label": y_test_cls,
            "pred_label_embedding": pred_emb_cls,
            "pred_label_tfidf": pred_tfidf_cls,
            "pred_label_concat": pred_concat_cls,
            "pred_label_meta": pred_meta_cls,
            "pred_label_routed": pred_routed_cls,
            "disagree": disagree.astype(int),
            "prob_embedding": test_emb_prob,
            "prob_tfidf": test_tfidf_prob,
            "prob_concat": test_concat_prob,
            "prob_meta": pred_meta_prob,
            "has_demographics": demographics_available_mask(test_df).astype(int)
            if use_demographics
            else 0,
            "true_mmse": y_test_reg,
            "pred_mmse_tfidf": pred_tfidf_reg,
            "pred_mmse_concat_demo": pred_concat_reg,
            "pred_mmse_routed": pred_routed_reg,
        }
    )
    if use_demographics:
        pred_df["age"] = test_df["age"].values
        pred_df["gender"] = test_df["gender"].values
        pred_df["educ"] = test_df["educ"].values

    meta_features = [
        "embedding_probability",
        "tfidf_probability",
        "concat_probability",
    ]
    if demo_encoder is not None:
        meta_features.extend(demo_encoder.feature_names())

    summary = {
        "method": "disagreement_routing_v3",
        "use_demographics": use_demographics,
        "rule": (
            "if embedding_cls != tfidf_cls -> use meta-learner; "
            "else -> use shared prediction"
        ),
        "route_regression": args.route_regression and use_demographics,
        "regression_model_when_routed": (
            "concat_plus_demographics"
            if args.route_regression and use_demographics
            else "tfidf"
        ),
        "meta_learner": "logistic_regression",
        "meta_features": meta_features,
        "meta_coefficients": meta_coefficient_summary(meta_model, demo_encoder),
        "meta_train_fit_accuracy": meta_train_metrics["accuracy"],
        "demographics_coverage": {
            "train_observed": int(demographics_available_mask(train_df).sum())
            if use_demographics
            else None,
            "test_observed": int(demographics_available_mask(test_df).sum())
            if use_demographics
            else None,
            "test_imputed": int((~demographics_available_mask(test_df)).sum())
            if use_demographics
            else None,
        },
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

    suffix = "_demo" if use_demographics else ""
    out_json = results_dir / f"test_disagreement_routing_v3{suffix}.json"
    out_csv = results_dir / f"test_predictions_disagreement_routing_v3{suffix}.csv"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    pred_df.to_csv(out_csv, index=False)

    print("=== Disagreement Routing v3 (test set) ===")
    print(f"Demographics enabled: {use_demographics}")
    print("Meta coefficients:", summary["meta_coefficients"])
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
    print_block("baseline/tfidf", baseline_cls["tfidf"], baseline_reg["tfidf"])
    if use_demographics:
        print_block(
            "baseline/concat_plus_demographics",
            baseline_cls["concat"],
            baseline_reg["concat_plus_demographics"],
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
