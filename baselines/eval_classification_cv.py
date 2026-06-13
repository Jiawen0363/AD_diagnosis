#!/usr/bin/env python3
"""Unified train CV evaluation for AD/Control classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from cv import load_or_create_folds
from data import load_train_data, load_train_data_with_rationale
from demographics import DemographicFeatureEncoder, build_demo_rationales
from eval_disagreement_routing_v3 import (
    build_embedding_config,
    build_meta_matrix,
    collect_oof_probabilities_embedding,
    collect_oof_probabilities_tfidf,
    route_with_meta_learner,
)
from features import (
    get_concat_text_embeddings,
    get_text_embeddings,
    get_triple_concat_text_embeddings,
)
from metrics import classification_metrics
from train import run_fold, run_vector_fold


def stack_tabular(x: np.ndarray, demo: np.ndarray | None) -> np.ndarray:
    if demo is None:
        return x
    return np.hstack([x, demo])


def mean_fold_metrics(rows: list[dict]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def evaluate_vector_cv(
    features: np.ndarray,
    y_cls: np.ndarray,
    cls_folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str = "svc",
) -> dict[str, float]:
    rows = [
        run_vector_fold(features, y_cls, tr, va, "classification", model_name)
        for tr, va in cls_folds
    ]
    return mean_fold_metrics(rows)


def evaluate_tfidf_cv(
    texts: list[str],
    y_cls: np.ndarray,
    cls_folds: list[tuple[np.ndarray, np.ndarray]],
    tfidf_config: dict,
    model_name: str = "svc",
) -> dict[str, float]:
    rows = [
        run_fold(
            texts=texts,
            y=y_cls,
            train_idx=tr,
            val_idx=va,
            task="classification",
            model_name=model_name,
            tfidf_config=tfidf_config,
        )
        for tr, va in cls_folds
    ]
    return mean_fold_metrics(rows)


def evaluate_v3_routing(
    y_cls: np.ndarray,
    oof_emb_prob: np.ndarray,
    oof_tfidf_prob: np.ndarray,
    oof_concat_prob: np.ndarray,
    demo_tab: np.ndarray | None,
) -> dict:
    oof_emb_pred = (oof_emb_prob >= 0.5).astype(int)
    oof_tfidf_pred = (oof_tfidf_prob >= 0.5).astype(int)
    disagree = oof_emb_pred != oof_tfidf_pred

    x_meta = build_meta_matrix(oof_emb_prob, oof_tfidf_prob, oof_concat_prob, demo_tab)
    meta = LogisticRegression(max_iter=1000).fit(x_meta, y_cls)
    meta_pred = meta.predict(x_meta).astype(int)
    routed = route_with_meta_learner(oof_tfidf_pred, meta_pred, disagree)
    cls_metrics = classification_metrics(y_cls, routed)

    return {
        "classification": cls_metrics,
        "routing": {
            "n_disagree": int(disagree.sum()),
            "n_agree": int((~disagree).sum()),
            "disagree_accuracy": float((routed[disagree] == y_cls[disagree]).mean())
            if disagree.any()
            else None,
            "agree_accuracy": float((routed[~disagree] == y_cls[~disagree]).mean())
            if (~disagree).any()
            else None,
        },
    }


def append_single_model(
    results: list[dict],
    *,
    model: str,
    feature_type: str,
    demographics: str,
    classifier: str,
    metrics: dict[str, float],
) -> None:
    results.append(
        {
            "model": model,
            "feature_type": feature_type,
            "demographics": demographics,
            "classifier": classifier,
            "evaluation": "5fold_cv",
            **metrics,
        }
    )


def main() -> None:
    with open(SCRIPT_DIR.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    embedding_config = build_embedding_config(config)
    results_dir = SCRIPT_DIR.parent / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, _, subject_ids = load_or_create_folds()
    train_df = load_train_data(with_demographics=True)
    train_df_r = load_train_data_with_rationale(with_demographics=True)
    y_cls = train_df["label"].astype(int).to_numpy()
    texts = train_df["text"].tolist()

    demo_encoder = DemographicFeatureEncoder.fit(train_df)
    demo_tab = demo_encoder.transform(train_df)
    demo_texts = build_demo_rationales(train_df)

    print("Loading embeddings...")
    x_emb = get_text_embeddings(
        texts=texts,
        subject_ids=subject_ids,
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train",
    )
    x_concat = get_concat_text_embeddings(
        transcripts=train_df_r["text"].tolist(),
        rationales=train_df_r["rationale_text"].tolist(),
        subject_ids=train_df_r["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_concat",
    )
    x_triple = get_triple_concat_text_embeddings(
        transcripts=train_df_r["text"].tolist(),
        rationales=train_df_r["rationale_text"].tolist(),
        demo_rationales=demo_texts,
        subject_ids=train_df_r["subject_id"].tolist(),
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_demo_concat",
    )

    results: list[dict] = []
    classifier = "svc_linear"

    print("Evaluating single-model baselines...")
    append_single_model(
        results,
        model="tfidf",
        feature_type="tfidf_transcript",
        demographics="none",
        classifier=classifier,
        metrics=evaluate_tfidf_cv(texts, y_cls, cls_folds, config["tfidf"]),
    )
    append_single_model(
        results,
        model="embedding",
        feature_type="embedding_transcript",
        demographics="none",
        classifier=classifier,
        metrics=evaluate_vector_cv(x_emb, y_cls, cls_folds),
    )
    append_single_model(
        results,
        model="embedding+tabular",
        feature_type="embedding_transcript",
        demographics="tabular",
        classifier=classifier,
        metrics=evaluate_vector_cv(stack_tabular(x_emb, demo_tab), y_cls, cls_folds),
    )
    append_single_model(
        results,
        model="concat",
        feature_type="concat_transcript_rationale",
        demographics="none",
        classifier=classifier,
        metrics=evaluate_vector_cv(x_concat, y_cls, cls_folds),
    )
    append_single_model(
        results,
        model="concat+tabular",
        feature_type="concat_transcript_rationale",
        demographics="tabular",
        classifier=classifier,
        metrics=evaluate_vector_cv(stack_tabular(x_concat, demo_tab), y_cls, cls_folds),
    )
    append_single_model(
        results,
        model="triple_concat",
        feature_type="concat_transcript_rationale_demo_rationale",
        demographics="template_demo_rationale",
        classifier=classifier,
        metrics=evaluate_vector_cv(x_triple, y_cls, cls_folds),
    )
    append_single_model(
        results,
        model="triple_concat+tabular",
        feature_type="concat_transcript_rationale_demo_rationale",
        demographics="template_demo_rationale+tabular",
        classifier=classifier,
        metrics=evaluate_vector_cv(stack_tabular(x_triple, demo_tab), y_cls, cls_folds),
    )

    print("Evaluating v3 disagreement routing...")
    oof_emb_prob = collect_oof_probabilities_embedding(x_emb, y_cls, cls_folds)
    oof_tfidf_prob = collect_oof_probabilities_tfidf(
        texts, y_cls, cls_folds, config["tfidf"]
    )
    oof_concat_prob = collect_oof_probabilities_embedding(x_concat, y_cls, cls_folds)

    for use_demo, demo_label in ((False, "none"), (True, "tabular_meta")):
        v3 = evaluate_v3_routing(
            y_cls,
            oof_emb_prob,
            oof_tfidf_prob,
            oof_concat_prob,
            demo_tab if use_demo else None,
        )
        row = {
            "model": "v3_routing",
            "feature_type": "routing_emb_tfidf_concat",
            "demographics": demo_label,
            "classifier": "logistic_regression_meta",
            "evaluation": "oof_probabilities",
            **v3["classification"],
            "routing": v3["routing"],
        }
        results.append(row)

    payload = {
        "protocol": {
            "task": "classification",
            "dataset": "train",
            "n_samples": len(train_df),
            "n_folds": len(cls_folds),
            "folds_file": str(results_dir / "folds.json"),
            "default_classifier": classifier,
            "embedding_model": embedding_config["model_name"],
            "demo_rationale_source": "template",
            "v3_rule": "if embedding_pred == tfidf_pred -> tfidf_pred else meta_pred",
        },
        "results": results,
    }

    out_path = results_dir / "cv_classification_full.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Train CV: classification (unified) ===")
    ranked = sorted(results, key=lambda r: r["f1"], reverse=True)
    for row in ranked:
        demo = row["demographics"]
        print(
            f"[{row['model']:<22}] demo={demo:<28} "
            f"f1={row['f1']:.3f} acc={row['accuracy']:.3f}"
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
