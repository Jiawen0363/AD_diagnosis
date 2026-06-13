#!/usr/bin/env python3
"""Train CV ablation: language-only vs language+demographics routing."""

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
from demographics import DemographicFeatureEncoder
from eval_disagreement_routing_v3 import (
    build_embedding_config,
    build_meta_matrix,
    collect_oof_probabilities_embedding,
    collect_oof_probabilities_tfidf,
    route_with_meta_learner,
)
from features import get_concat_text_embeddings, get_text_embeddings
from metrics import classification_metrics, regression_metrics
from train import get_regressor


def _stack(language: np.ndarray, demo: np.ndarray | None) -> np.ndarray:
    if demo is None:
        return language
    return np.hstack([language, demo])


def main() -> None:
    with open(SCRIPT_DIR.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    embedding_config = build_embedding_config(config)
    results_dir = SCRIPT_DIR.parent / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, reg_folds, _ = load_or_create_folds()
    train_df = load_train_data(with_demographics=True)
    train_df_r = load_train_data_with_rationale(with_demographics=True)
    y_cls = train_df["label"].astype(int).to_numpy()
    y_reg = train_df["mmse"].astype(float).to_numpy()
    texts = train_df["text"].tolist()
    subject_ids = train_df["subject_id"].tolist()

    demo_encoder = DemographicFeatureEncoder.fit(train_df)
    demo_all = demo_encoder.transform(train_df)

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

    oof_emb_prob = collect_oof_probabilities_embedding(x_emb, y_cls, cls_folds)
    oof_tfidf_prob = collect_oof_probabilities_tfidf(
        texts, y_cls, cls_folds, config["tfidf"]
    )
    oof_concat_prob = collect_oof_probabilities_embedding(
        x_concat, y_cls, cls_folds
    )

    oof_emb_pred = (oof_emb_prob >= 0.5).astype(int)
    oof_tfidf_pred = (oof_tfidf_prob >= 0.5).astype(int)
    disagree = oof_emb_pred != oof_tfidf_pred

    results = []
    for use_demo in (False, True):
        demo = demo_all if use_demo else None
        x_meta = build_meta_matrix(oof_emb_prob, oof_tfidf_prob, oof_concat_prob, demo)
        meta = LogisticRegression(max_iter=1000).fit(x_meta, y_cls)
        meta_pred = meta.predict(x_meta).astype(int)
        routed = route_with_meta_learner(oof_tfidf_pred, meta_pred, disagree)
        cls_metrics = classification_metrics(y_cls, routed)

        reg_preds = np.zeros(len(y_reg), dtype=np.float64)
        for train_idx, val_idx in reg_folds:
            if use_demo:
                x_train = _stack(x_concat[train_idx], demo_all[train_idx])
                x_val = _stack(x_concat[val_idx], demo_all[val_idx])
            else:
                x_train = x_concat[train_idx]
                x_val = x_concat[val_idx]
            reg = get_regressor("ridge")
            reg.fit(x_train, y_reg[train_idx])
            reg_preds[val_idx] = reg.predict(x_val)
        reg_metrics = regression_metrics(y_reg, reg_preds)

        results.append(
            {
                "use_demographics": use_demo,
                "classification": cls_metrics,
                "regression": reg_metrics,
                "routing": {
                    "n_disagree": int(disagree.sum()),
                    "disagree_accuracy": float((routed[disagree] == y_cls[disagree]).mean()),
                },
            }
        )

    out_path = results_dir / "cv_demographics_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Train CV demographics ablation ===")
    for row in results:
        label = "with_demographics" if row["use_demographics"] else "language_only"
        cls = row["classification"]
        reg = row["regression"]
        print(
            f"[{label}] cls f1={cls['f1']:.3f}, "
            f"reg rmse={reg['rmse']:.3f}, r={reg['pearson_r']:.3f}"
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
