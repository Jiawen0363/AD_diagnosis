#!/usr/bin/env python3
"""Train CV: compare tabular demo vs demo_rationale triple concat."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evaluation"))

from cv import load_or_create_folds
from data import load_train_data_with_rationale
from demographics import DemographicFeatureEncoder, build_demo_rationales
from eval_disagreement_routing_v3 import build_embedding_config
from features import get_concat_text_embeddings, get_triple_concat_text_embeddings
from train import run_vector_fold


def stack_demo(x: np.ndarray, demo: np.ndarray | None) -> np.ndarray:
    if demo is None:
        return x
    return np.hstack([x, demo])


def main() -> None:
    with open(SCRIPT_DIR.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    embedding_config = build_embedding_config(config)
    results_dir = SCRIPT_DIR.parent / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cls_folds, reg_folds, _ = load_or_create_folds()
    train_df = load_train_data_with_rationale(with_demographics=True)
    y_cls = train_df["label"].astype(int).to_numpy()
    y_reg = train_df["mmse"].astype(float).to_numpy()
    subject_ids = train_df["subject_id"].tolist()
    demo_encoder = DemographicFeatureEncoder.fit(train_df)
    demo_tab = demo_encoder.transform(train_df)
    demo_texts = build_demo_rationales(train_df)

    x_concat = get_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        subject_ids=subject_ids,
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_concat",
    )
    x_triple = get_triple_concat_text_embeddings(
        transcripts=train_df["text"].tolist(),
        rationales=train_df["rationale_text"].tolist(),
        demo_rationales=demo_texts,
        subject_ids=subject_ids,
        embedding_config=embedding_config,
        project_root=SCRIPT_DIR.parent,
        cache_suffix="train_transcript_rationale_demo_concat",
    )

    configs = [
        ("concat_language", x_concat, False),
        ("concat_language+tabular_demo", x_concat, True),
        ("triple_concat_demo_rationale", x_triple, False),
        ("triple_concat+demo_rationale+tabular", x_triple, True),
    ]

    results = []
    for name, x_base, use_tabular in configs:
        x = stack_demo(x_base, demo_tab if use_tabular else None)
        cls_rows = [
            run_vector_fold(x, y_cls, tr, va, "classification", "svc")
            for tr, va in cls_folds
        ]
        reg_rows = [
            run_vector_fold(x, y_reg, tr, va, "regression", "ridge")
            for tr, va in reg_folds
        ]
        results.append(
            {
                "model": name,
                "classification": {
                    k: float(np.mean([r[k] for r in cls_rows]))
                    for k in cls_rows[0]
                },
                "regression": {
                    k: float(np.mean([r[k] for r in reg_rows]))
                    for k in reg_rows[0]
                },
            }
        )

    out_path = results_dir / "cv_demo_rationale_triple_concat.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Train CV: demo_rationale experiment ===")
    for row in results:
        cls = row["classification"]
        reg = row["regression"]
        print(
            f"[{row['model']}] "
            f"cls f1={cls['f1']:.3f}, "
            f"reg rmse={reg['rmse']:.3f}, r={reg['pearson_r']:.3f}"
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
