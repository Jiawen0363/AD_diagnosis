"""Fixed 5-fold cross-validation splits shared across baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

from data import PROJECT_ROOT, get_xy, load_train_data

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_cv_splits(
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(y_cls))

    cls_splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    cls_folds = [
        (train_idx, val_idx)
        for train_idx, val_idx in cls_splitter.split(indices, y_cls)
    ]

    reg_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    reg_folds = [
        (train_idx, val_idx)
        for train_idx, val_idx in reg_splitter.split(indices, y_reg)
    ]

    return cls_folds, reg_folds


def save_folds(
    cls_folds: list[tuple[np.ndarray, np.ndarray]],
    reg_folds: list[tuple[np.ndarray, np.ndarray]],
    subject_ids: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classification": [
            {
                "fold": i,
                "train_subjects": [subject_ids[j] for j in train_idx],
                "val_subjects": [subject_ids[j] for j in val_idx],
            }
            for i, (train_idx, val_idx) in enumerate(cls_folds)
        ],
        "regression": [
            {
                "fold": i,
                "train_subjects": [subject_ids[j] for j in train_idx],
                "val_subjects": [subject_ids[j] for j in val_idx],
            }
            for i, (train_idx, val_idx) in enumerate(reg_folds)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_or_create_folds(results_dir: Path | None = None) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
    list[str],
]:
    config = load_config()
    results_dir = results_dir or PROJECT_ROOT / config["results_dir"]
    folds_path = results_dir / "folds.json"

    train_df = load_train_data()
    subject_ids = train_df["subject_id"].tolist()
    _, y_cls, y_reg, _ = get_xy(train_df)

    if folds_path.exists():
        with open(folds_path) as f:
            saved = json.load(f)

        subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
        cls_folds = []
        for fold in saved["classification"]:
            train_idx = np.array([subject_to_idx[s] for s in fold["train_subjects"]])
            val_idx = np.array([subject_to_idx[s] for s in fold["val_subjects"]])
            cls_folds.append((train_idx, val_idx))

        reg_folds = []
        for fold in saved["regression"]:
            train_idx = np.array([subject_to_idx[s] for s in fold["train_subjects"]])
            val_idx = np.array([subject_to_idx[s] for s in fold["val_subjects"]])
            reg_folds.append((train_idx, val_idx))

        return cls_folds, reg_folds, subject_ids

    cv_cfg = config["cv"]
    cls_folds, reg_folds = get_cv_splits(
        y_cls,
        y_reg,
        n_splits=cv_cfg["n_splits"],
        random_state=cv_cfg["random_state"],
    )
    save_folds(cls_folds, reg_folds, subject_ids, folds_path)
    return cls_folds, reg_folds, subject_ids
