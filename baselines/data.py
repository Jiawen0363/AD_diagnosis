"""Data loading utilities for AD_diagnosis transcript baselines."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBJECT_ID_PATTERN = re.compile(r"(adrso\d+|adrsdt\d+)")


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    return " ".join(text.split())


def extract_subject_id(file_path: str) -> str:
    match = SUBJECT_ID_PATTERN.search(str(file_path))
    if not match:
        raise ValueError(f"Could not extract subject id from: {file_path}")
    return match.group(1)


def load_transcript_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["subject_id"] = df["file"].map(extract_subject_id)
    df["text"] = df["Speech"].map(preprocess_text)
    return df


def load_train_data(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    ad = load_transcript_csv(data_dir / "ad_s2t_wav2vec.csv")
    control = load_transcript_csv(data_dir / "control_s2t_wav2vec.csv")
    train = pd.concat([ad, control], ignore_index=True)
    train = train.sort_values("subject_id").reset_index(drop=True)
    return train


def load_test_data(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    test = load_transcript_csv(data_dir / "test_s2t_wav2vec.csv")
    return test.sort_values("subject_id").reset_index(drop=True)


def get_xy(df: pd.DataFrame, text_col: str = "text"):
    texts = df[text_col].tolist()
    y_cls = df["label"].astype(int).values
    y_reg = df["mmse"].astype(float).values
    subject_ids = df["subject_id"].tolist()
    return texts, y_cls, y_reg, subject_ids
