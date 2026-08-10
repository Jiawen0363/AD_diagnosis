"""Data loading utilities for AD_diagnosis transcript baselines."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUNDTRUTH_FILENAME = "training-groundtruth.csv"
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


def load_groundtruth(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    gt = pd.read_csv(data_dir / GROUNDTRUTH_FILENAME)
    gt["subject_id"] = gt["adressfname"].astype(str).str.strip().str.lower()
    return gt[["subject_id", "age", "gender", "educ"]]


def attach_demographics(
    df: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    groundtruth = load_groundtruth(data_dir)
    merged = df.merge(groundtruth, on="subject_id", how="left", validate="one_to_one")
    merged["has_demographics"] = merged["age"].notna()
    return merged


def load_train_data(
    data_dir: Path | None = None,
    *,
    with_demographics: bool = False,
) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    ad = load_transcript_csv(data_dir / "ad_s2t_wav2vec.csv")
    control = load_transcript_csv(data_dir / "control_s2t_wav2vec.csv")
    train = pd.concat([ad, control], ignore_index=True)
    train = train.sort_values("subject_id").reset_index(drop=True)
    if with_demographics:
        train = attach_demographics(train, data_dir)
    return train


def load_test_data(
    data_dir: Path | None = None,
    *,
    with_demographics: bool = False,
) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    test = load_transcript_csv(data_dir / "test_s2t_wav2vec.csv")
    test = test.sort_values("subject_id").reset_index(drop=True)
    if with_demographics:
        test = attach_demographics(test, data_dir)
    return test


def get_xy(df: pd.DataFrame, text_col: str = "text"):
    texts = df[text_col].tolist()
    y_cls = df["label"].astype(int).values
    y_reg = df["mmse"].astype(float).values
    subject_ids = df["subject_id"].tolist()
    return texts, y_cls, y_reg, subject_ids


def rationale_json_name(split: str, suffix: str = "") -> str:
    """Return rationale JSON filename for ad/control/test split."""
    base = f"{split}_s2t_wav2vec_rationale"
    if not suffix:
        return f"{base}.json"
    normalized = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{base}{normalized}.json"


def load_rationale_json(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    df["subject_id"] = df["file"].map(extract_subject_id)
    df["rationale_text"] = df["rationale"].map(preprocess_text)
    return df[["subject_id", "rationale_text"]]


def load_demo_rationale_json(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    if "subject_id" not in df.columns:
        df["subject_id"] = df["file"].map(extract_subject_id)
    df["demo_rationale_text"] = df["demo_rationale"].map(preprocess_text)
    return df[["subject_id", "demo_rationale_text"]]


def _merge_demo_rationales(
    df: pd.DataFrame,
    rationale_dir: Path,
    train_or_test: str,
) -> pd.DataFrame:
    demo_path = rationale_dir / f"{train_or_test}_s2t_wav2vec_demo_rationale.json"
    if train_or_test == "train":
        demo_frames = [
            load_demo_rationale_json(rationale_dir / "ad_s2t_wav2vec_demo_rationale.json"),
            load_demo_rationale_json(rationale_dir / "control_s2t_wav2vec_demo_rationale.json"),
        ]
        demos = pd.concat(demo_frames, ignore_index=True)
    else:
        demos = load_demo_rationale_json(demo_path)

    merged = df.merge(demos, on="subject_id", how="left", validate="one_to_one")
    missing = merged["demo_rationale_text"].isna().sum()
    if missing:
        raise ValueError(f"Missing demo rationales for {missing} samples from {demo_path}.")
    empty = merged["demo_rationale_text"].astype(str).str.strip().eq("").sum()
    if empty:
        raise ValueError(f"Found {empty} empty demo rationales.")
    return merged


def load_train_data_with_rationale(
    data_dir: Path | None = None,
    rationale_dir: Path | None = None,
    *,
    rationale_suffix: str = "",
    with_demographics: bool = False,
    with_demo_rationale: bool = False,
) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    rationale_dir = rationale_dir or data_dir / "rationale"

    train = load_train_data(data_dir, with_demographics=with_demographics)
    rationale_frames = [
        load_rationale_json(
            rationale_dir / rationale_json_name("ad", rationale_suffix)
        ),
        load_rationale_json(
            rationale_dir / rationale_json_name("control", rationale_suffix)
        ),
    ]
    rationales = pd.concat(rationale_frames, ignore_index=True)
    merged = train.merge(rationales, on="subject_id", how="left", validate="one_to_one")

    missing = merged["rationale_text"].isna().sum()
    if missing:
        raise ValueError(f"Missing rationales for {missing} train samples.")

    empty = merged["rationale_text"].astype(str).str.strip().eq("").sum()
    if empty:
        raise ValueError(f"Found {empty} empty rationales in train data.")

    if with_demo_rationale:
        merged = _merge_demo_rationales(merged, rationale_dir, "train")

    return merged.sort_values("subject_id").reset_index(drop=True)


def load_test_data_with_rationale(
    data_dir: Path | None = None,
    rationale_dir: Path | None = None,
    *,
    rationale_suffix: str = "",
    with_demographics: bool = False,
    with_demo_rationale: bool = False,
) -> pd.DataFrame:
    data_dir = data_dir or PROJECT_ROOT / "data"
    rationale_dir = rationale_dir or data_dir / "rationale"

    test = load_test_data(data_dir, with_demographics=with_demographics)
    rationales = load_rationale_json(
        rationale_dir / rationale_json_name("test", rationale_suffix)
    )
    merged = test.merge(rationales, on="subject_id", how="left", validate="one_to_one")

    missing = merged["rationale_text"].isna().sum()
    if missing:
        raise ValueError(f"Missing rationales for {missing} test samples.")

    empty = merged["rationale_text"].astype(str).str.strip().eq("").sum()
    if empty:
        raise ValueError(f"Found {empty} empty rationales in test data.")

    if with_demo_rationale:
        merged = _merge_demo_rationales(merged, rationale_dir, "test")

    return merged.sort_values("subject_id").reset_index(drop=True)


def get_xy_with_rationale(df: pd.DataFrame):
    transcripts = df["text"].tolist()
    rationales = df["rationale_text"].tolist()
    y_cls = df["label"].astype(int).values
    y_reg = df["mmse"].astype(float).values
    subject_ids = df["subject_id"].tolist()
    return transcripts, rationales, y_cls, y_reg, subject_ids
