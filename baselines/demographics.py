"""Demographic feature encoding from groundtruth metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEMOGRAPHIC_COLUMNS = ("age", "gender", "educ")


@dataclass
class DemographicFeatureEncoder:
    """Fit imputation/scaling on train demographics; transform train/test arrays."""

    age_median: float
    educ_median: float
    age_mean: float
    age_std: float
    educ_mean: float
    educ_std: float

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "DemographicFeatureEncoder":
        age = df["age"].astype(float)
        educ = df["educ"].astype(float)
        age_median = float(age.median())
        educ_median = float(educ.median())
        age_filled = age.fillna(age_median)
        educ_filled = educ.fillna(educ_median)
        age_std = float(age_filled.std(ddof=0))
        educ_std = float(educ_filled.std(ddof=0))
        return cls(
            age_median=age_median,
            educ_median=educ_median,
            age_mean=float(age_filled.mean()),
            age_std=age_std if age_std > 0 else 1.0,
            educ_mean=float(educ_filled.mean()),
            educ_std=educ_std if educ_std > 0 else 1.0,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        age = df["age"].astype(float).fillna(self.age_median)
        educ = df["educ"].astype(float).fillna(self.educ_median)
        gender = df["gender"].astype(str).str.strip().str.lower()
        gender_male = gender.eq("male").astype(float)
        gender_missing = gender.isna() | gender.eq("") | gender.eq("nan")
        gender_male = gender_male.mask(gender_missing, 0.5)

        age_norm = (age - self.age_mean) / self.age_std
        educ_norm = (educ - self.educ_mean) / self.educ_std
        return np.column_stack(
            [
                age_norm.to_numpy(dtype=np.float64),
                gender_male.to_numpy(dtype=np.float64),
                educ_norm.to_numpy(dtype=np.float64),
            ]
        )

    def feature_names(self) -> list[str]:
        return ["age_norm", "gender_male", "educ_norm"]


def demographics_available_mask(df: pd.DataFrame) -> np.ndarray:
    return df["age"].notna().to_numpy()


def format_education_years(educ: float | int | str | None) -> str:
    if educ is None or (isinstance(educ, float) and np.isnan(educ)):
        return "not recorded"
    return f"{int(float(educ))}"


def build_template_demo_rationale(row: pd.Series) -> str:
    """Deterministic demographic context text for quick pipeline experiments."""
    age = row.get("age")
    gender = str(row.get("gender", "not recorded")).strip().lower()
    educ = format_education_years(row.get("educ"))
    if pd.isna(age):
        age_text = "not recorded"
    else:
        age_text = str(int(float(age)))

    return (
        f"This speaker is a {age_text}-year-old {gender} with {educ} years of formal education. "
        "Age and education are common background factors when interpreting spontaneous speech "
        "and cognitive screening tasks, because they can influence vocabulary, schooling-related "
        "language exposure, and typical performance ranges on tasks such as picture description. "
        "These demographics provide contextual information only and are not sufficient on their own "
        "to determine diagnostic status."
    )


def build_demo_rationales(df: pd.DataFrame) -> list[str]:
    return [build_template_demo_rationale(row) for _, row in df.iterrows()]
