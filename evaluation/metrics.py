"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson_r = float(pearsonr(y_true, y_pred).statistic)
    else:
        pearson_r = float("nan")
    return {"rmse": rmse, "mae": mae, "pearson_r": pearson_r}
