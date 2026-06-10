"""Core training and evaluation logic for a single CV fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from metrics import classification_metrics, regression_metrics

from features import build_tfidf_pipeline


def get_classifier(name: str):
    if name == "lr":
        return LogisticRegression(max_iter=1000)
    if name == "svc":
        return SVC(kernel="linear")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    if name == "ridge":
        return RidgeClassifier()
    raise ValueError(f"Unknown classifier: {name}")


def get_regressor(name: str):
    if name == "lr":
        return LinearRegression()
    if name == "ridge":
        return Ridge()
    if name == "svr":
        return SVR(kernel="rbf")
    if name == "rfr":
        return RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown regressor: {name}")


def run_fold(
    texts: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: str,
    model_name: str,
    tfidf_config: dict,
) -> dict[str, float]:
    x_train = [texts[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    if task == "classification":
        estimator = get_classifier(model_name)
        pipeline = build_tfidf_pipeline(estimator, tfidf_config)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        return classification_metrics(y_val, y_pred)

    if task == "regression":
        estimator = get_regressor(model_name)
        pipeline = build_tfidf_pipeline(estimator, tfidf_config)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        return regression_metrics(y_val, y_pred)

    raise ValueError(f"Unknown task: {task}")


def run_vector_fold(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: str,
    model_name: str,
) -> dict[str, float]:
    x_train = features[train_idx]
    x_val = features[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    if task == "classification":
        estimator = get_classifier(model_name)
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_val)
        return classification_metrics(y_val, y_pred)

    if task == "regression":
        estimator = get_regressor(model_name)
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_val)
        return regression_metrics(y_val, y_pred)

    raise ValueError(f"Unknown task: {task}")
