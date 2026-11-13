from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42
_YES_NO = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0}

def _to_matrix(features) -> np.ndarray:
    if hasattr(features, "columns"):
        frame = features.copy()
        for col in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
                frame[col] = frame[col].map(lambda v: _YES_NO.get(str(v).strip().lower(), v))
        arr = frame.to_numpy(dtype=float)
    else:
        arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _to_vector(target) -> np.ndarray:
    return np.asarray(target, dtype=float).reshape(-1)

def _standardize_params(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def _unscale(w_scaled: np.ndarray, b_scaled: float, mean: np.ndarray, std: np.ndarray):
    w = w_scaled / std
    b = float(b_scaled - np.sum(w_scaled * mean / std))
    return w, b

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def classification_metrics(y_true, proba, pred) -> ClassificationMetrics:
    y_true = _to_vector(y_true)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )

def fit_linear_gd(x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int):
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    history: list[float] = []
    for _ in range(epochs):
        residual = x @ w + b - y
        history.append(float(np.mean(residual ** 2)))
        grad_w = 2.0 / n * (x.T @ residual)
        grad_b = 2.0 / n * residual.sum()
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return w, b, history


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    x_train = _to_matrix(data.train_features)
    y_train = _to_vector(data.train_target)
    x_test = _to_matrix(data.test_features)

    mean, std = _standardize_params(x_train)
    w_s, b_s, history = fit_linear_gd((x_train - mean) / std, y_train, data.learning_rate, data.epochs)

    weights, bias = _unscale(w_s, b_s, mean, std)
    predictions = x_test @ weights + bias
    return RegressionResult(weights=weights, bias=bias, loss_history=history, predictions=predictions)

def fit_logistic_gd(x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int):
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    history: list[float] = []
    for _ in range(epochs):
        p = _sigmoid(x @ w + b)
        history.append(_bce(y, p))
        error = p - y  # градиент BCE по логиту
        w -= learning_rate * (x.T @ error) / n
        b -= learning_rate * error.mean()
    return w, b, history


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    x_train = _to_matrix(data.train_features)
    y_train = _to_vector(data.train_target)
    x_test = _to_matrix(data.test_features)

    mean, std = _standardize_params(x_train)
    w_s, b_s, history = fit_logistic_gd((x_train - mean) / std, y_train, data.learning_rate, data.epochs)

    weights, bias = _unscale(w_s, b_s, mean, std)
    probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=history,
        probabilities=probabilities,
        predictions=predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    x_train = _to_matrix(data.train_features)
    y_train = _to_vector(data.train_target).astype(int)
    x_test = _to_matrix(data.test_features)
    y_test = _to_vector(data.test_target).astype(int)

    own = train_logistic_classifier(data)
    own_metrics = classification_metrics(y_test, own.probabilities, own.predictions)

    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
    tree.fit(x_train, y_train)
    tree_metrics = classification_metrics(y_test, tree.predict_proba(x_test)[:, 1], tree.predict(x_test))

    scaler = StandardScaler().fit(x_train)
    knn = KNeighborsClassifier(n_neighbors=45, weights="distance")
    knn.fit(scaler.transform(x_train), y_train)
    xs_test = scaler.transform(x_test)
    knn_metrics = classification_metrics(y_test, knn.predict_proba(xs_test)[:, 1], knn.predict(xs_test))

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )
