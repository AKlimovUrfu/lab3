"""Заготовки задач на линейные модели."""
from __future__ import annotations
import numpy as np
from grader_contracts.linear_models import (
    LinearRegressionInput, LinearRegressionOutput,
    LogisticRegressionInput, LogisticRegressionOutput
)

def linear_regression_gd(data: LinearRegressionInput) -> LinearRegressionOutput:
    X_train, y_train = np.array(data.X_train), np.array(data.y_train)
    X_test = np.array(data.X_test)
    lr, n_iters = data.lr, data.n_iters

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(n_iters):
        y_pred = np.dot(X_train, weights) + bias
        dw = (2 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (2 / n_samples) * np.sum(y_pred - y_train)
        weights -= lr * dw
        bias -= lr * db

    predictions = np.dot(X_test, weights) + bias
    return LinearRegressionOutput(predictions=predictions.tolist())

def logistic_regression_gd(data: LogisticRegressionInput) -> LogisticRegressionOutput:
    X_train, y_train = np.array(data.X_train), np.array(data.y_train)
    X_test = np.array(data.X_test)
    lr, n_iters = data.lr, data.n_iters

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(n_iters):
        linear_model = np.dot(X_train, weights) + bias
        y_pred = 1 / (1 + np.exp(-np.clip(linear_model, -250, 250)))
        dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)
        weights -= lr * dw
        bias -= lr * db

    test_model = np.dot(X_test, weights) + bias
    probabilities = 1 / (1 + np.exp(-np.clip(test_model, -250, 250)))
    predictions = np.where(probabilities >= 0.5, 1, 0)

    return LogisticRegressionOutput(
        predictions=predictions.tolist(),
        probabilities=probabilities.tolist()
    )
