from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegressionInput:
    train_features: Any
    train_target: Any
    test_features: Any
    test_target: Any
    learning_rate: float = 0.05
    epochs: int = 800


@dataclass(frozen=True)
class RegressionResult:
    weights: Any
    bias: float
    loss_history: list[float]
    predictions: Any


@dataclass(frozen=True)
class ClassificationInput:
    train_features: Any
    train_target: Any
    test_features: Any
    test_target: Any
    learning_rate: float = 0.1
    epochs: int = 1000


@dataclass(frozen=True)
class ClassificationResult:
    weights: Any
    bias: float
    loss_history: list[float]
    probabilities: Any
    predictions: Any


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BaselineComparisonResult:
    logistic_regression: ClassificationMetrics
    decision_tree: ClassificationMetrics
    knn: ClassificationMetrics
