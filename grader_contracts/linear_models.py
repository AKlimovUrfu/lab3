from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RegressionInput:
    features: Any
    target: Any
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
    features: Any
    target: Any
    learning_rate: float = 0.1
    epochs: int = 1000

@dataclass(frozen=True)
class ClassificationResult:
    weights: Any
    bias: float
    loss_history: list[float]
    probabilities: Any
    predictions: Any
