"""Собственные линейная и логистическая модели."""
from __future__ import annotations

from grader_contracts.linear_models import ClassificationInput, ClassificationResult, RegressionInput, RegressionResult


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите линейную регрессию градиентным спуском и верните историю MSE."""
    raise NotImplementedError


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите бинарный линейный классификатор с сигмоидой и BCE."""
    raise NotImplementedError
