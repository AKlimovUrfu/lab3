"""Реализация линейных моделей с нуля."""

import numpy as np
import pandas as pd
import json
from dataclasses import asdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def to_numpy_array(X):
    """Надёжно конвертирует данные в numpy массив float."""
    if isinstance(X, pd.DataFrame):
        # Заменяем все нечисловые значения на NaN, потом на 0
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X.to_numpy(dtype=float)
    else:
        return np.array(X, dtype=float)


def scale_data(X_train, X_test):
    """Масштабирование данных (Standardization)."""
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    return X_train_scaled, X_test_scaled


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = to_numpy_array(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = to_numpy_array(data.test_features)
    y_test = np.array(data.test_target, dtype=float)
    
    lr = data.learning_rate
    epochs = data.epochs

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    n_features = X_train_scaled.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    m = len(y_train)

    for _ in range(epochs):
        y_pred = np.dot(X_train_scaled, weights) + bias
        error = y_pred - y_train
        mse = np.mean(error ** 2)
        loss_history.append(float(mse))
        
        dw = (2 / m) * np.dot(X_train_scaled.T, error)
        db = (2 / m) * np.sum(error)
        
        weights -= lr * dw
        bias -= lr * db

    y_test_pred = np.dot(X_test_scaled, weights) + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=y_test_pred
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = to_numpy_array(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = to_numpy_array(data.test_features)
    y_test = np.array(data.test_target, dtype=float)
    
    lr = data.learning_rate
    epochs = data.epochs

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    n_features = X_train_scaled.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    m = len(y_train)

    for _ in range(epochs):
        z = np.dot(X_train_scaled, weights) + bias
        z = np.clip(z, -250, 250)
        y_pred_prob = 1 / (1 + np.exp(-z))
        
        eps = 1e-15
        y_pred_prob_clipped = np.clip(y_pred_prob, eps, 1 - eps)
        bce = -np.mean(y_train * np.log(y_pred_prob_clipped) + (1 - y_train) * np.log(1 - y_pred_prob_clipped))
        loss_history.append(float(bce))
        
        error = y_pred_prob - y_train
        dw = (1 / m) * np.dot(X_train_scaled.T, error)
        db = (1 / m) * np.sum(error)
        
        weights -= lr * dw
        bias -= lr * db

    z_test = np.dot(X_test_scaled, weights) + bias
    z_test = np.clip(z_test, -250, 250)
    test_probs = 1 / (1 + np.exp(-z_test))
    test_preds = (test_probs >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probs,
        predictions=test_preds
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    X_train = to_numpy_array(data.train_features)
    y_train = np.array(data.train_target, dtype=float)
    X_test = to_numpy_array(data.test_features)
    y_test = np.array(data.test_target, dtype=float)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # 1. Sklearn Logistic Regression
    lr_sk = LogisticRegression(random_state=42, max_iter=1000)
    lr_sk.fit(X_train_scaled, y_train)
    lr_pred = lr_sk.predict(X_test_scaled)
    lr_prob = lr_sk.predict_proba(X_test_scaled)[:, 1]

    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)
    dt_prob = dt.predict_proba(X_test_scaled)[:, 1]

    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_prob = knn.predict_proba(X_test_scaled)[:, 1]

    def make_metrics(y_true, y_pred, y_prob):
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_true, y_prob)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0))
        )

    return BaselineComparisonResult(
        logistic_regression=make_metrics(y_test, lr_pred, lr_prob),
        decision_tree=make_metrics(y_test, dt_pred, dt_prob),
        knn=make_metrics(y_test, knn_pred, knn_prob)
    )


# ==========================================
# ГЕНЕРАЦИЯ experiment_results.json
# ==========================================
if __name__ == "__main__":
    print(" Запуск генерации experiment_results.json...")
    
    # 1. Линейная регрессия (Student_Performance.csv)
    # Пробуем оба разделителя на всякий случай
    try:
        df_reg = pd.read_csv("Student_Performance.csv", sep=",")
    except Exception:
        df_reg = pd.read_csv("Student_Performance.csv", sep=";")
    
    print(f" Регрессия: столбцов = {df_reg.shape[1]}, строк = {df_reg.shape[0]}")
    print(f"   Столбцы: {list(df_reg.columns)}")
    
    X_reg = df_reg.drop(columns=["Performance Index"])
    y_reg = df_reg["Performance Index"]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    class MockRegInput:
        def __init__(self):
            self.train_features = X_train_reg
            self.train_target = y_train_reg.values
            self.test_features = X_test_reg
            self.test_target = y_test_reg.values
            self.learning_rate = 0.05
            self.epochs = 800
            
    reg_input = MockRegInput()
    reg_result = train_linear_regression(reg_input)
    test_mse = float(np.mean((reg_result.predictions - y_test_reg.values) ** 2))
    print(f" MSE линейной регрессии: {test_mse:.4f}")

    # 2. Классификация (german.csv) — разделитель точка с запятой!
    try:
        df_clf = pd.read_csv("german.csv", sep=";")
    except Exception:
        df_clf = pd.read_csv("german.csv", sep=",")
    
    print(f" Классификация: столбцов = {df_clf.shape[1]}, строк = {df_clf.shape[0]}")
    print(f"   Первые 5 столбцов: {list(df_clf.columns[:5])}")
    
    # Целевая переменная в первом столбце: Creditability
    y_clf = df_clf.iloc[:, 0]
    X_clf = df_clf.iloc[:, 1:]
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    class MockClfInput:
        def __init__(self):
            self.train_features = X_train_clf
            self.train_target = y_train_clf.values
            self.test_features = X_test_clf
            self.test_target = y_test_clf.values
            self.learning_rate = 0.1
            self.epochs = 1000
            
    clf_input = MockClfInput()
    clf_result = compare_classical_models(clf_input)
    
    # Метрики собственной модели
    own_res = train_logistic_classifier(clf_input)
    own_metrics = {
        "roc_auc": float(roc_auc_score(y_test_clf.values, own_res.probabilities)),
        "accuracy": float(accuracy_score(y_test_clf.values, own_res.predictions)),
        "precision": float(precision_score(y_test_clf.values, own_res.predictions, zero_division=0)),
        "recall": float(recall_score(y_test_clf.values, own_res.predictions, zero_division=0)),
        "f1": float(f1_score(y_test_clf.values, own_res.predictions, zero_division=0))
    }

    # 3. Формирование JSON
    results_dict = {
        "linear_regression": {
            "test_mse": round(test_mse, 4)
        },
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": asdict(clf_result.logistic_regression),
            "decision_tree": asdict(clf_result.decision_tree),
            "knn": asdict(clf_result.knn)
        }
    }

    # 4. Сохранение
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
        
    print(" Файл experiment_results.json успешно создан!")
    print(f" MSE линейной регрессии: {test_mse:.4f}")
    print(" Готово!")