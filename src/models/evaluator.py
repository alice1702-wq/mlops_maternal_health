"""Оценка качества ML-моделей: метрики, матрица ошибок, отчёт."""
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Оценщик моделей: метрики, матрица ошибок, отчёт.

    Attributes:
        model_name: имя модели (для подписи графиков).
        class_names: список имён классов.
        results: словарь {dataset_name: {metric: value}}.
    """

    def __init__(self, model_name: str, class_names: List[str]) -> None:
        self.model_name = model_name
        self.class_names = class_names
        self.results: Dict[str, Dict[str, float]] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
    ) -> Dict[str, float]:
        """Считает основные метрики.

        Args:
            y_true: истинные метки.
            y_pred: предсказанные метки.
            dataset_name: имя набора (Train / Test).

        Returns:
            Словарь с метриками.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }
        self.results[dataset_name] = metrics
        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
    ) -> None:
        """Строит матрицу ошибок через Plotly (интерактивная)."""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        annotations = np.array([
            [f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names,
            y=self.class_names,
            colorscale="Blues",
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate=(
                "Истинный: <b>%{y}</b><br>"
                "Предсказанный: <b>%{x}</b><br>"
                "Количество: %{z}<extra></extra>"
            ),
            colorbar=dict(title="Количество"),
        ))
        fig.update_layout(
            title=f"Матрица ошибок ({self.model_name}) — {dataset_name}",
            title_x=0.5,
            xaxis_title="Предсказанный",
            yaxis_title="Истинный",
            height=550,
            width=650,
            template="plotly_white",
        )
        fig.show()

    def print_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
    ) -> None:
        """Печатает classification_report."""
        print(f"\n--- Отчёт: {self.model_name} ({dataset_name}) ---")
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
        ))

    def show_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
    ) -> None:
        """Считает метрики, рисует матрицу, печатает отчёт."""
        metrics = self.evaluate(y_true, y_pred, dataset_name)
        self.plot_confusion_matrix(y_true, y_pred, dataset_name)
        self.print_report(y_true, y_pred, dataset_name)

        print(f"\n--- Ключевые метрики ({self.model_name}, {dataset_name}) ---")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")