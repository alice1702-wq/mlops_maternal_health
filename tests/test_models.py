"""Тесты метрик и анализа переобучения."""

import numpy as np

from src.models.evaluator import ModelEvaluator
from src.models.overfitting import OverfittingAnalyzer


def test_evaluator_returns_all_metrics():
    ev = ModelEvaluator("TestModel", ["a", "b", "c"])
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 0, 2])

    metrics = ev.evaluate(y_true, y_pred, "Test")

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1


def test_evaluator_stores_results():
    ev = ModelEvaluator("M", ["a", "b"])
    ev.evaluate(np.array([0, 1]), np.array([0, 1]), "Train")
    ev.evaluate(np.array([0, 1]), np.array([1, 1]), "Test")

    assert "Train" in ev.results
    assert "Test" in ev.results
    assert ev.results["Train"]["accuracy"] == 1.0


def test_overfitting_analyzer_detects_gap():
    """Модель с сильным переобучением должна получить 'Да'."""
    ev_good = ModelEvaluator("Good", ["a", "b"])
    ev_good.results = {
        "Train": {"f1_macro": 0.85, "accuracy": 0.85},
        "Test": {"f1_macro": 0.83, "accuracy": 0.83},
    }

    ev_overfit = ModelEvaluator("Overfit", ["a", "b"])
    ev_overfit.results = {
        "Train": {"f1_macro": 0.99, "accuracy": 0.99},
        "Test": {"f1_macro": 0.60, "accuracy": 0.60},
    }

    analyzer = OverfittingAnalyzer(threshold_gap=0.10, threshold_ratio=0.85)
    summary = analyzer.collect(
        {
            "Good": ev_good,
            "Overfit": ev_overfit,
        }
    )

    overfit_row = summary[summary["Model"] == "Overfit"].iloc[0]
    good_row = summary[summary["Model"] == "Good"].iloc[0]

    assert overfit_row["Overfit?"] == "Да"
    assert good_row["Overfit?"] == "Нет"
