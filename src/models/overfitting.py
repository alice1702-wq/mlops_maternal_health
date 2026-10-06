"""Анализ переобучения моделей."""
from typing import Dict

import numpy as np
import pandas as pd

from src.models.evaluator import ModelEvaluator


class OverfittingAnalyzer:
    """Анализ переобучения по разрыву train/test.

    Args:
        threshold_ratio: минимально допустимое отношение test/train F1.
        threshold_gap: максимально допустимая разница F1 (train - test).
    """

    def __init__(
        self,
        threshold_ratio: float = 0.85,
        threshold_gap: float = 0.10,
    ) -> None:
        self.threshold_ratio = threshold_ratio
        self.threshold_gap = threshold_gap
        self.summary: pd.DataFrame | None = None

    def collect(
        self,
        evaluators: Dict[str, ModelEvaluator],
    ) -> pd.DataFrame:
        """Собирает сводную таблицу по всем моделям.

        Args:
            evaluators: {model_name: ModelEvaluator}.

        Returns:
            DataFrame с метриками и флагом переобучения.
        """
        rows = []
        for name, ev in evaluators.items():
            tr = ev.results.get("Train", {})
            te = ev.results.get("Test", {})
            row = {"Model": name}
            for split, m in [("train", tr), ("test", te)]:
                row[f"F1_{split}"] = m.get("f1_macro", np.nan)
                row[f"Acc_{split}"] = m.get("accuracy", np.nan)
            row["F1_gap"] = row["F1_train"] - row["F1_test"]
            row["F1_ratio"] = (
                row["F1_test"] / row["F1_train"]
                if row["F1_train"] > 0 else np.nan
            )
            row["Overfit?"] = (
                "Да"
                if (row["F1_gap"] > self.threshold_gap
                    or row["F1_ratio"] < self.threshold_ratio)
                else "Нет"
            )
            rows.append(row)

        self.summary = (
            pd.DataFrame(rows)
            .sort_values("F1_test", ascending=False)
            .reset_index(drop=True)
        )
        return self.summary

    def print_summary(self) -> None:
        """Печатает сводную таблицу в консоль."""
        if self.summary is None:
            print("Сначала вызовите collect().")
            return

        print("\n" + "=" * 70)
        print("СВОДНАЯ ТАБЛИЦА МОДЕЛЕЙ")
        print("=" * 70)
        cols = ["Model", "F1_train", "F1_test", "F1_gap",
                "F1_ratio", "Overfit?"]
        print(self.summary[cols].to_string(index=False))
        print("=" * 70)