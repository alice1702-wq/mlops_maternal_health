"""Препроцессинг: Winsorizer с fit только на train.

Класс реализует паттерн fit/transform, как sklearn-трансформеры.
Это позволяет избежать data leakage: границы вычисляются ТОЛЬКО
на train, а к test применяются те же самые границы.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class Winsorizer:
    """Ограничивает выбросы по квантилям.

    Принцип работы:
        1. fit(X_train) — запоминает границы (lower, upper) для каждой колонки.
        2. transform(X) — обрезает значения по запомненным границам.
        3. fit_transform(X_train) — fit + transform за один шаг.

    Это предотвращает data leakage: границы считаются только
    на train-данных, а к test применяются те же значения.

    Attributes:
        cols: список колонок для обработки.
        lower_q: нижний квантиль (по умолчанию 0.01).
        upper_q: верхний квантиль (по умолчанию 0.99).
        bounds_: словарь {col: (low, high)} после fit().
    """

    def __init__(
        self,
        cols: List[str] | None = None,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
    ) -> None:
        self.cols = cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_: Dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame) -> "Winsorizer":
        """Вычисляет границы на train-данных."""
        if self.cols is None:
            self.cols = X.select_dtypes(include=np.number).columns.tolist()

        self.bounds_ = {
            col: (
                X[col].quantile(self.lower_q),
                X[col].quantile(self.upper_q),
            )
            for col in self.cols
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применяет запомненные границы к данным."""
        if not self.bounds_:
            raise RuntimeError("Winsorizer не обучен. Сначала вызовите fit().")

        X = X.copy()
        for col, (low, high) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low, upper=high)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """fit + transform одной командой (для train-данных)."""
        return self.fit(X).transform(X)

    def get_bounds(self) -> Dict[str, tuple]:
        """Возвращает границы (для логирования / метаданных)."""
        return self.bounds_.copy()
