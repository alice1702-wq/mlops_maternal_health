"""Обёртка над Optuna для подбора гиперпараметров."""
from typing import Any, Callable

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class OptunaTuner:
    """Универсальная ООП-обёртка над Optuna.

    Attributes:
        model_name: имя модели (для study_name).
        n_trials: количество испытаний.
        cv_splits: количество фолдов CV.
        random_state: seed для воспроизводимости.
        study: объект optuna.Study после tune().
        best_model: обученная модель после build_best_model().
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int = 30,
        cv_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.study: optuna.Study | None = None
        self.best_model: Any = None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        objective_fn: Callable,
    ) -> optuna.Study:
        """Запускает Optuna-оптимизацию.

        Args:
            X_train: признаки.
            y_train: метки.
            objective_fn: функция (trial, X, y) -> float.

        Returns:
            optuna.Study с результатами.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            study_name=f"{self.model_name}_study",
        )
        self.study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        print(f"\n✅ [{self.model_name}] Лучший F1-macro (CV): "
              f"{self.study.best_value:.4f}")
        print(f"   Лучшие параметры: {self.study.best_params}")
        return self.study

    def build_best_model(
        self,
        model_class: Callable,
        **fixed_params: Any,
    ) -> Any:
        """Создаёт модель с лучшими найденными параметрами.

        Args:
            model_class: класс модели (например, CatBoostClassifier).
            **fixed_params: параметры, которые не тюнятся, но нужны.

        Returns:
            Необученная модель с лучшими гиперпараметрами.
        """
        if self.study is None:
            raise RuntimeError("Сначала вызовите tune().")

        params = {**self.study.best_params, **fixed_params}
        self.best_model = model_class(**params)
        return self.best_model