"""Точка входа: обучение и сравнение трёх моделей.

Модели:
    1. Logistic Regression (baseline).
    2. Random Forest.
    3. CatBoost.

Все модели обучаются на одних данных, оцениваются на train/test,
результаты сравниваются в сводной таблице.
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, CV_SPLITS, RANDOM_STATE, TEST_SIZE, Paths
from src.data.loader import load_data
from src.features.build import build_features, get_numeric_columns
from src.features.preprocessing import Winsorizer
from src.models.evaluator import ModelEvaluator
from src.models.overfitting import OverfittingAnalyzer

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def get_models() -> dict:
    """Возвращает словарь {имя: (модель, нужно_ли_масштабирование)}.

    Масштабирование нужно только для Logistic Regression.
    Деревья (RF, CatBoost) к масштабу не чувствительны.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000,
                class_weight="balanced",
            ),
            True,   # нужен scaler
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = (
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                loss_function="MultiClass",
                eval_metric="TotalF1",
                random_seed=RANDOM_STATE,
                verbose=100,
                allow_writing_files=False,
            ),
            False,
        )
    return models


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = CV_SPLITS,
) -> dict:
    """5-fold CV для модели. Возвращает средние метрики."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
    )
    return {
        "cv_f1_macro_mean": float(results["test_f1_macro"].mean()),
        "cv_f1_macro_std": float(results["test_f1_macro"].std()),
        "cv_accuracy_mean": float(results["test_accuracy"].mean()),
    }


def main() -> None:
    """Главный пайплайн."""
    print("=" * 70)
    print("ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    # ─── 1. Данные ───────────────────────────────────────────
    print("\n[1/6] Загрузка данных...")
    df_raw = load_data()
    print(f"  Загружено: {df_raw.shape}")

    # ─── 2. Feature engineering ─────────────────────────────
    print("\n[2/6] Feature engineering...")
    df_fe = build_features(df_raw)
    print(f"  После FE: {df_fe.shape}")

    X = df_fe.drop(columns=["RiskLevel"])
    y = df_fe["RiskLevel"]

    # ─── 3. Split (ДО winsorization!) ───────────────────────
    print("\n[3/6] Train/test split (со стратификацией)...")
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ─── 4. Preprocessing (fit ТОЛЬКО на train) ─────────────
    print("\n[4/6] Препроцессинг (без data leakage)...")
    num_cols = get_numeric_columns(X_train)

    # Winsorization
    winsorizer = Winsorizer(cols=num_cols)
    X_train_w = winsorizer.fit_transform(X_train)
    X_test_w = winsorizer.transform(X_test)
    print(f"  Winsorizer обучен на {len(num_cols)} колонках")

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_w)
    X_test_s = scaler.transform(X_test_w)
    print("  Scaler обучен на train")

    # ─── 5. Обучение и оценка моделей ───────────────────────
    print("\n[5/6] Обучение моделей...")
    models = get_models()
    evaluators = {}
    trained_models = {}
    cv_results = {}

    for name, (model, needs_scaling) in models.items():
        print(f"\n  ▶ {name}...")
        X_tr = X_train_s if needs_scaling else X_train_w.values
        X_te = X_test_s if needs_scaling else X_test_w.values

        model.fit(X_tr, y_train)
        trained_models[name] = (model, needs_scaling)

        y_train_pred = np.asarray(model.predict(X_tr)).ravel()
        y_test_pred = np.asarray(model.predict(X_te)).ravel()

        ev = ModelEvaluator(name, class_names)
        ev.evaluate(y_train, y_train_pred, "Train")
        ev.evaluate(y_test, y_test_pred, "Test")
        evaluators[name] = ev

        # CV — только для лучших/интересных моделей (экономия времени)
        if name in ("Logistic Regression", "CatBoost"):
            print(f"    CV ({CV_SPLITS}-fold)...")
            cv_results[name] = cross_validate_model(model, X_tr, y_train)
            print(f"    CV F1-macro: {cv_results[name]['cv_f1_macro_mean']:.4f} "
                  f"± {cv_results[name]['cv_f1_macro_std']:.4f}")

    # ─── 6. Сводная таблица ─────────────────────────────────
    print("\n[6/6] Сравнение моделей...")
    analyzer = OverfittingAnalyzer()
    analyzer.collect(evaluators)
    analyzer.print_summary()

    # Лучшая модель
    best_row = analyzer.summary.iloc[0]
    best_name = best_row["Model"]
    print(f"\n🏆 Лучшая модель: {best_name}")
    print(f"   F1-macro (test): {best_row['F1_test']:.4f}")
    print(f"   Accuracy (test): {best_row['Acc_test']:.4f}")

    # ─── Сохранение артефактов ──────────────────────────────
    paths = Paths()
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    for name, (model, _) in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        if name == "CatBoost":
            fname = "catboost_model.joblib"
        joblib.dump(model, ARTIFACTS_DIR / fname)

    joblib.dump(scaler, paths.scaler)
    joblib.dump(winsorizer, paths.winsorizer)
    joblib.dump(encoder, paths.encoder)

    metadata = {
        "best_model": best_name,
        "class_names": class_names,
        "feature_names": X.columns.tolist(),
        "metrics": {
            name: {
                "train_f1": ev.results["Train"]["f1_macro"],
                "test_f1": ev.results["Test"]["f1_macro"],
                "test_accuracy": ev.results["Test"]["accuracy"],
            }
            for name, ev in evaluators.items()
        },
        "cv_results": cv_results,
        "winsorizer_bounds": winsorizer.get_bounds(),
    }
    paths.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("✅ АРТЕФАКТЫ СОХРАНЕНЫ В artifacts/:")
    for f in sorted(ARTIFACTS_DIR.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()