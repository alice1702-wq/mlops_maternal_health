"""Тесты feature engineering и preprocessing."""

import pandas as pd
import pytest

from src.features.build import (
    add_age_features,
    add_hemodynamic_features,
    add_risk_flags,
)
from src.features.preprocessing import Winsorizer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Мини-датасет для тестов."""
    return pd.DataFrame(
        {
            "Age": [15, 25, 40, 60],
            "SystolicBP": [110, 130, 150, 120],
            "DiastolicBP": [70, 85, 95, 80],
            "BS": [5.0, 6.5, 8.0, 7.2],
            "BodyTemp": [98.0, 98.6, 100.5, 99.0],
            "HeartRate": [70, 80, 95, 75],
        }
    )


# ─── Feature engineering ─────────────────────────────
def test_add_hemodynamic_features(sample_df):
    df = add_hemodynamic_features(sample_df)
    assert "PulsePressure" in df.columns
    assert "MAP" in df.columns
    assert df["PulsePressure"].iloc[0] == 110 - 70
    assert df["MAP"].iloc[0] == 70 + (110 - 70) / 3


def test_add_age_features(sample_df):
    df = add_age_features(sample_df)
    assert "AgeGroup" in df.columns
    assert "IsTeen" in df.columns
    assert "IsSenior" in df.columns
    assert df["IsTeen"].iloc[0] == 1  # 15 лет
    assert df["IsTeen"].iloc[1] == 0  # 25 лет
    assert df["IsSenior"].iloc[3] == 1  # 60 лет


def test_add_risk_flags(sample_df):
    # Сначала добавляем возрастные флаги — они нужны для RiskScore
    df = add_age_features(sample_df)
    df = add_risk_flags(df)
    expected_cols = [
        "BS_Category",
        "HighBP_Flag",
        "Tachycardia_Flag",
        "Fever_Flag",
        "RiskScore_Heuristic",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert df["HighBP_Flag"].iloc[2] == 1  # 150/95
    assert df["Tachycardia_Flag"].iloc[2] == 1  # 95 уд/мин
    assert df["Fever_Flag"].iloc[2] == 1  # 100.5°F


# ─── Winsorizer (data leakage) ───────────────────────
def test_winsorizer_fit_transform():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 1000],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )
    w = Winsorizer(cols=["x", "y"])
    df_out = w.fit_transform(df)

    # x — есть явный выброс (1000), должен обрезаться
    assert df_out["x"].max() < 1000
    # y — все значения в диапазоне, но 99-й перцентиль
    # может слегка обрезать максимум (59.5 вместо 60)
    assert df_out["y"].max() <= 60
    assert df_out["y"].min() >= 10


def test_winsorizer_uses_same_bounds_for_test():
    """Главный тест: границы фитятся на train, применяются к test."""
    train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
    test = pd.DataFrame({"x": [0, 50, 5000]})

    w = Winsorizer(cols=["x"], lower_q=0.1, upper_q=0.9)
    w.fit(train)
    test_out = w.transform(test)

    assert test_out["x"].iloc[2] <= train["x"].quantile(0.9)


def test_winsorizer_transform_before_fit_raises():
    w = Winsorizer(cols=["x"])
    with pytest.raises(RuntimeError, match="Winsorizer не обучен"):
        w.transform(pd.DataFrame({"x": [1, 2, 3]}))
