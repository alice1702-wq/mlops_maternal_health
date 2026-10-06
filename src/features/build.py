"""Feature engineering для датасета Maternal Health Risk.

Важно: эта функция НЕ делает winsorization — это работа preprocessing.py.
Причина: winsorization требует статистик (квантилей) с train-данных,
поэтому должна фититься отдельно, чтобы избежать data leakage.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    AGE_SENIOR_MIN,
    AGE_TEEN_MAX,
    BODY_TEMP_FEVER,
    BP_DIASTOLIC_HIGH,
    BP_SYSTOLIC_HIGH,
    BS_NORMAL_MAX,
    BS_PREDIABETES_MAX,
    HEART_RATE_TACHYCARDIA,
)


# ============================================================
# Гемодинамика
# ============================================================
def add_hemodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет PulsePressure и MAP (среднее артериальное давление)."""
    df = df.copy()
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["MAP"] = df["DiastolicBP"] + df["PulsePressure"] / 3
    return df


# ============================================================
# Возраст
# ============================================================
def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет возрастные категории и флаги (IsTeen, IsSenior)."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["IsTeen"] = (df["Age"] < AGE_TEEN_MAX).astype(int)
    df["IsSenior"] = (df["Age"] >= AGE_SENIOR_MIN).astype(int)
    return df


# ============================================================
# Флаги риска
# ============================================================
def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет бинарные флаги риска и композитный RiskScore."""
    df = df.copy()

    df["BS_Category"] = pd.cut(
        df["BS"],
        bins=[0, BS_NORMAL_MAX, BS_PREDIABETES_MAX, 100],
        labels=["normal", "prediabetes", "hyperglycemia"],
    )

    df["HighBP_Flag"] = (
        (df["SystolicBP"] >= BP_SYSTOLIC_HIGH) | (df["DiastolicBP"] >= BP_DIASTOLIC_HIGH)
    ).astype(int)
    df["Tachycardia_Flag"] = (df["HeartRate"] > HEART_RATE_TACHYCARDIA).astype(int)
    df["Fever_Flag"] = (df["BodyTemp"] >= BODY_TEMP_FEVER).astype(int)

    df["RiskScore_Heuristic"] = (
        df["HighBP_Flag"] * 2
        + (df["BS"] > BS_PREDIABETES_MAX).astype(int) * 2
        + df["IsTeen"]
        + df["IsSenior"]
        + df["Tachycardia_Flag"]
        + df["Fever_Flag"]
    )
    return df


# ============================================================
# One-Hot Encoding
# ============================================================
def encode_categoricals(
    df: pd.DataFrame,
    cols: List[str] | None = None,
) -> pd.DataFrame:
    """One-Hot Encoding для категориальных признаков."""
    cols = cols or ["AgeGroup", "BS_Category"]
    return pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)


# ============================================================
# Главная функция
# ============================================================
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн feature engineering (без winsorization).

    Шаги:
        1. Гемодинамические признаки (PulsePressure, MAP).
        2. Возрастные категории.
        3. Флаги риска + композитный скор.
        4. One-Hot Encoding.

    Winsorization здесь НЕ выполняется — она делается отдельно
    в src.features.preprocessing.Winsorizer (fit только на train).

    Args:
        df_raw: исходный DataFrame.

    Returns:
        DataFrame с обогащёнными признаками.
    """
    df = add_hemodynamic_features(df_raw)
    df = add_age_features(df)
    df = add_risk_flags(df)
    df = encode_categoricals(df)
    return df


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Возвращает список числовых столбцов (без целевой переменной).

    Используется в Winsorizer для фита только на нужных колонках.
    """
    cols = df.select_dtypes(include=np.number).columns.tolist()
    if "RiskLevel" in cols:
        cols.remove("RiskLevel")
    return cols
