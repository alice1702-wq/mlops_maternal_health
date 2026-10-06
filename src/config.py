"""Конфигурация проекта MLOps: Maternal Health Risk."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

# ============================================================
# Общие настройки
# ============================================================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 5

# ============================================================
# Пути
# ============================================================
ARTIFACTS_DIR: Path = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATA_FILE: Path = Path("maternal_health.xlsx")

# ============================================================
# Источник данных
# ============================================================
PUBLIC_LINK: str = "https://disk.yandex.ru/i/ffW4f5YfUfp2vA"
SHEET_NAME: str = "Таблица1"

# ============================================================
# Медицинские пороги (по рекомендациям ВОЗ)
# ============================================================
# Глюкоза (BS)
BS_NORMAL_MAX: float = 6.1  # < 6.1 — норма
BS_PREDIABETES_MAX: float = 7.0  # 6.1–7.0 — преддиабет; > 7.0 — гипергликемия

# Давление
BP_SYSTOLIC_HIGH: int = 140
BP_DIASTOLIC_HIGH: int = 90

# Пульс
HEART_RATE_TACHYCARDIA: int = 90

# Температура (Fahrenheit)
BODY_TEMP_FEVER: float = 100.4

# Возрастные группы
AGE_BINS: List[int] = [0, 18, 25, 35, 50, 100]
AGE_LABELS: List[str] = ["teen", "young", "adult", "mature", "senior"]
AGE_TEEN_MAX: int = 18
AGE_SENIOR_MIN: int = 45


# ============================================================
# Датаклассы для путей
# ============================================================
@dataclass
class Paths:
    """Пути к артефактам проекта."""

    # Модели
    logistic_regression: Path = ARTIFACTS_DIR / "logistic_regression.joblib"
    random_forest: Path = ARTIFACTS_DIR / "random_forest.joblib"
    catboost: Path = ARTIFACTS_DIR / "catboost_model.joblib"

    # Препроцессинг
    scaler: Path = ARTIFACTS_DIR / "scaler.joblib"
    winsorizer: Path = ARTIFACTS_DIR / "winsorizer.joblib"
    encoder: Path = ARTIFACTS_DIR / "label_encoder.joblib"

    # Метаданные
    metadata: Path = ARTIFACTS_DIR / "metadata.json"
