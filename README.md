# MLOps: Maternal Health Risk Classification

[![CI](https://github.com/alice1702-wq/mlops_maternal_health/actions/workflows/ci.yml/badge.svg)](https://github.com/alice1702-wq/mlops_maternal_health/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Poetry](https://img.shields.io/badge/poetry-2.5-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

Production-ready ML-проект для классификации уровня риска беременности по клиническим показателям.

## Задача

По 6 клиническим признакам (возраст, давление, глюкоза, температура, пульс) определить уровень риска беременности: `low risk`, `mid risk`, `high risk`.

**Датасет:** [Maternal Health Risk Data Set](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) — 1014 наблюдений, 3 класса.

## Результаты

| Модель | F1-macro (test) | Accuracy (test) |
|---|---|---|
| **CatBoost** | **0.867** | **0.862** |
| Random Forest | 0.819 | 0.816 |
| Logistic Regression | 0.665 | 0.670 |

**Метрика:** macro F1-score — обоснован дисбалансом классов (40/32/28%).

## Структура проекта

```
├── src/
│   ├── config.py              # константы и пути
│   ├── data/
│   │   └── loader.py          # загрузка с Яндекс.Диска
│   ├── features/
│   │   ├── build.py           # feature engineering
│   │   └── preprocessing.py   # Winsorizer (без data leakage)
│   ├── models/
│   │   ├── evaluator.py       # метрики, матрицы ошибок
│   │   ├── overfitting.py     # анализ переобучения
│   │   ├── tuner.py           # Optuna wrapper
│   │   └── train.py           # точка входа
│   └── visualization/
│       └── plots.py           # функции графиков
├── tests/                     # 11 unit-тестов
├── notebooks/
│   └── eda.ipynb              # полный EDA
├── .github/workflows/ci.yml   # GitHub Actions
├── .pre-commit-config.yaml
├── pyproject.toml
└── poetry.lock
```

## Установка

```bash
git clone https://github.com/alice1702-wq/mlops_maternal_health.git
cd mlops_maternal_health
poetry install
```

## Запуск

### Обучение моделей

```bash
poetry run train
```

Артефакты сохраняются в `artifacts/`:
- `catboost_model.joblib`, `random_forest.joblib`, `logistic_regression.joblib`
- `scaler.joblib`, `winsorizer.joblib`, `label_encoder.joblib`
- `metadata.json` (метрики, гиперпараметры, границы winsorizer)

### EDA

Открой `notebooks/eda.ipynb` в Jupyter или VSCode.

## Тесты и линтеры

```bash
poetry run pytest -v          # 11 тестов
poetry run flake8 src/ tests/ # стиль
poetry run black --check src/ # форматирование
```

## Технологии

- **Python 3.11**, **Poetry** — управление зависимостями
- **scikit-learn**, **CatBoost**, **Optuna** — ML
- **pandas**, **numpy**, **phik** — данные
- **Plotly**, **SHAP** — визуализация и интерпретация
- **pytest**, **black**, **flake8**, **pre-commit** — качество кода
- **GitHub Actions** — CI/CD

## Методология

### Особенности реализации

1. **Стратифицированный split** для сохранения пропорций классов.
2. **Feature engineering** с медицинским обоснованием: PulsePressure, MAP, категории возраста и глюкозы по порогам ВОЗ.
3. **CV** (5-fold) для оценки стабильности.
4. **Анализ переобучения** через `OverfittingAnalyzer` (gap + ratio).
5. **SHAP-анализ** для интерпретации модели.

### Feature Engineering

- `PulsePressure` = SystolicBP − DiastolicBP
- `MAP` = DiastolicBP + PulsePressure / 3
- `AgeGroup` (5 категорий), `IsTeen`, `IsSenior`
- `BS_Category` (норма / преддиабет / гипергликемия)
- `HighBP_Flag`, `Tachycardia_Flag`, `Fever_Flag`
- `RiskScore_Heuristic` — композитный скор

## Данные

Данные автоматически скачиваются с Яндекс.Диска при первом запуске (`src/data/loader.py`). В репозиторий **не коммитятся**.

## Автор

**Селезнева Алиса**
Магистратура, 2 курс

Уральский федеральный университет, ИРИТ-РТФ

Курс: MLOps (преподаватель — А. А. Кошелев)

## Лицензия

MIT
