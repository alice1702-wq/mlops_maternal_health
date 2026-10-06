# MLOps: Maternal Health Risk Classification

## Описание
ML-проект для классификации уровня риска беременности по клиническим показателям (датасет Maternal Health Risk).

## Стек
- Python 3.11, Poetry
- scikit-learn, CatBoost, Optuna, SHAP
- Plotly, Phik, pandas, numpy
- pytest, flake8, black, pre-commit

## Структура проекта
```
├── src/               # исходный код
│   ├── data/          # загрузка данных
│   ├── features/      # feature engineering
│   ├── models/        # обучение и оценка моделей
│   └── visualization/ # графики
├── tests/             # тесты
├── notebooks/         # Jupyter-ноутбуки
├── artifacts/         # сохранённые модели (в .gitignore)
├── pyproject.toml     # зависимости и настройки Poetry
└── poetry.lock        # зафиксированные версии
```

## Установка
```bash
poetry install
```

## Запуск обучения
```bash
poetry run train
```

## Тесты и линтеры
```bash
poetry run pytest
poetry run flake8 src/
poetry run black --check src/
```

## Автор
Селезнева Алиса, магистратура 2 курс
