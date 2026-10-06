"""Функции визуализации для EDA и отчётов.

Все графики — интерактивные (Plotly). Логика: функция принимает
DataFrame и возвращает Figure, не вызывая fig.show() — это делает
вызывающий код (ноутбук или скрипт).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_MAP_RISK = {
    "low risk": "#2ecc71",
    "mid risk": "#f39c12",
    "high risk": "#e74c3c",
}


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> go.Figure:
    """Гистограммы распределений числовых признаков (2×3 сетка)."""
    n = len(numeric_cols)
    rows = (n + 2) // 3
    fig = make_subplots(
        rows=rows, cols=3,
        subplot_titles=[f"Распределение: {c}" for c in numeric_cols],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    for i, col in enumerate(numeric_cols):
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_trace(
            go.Histogram(
                x=df[col], name=col, nbinsx=30,
                marker_color="#3498db", opacity=0.75,
                hovertemplate=(
                    "<b>%{x}</b><br>Количество: %{y}<extra></extra>"
                ),
            ),
            row=r, col=c,
        )

    fig.update_layout(
        title_text="Распределения числовых признаков",
        title_x=0.5, height=300 * rows,
        showlegend=False, template="plotly_white",
    )
    return fig


def plot_class_balance(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Столбчатая диаграмма баланса классов."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "Count"]

    fig = px.bar(
        counts, x=target_col, y="Count",
        color=target_col, text="Count",
        color_discrete_map=COLOR_MAP_RISK,
        title="Распределение классов (RiskLevel)",
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Количество: %{y}<br>"
            "Доля: %{customdata:.1%}<extra></extra>"
        ),
        customdata=(counts["Count"] / counts["Count"].sum()).values,
    )
    fig.update_layout(
        title_x=0.5, height=500,
        showlegend=False, template="plotly_white",
    )
    return fig


def plot_phik_correlation(
    df: pd.DataFrame,
    interval_cols: List[str],
) -> go.Figure:
    """Матрица корреляций Phik (для смешанных типов данных)."""
    import phik  # локальный импорт — редко нужен

    phik_matrix = df.phik_matrix(interval_cols=interval_cols)

    fig = go.Figure(data=go.Heatmap(
        z=phik_matrix.values,
        x=phik_matrix.columns,
        y=phik_matrix.index,
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=np.round(phik_matrix.values, 2),
        texttemplate="%{text}", textfont={"size": 11},
        hovertemplate=(
            "<b>%{y}</b> ↔ <b>%{x}</b><br>"
            "Φk = %{z:.3f}<extra></extra>"
        ),
        colorbar=dict(title="Φk"),
    ))
    fig.update_layout(
        title="Корреляционная матрица Phik",
        title_x=0.5, height=700, width=800,
        template="plotly_white",
    )
    return fig


def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> go.Figure:
    """Box plots для выявления выбросов (2×3 сетка)."""
    n = len(numeric_cols)
    rows = (n + 2) // 3
    fig = make_subplots(
        rows=rows, cols=3,
        subplot_titles=[f"Box plot: {c}" for c in numeric_cols],
    )
    for i, col in enumerate(numeric_cols):
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_trace(
            go.Box(
                y=df[col], name=col,
                marker_color="#3498db",
                boxmean="sd",
                hovertemplate="<b>%{y:.2f}</b><extra></extra>",
            ),
            row=r, col=c,
        )
    fig.update_layout(
        title_text="Box plots — выявление выбросов",
        title_x=0.5, height=300 * rows,
        showlegend=False, template="plotly_white",
    )
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str = "Model",
    dataset_name: str = "Test",
) -> go.Figure:
    """Матрица ошибок с абсолютными значениями и процентами."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    annotations = np.array([
        [f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names, y=class_names,
        colorscale="Blues",
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 13},
        hovertemplate=(
            "Истинный: <b>%{y}</b><br>"
            "Предсказанный: <b>%{x}</b><br>"
            "Количество: %{z}<extra></extra>"
        ),
        colorbar=dict(title="Количество"),
    ))
    fig.update_layout(
        title=f"Матрица ошибок ({model_name}) — {dataset_name}",
        title_x=0.5,
        xaxis_title="Предсказанный",
        yaxis_title="Истинный",
        height=550, width=650,
        template="plotly_white",
    )
    return fig


def plot_feature_importance(
    importances: Dict[str, float],
    model_name: str = "Model",
    color_scale: str = "Blues",
) -> go.Figure:
    """Горизонтальная столбчатая диаграмма важности признаков."""
    df = pd.DataFrame({
        "Feature": list(importances.keys()),
        "Importance": list(importances.values()),
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        df, x="Importance", y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale=color_scale,
        title=f"Важность признаков ({model_name})",
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Важность: %{x:.4f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title_x=0.5, height=600,
        template="plotly_white",
        coloraxis_showscale=False,
    )
    return fig