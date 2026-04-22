"""
Reporte de métricas.

Este módulo implementa la generación de reportes de métricas
tanto de clasificación como de regresión.

Para clasificación incluye:
- Accuracy
- Precision
- Recall
- F1 Score

Para regresión incluye:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coeficiente de determinación (R^2)

:authors: Joaquín Palacio Feijóo
:date: 15/04/2026
"""

from typing import Any

import pandas as pd
from numpy.typing import ArrayLike

from whiteboxml import metricas


def classification_summary(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
    pos_label: Any = 1,
) -> pd.DataFrame:
    """
    Generación de reporte de métricas de clasificación.

    Incluye:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: tipo de promedio ("binary", "micro", "macro", "weighted", None)
    :param pos_label: etiqueta positiva para el caso binario
    :return: DataFrame con las métricas
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """
    accuracy = metricas.accuracy(y_true, y_pred)

    precision = metricas.precision(y_true, y_pred, average=average, pos_label=pos_label)
    recall = metricas.recall(y_true, y_pred, average=average, pos_label=pos_label)
    f1 = metricas.f1_score(y_true, y_pred, average=average, pos_label=pos_label)

    return pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
        }
    )


def regression_summary(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """
    Generación de reporte de métricas de regresión.

    Incluye:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Coeficiente de determinación (R^2)

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: DataFrame con las métricas
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """
    mse = metricas.mean_squared_error(y_true, y_pred)
    mae = metricas.mean_absolute_error(y_true, y_pred)
    r2 = metricas.r2(y_true, y_pred)

    return pd.DataFrame(
        {
            "MSE": [mse],
            "MAE": [mae],
            "R^2": [r2],
        }
    )
