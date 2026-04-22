"""
Módulo para cálculo y visualización de curvas ROC.

:authors: Claudio Gabriel Alonso
:date: 21/04/2026
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def calculate_auc(fpr: ArrayLike, tpr: ArrayLike) -> float:
    """
    Calcula el área bajo la curva ROC (AUC) usando integración numérica.

    :param fpr: False Positive Rate.
    :param tpr: True Positive Rate.
    :return: Valor del AUC.

    :authors: Claudio Gabriel Alonso
    :date: 20/04/2026
    """

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    return np.trapz(tpr, fpr)


def calculate_roc_curve(
    y_true: ArrayLike, y_pred_proba: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva ROC a partir de etiquetas reales y probabilidades predichas.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas para la clase positiva.
    :return: Tupla (fpr, tpr, thresholds).

    :authors: Claudio Gabriel Alonso
    :date: 20/04/2026
    """

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    thresholds = y_pred_proba[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)

    tpr = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    fpr = fp / n_neg if n_neg > 0 else np.zeros_like(fp)

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([1.0], thresholds))

    return fpr, tpr, thresholds


def plot_roc_curve(
    y_true: ArrayLike, y_pred_proba: ArrayLike, show_diagonal: bool = True
) -> None:
    """
    Calcula y grafica la curva ROC a partir de los datos de entrada.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas.
    :param show_diagonal: Si True, muestra la línea base.
    :return: None.

    :authors: Claudio Gabriel Alonso
    :date: 20/04/2026
    """

    fpr, tpr, _ = calculate_roc_curve(y_true, y_pred_proba)
    auc = calculate_auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

    if show_diagonal:
        plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()
