"""
Visualización: Módulo de visualización de WhiteBoxML.

:authors: Nahuel Nicolas Alvarez
:date: 15/04/2026
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from whiteboxml.utils import _validacion_inputs


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None
) -> np.ndarray:
    """
    Genera una matriz de confusión a partir de los valores reales y predichos.

    Las filas representan las clases reales y las columnas las clases predichas.

    :param y_true: Array de valores reales.
    :param y_pred: Array de valores predichos.
    :param labels: Lista de etiquetas para las clases.
        Si es None, se inferirán de los datos.
    :return: Matriz de confusión como un array de NumPy.

    :authors: Nahuel Nicolas Alvarez
    :date: 19/04/2026
    """
    y_true, y_pred = _validacion_inputs(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    return np.array(
        [[np.sum((y_true == i) & (y_pred == j)) for j in labels] for i in labels],
        dtype=int,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: ArrayLike,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = "Blues",
) -> tuple[Figure, Axes]:
    """
    Genera una visualización de una matriz de confusión usando Matplotlib.

    :param cm: Matriz de confusión calculada previamente.
    :param labels: Etiquetas de las clases a mostrar en los ejes.
    :param figsize: Tamaño de la figura.
    :param cmap: Mapa de colores utilizado para el gráfico.
    :return: Figura y ejes del gráfico generado.

    :authors: Nahuel Nicolas Alvarez
    :date: 19/04/2026
    """
    cm = np.asarray(cm)
    labels = np.asarray(labels)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm debe ser una matriz cuadrada de dos dimensiones.")

    if cm.shape[0] != len(labels):
        raise ValueError(
            "La cantidad de labels debe coincidir con las dimensiones de la matriz."
        )

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicción",
        ylabel="Valor real",
        title="Matriz de confusión",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    return fig, ax
