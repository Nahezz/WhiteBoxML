"""
Tests de matriz de confusión para el módulo de reporting.

:authors: Nahuel Nicolas Alvarez
:date: 19/04/2026
"""

import numpy as np

from whiteboxml.reporting.visualizacion.confusion_matrix import confusion_matrix


def test_confusion_matrix_binaria_basica():
    """
    Test matriz de confusión binaria básica.
    :authors: Alvarez Nahuel Nicolas
    :date: 19/04/2026
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    expected = np.array([[1, 1], [1, 1]])

    result = confusion_matrix(y_true, y_pred)

    assert np.array_equal(result, expected)
