"""
Tests del módulo de reportes
"""

import pandas as pd
from pandas.testing import assert_frame_equal

from whiteboxml import reporting


def test_report_classification_perfect():
    """
    Test reporte de clasificación perfecto
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1": [1.0],
        }
    )

    df = reporting.classification_summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_report_regression_perfect():
    """
    Test reporte de regresión perfecto
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "MSE": [0.0],
            "MAE": [0.0],
            "R^2": [1.0],
        }
    )

    df = reporting.regression_summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_report_classification_imperfect():
    """
    Test reporte de clasificación imperfecto
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [0.5],
            "Precision": [1 / 3],
            "Recall": [1.0],
            "F1": [0.5],
        }
    )

    df = reporting.classification_summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_report_regression_imperfect():
    """
    Test reporte de regresión imperfecto
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "MSE": [0.5],
            "MAE": [0.5],
            "R^2": [-1.6666666666666665],
        }
    )

    df = reporting.regression_summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_report_classification_macro():
    """
    Test reporte de clasificación usando average='macro' explícito
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [0.5],
            "Precision": [2 / 3],
            "Recall": [2 / 3],
            "F1": [0.5],
        }
    )

    df = reporting.classification_summary(y_true, y_pred, average="macro")

    assert_frame_equal(df, expected_df)


def test_report_classification_micro():
    """
    Test reporte de clasificación usando average='micro' explícito
    :authors: Joaquín Palacio Feijóo
    :date: 15/04/2026
    """

    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [0.5],
            "Precision": [0.5],
            "Recall": [0.5],
            "F1": [0.5],
        }
    )

    df = reporting.classification_summary(y_true, y_pred, average="micro")

    assert_frame_equal(df, expected_df)
