"""
Unit Tests for src.evaluate: Metrics Calculation, Model Comparison, and Diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compare_models,
    evaluate_binary_classification,
    evaluate_multiclass_risk,
    evaluate_regression,
)


class TestEvaluationMetrics:
    """Tests metric calculations across regression, binary, and multi-class objectives."""

    def test_evaluate_regression(self):
        y_true = np.array([60.0, 75.0, 80.0, 45.0, 90.0])
        y_pred = np.array([58.0, 73.0, 82.0, 48.0, 88.0])

        metrics = evaluate_regression(y_true, y_pred, model_name="TestRegressor", generate_plots=False)

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert metrics["MAE"] == pytest.approx(2.2, rel=1e-2)
        assert metrics["R2"] > 0.90

    def test_evaluate_binary_classification(self):
        y_true = np.array(["Pass", "Pass", "Fail", "Pass", "Fail"])
        y_pred = np.array(["Pass", "Pass", "Fail", "Fail", "Fail"])

        metrics = evaluate_binary_classification(
            y_true, y_pred, model_name="TestBinary", generate_plots=False
        )

        assert metrics["Accuracy"] == 0.80
        assert metrics["F1"] > 0.0
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 2

    def test_evaluate_multiclass_risk(self):
        y_true = np.array(["Normal", "Warning 1", "Warning 2", "Normal", "Warning 2"])
        y_pred = np.array(["Normal", "Warning 1", "Warning 2", "Warning 1", "Warning 2"])

        metrics = evaluate_multiclass_risk(
            y_true, y_pred, model_name="TestRisk", generate_plots=False
        )

        assert "Accuracy" in metrics
        assert "Macro_F1" in metrics
        assert "Warning_2_Recall" in metrics
        # Both Warning 2 cases were predicted correctly -> Warning 2 Recall = 1.0
        assert metrics["Warning_2_Recall"] == 1.0
        assert "per_class" in metrics
        assert "Warning 2" in metrics["per_class"]

    def test_compare_models_ranking(self):
        results = [
            {"model_name": "Model A", "Warning_2_Recall": 0.75, "Macro_F1": 0.80},
            {"model_name": "Model B", "Warning_2_Recall": 0.90, "Macro_F1": 0.85},
            {"model_name": "Model C", "Warning_2_Recall": 0.60, "Macro_F1": 0.70},
        ]

        df_comp = compare_models(results, task="multiclass_risk")

        assert len(df_comp) == 3
        # Model B has highest Warning 2 Recall and should be at index 0
        assert df_comp.iloc[0]["Model"] == "Model B"
