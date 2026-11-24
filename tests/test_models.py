"""
Unit Tests for src.models: Regression, Binary Classification, Multi-Class Risk Models,
Stacking Ensembles, and Tabular Neural Network with Focal Loss.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_BINARY, TARGET_RISK, TARGET_SCORE
from src.data import generate_student_dataset
from src.features import EducationalPreprocessingPipeline, prepare_train_test_split
from src.models import (
    PyTorchTabularEstimator,
    get_binary_classification_models,
    get_multiclass_risk_models,
    get_regression_models,
    train_and_save_model,
)


@pytest.fixture
def preprocessed_splits():
    """Generates synthetic preprocessed features and target splits."""
    df = generate_student_dataset(n_students=120, n_courses=3, random_seed=42)

    df_train, df_test, y_train_risk, y_test_risk = prepare_train_test_split(
        df, target_col=TARGET_RISK, test_size=0.25, random_state=42
    )

    pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
    X_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)

    y_train_score = df_train[TARGET_SCORE]
    y_test_score = df_test[TARGET_SCORE]

    y_train_binary = df_train[TARGET_BINARY]
    y_test_binary = df_test[TARGET_BINARY]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_risk": y_train_risk,
        "y_test_risk": y_test_risk,
        "y_train_score": y_train_score,
        "y_test_score": y_test_score,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "pipeline": pipeline,
    }


class TestModelTrainingAndInference:
    """Tests training and prediction across regression, binary, and multi-class models."""

    def test_regression_models(self, preprocessed_splits):
        reg_models = get_regression_models(random_state=42)
        X_train = preprocessed_splits["X_train"]
        y_train = preprocessed_splits["y_train_score"]
        X_test = preprocessed_splits["X_test"]

        for name in ["Linear Regression", "Random Forest Regressor", "Stacking Regressor"]:
            assert name in reg_models, f"Model {name} missing from registry"
            model = reg_models[name]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            assert len(preds) == len(X_test)
            assert not np.isnan(preds).any()
            # Predictions should be reasonable scores
            assert preds.mean() > 0.0

    def test_binary_classification_models(self, preprocessed_splits):
        bin_models = get_binary_classification_models(random_state=42)
        X_train = preprocessed_splits["X_train"]
        y_train = preprocessed_splits["y_train_binary"]
        X_test = preprocessed_splits["X_test"]

        for name in ["Logistic Regression", "Random Forest Classifier", "Stacking Classifier"]:
            assert name in bin_models
            model = bin_models[name]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            assert len(preds) == len(X_test)
            assert set(preds).issubset({"Pass", "Fail"})

    def test_multiclass_risk_models(self, preprocessed_splits):
        risk_models = get_multiclass_risk_models(random_state=42)
        X_train = preprocessed_splits["X_train"]
        y_train = preprocessed_splits["y_train_risk"]
        X_test = preprocessed_splits["X_test"]

        for name in ["Logistic Regression", "Random Forest Classifier", "Stacking Classifier"]:
            assert name in risk_models
            model = risk_models[name]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            assert len(preds) == len(X_test)
            assert set(preds).issubset({"Normal", "Warning 1", "Warning 2"})

    def test_pytorch_tabular_estimator(self, preprocessed_splits):
        pipe = preprocessed_splits["pipeline"]
        estimator = PyTorchTabularEstimator(
            task="multiclass",
            sal_indices=pipe.sal_indices_,
            ec_indices=pipe.ec_indices_,
            epochs=5,
            random_state=42,
        )

        X_train = preprocessed_splits["X_train"]
        y_train = preprocessed_splits["y_train_risk"]
        X_test = preprocessed_splits["X_test"]

        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        probs = estimator.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert len(probs) == len(X_test)
        assert probs.shape[1] == 3
        # Probabilities should sum to approximately 1.0
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
