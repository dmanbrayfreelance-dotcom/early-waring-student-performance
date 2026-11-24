"""
Test Suite: tests.test_data_prep
Description: Unit and integration tests for UCI dataset loading, Qin et al. (2023)
             feature engineering, strict early-warning leakage prevention, and
             preprocessing pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    build_feature_views,
    create_targets,
    engineer_features_qin_et_al,
    load_uci_data,
    prepare_dataset,
)


def test_load_uci_data_mat():
    df = load_uci_data(subject="mat")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 395
    assert "student_id" in df.columns
    assert "G1" in df.columns
    assert "G2" in df.columns
    assert "G3" in df.columns
    assert df["subject"].iloc[0] == "Mathematics"


def test_load_uci_data_por():
    df = load_uci_data(subject="por")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 649
    assert "student_id" in df.columns
    assert df["subject"].iloc[0] == "Portuguese"


def test_engineer_features_qin_et_al():
    df = load_uci_data(subject="mat")
    df_eng = engineer_features_qin_et_al(df)

    expected_cols = [
        "study_to_absence_ratio",
        "relative_g1_cohort",
        "relative_absence_cohort",
        "parental_edu_avg",
        "alcohol_workday_ratio",
        "total_alcohol",
        "social_study_ratio",
        "academic_strain",
    ]
    for col in expected_cols:
        assert col in df_eng.columns, f"Engineered column {col} missing from output."
        assert not df_eng[col].isnull().all(), f"Engineered column {col} is all null."

    # Validate study_to_absence_ratio calculation
    row0 = df_eng.iloc[0]
    expected_ratio = row0["studytime"] / (row0["absences"] + 1.0)
    assert np.isclose(row0["study_to_absence_ratio"], expected_ratio)


def test_strict_early_warning_leakage_exclusion():
    df = load_uci_data(subject="mat")
    df_eng = engineer_features_qin_et_al(df)

    # 1. Early-Warning view: strictly no downstream marks
    X_early, cat_cols_e, num_cols_e = build_feature_views(df_eng, mode="early_warning")
    assert "G2" not in X_early.columns, "Data Leakage Alert: G2 found in early_warning view!"
    assert "G3" not in X_early.columns, "Data Leakage Alert: G3 found in early_warning view!"
    assert "student_id" not in X_early.columns
    assert "G1" in X_early.columns

    # 2. Full-Feature view: includes G2, excludes G3
    X_full, cat_cols_f, num_cols_f = build_feature_views(df_eng, mode="full_feature")
    assert "G2" in X_full.columns, "G2 expected in full_feature benchmark view."
    assert "G3" not in X_full.columns, "Target G3 should not be in feature matrix."


def test_create_targets():
    df = load_uci_data(subject="mat")
    y_score, y_binary, y_risk = create_targets(df, pass_threshold=10.0)

    assert len(y_score) == len(df)
    assert len(y_binary) == len(df)
    assert len(y_risk) == len(df)

    # Validate pass/fail threshold
    for g3, p_val in zip(y_score, y_binary):
        if g3 >= 10.0:
            assert p_val == 1
        else:
            assert p_val == 0

    # Validate risk tiers are integers 0, 1, 2
    unique_risks = set(y_risk.unique())
    assert unique_risks.issubset({0, 1, 2})


def test_prepare_dataset_pipeline():
    bundle = prepare_dataset(subject="mat", mode="early_warning", test_size=0.2, random_state=42)

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    preprocessor = bundle["preprocessor"]

    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == 316
    assert len(X_test) == 79
    assert hasattr(preprocessor, "transform")
