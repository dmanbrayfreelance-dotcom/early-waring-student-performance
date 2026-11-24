"""
Unit Tests for src.data: Dataset Generation, Schema Validation, CSV I/O, and Target Distributions.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data import (
    DEFAULT_PASS_THRESHOLD,
    EC_COURSE_FEATURES,
    FUTURE_ASSESSMENT_FEATURES,
    SAL_ALL_FEATURES,
    SAL_BEHAVIORAL_FEATURES,
    SAL_HISTORICAL_FEATURES,
    STUDENT_IDENTIFIERS,
    TARGET_COLUMNS,
    compute_risk_label,
    export_dataset_to_csv,
    generate_student_dataset,
    load_dataset_from_csv,
    summarize_dataset,
    validate_dataset_schema,
)


class TestDatasetGeneration:
    """Tests synthetic dataset generation parameters, structure, and shapes."""

    def test_dataset_generation_shape(self):
        n_students = 150
        n_courses = 5
        df = generate_student_dataset(n_students=n_students, n_courses=n_courses, random_seed=42)

        assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"
        assert len(df) == n_students, f"Expected {n_students} rows, got {len(df)}"
        assert not df.empty, "Generated DataFrame must not be empty"

    def test_required_columns_present(self):
        df = generate_student_dataset(n_students=50, n_courses=4, random_seed=123)

        # Check Identifiers
        for col in STUDENT_IDENTIFIERS:
            assert col in df.columns, f"Identifier '{col}' missing from generated dataset"

        # Check SAL Features
        for col in SAL_HISTORICAL_FEATURES:
            assert col in df.columns, f"SAL historical feature '{col}' missing"
        for col in SAL_BEHAVIORAL_FEATURES:
            assert col in df.columns, f"SAL behavioral feature '{col}' missing"

        # Check EC Features
        for col in EC_COURSE_FEATURES:
            assert col in df.columns, f"EC course feature '{col}' missing"

        # Check Future Assessment Features
        for col in FUTURE_ASSESSMENT_FEATURES:
            assert col in df.columns, f"Future assessment feature '{col}' missing"

        # Check Targets
        for col in TARGET_COLUMNS:
            assert col in df.columns, f"Target column '{col}' missing"

    def test_correct_data_types(self):
        df = generate_student_dataset(n_students=60, n_courses=3, random_seed=99)

        # Numeric SAL features
        numeric_sal = [
            "previous_gpa",
            "prev_sub_score_1",
            "mean_hist_score",
            "daily_study_hours",
            "absence_ratio",
            "assignment_completion_pct",
        ]
        for col in numeric_sal:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' must be numeric"

        # Categorical EC features
        cat_ec = ["course_id", "subject_category", "department", "learning_mode"]
        for col in cat_ec:
            assert pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]), f"Column '{col}' must be string/object"

        # Targets
        assert pd.api.types.is_numeric_dtype(df["final_score"]), "'final_score' must be numeric"
        assert df["pass_fail"].isin(["Pass", "Fail"]).all(), "'pass_fail' values must be 'Pass' or 'Fail'"
        assert df["risk_level"].isin(["Normal", "Warning 1", "Warning 2"]).all(), "'risk_level' must be Normal/Warning 1/Warning 2"

    def test_target_distributions(self):
        df = generate_student_dataset(n_students=500, n_courses=6, random_seed=42)

        # Final score range
        assert (df["final_score"] >= 0.0).all() and (df["final_score"] <= 100.0).all(), (
            "Final scores must be bounded in [0, 100]"
        )

        # Pass / Fail consistency with threshold
        pass_mask = df["final_score"] >= DEFAULT_PASS_THRESHOLD
        assert (df.loc[pass_mask, "pass_fail"] == "Pass").all(), "Scores >= 40 must be labeled 'Pass'"
        assert (df.loc[~pass_mask, "pass_fail"] == "Fail").all(), "Scores < 40 must be labeled 'Fail'"

        # Risk distribution representation (all 3 classes should be present)
        risk_counts = df["risk_level"].value_counts().to_dict()
        assert "Normal" in risk_counts, "Normal risk level must be present"
        assert "Warning 1" in risk_counts, "Warning 1 risk level must be present"
        assert "Warning 2" in risk_counts, "Warning 2 risk level must be present"
        assert risk_counts["Normal"] > risk_counts["Warning 2"], "Normal students should typically exceed high-risk students"

    def test_missing_value_compatibility(self):
        missing_rate = 0.05
        df = generate_student_dataset(n_students=300, random_seed=42, missing_rate=missing_rate)

        # Targets and Identifiers must NEVER have missing values
        for col in TARGET_COLUMNS + STUDENT_IDENTIFIERS:
            assert df[col].isnull().sum() == 0, f"Target or identifier '{col}' must not contain missing values"

        # Check that missingness was injected in target behavioral fields
        total_missing = df.isnull().sum().sum()
        assert total_missing > 0, "Expected missing values in injected behavioral columns"


class TestReproducibility:
    """Verifies that seed controls output deterministically."""

    def test_reproducibility_same_seed(self):
        df1 = generate_student_dataset(n_students=100, random_seed=777)
        df2 = generate_student_dataset(n_students=100, random_seed=777)

        pd.testing.assert_frame_equal(df1, df2), "Datasets generated with identical seeds must be identical"

    def test_reproducibility_different_seed(self):
        df1 = generate_student_dataset(n_students=100, random_seed=111)
        df2 = generate_student_dataset(n_students=100, random_seed=222)

        assert not df1["final_score"].equals(df2["final_score"]), "Different seeds must generate distinct scores"


class TestRiskLabelLogic:
    """Tests the multi-factor risk categorization rubric."""

    def test_critical_risk_on_failure(self):
        row = {"final_score": 32.0, "previous_gpa": 7.0, "prev_sem_attendance": 85.0}
        assert compute_risk_label(row) == "Warning 2", "Failing score must yield Warning 2"

    def test_critical_risk_on_severe_distress(self):
        # Passing score, but severe GPA and attendance collapse
        row = {
            "final_score": 45.0,
            "previous_gpa": 4.2,
            "prev_sem_attendance": 55.0,
            "num_backlogs": 0,
            "absence_ratio": 0.1,
        }
        assert compute_risk_label(row) == "Warning 2", "Low GPA + low attendance must trigger Warning 2"

    def test_warning_1_attention_needed(self):
        row = {
            "final_score": 48.0,
            "previous_gpa": 7.5,
            "prev_sem_attendance": 82.0,
            "num_backlogs": 0,
            "daily_study_hours": 3.0,
            "assignment_completion_pct": 85.0,
        }
        assert compute_risk_label(row) == "Warning 1", "Score < 52 must trigger Warning 1"

    def test_normal_risk_for_good_standing(self):
        row = {
            "final_score": 78.0,
            "previous_gpa": 8.5,
            "prev_sem_attendance": 90.0,
            "num_backlogs": 0,
            "absence_ratio": 0.05,
            "consecutive_absence_days": 1,
            "daily_study_hours": 4.5,
            "assignment_completion_pct": 95.0,
        }
        assert compute_risk_label(row) == "Normal", "High performing student must be labeled Normal"


class TestSchemaValidationAndRejection:
    """Tests schema validation, missing column rejection, and corrupted value detection."""

    def test_valid_schema_passes(self):
        df = generate_student_dataset(n_students=50, random_seed=42)
        is_valid, issues, meta = validate_dataset_schema(df, is_training=True)

        assert is_valid is True, f"Valid dataset failed schema check: {issues}"
        assert len(meta["sal_features_missing"]) == 0
        assert len(meta["ec_features_missing"]) == 0
        assert len(meta["targets_missing"]) == 0

    def test_missing_required_sal_feature_rejected(self):
        df = generate_student_dataset(n_students=30, random_seed=42)
        df_corrupt = df.drop(columns=["previous_gpa"])

        is_valid, issues, meta = validate_dataset_schema(df_corrupt, is_training=True)
        assert is_valid is False, "Dataset missing 'previous_gpa' must be rejected"
        assert "previous_gpa" in meta["sal_features_missing"]

    def test_missing_required_ec_feature_rejected(self):
        df = generate_student_dataset(n_students=30, random_seed=42)
        df_corrupt = df.drop(columns=["course_difficulty"])

        is_valid, issues, meta = validate_dataset_schema(df_corrupt, is_training=True)
        assert is_valid is False, "Dataset missing 'course_difficulty' must be rejected"
        assert "course_difficulty" in meta["ec_features_missing"]

    def test_missing_target_in_training_mode_rejected(self):
        df = generate_student_dataset(n_students=30, random_seed=42)
        df_inference = df.drop(columns=["final_score"])

        is_valid, issues, meta = validate_dataset_schema(df_inference, is_training=True)
        assert is_valid is False, "Dataset missing 'final_score' in training mode must fail"

    def test_empty_dataframe_rejected(self):
        df_empty = pd.DataFrame()
        is_valid, issues, _ = validate_dataset_schema(df_empty)
        assert is_valid is False
        assert any("CRITICAL: DataFrame is empty" in issue for issue in issues)


class TestCSVExportAndLoad:
    """Tests exporting to CSV and reloading with schema validation."""

    def test_export_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_students.csv"
            df_original = generate_student_dataset(n_students=80, random_seed=42)

            # Export
            exported_path = export_dataset_to_csv(df_original, tmp_path)
            assert exported_path.exists(), "Exported CSV file must exist on disk"

            # Load back
            df_loaded, report = load_dataset_from_csv(exported_path, validate=True, is_training=True)

            assert report["is_valid"] is True
            assert len(df_loaded) == len(df_original)
            assert set(df_loaded.columns) == set(df_original.columns)

    def test_load_nonexistent_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_from_csv("non_existent_directory/fake_file.csv")

    def test_summarize_dataset_structure(self):
        df = generate_student_dataset(n_students=120, random_seed=42)
        summary = summarize_dataset(df)

        assert summary["total_records"] == 120
        assert summary["feature_counts"]["sal_features"] == len(SAL_ALL_FEATURES)
        assert summary["feature_counts"]["ec_features"] == len(EC_COURSE_FEATURES)
        assert "final_score" in summary["target_distributions"]
        assert "pass_fail" in summary["target_distributions"]
        assert "risk_level" in summary["target_distributions"]
