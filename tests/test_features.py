"""
Unit Tests for src.features: SAL-EC Feature Engineering, Leak-Free Preprocessing,
Privacy Anonymization, Early-Warning Guards, and SMOTE Resampling.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_TARGETS, FUTURE_ASSESSMENT_FEATURES, TARGET_RISK
from src.data import generate_student_dataset
from src.features import (
    CourseRelativeRankTransformer,
    EducationalPreprocessingPipeline,
    anonymize_student_records,
    apply_early_warning_guard,
    apply_smote_resampling,
    calculate_elearning_features,
    compute_class_weights_dict,
    prepare_train_test_split,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provides a deterministic 100-record student dataset for unit tests."""
    return generate_student_dataset(n_students=100, n_courses=4, random_seed=42, missing_rate=0.03)


class TestFeatureEngineeringTransforms:
    """Tests course relative ranking and e-learning feature engineering."""

    def test_course_relative_rank_fit_transform(self, sample_raw_data: pd.DataFrame):
        transformer = CourseRelativeRankTransformer(score_col="mean_hist_score", course_id_col="course_id")
        transformer.fit(sample_raw_data)

        assert len(transformer.course_stats_) > 0, "Transformer should learn course stats"
        assert "mean" in transformer.global_stats_

        df_transformed = transformer.transform(sample_raw_data)
        assert "student_course_relative_rank" in df_transformed.columns
        assert "course_percentile" in df_transformed.columns

        # Verify values bounded in [0, 1]
        ranks = df_transformed["student_course_relative_rank"]
        assert (ranks >= 0.0).all() and (ranks <= 1.0).all()

    def test_course_relative_rank_unseen_course_fallback(self, sample_raw_data: pd.DataFrame):
        transformer = CourseRelativeRankTransformer()
        transformer.fit(sample_raw_data)

        # Create a new student with an unseen course
        unseen_row = sample_raw_data.iloc[:2].copy()
        unseen_row["course_id"] = "UNSEEN_COURSE_999"
        unseen_row["mean_hist_score"] = 70.0

        df_out = transformer.transform(unseen_row)
        assert not df_out["student_course_relative_rank"].isnull().any(), "Unseen course should not produce NaN"

    def test_elearning_features(self, sample_raw_data: pd.DataFrame):
        df_elearn = calculate_elearning_features(sample_raw_data, threshold=0.5)
        assert "elearning_ratio" in df_elearn.columns
        assert "elearning_indicator" in df_elearn.columns

        ratios = df_elearn["elearning_ratio"]
        assert (ratios >= 0.0).all() and (ratios <= 1.0).all()
        assert set(df_elearn["elearning_indicator"].unique()).issubset({0, 1})


class TestPrivacyAndLeakageGuards:
    """Tests student anonymization and early-warning leakage guards."""

    def test_privacy_anonymization(self, sample_raw_data: pd.DataFrame):
        df_anon, mapping = anonymize_student_records(sample_raw_data, method="anonymous_id")

        assert "student_name" not in df_anon.columns, "Student name must be stripped"
        assert "student_id" not in df_anon.columns, "Original student_id must be stripped"
        assert "anonymous_student_id" in df_anon.columns, "Anonymous ID column must be present"
        assert len(mapping) == len(sample_raw_data["student_id"].unique())

    def test_early_warning_guard_mode_true(self, sample_raw_data: pd.DataFrame):
        df_guarded, log = apply_early_warning_guard(sample_raw_data, early_warning_mode=True)

        # All target columns must be removed
        for tgt in ALL_TARGETS:
            assert tgt not in df_guarded.columns, f"Target '{tgt}' leaked in features"

        # All future assessment columns must be removed
        for future_col in FUTURE_ASSESSMENT_FEATURES:
            assert future_col not in df_guarded.columns, f"Future mark '{future_col}' leaked in early-warning mode"

        # Exclusion log must contain entries
        assert len(log) > 0
        excluded_names = [item["column"] for item in log]
        assert "final_score" in excluded_names
        assert "final_exam_mark" in excluded_names

    def test_early_warning_guard_mode_false(self, sample_raw_data: pd.DataFrame):
        df_guarded, log = apply_early_warning_guard(sample_raw_data, early_warning_mode=False)

        # Targets must still be removed
        for tgt in ALL_TARGETS:
            assert tgt not in df_guarded.columns, f"Target '{tgt}' must never be in feature inputs"

        # Current assessment marks are permitted in full mode
        assert "final_exam_mark" in df_guarded.columns


class TestPreprocessingPipeline:
    """Tests the leak-free preprocessing pipeline and metadata tracking."""

    def test_pipeline_fit_transform(self, sample_raw_data: pd.DataFrame):
        pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
        X_mat = pipeline.fit_transform(sample_raw_data)

        assert isinstance(X_mat, np.ndarray), "Output must be a 2D numpy array"
        assert len(X_mat) == len(sample_raw_data)
        assert not np.isnan(X_mat).any(), "Processed feature matrix must not contain NaNs"
        assert len(pipeline.feature_names_out_) == X_mat.shape[1]

    def test_sal_and_ec_indices_tracking(self, sample_raw_data: pd.DataFrame):
        pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
        pipeline.fit(sample_raw_data)

        assert len(pipeline.sal_indices_) > 0, "SAL indices list must not be empty"
        assert len(pipeline.ec_indices_) > 0, "EC indices list must not be empty"
        # Indices should be mutually exclusive and cover all features
        total_tracked = len(pipeline.sal_indices_) + len(pipeline.ec_indices_)
        assert total_tracked == len(pipeline.feature_names_out_)


class TestTrainTestAndSMOTE:
    """Tests train/test partition isolation and SMOTE resampling."""

    def test_train_test_split_isolation(self, sample_raw_data: pd.DataFrame):
        df_train, df_test, y_train, y_test = prepare_train_test_split(
            sample_raw_data,
            target_col=TARGET_RISK,
            test_size=0.25,
            random_state=42,
        )

        assert len(df_train) == 75
        assert len(df_test) == 25
        # No index overlap
        assert len(set(df_train.index).intersection(set(df_test.index))) == 0

    def test_smote_resampling_applied_to_training_only(self, sample_raw_data: pd.DataFrame):
        df_train, df_test, y_train, y_test = prepare_train_test_split(
            sample_raw_data,
            target_col=TARGET_RISK,
            test_size=0.2,
            random_state=42,
        )

        pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
        X_train = pipeline.fit_transform(df_train)
        X_test = pipeline.transform(df_test)

        # Apply SMOTE to training set only
        X_res, y_res, meta = apply_smote_resampling(X_train, y_train, random_state=42)

        assert meta["resampled"] is True
        assert len(X_res) >= len(X_train), "Resampled training set should have >= original samples"
        # Test set is untouched
        assert len(X_test) == len(df_test), "Test set must not be resampled"

    def test_class_weights_computation(self, sample_raw_data: pd.DataFrame):
        y = sample_raw_data[TARGET_RISK]
        weights = compute_class_weights_dict(y)

        assert len(weights) == len(y.unique())
        # Minority class should have higher weight than majority class
        counts = y.value_counts()
        majority_cls = counts.index[0]
        minority_cls = counts.index[-1]
        assert weights[minority_cls] >= weights[majority_cls]
