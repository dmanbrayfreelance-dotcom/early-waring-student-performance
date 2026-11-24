"""
Unit Tests for src.explain: Model Explainability (SHAP & Permutation Fallback),
SAL vs EC Contribution Breakdown, and Faculty Academic Intervention Engine.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import TARGET_RISK
from src.data import generate_student_dataset
from src.explain import FacultyInterventionEngine, ModelExplainer
from src.features import EducationalPreprocessingPipeline, prepare_train_test_split


@pytest.fixture
def trained_explainer_setup():
    df = generate_student_dataset(n_students=100, n_courses=3, random_seed=42)
    df_train, df_test, y_train, y_test = prepare_train_test_split(
        df, target_col=TARGET_RISK, test_size=0.25, random_state=42
    )

    pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
    X_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)

    model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    explainer = ModelExplainer(
        model=model,
        feature_names=pipeline.feature_names_out_,
        sal_indices=pipeline.sal_indices_,
        ec_indices=pipeline.ec_indices_,
        task="multiclass_risk",
    )
    explainer.fit(X_train)

    return {
        "explainer": explainer,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "pipeline": pipeline,
        "raw_test_students": df_test,
    }


class TestModelExplainability:
    """Tests global and local explainability."""

    def test_explain_global(self, trained_explainer_setup):
        explainer = trained_explainer_setup["explainer"]
        X_test = trained_explainer_setup["X_test"]

        report = explainer.explain_global(X_test)

        assert "method" in report
        assert "feature_importance" in report
        assert len(report["feature_importance"]) == len(explainer.feature_names)
        assert "sal_vs_ec" in report

        sal_ec = report["sal_vs_ec"]
        assert "sal_percentage" in sal_ec
        assert "ec_percentage" in sal_ec
        # SAL + EC should sum to approximately 100%
        assert sal_ec["sal_percentage"] + sal_ec["ec_percentage"] == pytest.approx(100.0, abs=0.5)

    def test_explain_local_instance(self, trained_explainer_setup):
        explainer = trained_explainer_setup["explainer"]
        X_test = trained_explainer_setup["X_test"]

        # Explain the first test student
        local_report = explainer.explain_local_instance(X_test[0], target_class_idx=1)

        assert "top_risk_drivers" in local_report
        assert "top_protective_factors" in local_report
        assert len(local_report["all_attributions"]) == len(explainer.feature_names)


class TestFacultyInterventionEngine:
    """Tests rule-based academic intervention triggers."""

    def test_intervention_generation_for_struggling_student(self):
        engine = FacultyInterventionEngine()
        struggling_profile = {
            "prev_sem_attendance": 62.0,
            "absence_count": 18,
            "absence_ratio": 0.32,
            "consecutive_absence_days": 6,
            "daily_study_hours": 1.2,
            "previous_gpa": 4.1,
            "num_backlogs": 2,
            "assignment_completion_pct": 52.0,
            "lms_login_freq": 5.0,
            "course_difficulty": 4.2,
        }

        recommendations = engine.generate_recommendations(
            struggling_profile,
            predicted_risk="Warning 2",
            predicted_score=36.0,
        )

        assert len(recommendations) >= 3, "Struggling student should trigger multiple interventions"

        categories = [r["category"] for r in recommendations]
        assert any("Attendance" in cat for cat in categories)
        assert any("Curricular Remediation" in cat for cat in categories)
        assert any("Study Habits" in cat for cat in categories)

        # Check urgent priority presence
        priorities = [r["priority"] for r in recommendations]
        assert "Urgent" in priorities

    def test_intervention_generation_for_normal_student(self):
        engine = FacultyInterventionEngine()
        good_profile = {
            "prev_sem_attendance": 92.0,
            "absence_count": 2,
            "absence_ratio": 0.04,
            "consecutive_absence_days": 1,
            "daily_study_hours": 4.5,
            "previous_gpa": 8.8,
            "num_backlogs": 0,
            "assignment_completion_pct": 98.0,
            "lms_login_freq": 24.0,
            "course_difficulty": 2.8,
        }

        recommendations = engine.generate_recommendations(
            good_profile,
            predicted_risk="Normal",
            predicted_score=84.0,
        )

        assert len(recommendations) == 1
        assert recommendations[0]["category"] == "Academic Commendation"
