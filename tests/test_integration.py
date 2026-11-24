"""
Integration Test: End-to-End Workflow Verification for the SAL-EC Educational System.

Tests complete sequence:
  Synthetic Data Generation -> Schema Validation -> Privacy Anonymization ->
  Early-Warning Guard -> Preprocessing Pipeline -> Model Training ->
  Multi-Objective Evaluation -> Global & Local Explanation ->
  Faculty Recommendations -> Batch CSV Inference & Export.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_BINARY, TARGET_RISK, TARGET_SCORE
from src.data import (
    export_dataset_to_csv,
    generate_student_dataset,
    load_dataset_from_csv,
    validate_dataset_schema,
)
from src.evaluate import (
    compare_models,
    evaluate_binary_classification,
    evaluate_multiclass_risk,
    evaluate_regression,
)
from src.explain import FacultyInterventionEngine, ModelExplainer
from src.features import (
    EducationalPreprocessingPipeline,
    anonymize_student_records,
    apply_early_warning_guard,
    apply_smote_resampling,
    prepare_train_test_split,
)
from src.models import (
    get_binary_classification_models,
    get_multiclass_risk_models,
    get_regression_models,
    train_and_save_model,
)


def test_full_end_to_end_pipeline():
    """
    Validates that all architectural modules interface seamlessly from raw data
    to scored predictions and faculty reports.
    """
    # 1. Dataset Generation
    df_raw = generate_student_dataset(n_students=150, n_courses=4, random_seed=42)
    assert len(df_raw) == 150
    assert not df_raw.empty

    # 2. Schema Validation
    is_valid, issues, meta = validate_dataset_schema(df_raw, is_training=True)
    assert is_valid is True

    # 3. Privacy Anonymization
    df_anon, id_map = anonymize_student_records(df_raw, method="anonymous_id")
    assert "student_name" not in df_anon.columns
    assert "student_id" not in df_anon.columns
    assert "anonymous_student_id" in df_anon.columns

    # 4. Train / Test Split Isolation
    df_train, df_test, y_train_risk, y_test_risk = prepare_train_test_split(
        df_anon, target_col=TARGET_RISK, test_size=0.20, random_state=42
    )
    assert len(df_train) == 120
    assert len(df_test) == 30

    y_train_score = df_train[TARGET_SCORE]
    y_test_score = df_test[TARGET_SCORE]

    y_train_binary = df_train[TARGET_BINARY]
    y_test_binary = df_test[TARGET_BINARY]

    # 5. Preprocessing Pipeline (Fitted ONLY on training data)
    pipeline = EducationalPreprocessingPipeline(early_warning_mode=True)
    X_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)

    assert X_train.shape[0] == 120
    assert X_test.shape[0] == 30
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # 6. SMOTE Resampling (Training Fold ONLY)
    X_train_res, y_train_risk_res, smote_meta = apply_smote_resampling(
        X_train, y_train_risk, random_state=42
    )
    assert smote_meta["resampled"] is True
    assert len(X_train_res) >= len(X_train)

    # 7. Model Training & Evaluation
    # A. Regression
    reg_models = get_regression_models(random_state=42)
    reg_model = reg_models["Random Forest Regressor"]
    reg_model.fit(X_train, y_train_score)
    reg_preds = reg_model.predict(X_test)
    reg_eval = evaluate_regression(y_test_score, reg_preds, model_name="RandomForestRegressor", generate_plots=False)
    assert "R2" in reg_eval

    # B. Binary Classification
    bin_models = get_binary_classification_models(random_state=42)
    bin_model = bin_models["Random Forest Classifier"]
    bin_model.fit(X_train, y_train_binary)
    bin_preds = bin_model.predict(X_test)
    bin_eval = evaluate_binary_classification(y_test_binary, bin_preds, model_name="RandomForestBinary", generate_plots=False)
    assert "F1" in bin_eval

    # C. Multi-Class Risk Classification
    risk_models = get_multiclass_risk_models(random_state=42)
    risk_model = risk_models["Random Forest Classifier"]
    risk_model.fit(X_train_res, y_train_risk_res)
    risk_preds = risk_model.predict(X_test)
    risk_probs = risk_model.predict_proba(X_test)
    risk_eval = evaluate_multiclass_risk(y_test_risk, risk_preds, y_prob=risk_probs, model_name="RandomForestRisk", generate_plots=False)
    assert "Macro_F1" in risk_eval
    assert "Warning_2_Recall" in risk_eval

    # 8. Model Comparison
    comparison_df = compare_models([risk_eval], task="multiclass_risk")
    assert len(comparison_df) == 1

    # 9. Explainability (Global and Local)
    explainer = ModelExplainer(
        model=risk_model,
        feature_names=pipeline.feature_names_out_,
        sal_indices=pipeline.sal_indices_,
        ec_indices=pipeline.ec_indices_,
        task="multiclass_risk",
    )
    explainer.fit(X_train)

    global_exp = explainer.explain_global(X_test)
    assert "sal_vs_ec" in global_exp
    assert "feature_importance" in global_exp

    local_exp = explainer.explain_local_instance(X_test[0], target_class_idx=1)
    assert "top_risk_drivers" in local_exp
    assert "top_protective_factors" in local_exp

    # 10. Faculty Academic Intervention Engine
    engine = FacultyInterventionEngine()
    first_student_profile = df_test.iloc[0].to_dict()
    recommendations = engine.generate_recommendations(
        first_student_profile,
        predicted_risk=str(risk_preds[0]),
        predicted_score=float(reg_preds[0]),
        top_risk_drivers=local_exp["top_risk_drivers"],
    )
    assert len(recommendations) >= 1
    assert "action_items" in recommendations[0]

    # 11. Batch CSV Scoring Round-Trip
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch_input_path = Path(tmp_dir) / "incoming_students.csv"
        export_dataset_to_csv(df_test, batch_input_path)

        # Reload & validate
        loaded_df, val_report = load_dataset_from_csv(batch_input_path, validate=True, is_training=False)
        assert val_report["is_valid"] is True

        # Transform and score
        X_batch = pipeline.transform(loaded_df)
        batch_scores = reg_model.predict(X_batch)
        batch_pass_fail = bin_model.predict(X_batch)
        batch_risks = risk_model.predict(X_batch)

        results_df = loaded_df.copy()
        results_df["predicted_final_score"] = np.round(batch_scores, 2)
        results_df["predicted_pass_fail"] = batch_pass_fail
        results_df["predicted_risk_level"] = batch_risks

        batch_output_path = Path(tmp_dir) / "scored_students.csv"
        export_dataset_to_csv(results_df, batch_output_path)
        assert batch_output_path.exists()
