"""
Script: train.py
Description: CLI training and benchmarking pipeline for Explainable Student Academic
             Performance Prediction and At-Risk Early-Warning Dashboard.
             Implements strict early-warning leakage guards, Stacking Ensembles across
             7 model families, Qin et al. (2023) feature interactions, SHAP explainers,
             and artifact persistence for instant Streamlit UI rendering.

Usage:
  python train.py --mode early_warning --dataset mat
  python train.py --mode both --dataset mat --use-smote
  python train.py --fast-dev
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    EXPLAIN_DIR,
    INVERSE_UCI_RISK_MAPPING,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PREPROCESSORS_DIR,
    RANDOM_SEED,
    UCI_RISK_MAPPING,
    UCI_RISK_TIERS,
)
from src.data_prep import prepare_dataset
from src.evaluate import (
    compare_models,
    evaluate_binary_classification,
    evaluate_multiclass_risk,
    evaluate_regression,
)
from src.explain import ModelExplainer
from src.models import (
    get_binary_classification_models,
    get_multiclass_risk_models,
    get_regression_models,
    train_and_save_model,
)
from src.utils import get_logger, save_artifact, save_json

logger = get_logger("TRAIN_PIPELINE")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Explainable Student Performance Prediction & Early-Warning Training Pipeline."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["early_warning", "full_feature", "both"],
        default="early_warning",
        help="Feature view mode: 'early_warning' (strictly excludes downstream marks), 'full_feature' (includes G2), or 'both'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mat", "por", "both"],
        default="mat",
        help="UCI Dataset to train on: 'mat' (Mathematics), 'por' (Portuguese), or 'both'.",
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        default=False,
        help="Apply SMOTE resampling to the training split for minority risk tiers.",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        default=False,
        help="Fast development mode: train a curated subset of representative models for rapid verification.",
    )
    return parser.parse_args()


def run_training_experiment(
    mode: str,
    subject: str,
    use_smote: bool = False,
    fast_dev: bool = False,
) -> Dict[str, Any]:
    """
    Executes a complete training and benchmarking cycle for a specific mode and dataset.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING EXPERIMENT: Mode='%s', Dataset='%s', SMOTE=%s, FastDev=%s", mode, subject, use_smote, fast_dev)
    logger.info("=" * 80)

    # 1. Prepare Dataset (with strict early_warning leakage boundaries)
    data_bundle = prepare_dataset(
        subject=subject,
        mode=mode,
        use_smote=use_smote,
        random_state=RANDOM_SEED,
    )

    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_train = data_bundle["y_train"]
    y_test = data_bundle["y_test"]
    preprocessor = data_bundle["preprocessor"]
    feature_names = data_bundle["feature_names"]
    test_students = data_bundle["test_student_df"]

    # Persist fitted preprocessor
    prep_filename = f"{mode}_{subject}_preprocessor.joblib"
    save_artifact(preprocessor, PREPROCESSORS_DIR / prep_filename)
    # Also save as default pipeline
    save_artifact(preprocessor, PREPROCESSORS_DIR / "fitted_pipeline.joblib")
    save_artifact(feature_names, PREPROCESSORS_DIR / f"{mode}_{subject}_feature_names.joblib")
    save_artifact(feature_names, PREPROCESSORS_DIR / "feature_names.joblib")

    # ==========================================================================
    # TASK A: CONTINUOUS SCORE REGRESSION (G3)
    # ==========================================================================
    logger.info("--- Training Task A: Continuous Final Score Regression ---")
    reg_models = get_regression_models(random_state=RANDOM_SEED)
    if fast_dev:
        # In fast dev mode, select key representative estimators
        reg_models = {
            k: v for k, v in reg_models.items()
            if k in ["Linear Regression", "Random Forest Regressor", "Stacking Regressor"]
        }

    reg_results: List[Dict[str, Any]] = []
    fitted_reg_models: Dict[str, Any] = {}

    for name, model in reg_models.items():
        clean_name = name.lower().replace(" ", "_")
        train_and_save_model(
            model=model,
            model_name=name,
            task=f"{mode}_{subject}_regression",
            X_train=X_train,
            y_train=y_train["score"],
            save=True,
        )
        # Also save with standard prefix for app.py default loading
        train_and_save_model(
            model=model,
            model_name=name,
            task="regression",
            X_train=X_train,
            y_train=y_train["score"],
            save=True,
        )
        fitted_reg_models[name] = model

        # Evaluate on test set
        preds = model.predict(X_test)
        metrics = evaluate_regression(
            y_true=y_test["score"],
            y_pred=preds,
            model_name=name,
            generate_plots=False,
        )
        reg_results.append(metrics)

    df_reg_metrics = pd.DataFrame(reg_results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    df_reg_metrics.to_csv(METRICS_DIR / f"{mode}_{subject}_regression_metrics.csv", index=False)
    save_json(reg_results, METRICS_DIR / f"{mode}_{subject}_regression_metrics.json")

    # ==========================================================================
    # TASK B: BINARY ACADEMIC CLASSIFICATION (PASS / FAIL)
    # ==========================================================================
    logger.info("--- Training Task B: Binary Pass/Fail Classification ---")
    bin_models = get_binary_classification_models(random_state=RANDOM_SEED)
    if fast_dev:
        bin_models = {
            k: v for k, v in bin_models.items()
            if k in ["Logistic Regression", "Random Forest Classifier", "Stacking Classifier"]
        }

    bin_results: List[Dict[str, Any]] = []
    fitted_bin_models: Dict[str, Any] = {}

    y_test_bin_labels = np.where(y_test["binary"] == 1, "Pass", "Fail")

    for name, model in bin_models.items():
        train_and_save_model(
            model=model,
            model_name=name,
            task=f"{mode}_{subject}_binary",
            X_train=X_train,
            y_train=y_train["binary"],
            save=True,
        )
        train_and_save_model(
            model=model,
            model_name=name,
            task="binary",
            X_train=X_train,
            y_train=y_train["binary"],
            save=True,
        )
        fitted_bin_models[name] = model

        preds_int = model.predict(X_test)
        preds_labels = np.where(np.array(preds_int) == 1, "Pass", "Fail")
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_binary_classification(
            y_true=y_test_bin_labels,
            y_pred=preds_labels,
            y_prob=probs,
            model_name=name,
            pos_label="Pass",
            generate_plots=False,
        )
        bin_results.append(metrics)

    df_bin_metrics = pd.DataFrame(bin_results).sort_values(by="F1", ascending=False).reset_index(drop=True)
    df_bin_metrics.to_csv(METRICS_DIR / f"{mode}_{subject}_binary_metrics.csv", index=False)
    save_json(bin_results, METRICS_DIR / f"{mode}_{subject}_binary_metrics.json")

    # ==========================================================================
    # TASK C: THREE-LEVEL ACADEMIC RISK CLASSIFICATION (QIN ET AL. 2023)
    # ==========================================================================
    logger.info("--- Training Task C: Multi-Tier Academic Risk Classification ---")
    risk_models = get_multiclass_risk_models(random_state=RANDOM_SEED)
    if fast_dev:
        risk_models = {
            k: v for k, v in risk_models.items()
            if k in ["Logistic Regression", "Random Forest Classifier", "Stacking Classifier"]
        }

    risk_results: List[Dict[str, Any]] = []
    fitted_risk_models: Dict[str, Any] = {}

    # Use SMOTE training split if available
    X_train_risk = y_train.get("X_train_smote_risk", X_train)
    y_train_risk = y_train.get("risk_smote", y_train["risk"])

    y_test_risk_labels = [INVERSE_UCI_RISK_MAPPING[val] for val in y_test["risk"]]

    for name, model in risk_models.items():
        train_and_save_model(
            model=model,
            model_name=name,
            task=f"{mode}_{subject}_risk",
            X_train=X_train_risk,
            y_train=y_train_risk,
            save=True,
        )
        train_and_save_model(
            model=model,
            model_name=name,
            task="risk",
            X_train=X_train_risk,
            y_train=y_train_risk,
            save=True,
        )
        fitted_risk_models[name] = model

        preds_code = model.predict(X_test)
        preds_risk_labels = [INVERSE_UCI_RISK_MAPPING.get(int(c), "Low Risk") for c in preds_code]
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_multiclass_risk(
            y_true=y_test_risk_labels,
            y_pred=preds_risk_labels,
            y_prob=probs,
            model_name=name,
            class_names=UCI_RISK_TIERS,
            generate_plots=False,
        )
        risk_results.append(metrics)

    df_risk_metrics = pd.DataFrame(risk_results).sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    df_risk_metrics.to_csv(METRICS_DIR / f"{mode}_{subject}_risk_metrics.csv", index=False)
    save_json(risk_results, METRICS_DIR / f"{mode}_{subject}_risk_metrics.json")

    # ==========================================================================
    # EXPLAINABILITY & SHAP ARTIFACT GENERATION
    # ==========================================================================
    logger.info("--- Generating Global Explainability & SHAP Values ---")
    # Select best risk model for explanations
    best_risk_model_name = df_risk_metrics.iloc[0]["model_name"]
    best_risk_model = fitted_risk_models[best_risk_model_name]

    explainer = ModelExplainer(
        model=best_risk_model,
        feature_names=feature_names,
        task="multiclass_risk",
    )
    explainer.fit(X_train)
    global_exp = explainer.explain_global(X_test)

    # Persist explainer and precomputed explanation
    save_artifact(explainer, EXPLAIN_DIR / f"{mode}_{subject}_explainer.joblib")
    save_artifact(explainer, EXPLAIN_DIR / "explainer.joblib")
    save_artifact(global_exp, EXPLAIN_DIR / f"{mode}_{subject}_global_explanation.joblib")
    save_artifact(global_exp, EXPLAIN_DIR / "global_explanation.joblib")

    # ==========================================================================
    # TEST PREDICTIONS DATASET FOR INSTANT APP RENDERING
    # ==========================================================================
    logger.info("--- Generating and Persisting Test Set Watchlist Predictions ---")
    best_reg_name = df_reg_metrics.iloc[0]["model_name"]
    best_reg_model = fitted_reg_models[best_reg_name]

    pred_scores = best_reg_model.predict(X_test)
    pred_risk_codes = best_risk_model.predict(X_test)
    pred_risk_probs = best_risk_model.predict_proba(X_test) if hasattr(best_risk_model, "predict_proba") else np.zeros((len(X_test), 3))

    predictions_df = test_students.copy()
    predictions_df["actual_final_score"] = y_test["score"]
    predictions_df["predicted_final_score"] = np.round(pred_scores, 2)
    predictions_df["actual_pass_fail"] = y_test_bin_labels
    predictions_df["predicted_pass_fail"] = np.where(pred_scores >= 10.0, "Pass", "Fail")
    predictions_df["actual_risk_tier"] = y_test_risk_labels
    predictions_df["predicted_risk_tier"] = [INVERSE_UCI_RISK_MAPPING.get(int(c), "Low Risk") for c in pred_risk_codes]

    if pred_risk_probs.shape[1] >= 3:
        predictions_df["prob_low_risk"] = np.round(pred_risk_probs[:, 0], 3)
        predictions_df["prob_medium_risk"] = np.round(pred_risk_probs[:, 1], 3)
        predictions_df["prob_high_risk"] = np.round(pred_risk_probs[:, 2], 3)
        predictions_df["risk_score"] = np.round(pred_risk_probs[:, 2] * 100.0, 1)
    else:
        predictions_df["risk_score"] = np.where(predictions_df["predicted_risk_tier"] == "High Risk", 90.0, 20.0)

    # Save predictions
    pred_csv_path = PREDICTIONS_DIR / f"{mode}_{subject}_test_predictions.csv"
    predictions_df.to_csv(pred_csv_path, index=False)
    predictions_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    elapsed = time.time() - start_time
    logger.info("Completed experiment [%s, %s] in %.2f seconds.", mode, subject, elapsed)

    # ==========================================================================
    # DISPLAY BENCHMARK SUMMARY TABLES
    # ==========================================================================
    print("\n" + "=" * 90)
    print(f"BENCHMARK SUMMARY: Mode = '{mode}' | Dataset = '{subject.upper()}' | Elapsed = {elapsed:.1f}s")
    print("=" * 90)

    print("\n[TASK A: REGRESSION - FINAL GRADE G3 (0 - 20)]")
    print(df_reg_metrics[["model_name", "RMSE", "MAE", "R2"]].to_string(index=False))

    print("\n[TASK B: BINARY CLASSIFICATION - PASS (>=10) / FAIL (<10)]")
    cols_bin = [c for c in ["model_name", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if c in df_bin_metrics.columns]
    print(df_bin_metrics[cols_bin].to_string(index=False))

    print("\n[TASK C: MULTI-TIER RISK (IEEE Qin et al. 2023 Prioritizing High-Risk Recall)]")
    cols_risk = [c for c in ["model_name", "Accuracy", "Macro_F1", "Weighted_F1", "High_Risk_Recall"] if c in df_risk_metrics.columns]
    print(df_risk_metrics[cols_risk].to_string(index=False))
    print("=" * 90 + "\n")

    return {
        "mode": mode,
        "dataset": subject,
        "reg_metrics": df_reg_metrics,
        "bin_metrics": df_bin_metrics,
        "risk_metrics": df_risk_metrics,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """CLI Entrypoint for Model Training and Benchmarking."""
    args = parse_arguments()

    modes = ["early_warning", "full_feature"] if args.mode == "both" else [args.mode]
    subjects = ["mat", "por"] if args.dataset == "both" else [args.dataset]

    results = []
    for m in modes:
        for s in subjects:
            res = run_training_experiment(
                mode=m,
                subject=s,
                use_smote=args.use_smote,
                fast_dev=args.fast_dev,
            )
            results.append(res)

    logger.info("All training experiments completed successfully.")


if __name__ == "__main__":
    main()
