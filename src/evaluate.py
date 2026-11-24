"""
Module: src.evaluate
Description: Comprehensive research evaluation metrics, comparison tables, and diagnostic
             plotting for regression, binary pass/fail, and multi-class academic risk models.

Key Evaluation Dimensions:
  - Regression: MAE, MSE, RMSE, R2, Actual vs Predicted, Residual analysis
  - Binary: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
  - Multi-Class Risk: Accuracy, Macro-F1, Weighted-F1, Warning 2 Recall, Per-Class metrics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS_DIR, PLOTS_DIR, REPORTS_DIR, RISK_CLASS_NAMES
from src.utils import get_logger, save_json, save_plot

logger = get_logger("SAL_EC_EVALUATE")


# ==============================================================================
# 1. REGRESSION EVALUATION METRICS & CHARTS
# ==============================================================================

def evaluate_regression(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str = "Regressor",
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Computes standard regression evaluation metrics and diagnostic plots.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "task": "regression",
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "num_test_samples": len(y_t),
    }

    if generate_plots:
        # 1. Actual vs Predicted Scatter Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_t, y_p, alpha=0.6, color="#2b5c8f", edgecolors="none", s=30)
        min_val = min(float(y_t.min()), float(y_p.min()))
        max_val = max(float(y_t.max()), float(y_p.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal (y = x)")
        ax.set_title(f"Actual vs Predicted Score: {model_name} (R² = {r2:.3f})")
        ax.set_xlabel("Actual Final Score")
        ax.set_ylabel("Predicted Final Score")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        clean_name = model_name.lower().replace(" ", "_")
        plot_path = PLOTS_DIR / f"regression_actual_vs_pred_{clean_name}.png"
        save_plot(fig, plot_path)
        plt.close(fig)

        # 2. Residual Distribution Plot
        residuals = y_t - y_p
        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax_res, color="#d95f02", bins=20)
        ax_res.axvline(0, color="black", linestyle="--", lw=1)
        ax_res.set_title(f"Residual Distribution: {model_name}")
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.grid(True, linestyle=":", alpha=0.6)

        res_path = PLOTS_DIR / f"regression_residuals_{clean_name}.png"
        save_plot(fig_res, res_path)
        plt.close(fig_res)

    logger.info("Regression eval [%s]: MAE=%.2f, RMSE=%.2f, R2=%.3f", model_name, mae, rmse, r2)
    return metrics


# ==============================================================================
# 2. BINARY CLASSIFICATION METRICS & CHARTS
# ==============================================================================

def evaluate_binary_classification(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Classifier",
    pos_label: str = "Pass",
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Computes binary evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    and confusion matrix heatmaps.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=["Fail", "Pass"])

    roc_auc: Optional[float] = None
    if y_prob is not None:
        try:
            # If probabilities are 2D, extract probability for positive class
            if y_prob.ndim == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            roc_auc = float(roc_auc_score((y_t == pos_label).astype(int), prob_pos))
        except Exception:
            pass

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "task": "binary_classification",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(y_t),
    }

    if generate_plots:
        clean_name = model_name.lower().replace(" ", "_")

        # Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {model_name} (F1 = {f1:.3f})")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        cm_path = PLOTS_DIR / f"binary_cm_{clean_name}.png"
        save_plot(fig, cm_path)
        plt.close(fig)

    logger.info("Binary eval [%s]: Acc=%.3f, Prec=%.3f, Rec=%.3f, F1=%.3f", model_name, acc, prec, rec, f1)
    return metrics


# ==============================================================================
# 3. MULTI-CLASS RISK CLASSIFICATION METRICS & CHARTS
# ==============================================================================

def evaluate_multiclass_risk(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "RiskClassifier",
    class_names: Optional[List[str]] = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Computes multi-class risk classification metrics with special research focus
    on Warning 2 Recall and Macro-F1.
    """
    classes = class_names or RISK_CLASS_NAMES
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=classes, average="weighted", zero_division=0))

    # Per-Class Precision, Recall, F1
    per_cls_prec = precision_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    per_cls_rec = recall_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    per_cls_f1 = f1_score(y_t, y_p, labels=classes, average=None, zero_division=0)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, c_name in enumerate(classes):
        per_class_metrics[c_name] = {
            "Precision": round(float(per_cls_prec[idx]), 4),
            "Recall": round(float(per_cls_rec[idx]), 4),
            "F1": round(float(per_cls_f1[idx]), 4),
        }

    cm = confusion_matrix(y_t, y_p, labels=classes)

    # Multiclass ROC-AUC (One-vs-Rest)
    multiclass_auc: Optional[float] = None
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == len(classes):
        try:
            multiclass_auc = float(roc_auc_score(y_t, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    warning_2_recall = per_class_metrics.get("Warning 2", {}).get("Recall", 0.0)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "task": "multiclass_risk",
        "Accuracy": round(acc, 4),
        "Macro_Precision": round(macro_prec, 4),
        "Macro_Recall": round(macro_rec, 4),
        "Macro_F1": round(macro_f1, 4),
        "Weighted_F1": round(weighted_f1, 4),
        "Warning_2_Recall": round(warning_2_recall, 4),
        "Multiclass_ROC_AUC": round(multiclass_auc, 4) if multiclass_auc is not None else None,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(y_t),
    }

    if generate_plots:
        clean_name = model_name.lower().replace(" ", "_")

        # 1. Multiclass Confusion Matrix Heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax_cm,
        )
        ax_cm.set_title(f"Risk Confusion Matrix: {model_name} (Macro-F1={macro_f1:.3f})")
        ax_cm.set_xlabel("Predicted Risk Level")
        ax_cm.set_ylabel("True Risk Level")

        cm_path = PLOTS_DIR / f"risk_cm_{clean_name}.png"
        save_plot(fig_cm, cm_path)
        plt.close(fig_cm)

        # 2. Per-Class F1 Score Comparison Bar Chart
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        f1_vals = [per_class_metrics[c]["F1"] for c in classes]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # Green for Normal, Orange for W1, Red for W2
        ax_bar.bar(classes, f1_vals, color=colors, alpha=0.85, width=0.55)
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("F1 Score")
        ax_bar.set_title(f"Per-Class Performance: {model_name}")
        for i, val in enumerate(f1_vals):
            ax_bar.text(i, val + 0.02, f"{val:.2f}", ha="center", fontweight="bold")
        ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

        bar_path = PLOTS_DIR / f"risk_per_class_f1_{clean_name}.png"
        save_plot(fig_bar, bar_path)
        plt.close(fig_bar)

    logger.info(
        "Multi-Class Risk [%s]: Acc=%.3f, Macro-F1=%.3f, W2-Recall=%.3f",
        model_name,
        acc,
        macro_f1,
        warning_2_recall,
    )
    return metrics


# ==============================================================================
# 4. MODEL COMPARISON TABLE & EXPORT
# ==============================================================================

def compare_models(
    results: List[Dict[str, Any]],
    task: str = "multiclass_risk",
) -> pd.DataFrame:
    """
    Aggregates evaluation dictionaries into a formatted benchmark comparison table.
    """
    rows: List[Dict[str, Any]] = []

    for res in results:
        m_name = res.get("model_name", "Unknown")
        if task == "regression":
            rows.append({
                "Model": m_name,
                "MAE": res.get("MAE"),
                "RMSE": res.get("RMSE"),
                "R² Score": res.get("R2"),
            })
        elif task == "binary":
            rows.append({
                "Model": m_name,
                "Accuracy": res.get("Accuracy"),
                "Precision": res.get("Precision"),
                "Recall": res.get("Recall"),
                "F1 Score": res.get("F1"),
                "ROC-AUC": res.get("ROC_AUC"),
            })
        else:  # multiclass_risk
            rows.append({
                "Model": m_name,
                "Accuracy": res.get("Accuracy"),
                "Macro-F1": res.get("Macro_F1"),
                "Weighted-F1": res.get("Weighted_F1"),
                "Warning 2 Recall": res.get("Warning_2_Recall"),
                "ROC-AUC": res.get("Multiclass_ROC_AUC"),
            })

    df_comp = pd.DataFrame(rows)

    # Sort by primary target metric
    if task == "regression" and "R² Score" in df_comp.columns:
        df_comp = df_comp.sort_values(by="R² Score", ascending=False)
    elif task == "binary" and "F1 Score" in df_comp.columns:
        df_comp = df_comp.sort_values(by="F1 Score", ascending=False)
    elif "Warning 2 Recall" in df_comp.columns and "Macro-F1" in df_comp.columns:
        # Research priority: Warning 2 Recall then Macro-F1
        df_comp = df_comp.sort_values(by=["Warning 2 Recall", "Macro-F1"], ascending=[False, False])

    return df_comp.reset_index(drop=True)
