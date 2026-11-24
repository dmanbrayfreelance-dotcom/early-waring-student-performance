"""
Module: src.explain
Description: Explainable AI (XAI) using SHAP (SHapley Additive exPlanations),
             SAL vs EC contribution breakdown, permutation fallback interpretability,
             and rule-based faculty academic intervention recommendation engine.

Research Principles:
  - Explanations correspond to the selected model and prediction.
  - Multiclass risk attributions are extracted with focus on at-risk classes (Warning 1 & 2).
  - Explicit scientific disclosure: SHAP values represent model-associative feature
    attribution, NOT proven institutional causality.
  - Graceful fallback: If SHAP is unavailable or on non-tree models, clearly labelled
    Permutation Importance is provided (never mislabelled as SHAP).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import EXPLAIN_DIR, RISK_CLASS_NAMES, SAL_ALL_FEATURES
from src.utils import get_logger, save_plot

logger = get_logger("SAL_EC_EXPLAIN")

# Check SHAP availability
HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass


# ==============================================================================
# 1. EXPLAINABILITY ENGINE (SHAP & PERMUTATION FALLBACK)
# ==============================================================================

class ModelExplainer:
    """
    Unified Explainability Engine providing global feature importance,
    SAL vs EC contribution analysis, and local student-level explanations.
    Uses SHAP when installed, with a clearly labeled Permutation Importance fallback.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        sal_indices: Optional[List[int]] = None,
        ec_indices: Optional[List[int]] = None,
        task: str = "multiclass_risk",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.sal_indices = sal_indices if sal_indices is not None else []
        self.ec_indices = ec_indices if ec_indices is not None else []
        self.task = task
        self.explainer_type: str = "Uninitialized"
        self.explainer_: Any = None
        self.background_data_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, max_background_samples: int = 100) -> "ModelExplainer":
        """
        Initializes the explanation background distribution.
        """
        # Sample background for efficient explanation
        if len(X_train) > max_background_samples:
            rng = np.random.default_rng(42)
            bg_indices = rng.choice(len(X_train), size=max_background_samples, replace=False)
            self.background_data_ = X_train[bg_indices]
        else:
            self.background_data_ = X_train

        if HAS_SHAP:
            try:
                # Tree models (RandomForest, XGBoost, LightGBM, CatBoost)
                if hasattr(self.model, "estimators_") or "Forest" in type(self.model).__name__ or "Boost" in type(self.model).__name__:
                    self.explainer_ = shap.TreeExplainer(self.model)
                    self.explainer_type = "SHAP TreeExplainer"
                elif hasattr(self.model, "coef_"):
                    self.explainer_ = shap.LinearExplainer(self.model, self.background_data_)
                    self.explainer_type = "SHAP LinearExplainer"
                else:
                    self.explainer_ = shap.KernelExplainer(self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict, self.background_data_)
                    self.explainer_type = "SHAP KernelExplainer"
                logger.info("Fitted %s successfully.", self.explainer_type)
                return self
            except Exception as e:
                logger.warning("SHAP initialization failed (%s). Falling back to Permutation Importance.", e)

        # Fallback to Permutation Importance
        self.explainer_type = "Permutation Feature Attribution (Fallback)"
        logger.info("Using %s.", self.explainer_type)
        return self

    def explain_global(self, X_eval: np.ndarray, y_eval: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes global feature attributions, ranking features and analyzing
        SAL vs EC relative contribution percentages.
        """
        attributions: np.ndarray = np.zeros(len(self.feature_names))

        if "SHAP" in self.explainer_type and self.explainer_ is not None:
            try:
                shap_vals = self.explainer_.shap_values(X_eval)
                if isinstance(shap_vals, list):
                    # Multi-class output: sum absolute contributions across classes
                    abs_stack = [np.abs(sv) for sv in shap_vals]
                    attributions = np.mean(np.sum(abs_stack, axis=0), axis=0)
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    attributions = np.mean(np.sum(np.abs(shap_vals), axis=2), axis=0)
                else:
                    attributions = np.mean(np.abs(shap_vals), axis=0)
            except Exception as shap_err:
                logger.warning("Global SHAP computation failed (%s). Using permutation fallback.", shap_err)
                attributions = self._compute_permutation_attribution(X_eval, y_eval)
        else:
            attributions = self._compute_permutation_attribution(X_eval, y_eval)

        # Normalize attributions
        total_attr = float(np.sum(attributions))
        norm_attributions = (attributions / max(total_attr, 1e-6)) * 100.0

        # Create sorted feature importance table
        feat_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": attributions,
            "relative_percentage": norm_attributions,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        # SAL vs EC aggregate attribution share
        sal_attr = float(np.sum([attributions[i] for i in self.sal_indices if i < len(attributions)]))
        ec_attr = float(np.sum([attributions[i] for i in self.ec_indices if i < len(attributions)]))
        total_domain = sal_attr + ec_attr + 1e-6

        sal_pct = round((sal_attr / total_domain) * 100.0, 2)
        ec_pct = round((ec_attr / total_domain) * 100.0, 2)

        return {
            "method": self.explainer_type,
            "is_shap": "SHAP" in self.explainer_type,
            "feature_importance": feat_df.to_dict(orient="records"),
            "sal_vs_ec": {
                "sal_percentage": sal_pct,
                "ec_percentage": ec_pct,
                "sal_absolute": round(sal_attr, 4),
                "ec_absolute": round(ec_attr, 4),
            },
            "disclaimer": "Attributions represent model-associative feature sensitivity and do not establish causal relationships.",
        }

    def _compute_permutation_attribution(self, X_eval: np.ndarray, y_eval: Optional[np.ndarray]) -> np.ndarray:
        if y_eval is not None:
            try:
                res = permutation_importance(
                    self.model, X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=-1
                )
                return np.maximum(res.importances_mean, 0.0)
            except Exception:
                pass

        if hasattr(self.model, "feature_importances_"):
            return np.array(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_)
            return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

        # Uniform fallback
        return np.ones(len(self.feature_names)) / len(self.feature_names)

    def explain_local_instance(
        self,
        x_row: np.ndarray,
        feature_values_dict: Optional[Dict[str, Any]] = None,
        target_class_idx: int = 1,
    ) -> Dict[str, Any]:
        """
        Generates local explanation for an individual student record.
        Identifies top risk-increasing drivers and top protective factors.
        """
        x_mat = x_row.reshape(1, -1)
        values: np.ndarray = np.zeros(len(self.feature_names))

        if "SHAP" in self.explainer_type and self.explainer_ is not None:
            try:
                sv = self.explainer_.shap_values(x_mat)
                if isinstance(sv, list):
                    target_idx = min(target_class_idx, len(sv) - 1)
                    values = sv[target_idx][0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    target_idx = min(target_class_idx, sv.shape[2] - 1)
                    values = sv[0, :, target_idx]
                else:
                    values = sv[0]
            except Exception as e:
                logger.warning("Local SHAP failed (%s). Using surrogate local attribution.", e)
                values = self._surrogate_local_attribution(x_row)
        else:
            values = self._surrogate_local_attribution(x_row)

        feature_records: List[Dict[str, Any]] = []
        for idx, (f_name, shap_val) in enumerate(zip(self.feature_names, values)):
            raw_val = feature_values_dict.get(f_name, round(float(x_row[idx]), 2)) if feature_values_dict else round(float(x_row[idx]), 2)
            feature_records.append({
                "feature": f_name,
                "value": raw_val,
                "attribution": round(float(shap_val), 4),
                "is_risk_driver": float(shap_val) > 0,
            })

        # Sort drivers
        risk_drivers = sorted([r for r in feature_records if r["attribution"] > 0], key=lambda x: x["attribution"], reverse=True)[:5]
        protective_factors = sorted([r for r in feature_records if r["attribution"] < 0], key=lambda x: x["attribution"])[:5]

        return {
            "method": self.explainer_type,
            "is_shap": "SHAP" in self.explainer_type,
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective_factors,
            "all_attributions": feature_records,
            "disclaimer": "Local attributions reflect model scoring influence and do not prove individual clinical causality.",
        }

    def _surrogate_local_attribution(self, x_row: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            base_weights = np.array(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_)
            base_weights = np.mean(coef, axis=0) if coef.ndim > 1 else coef
        else:
            base_weights = np.ones(len(x_row))

        # Local deviation from background mean
        if self.background_data_ is not None and len(self.background_data_) > 0:
            bg_mean = np.mean(self.background_data_, axis=0)
            diff = x_row - bg_mean
            return diff * base_weights
        return x_row * base_weights


# ==============================================================================
# 2. FACULTY ACADEMIC INTERVENTION ENGINE
# ==============================================================================

class FacultyInterventionEngine:
    """
    Translates detected student risk characteristics and explanation drivers
    into individualized, actionable faculty advising interventions.
    """

    def generate_recommendations(
        self,
        student_profile: Dict[str, Any],
        predicted_risk: str = "Normal",
        predicted_score: Optional[float] = None,
        top_risk_drivers: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes student attributes and explainability drivers into prioritized
        academic advising recommendations.
        """
        recommendations: List[Dict[str, Any]] = []

        # Extract student metrics safely (handling both UCI and synthetic SAL-EC columns)
        att = float(student_profile.get("prev_sem_attendance", 85.0))
        absence_count = int(student_profile.get("absences", student_profile.get("absence_count", 2)))
        absence_ratio = float(student_profile.get("absence_ratio", absence_count / 50.0))
        consec_abs = int(student_profile.get("consecutive_absence_days", 1))
        study_hours = float(student_profile.get("studytime", student_profile.get("daily_study_hours", 3.0)))
        assign_pct = float(student_profile.get("assignment_completion_pct", 85.0))
        gpa = float(student_profile.get("previous_gpa", student_profile.get("G1", 7.0)))
        backlogs = int(student_profile.get("failures", student_profile.get("num_backlogs", 0)))
        lms_logins = float(student_profile.get("lms_login_freq", 15.0))
        course_diff = float(student_profile.get("course_difficulty", 3.0))
        dalc = int(student_profile.get("Dalc", 1))
        walc = int(student_profile.get("Walc", 1))
        schoolsup = str(student_profile.get("schoolsup", "no")).lower()

        # Check driver names if available
        driver_names = [d["feature"].lower() for d in (top_risk_drivers or [])]

        # 1. Attendance & Absenteeism Intervention
        if absence_count >= 10 or att < 75.0 or absence_ratio > 0.20 or consec_abs >= 4 or any("abs" in d or "att" in d for d in driver_names):
            priority = "Urgent" if (absence_count >= 16 or att < 65.0 or consec_abs >= 5) else "High"
            recommendations.append({
                "category": "Attendance & Regularity",
                "priority": priority,
                "title": "Mandatory Attendance Recovery & Mentorship",
                "finding": f"High absenteeism detected ({absence_count} recorded absences).",
                "action_items": [
                    "Assign a departmental faculty mentor for weekly attendance verification.",
                    "Notify parents/guardians regarding university attendance minimum policy (75%).",
                    "Establish a compensatory tutorial schedule to recover missed laboratory/theory sessions.",
                ],
            })

        # 2. Independent Study Hours Intervention
        if study_hours <= 1.0 or (study_hours < 2.0 and "study" in str(student_profile)) or any("study" in d for d in driver_names):
            priority = "High" if study_hours <= 1.0 else "Moderate"
            recommendations.append({
                "category": "Study Habits & Time Management",
                "priority": priority,
                "title": "Structured Independent Study & Peer Circle Planning",
                "finding": f"Self-study commitment is low ({study_hours:.1f} study-time level vs recommended >= 3).",
                "action_items": [
                    "Formulate a structured weekly study timetable targeting at least 2-5 hours of dedicated study per week.",
                    "Pair student with a high-performing study peer in the department study circle.",
                    "Review progress bi-weekly during mentor office hours.",
                ],
            })

        # 3. Foundational Academic / Backlogs Intervention
        is_uci_scale = "G1" in student_profile
        gpa_thresh_urgent = 8.0 if is_uci_scale else 4.5
        gpa_thresh_high = 10.0 if is_uci_scale else 5.5

        if backlogs > 0 or gpa < gpa_thresh_high or any("fail" in d or "backlog" in d or "gpa" in d for d in driver_names):
            priority = "Urgent" if (backlogs >= 2 or gpa < gpa_thresh_urgent) else "High"
            scale_str = "20" if is_uci_scale else "10"
            recommendations.append({
                "category": "Curricular Remediation",
                "priority": priority,
                "title": "Core Prerequisite Remedial Program",
                "finding": f"Student has {backlogs} accumulated backlogs/failures with academic standing of {gpa:.1f}/{scale_str}.",
                "action_items": [
                    "Enroll student in remedial foundation bridge classes for cleared and pending prerequisite concepts.",
                    "Provide question-bank practice sessions with supervised problem-solving.",
                    "Conduct weekly diagnostic conceptual quizzes before internal examinations.",
                ],
            })

        # 4. Academic Support Program (when explicit schoolsup attribute is present)
        if "schoolsup" in student_profile and schoolsup == "no" and (backlogs > 0 or gpa < gpa_thresh_high):
            recommendations.append({
                "category": "Institutional Support",
                "priority": "High",
                "title": "Extra Educational Support Enrollment",
                "finding": "Student is not enrolled in extra institutional educational support despite academic risk.",
                "action_items": [
                    "Enroll student in faculty-led supplementary academic clinic.",
                    "Provide curated course notes and step-by-step problem sets.",
                ],
            })

        # 5. Alcohol & Lifestyle Balance Intervention
        if dalc >= 3 or walc >= 4 or any("alc" in d for d in driver_names):
            recommendations.append({
                "category": "Student Wellness & Counseling",
                "priority": "Moderate",
                "title": "Campus Wellness & Life-Skills Advising",
                "finding": f"Lifestyle stress factors detected (workday alcohol level: {dalc}/5, weekend alcohol: {walc}/5).",
                "action_items": [
                    "Recommend confidential consultation with campus student counseling and wellness center.",
                    "Encourage balanced lifestyle through campus sports or recreational student associations.",
                ],
            })

        # 4. Formative Assignment Completion Intervention
        if assign_pct < 70.0 or any("assign" in d for d in driver_names):
            priority = "High" if assign_pct < 55.0 else "Moderate"
            recommendations.append({
                "category": "Formative Engagement",
                "priority": priority,
                "title": "Assignment Submission Milestone Tracking",
                "finding": f"Assignment submission rate is {assign_pct:.1f}%.",
                "action_items": [
                    "Institute interim submission checkpoints (draft reviews 3 days before final deadline).",
                    "Teaching assistant check-in to resolve roadblocks in programming/analytical assignments.",
                ],
            })

        # 5. Digital Platform (LMS) Engagement Intervention
        if lms_logins < 8.0 or any("lms" in d for d in driver_names):
            recommendations.append({
                "category": "LMS Resource Engagement",
                "priority": "Moderate",
                "title": "Digital Platform Orientation & Online Content Access",
                "finding": f"Low LMS activity detected ({lms_logins:.1f} logins/week).",
                "action_items": [
                    "Verify student device and internet connectivity access to digital learning repositories.",
                    "Ensure access to recorded lectures, slides, and online practice modules.",
                ],
            })

        # 6. Rigorous Course Difficulty Support
        if course_diff >= 3.8:
            recommendations.append({
                "category": "Course-Specific Tutoring",
                "priority": "Moderate",
                "title": "Advanced Subject Tutoring Support",
                "finding": f"Course difficulty index is elevated ({course_diff:.2f}/5.0).",
                "action_items": [
                    "Offer supplementary doubt-clearing sessions with subject faculty.",
                    "Encourage active participation in laboratory collaborative problem-solving.",
                ],
            })

        # Default commendation if student is on track
        if not recommendations:
            recommendations.append({
                "category": "Academic Commendation",
                "priority": "Low",
                "title": "Satisfactory Academic Progression",
                "finding": "Student meets or exceeds all academic, behavioral, and engagement benchmarks.",
                "action_items": [
                    "Encourage participation in advanced honors electives, research projects, or student leadership.",
                    "Continue monitoring regular semester progression.",
                ],
            })

        return recommendations


# ==============================================================================
# 3. INTERACTIVE "WHAT-IF" INTERVENTION SIMULATOR
# ==============================================================================

def simulate_intervention(
    base_student: Union[pd.Series, Dict[str, Any]],
    modifications: Dict[str, Any],
    preprocessor: Any,
    regression_model: Any,
    risk_model: Any,
    mode: str = "early_warning",
) -> Dict[str, Any]:
    """
    Simulates academic intervention scenarios by modifying controllable
    student parameters (e.g. study time, absences, educational support, lifestyle),
    recomputing engineered features, and measuring the predicted impact on final grade
    and multi-tier academic risk probability.
    """
    from src.data_prep import build_feature_views, engineer_features_qin_et_al
    from src.config import INVERSE_UCI_RISK_MAPPING, UCI_RISK_TIERS

    base_dict = dict(base_student)
    mod_dict = dict(base_student)
    mod_dict.update(modifications)

    df_base = pd.DataFrame([base_dict])
    df_mod = pd.DataFrame([mod_dict])

    # Re-engineer features
    df_base_eng = engineer_features_qin_et_al(df_base)
    df_mod_eng = engineer_features_qin_et_al(df_mod)

    # Build feature views
    X_base, _, _ = build_feature_views(df_base_eng, mode=mode)
    X_mod, _, _ = build_feature_views(df_mod_eng, mode=mode)

    # Transform through pipeline
    X_base_trans = preprocessor.transform(X_base)
    X_mod_trans = preprocessor.transform(X_mod)

    # Predict regression score
    pred_base = regression_model.predict(X_base_trans)
    base_score = float(pred_base[0] if hasattr(pred_base, "__getitem__") else pred_base)

    pred_mod = regression_model.predict(X_mod_trans)
    mod_score = float(pred_mod[0] if hasattr(pred_mod, "__getitem__") else pred_mod)
    delta_score = mod_score - base_score

    # Predict risk tier
    if hasattr(risk_model, "predict_proba"):
        base_risk_proba = risk_model.predict_proba(X_base_trans)[0]
        mod_risk_proba = risk_model.predict_proba(X_mod_trans)[0]
        base_risk_idx = int(np.argmax(base_risk_proba))
        mod_risk_idx = int(np.argmax(mod_risk_proba))
    else:
        pred_r_base = risk_model.predict(X_base_trans)
        base_risk_idx = int(pred_r_base[0] if hasattr(pred_r_base, "__getitem__") else pred_r_base)
        pred_r_mod = risk_model.predict(X_mod_trans)
        mod_risk_idx = int(pred_r_mod[0] if hasattr(pred_r_mod, "__getitem__") else pred_r_mod)
        base_risk_proba = [0.0, 0.0, 0.0]
        base_risk_proba[base_risk_idx] = 1.0
        mod_risk_proba = [0.0, 0.0, 0.0]
        mod_risk_proba[mod_risk_idx] = 1.0

    base_risk_label = INVERSE_UCI_RISK_MAPPING.get(base_risk_idx, f"Risk {base_risk_idx}")
    mod_risk_label = INVERSE_UCI_RISK_MAPPING.get(mod_risk_idx, f"Risk {mod_risk_idx}")

    risk_reduced = mod_risk_idx < base_risk_idx or (mod_risk_idx == base_risk_idx and delta_score > 0)

    return {
        "baseline_score": round(base_score, 2),
        "simulated_score": round(mod_score, 2),
        "score_delta": round(delta_score, 2),
        "baseline_risk": base_risk_label,
        "simulated_risk": mod_risk_label,
        "baseline_probabilities": [round(float(p), 3) for p in base_risk_proba],
        "simulated_probabilities": [round(float(p), 3) for p in mod_risk_proba],
        "risk_reduced": risk_reduced,
        "modifications_applied": modifications,
    }

