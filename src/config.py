"""
Module: src.config
Description: Centralized configuration management for the SAL-EC educational early-warning
             and student performance prediction architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

# ==============================================================================
# PROJECT DIRECTORY PATHS
# ==============================================================================

BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_SAMPLE_DIR: Path = DATA_DIR / "sample"

# Artifact storage
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR: Path = ARTIFACTS_DIR / "preprocessors"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
EXPLAIN_DIR: Path = ARTIFACTS_DIR / "explainability"
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure directories exist
for _dir in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    MODELS_DIR,
    PREPROCESSORS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    EXPLAIN_DIR,
    PREDICTIONS_DIR,
    LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# Default sample dataset path
DEFAULT_RAW_DATASET: Path = DATA_RAW_DIR / "student_data_sample.csv"
UCI_MAT_DATASET: Path = DATA_RAW_DIR / "student-mat.csv"
UCI_POR_DATASET: Path = DATA_RAW_DIR / "student-por.csv"


# ==============================================================================
# PREDICTION TARGETS & THRESHOLDS
# ==============================================================================

TARGET_SCORE: str = "final_score"        # Objective 1: Continuous regression [0 - 100]
TARGET_BINARY: str = "pass_fail"         # Objective 2: Binary classification ("Pass" / "Fail")
TARGET_RISK: str = "risk_level"          # Objective 3: 3-class risk ("Normal", "Warning 1", "Warning 2")

ALL_TARGETS: List[str] = [TARGET_SCORE, TARGET_BINARY, TARGET_RISK]

DEFAULT_PASS_THRESHOLD: float = 40.0
DEFAULT_ELEARNING_THRESHOLD: float = 0.5

# Multi-level risk classification mappings
RISK_MAPPING: Dict[str, int] = {
    "Normal": 0,
    "Warning 1": 1,
    "Warning 2": 2,
}
INVERSE_RISK_MAPPING: Dict[int, str] = {v: k for k, v in RISK_MAPPING.items()}
RISK_CLASS_NAMES: List[str] = ["Normal", "Warning 1", "Warning 2"]

BINARY_MAPPING: Dict[str, int] = {
    "Fail": 0,
    "Pass": 1,
}
INVERSE_BINARY_MAPPING: Dict[int, str] = {v: k for k, v in BINARY_MAPPING.items()}


# ==============================================================================
# FEATURE GROUP METADATA (SAL-EC ARCHITECTURE)
# ==============================================================================

STUDENT_IDENTIFIERS: List[str] = [
    "student_id",
    "student_name",
]

# Student Academic Level (SAL) - Historical
SAL_HISTORICAL_FEATURES: List[str] = [
    "previous_gpa",
    "prev_sub_score_1",
    "prev_sub_score_2",
    "prev_sub_score_3",
    "min_hist_score",
    "max_hist_score",
    "mean_hist_score",
    "var_hist_score",
    "prev_sem_attendance",
    "academic_improvement_trend",
    "num_backlogs",
    "prev_warning_count",
]

# Student Academic Level (SAL) - Daily Behavioral
SAL_BEHAVIORAL_FEATURES: List[str] = [
    "daily_study_hours",
    "weekly_study_hours",
    "absence_count",
    "absence_ratio",
    "library_visit_freq",
    "library_activity_ratio",
    "late_night_study_ratio",
    "assignment_completion_pct",
    "lms_login_freq",
    "learning_activity_duration",
    "quiz_participation_rate",
    "consecutive_absence_days",
]

SAL_ALL_FEATURES: List[str] = SAL_HISTORICAL_FEATURES + SAL_BEHAVIORAL_FEATURES

# External Characteristics (EC) - Course & Learning Environment
EC_COURSE_FEATURES: List[str] = [
    "course_id",
    "course_credits",
    "course_difficulty",
    "subject_category",
    "department",
    "semester",
    "learning_mode",
    "total_class_hours",
    "online_class_hours",
    "elearning_ratio",
    "elearning_indicator",
    "hist_course_avg_score",
    "hist_course_score_var",
    "course_difficulty_distribution",
    "student_course_relative_rank",
    "course_percentile",
]

# Categorical vs Numeric EC Features
EC_CATEGORICAL_FEATURES: List[str] = [
    "course_id",
    "subject_category",
    "department",
    "learning_mode",
    "course_difficulty_distribution",
]

EC_NUMERIC_FEATURES: List[str] = [
    "course_credits",
    "course_difficulty",
    "semester",
    "total_class_hours",
    "online_class_hours",
    "elearning_ratio",
    "elearning_indicator",
    "hist_course_avg_score",
    "hist_course_score_var",
    "student_course_relative_rank",
    "course_percentile",
]

# Downstream Future Assessment Marks (Excluded in Early-Warning Mode)
FUTURE_ASSESSMENT_FEATURES: List[str] = [
    "mid_term_2_score",
    "final_exam_mark",
    "post_term_lab_eval",
    "future_internal_marks",
]

# Regex patterns for detecting future assessment leakage
FUTURE_COLUMN_PATTERNS: List[str] = [
    r".*final.*exam.*",
    r".*mid.*term.*2.*",
    r".*post.*term.*",
    r".*future.*internal.*",
    r".*end.*sem.*",
    r".*subsequent.*",
    r".*post.*eval.*",
]


# ==============================================================================
# UCI STUDENT PERFORMANCE & QIN ET AL. (2023) IEEE METADATA
# ==============================================================================

UCI_TARGET_SCORE: str = "G3"
UCI_PASS_THRESHOLD: float = 10.0

# Qin et al. (2023) Subjective vs Objective Elements
UCI_DEMOGRAPHIC_COLS: List[str] = ["school", "sex", "age", "address", "famsize", "Pstatus"]
UCI_FAMILY_COLS: List[str] = ["Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "famrel", "famsup"]
UCI_OBJECTIVE_COLS: List[str] = UCI_DEMOGRAPHIC_COLS + UCI_FAMILY_COLS

UCI_SUBJECTIVE_BEHAVIORAL_COLS: List[str] = [
    "studytime", "failures", "schoolsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic", "freetime",
    "goout", "Dalc", "Walc", "health", "absences"
]

UCI_EARLY_GRADE_COLS: List[str] = ["G1"]
UCI_DOWNSTREAM_GRADE_COLS: List[str] = ["G2", "G3"]

UCI_CATEGORICAL_COLS: List[str] = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic"
]

UCI_BASE_NUMERIC_COLS: List[str] = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences", "G1"
]

UCI_ENGINEERED_COLS: List[str] = [
    "study_to_absence_ratio",
    "relative_g1_cohort",
    "relative_absence_cohort",
    "parental_edu_avg",
    "alcohol_workday_ratio",
    "total_alcohol",
    "social_study_ratio",
]

UCI_RISK_TIERS: List[str] = ["Low Risk", "Medium Risk", "High Risk"]
UCI_RISK_MAPPING: Dict[str, int] = {
    "Low Risk": 0,
    "Medium Risk": 1,
    "High Risk": 2,
}
INVERSE_UCI_RISK_MAPPING: Dict[int, str] = {v: k for k, v in UCI_RISK_MAPPING.items()}


# ==============================================================================
# EXPERIMENT & MODEL PARAMETERS
# ==============================================================================

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
MISSING_RATE_DEFAULT: float = 0.02

# Supported Model Registries
REGRESSION_MODELS: List[str] = [
    "Linear Regression",
    "Random Forest Regressor",
    "XGBoost Regressor",
    "LightGBM Regressor",
    "CatBoost Regressor",
    "Stacking Regressor",
]

BINARY_MODELS: List[str] = [
    "Logistic Regression",
    "Random Forest Classifier",
    "XGBoost Classifier",
    "LightGBM Classifier",
    "CatBoost Classifier",
    "Support Vector Machine",
    "PyTorch Neural Network",
    "Stacking Classifier",
]

RISK_MODELS: List[str] = [
    "Logistic Regression",
    "Random Forest Classifier",
    "XGBoost Classifier",
    "LightGBM Classifier",
    "CatBoost Classifier",
    "Support Vector Machine",
    "PyTorch Neural Network",
    "Stacking Classifier",
]
