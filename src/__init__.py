"""
Student Academic Performance Prediction & At-Risk Early-Warning System (SAL-EC).

A research-oriented explainable AI framework combining Student Academic Level (SAL)
and External Characteristics (EC) tabular fusion, ensemble learning, cost-sensitive
imbalance handling, and SHAP interpretability.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from src.data import (
    generate_student_dataset,
    load_dataset_from_csv,
    export_dataset_to_csv,
    validate_dataset_schema,
    summarize_dataset,
    STUDENT_IDENTIFIERS,
    SAL_HISTORICAL_FEATURES,
    SAL_BEHAVIORAL_FEATURES,
    SAL_ALL_FEATURES,
    EC_COURSE_FEATURES,
    FUTURE_ASSESSMENT_FEATURES,
    TARGET_COLUMNS,
)

__all__ = [
    "generate_student_dataset",
    "load_dataset_from_csv",
    "export_dataset_to_csv",
    "validate_dataset_schema",
    "summarize_dataset",
    "STUDENT_IDENTIFIERS",
    "SAL_HISTORICAL_FEATURES",
    "SAL_BEHAVIORAL_FEATURES",
    "SAL_ALL_FEATURES",
    "EC_COURSE_FEATURES",
    "FUTURE_ASSESSMENT_FEATURES",
    "TARGET_COLUMNS",
]
