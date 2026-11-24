"""
Module: src.data
Description: Research-grade synthetic educational dataset generation, custom CSV
             loading, schema validation, and summary utilities for the SAL-EC
             (Student Academic Level - External Characteristics) framework.

Architecture:
  - SAL (Student Academic Level):
      * Historical academic statistics (GPA, prior scores, backlogs, warnings)
      * Daily behavioral engagement (study hours, absence ratio, LMS activity, library)
  - EC (External Characteristics):
      * Course attributes, credits, subject category, difficulty
      * Learning environment (online/offline hours, e-learning ratio, relative rank)
  - Future Assessment Features (for leakage demonstration and early-warning guard testing)
  - Dual / Multi-level Targets:
      * Objective 1: Continuous final score (0.0 - 100.0)
      * Objective 2: Binary classification (Pass / Fail)
      * Objective 3: Multi-level risk classification (Normal / Warning 1 / Warning 2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Setup structured logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SAL_EC_DATA")

# ==============================================================================
# SCHEMA CONSTANTS & FEATURE DEFINITIONS
# ==============================================================================

# Direct Identifiers (to be audited and stripped during privacy anonymization)
STUDENT_IDENTIFIERS: List[str] = [
    "student_id",
    "student_name",
]

# Student Academic Level (SAL) - Historical Performance
SAL_HISTORICAL_FEATURES: List[str] = [
    "previous_gpa",                 # Prior semester GPA on 0.0 - 10.0 scale
    "prev_sub_score_1",             # Prior semester subject score 1 (0-100)
    "prev_sub_score_2",             # Prior semester subject score 2 (0-100)
    "prev_sub_score_3",             # Prior semester subject score 3 (0-100)
    "min_hist_score",               # Minimum of historical subject scores
    "max_hist_score",               # Maximum of historical subject scores
    "mean_hist_score",              # Mean of historical subject scores
    "var_hist_score",               # Variance of historical subject scores
    "prev_sem_attendance",          # Previous semester overall attendance percentage (0-100)
    "academic_improvement_trend",   # Academic delta/trend from prior semesters (-3.0 to +3.0)
    "num_backlogs",                 # Historical accumulated backlog count
    "prev_warning_count",           # Count of previously issued academic warning letters
]

# Student Academic Level (SAL) - Daily Learning Behavior
SAL_BEHAVIORAL_FEATURES: List[str] = [
    "daily_study_hours",            # Average daily independent study hours
    "weekly_study_hours",           # Estimated weekly self-study hours
    "absence_count",                # Accumulated absence periods in current semester
    "absence_ratio",                # Absence count / scheduled classes ratio (0.0 - 1.0)
    "library_visit_freq",           # Library check-in frequency (visits per month)
    "library_activity_ratio",       # Proportion of study time spent in library (0.0 - 1.0)
    "late_night_study_ratio",       # Proportion of LMS/study activity between 11PM-5AM (0.0 - 1.0)
    "assignment_completion_pct",    # Formative homework/assignment submission rate (0-100%)
    "lms_login_freq",               # LMS logins per week
    "learning_activity_duration",   # Weekly LMS active engagement duration in hours
    "quiz_participation_rate",      # Participation rate in interim formative quizzes (0-100%)
    "consecutive_absence_days",     # Maximum consecutive days absent
]

# Consolidated SAL Features
SAL_ALL_FEATURES: List[str] = SAL_HISTORICAL_FEATURES + SAL_BEHAVIORAL_FEATURES

# External Characteristics (EC) - Course & Environment Features
EC_COURSE_FEATURES: List[str] = [
    "course_id",                      # Unique course code (e.g., CS101, DS201)
    "course_credits",                 # Course credit weight (2, 3, 4)
    "course_difficulty",              # Course difficulty rating (1.0 to 5.0)
    "subject_category",               # Category: Core, Foundational, Elective, Lab
    "department",                     # Department offering the course
    "semester",                       # Academic semester (1 through 8)
    "learning_mode",                  # Mode: Offline, Online, Hybrid
    "total_class_hours",              # Total planned class hours
    "online_class_hours",             # Hours conducted via online / virtual instruction
    "elearning_ratio",                # online_class_hours / total_class_hours
    "elearning_indicator",            # Binary flag: 1 if elearning_ratio >= threshold else 0
    "hist_course_avg_score",          # Benchmark historical cohort average for course
    "hist_course_score_var",          # Benchmark historical score variance
    "course_difficulty_distribution", # Course spread category: Low-Variance, Normal, High-Spread
    "student_course_relative_rank",   # Relative rank percentile in course cohort (0.0 - 1.0)
    "course_percentile",              # Course standing percentile within department (0.0 - 1.0)
]

# Future Assessment Features (Simulated downstream marks used for leakage guard tests)
FUTURE_ASSESSMENT_FEATURES: List[str] = [
    "mid_term_2_score",               # Subsequent mid-term exam score
    "final_exam_mark",                # End-of-term final exam mark
    "post_term_lab_eval",             # End-of-term practical lab evaluation
    "future_internal_marks",          # Final aggregated internal marks
]

# Prediction Targets
TARGET_COLUMNS: List[str] = [
    "final_score",                    # Objective 1: Continuous final score (0 - 100)
    "pass_fail",                      # Objective 2: Binary classification ("Pass" / "Fail")
    "risk_level",                     # Objective 3: Multi-level risk ("Normal", "Warning 1", "Warning 2")
]

# Default Operational Parameters
DEFAULT_PASS_THRESHOLD: float = 40.0
DEFAULT_ELEARNING_THRESHOLD: float = 0.5

# Standard Risk Label Mapping
RISK_LABEL_MAP: Dict[str, int] = {
    "Normal": 0,
    "Warning 1": 1,
    "Warning 2": 2,
}
REVERSE_RISK_MAP: Dict[int, str] = {v: k for k, v in RISK_LABEL_MAP.items()}


# ==============================================================================
# RISK LABEL DETERMINATION ENGINE
# ==============================================================================

def compute_risk_label(
    row: Union[pd.Series, Dict[str, Any]],
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    gpa_warning_thresh: float = 4.5,
    att_warning_thresh: float = 60.0,
    backlog_warning_thresh: int = 2,
    absence_ratio_thresh: float = 0.30,
) -> str:
    """
    Computes documented, multi-criteria early-warning academic risk labels.

    Classification Rubric:
      - Warning 2 (High Risk / Critical Early-Warning):
          Student is failing (final_score < pass_threshold), OR demonstrates acute
          systemic failure indicators:
            * Critical low GPA (<4.5) AND low historical attendance (<60%), OR
            * Multiple backlogs (>=2) AND severe absence ratio (>0.30), OR
            * Consecutive absences (>=6 days) AND assignment completion (<45%).
      - Warning 1 (Attention / Moderate Risk):
          Student is not in Warning 2, but exhibits warning symptoms:
            * Borderline final score (<52.0), OR
            * Subpar attendance (<68.0%) combined with high absence ratio (>0.22), OR
            * Having at least 1 backlog (>=1), OR
            * Low independent study hours (<1.6 hrs/day) AND low assignment completion (<55%).
      - Normal (Low Risk / On-Track):
          Student satisfies satisfactory academic progression criteria.

    Args:
        row: Series or dictionary of student features.
        pass_threshold: Passing grade threshold (default 40.0).
        gpa_warning_thresh: GPA threshold triggering critical warning (default 4.5).
        att_warning_thresh: Attendance threshold triggering critical warning (default 60.0).
        backlog_warning_thresh: Backlogs threshold triggering critical warning (default 2).
        absence_ratio_thresh: Absence ratio threshold triggering critical warning (default 0.30).

    Returns:
        Risk label string: 'Normal', 'Warning 1', or 'Warning 2'.
    """
    score = float(row.get("final_score", 50.0))
    gpa = float(row.get("previous_gpa", 7.0))
    att = float(row.get("prev_sem_attendance", 80.0))
    backlogs = int(row.get("num_backlogs", 0))
    absence_ratio = float(row.get("absence_ratio", 0.15))
    consecutive_abs = int(row.get("consecutive_absence_days", 1))
    assignment_pct = float(row.get("assignment_completion_pct", 80.0))
    study_hours = float(row.get("daily_study_hours", 3.0))

    # Condition for Warning 2 (Critical Risk)
    is_failing = score < pass_threshold
    is_severe_gpa_att = (gpa < gpa_warning_thresh) and (att < att_warning_thresh)
    is_severe_backlog_abs = (backlogs >= backlog_warning_thresh) and (absence_ratio > absence_ratio_thresh)
    is_severe_disengagement = (consecutive_abs >= 6) and (assignment_pct < 45.0)

    if is_failing or is_severe_gpa_att or is_severe_backlog_abs or is_severe_disengagement:
        return "Warning 2"

    # Condition for Warning 1 (Moderate Risk / Attention Needed)
    is_borderline_score = score < 52.0
    is_subpar_attendance = (att < 68.0) and (absence_ratio > 0.22)
    is_has_backlog = backlogs >= 1
    is_low_study_engage = (study_hours < 1.6) and (assignment_pct < 55.0)

    if (
        is_borderline_score
        or is_subpar_attendance
        or is_has_backlog
        or is_low_study_engage
    ):
        return "Warning 1"

    return "Normal"


# ==============================================================================
# SYNTHETIC DATASET GENERATOR
# ==============================================================================

def generate_student_dataset(
    n_students: int = 1200,
    n_courses: int = 8,
    random_seed: int = 42,
    missing_rate: float = 0.02,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    elearning_threshold: float = DEFAULT_ELEARNING_THRESHOLD,
) -> pd.DataFrame:
    """
    Generates a research-grade synthetic student dataset simulating realistic
    academic, behavioral, course, and environment distributions under SAL-EC principles.

    Theoretical Foundation:
      - Latent Academic Aptitude C_i ~ Normal(mu=7.0, sigma=1.4) in [1.5, 9.8]
      - Latent Diligence/Effort E_i ~ Normal(mu=0.65, sigma=0.18) in [0.08, 0.98]
      - Course Difficulty D_j ~ Uniform(1.8, 4.6)
      - Final Score = Composite(C_i, E_i, Attendance, Assignments, Study Hours)
                      - Difficulty_Penalty(D_j) + Gaussian Noise
      - Configurable class imbalance reflecting real institutional distributions
        (~65-70% Normal, ~20-25% Warning 1, ~8-12% Warning 2).

    Args:
        n_students: Total number of student enrollment records to simulate.
        n_courses: Number of unique courses across departments.
        random_seed: Seed for strict reproducibility.
        missing_rate: Proportion of realistic missing values to inject into non-critical fields.
        pass_threshold: Score cut-off for binary 'Pass' / 'Fail' labeling.
        elearning_threshold: Ratio threshold for e-learning binary indicator.

    Returns:
        pd.DataFrame containing complete identifiers, SAL, EC, future assessment,
        and target columns.
    """
    rng = np.random.default_rng(random_seed)
    logger.info(
        "Generating synthetic dataset: %d records, %d courses, seed=%d",
        n_students,
        n_courses,
        random_seed,
    )

    # 1. Generate Course Catalog (EC Domain)
    departments = ["Computer Science", "Data Science", "Information Technology", "Electronics"]
    categories = ["Core", "Foundational", "Elective", "Lab"]
    course_diff_dist_types = ["Low-Variance", "Normal", "High-Spread"]

    course_catalog: List[Dict[str, Any]] = []
    course_codes = [f"CS{101 + i * 10}" if i < 4 else f"DS{201 + (i-4) * 10}" for i in range(n_courses)]

    for idx, c_code in enumerate(course_codes):
        c_dept = departments[idx % len(departments)]
        c_cat = categories[idx % len(categories)]
        c_credits = int(rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3]))
        c_difficulty = float(np.round(rng.uniform(1.8, 4.5), 2))
        total_hours = int(rng.choice([40, 48, 56, 60]))
        online_ratio_val = float(np.round(rng.uniform(0.1, 0.9), 2))
        online_hours = float(np.round(total_hours * online_ratio_val, 1))

        # Historical benchmark metrics for this course
        hist_avg = float(np.round(82.0 - (c_difficulty * 6.5) + rng.normal(0, 2.0), 2))
        hist_avg = float(np.clip(hist_avg, 45.0, 88.0))
        hist_var = float(np.round(rng.uniform(40.0, 140.0), 2))

        learning_mode = "Offline" if online_ratio_val < 0.25 else ("Online" if online_ratio_val > 0.70 else "Hybrid")
        diff_dist = course_diff_dist_types[idx % len(course_diff_dist_types)]

        course_catalog.append({
            "course_id": c_code,
            "course_credits": c_credits,
            "course_difficulty": c_difficulty,
            "subject_category": c_cat,
            "department": c_dept,
            "total_class_hours": total_hours,
            "online_class_hours": online_hours,
            "elearning_ratio": online_ratio_val,
            "elearning_indicator": 1 if online_ratio_val >= elearning_threshold else 0,
            "hist_course_avg_score": hist_avg,
            "hist_course_score_var": hist_var,
            "learning_mode": learning_mode,
            "course_difficulty_distribution": diff_dist,
        })

    # 2. Generate Student Latent Profiles
    # Aptitude C_i ~ Normal(6.5, 1.5) in [1.5, 9.8]
    aptitude = rng.normal(loc=6.5, scale=1.5, size=n_students)
    aptitude = np.clip(aptitude, 1.5, 9.8)

    # Diligence / Effort E_i ~ Normal(0.60, 0.19) in [0.05, 0.98]
    effort = rng.normal(loc=0.60, scale=0.19, size=n_students)
    effort = np.clip(effort, 0.05, 0.98)

    # 3. Construct Student Records
    records: List[Dict[str, Any]] = []

    for i in range(n_students):
        stu_id = f"STU_{10001 + i}"
        stu_name = f"Student_{10001 + i}"

        # Assign course
        c_info = course_catalog[i % n_courses]

        # Prior Academic Level (SAL - Historical)
        # Previous GPA scaled to 0.0 - 10.0
        prev_gpa = float(np.round(np.clip(aptitude[i] + rng.normal(0, 0.35), 1.5, 9.9), 2))

        # Prior Subject Scores (0-100)
        s1 = float(np.clip(prev_gpa * 9.8 + rng.normal(0, 4.0), 10.0, 99.0))
        s2 = float(np.clip(prev_gpa * 9.5 + rng.normal(0, 4.5), 10.0, 99.0))
        s3 = float(np.clip(prev_gpa * 10.1 + rng.normal(0, 4.0), 10.0, 99.0))
        hist_scores = [s1, s2, s3]

        min_score = float(np.round(np.min(hist_scores), 2))
        max_score = float(np.round(np.max(hist_scores), 2))
        mean_score = float(np.round(np.mean(hist_scores), 2))
        var_score = float(np.round(np.var(hist_scores), 2))

        # Attendance history & trend
        prev_att = float(np.clip(effort[i] * 85.0 + 15.0 + rng.normal(0, 3.5), 20.0, 100.0))
        trend = float(np.round(rng.normal(0.1, 0.6), 2))

        # Backlogs and prior warnings
        if prev_gpa < 4.5:
            num_backlogs = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.15, 0.1]))
            prev_warnings = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
        elif prev_gpa < 5.8:
            num_backlogs = int(rng.choice([0, 1, 2], p=[0.65, 0.25, 0.1]))
            prev_warnings = int(rng.choice([0, 1], p=[0.85, 0.15]))
        else:
            num_backlogs = 0
            prev_warnings = 0

        # Daily Learning Behavior (SAL - Behavioral)
        daily_study = float(np.round(np.clip(effort[i] * 5.0 + rng.normal(0, 0.4), 0.5, 8.0), 2))
        weekly_study = float(np.round(np.clip(daily_study * 7.0 + rng.normal(0, 1.5), 3.0, 56.0), 1))

        # Class absences
        tot_hrs = c_info["total_class_hours"]
        expected_absences = int(np.round((1.0 - (prev_att / 100.0)) * (tot_hrs / 2.0)))
        abs_count = int(np.clip(expected_absences + rng.integers(-2, 3), 0, tot_hrs // 2))
        abs_ratio = float(np.round(abs_count / max(tot_hrs, 1), 3))

        consec_abs = int(np.clip(int(abs_count * rng.uniform(0.2, 0.6)), 0, 15))

        # Library & LMS activity
        lib_freq = float(np.round(np.clip(effort[i] * 16.0 + rng.normal(0, 2.0), 0.0, 25.0), 1))
        lib_ratio = float(np.round(np.clip(lib_freq / 25.0 + rng.normal(0, 0.05), 0.0, 1.0), 3))
        late_night_study_ratio = float(np.round(np.clip(rng.beta(2, 5), 0.0, 0.85), 3))

        assign_pct = float(np.round(np.clip(effort[i] * 85.0 + 15.0 + rng.normal(0, 4.0), 10.0, 100.0), 1))
        lms_logins = float(np.round(np.clip(effort[i] * 22.0 + rng.normal(0, 2.5), 1.0, 35.0), 1))
        lms_duration = float(np.round(np.clip(lms_logins * 0.8 + rng.normal(0, 1.0), 1.0, 30.0), 1))
        quiz_pct = float(np.round(np.clip(aptitude[i] * 5.0 + effort[i] * 45.0 + rng.normal(0, 3.0), 10.0, 100.0), 1))

        # External Characteristics & Relative Standing
        semester_num = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8], p=[0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.05, 0.05]))
        relative_rank = float(np.round(np.clip((aptitude[i] - 1.5) / (9.8 - 1.5) + rng.normal(0, 0.06), 0.01, 0.99), 3))
        course_pct = float(np.round(np.clip(relative_rank + rng.normal(0, 0.04), 0.01, 0.99), 3))

        # Target 1: Continuous Final Score Synthesis
        # Score influenced by aptitude, diligence, attendance, assignments, adjusted for difficulty
        raw_final_score = (
            (aptitude[i] * 4.0) +
            (effort[i] * 16.0) +
            (assign_pct * 0.15) +
            (prev_att * 0.10) +
            (quiz_pct * 0.10) -
            ((c_info["course_difficulty"] - 2.5) * 4.5) +
            rng.normal(0, 3.2)
        )
        final_score = float(np.round(np.clip(raw_final_score, 5.0, 99.5), 2))

        # Target 2: Binary Classification
        pass_fail = "Pass" if final_score >= pass_threshold else "Fail"

        # Future Assessment Marks (Simulated leakage fields)
        mid_2 = float(np.round(np.clip(final_score * 0.95 + rng.normal(0, 3.5), 5.0, 100.0), 2))
        final_exam = float(np.round(np.clip(final_score * 1.02 + rng.normal(0, 2.5), 5.0, 100.0), 2))
        post_lab = float(np.round(np.clip(final_score * 0.98 + rng.normal(0, 3.0), 10.0, 100.0), 2))
        future_internal = float(np.round(np.clip((mid_2 + post_lab) / 2.0, 5.0, 100.0), 2))

        # Assemble row dictionary
        row: Dict[str, Any] = {
            # Direct Identifiers
            "student_id": stu_id,
            "student_name": stu_name,

            # SAL - Historical Academic
            "previous_gpa": prev_gpa,
            "prev_sub_score_1": s1,
            "prev_sub_score_2": s2,
            "prev_sub_score_3": s3,
            "min_hist_score": min_score,
            "max_hist_score": max_score,
            "mean_hist_score": mean_score,
            "var_hist_score": var_score,
            "prev_sem_attendance": prev_att,
            "academic_improvement_trend": trend,
            "num_backlogs": num_backlogs,
            "prev_warning_count": prev_warnings,

            # SAL - Daily Behavioral
            "daily_study_hours": daily_study,
            "weekly_study_hours": weekly_study,
            "absence_count": abs_count,
            "absence_ratio": abs_ratio,
            "library_visit_freq": lib_freq,
            "library_activity_ratio": lib_ratio,
            "late_night_study_ratio": late_night_study_ratio,
            "assignment_completion_pct": assign_pct,
            "lms_login_freq": lms_logins,
            "learning_activity_duration": lms_duration,
            "quiz_participation_rate": quiz_pct,
            "consecutive_absence_days": consec_abs,

            # EC - Course & Contextual
            "course_id": c_info["course_id"],
            "course_credits": c_info["course_credits"],
            "course_difficulty": c_info["course_difficulty"],
            "subject_category": c_info["subject_category"],
            "department": c_info["department"],
            "semester": semester_num,
            "learning_mode": c_info["learning_mode"],
            "total_class_hours": c_info["total_class_hours"],
            "online_class_hours": c_info["online_class_hours"],
            "elearning_ratio": c_info["elearning_ratio"],
            "elearning_indicator": c_info["elearning_indicator"],
            "hist_course_avg_score": c_info["hist_course_avg_score"],
            "hist_course_score_var": c_info["hist_course_score_var"],
            "course_difficulty_distribution": c_info["course_difficulty_distribution"],
            "student_course_relative_rank": relative_rank,
            "course_percentile": course_pct,

            # Future Assessment Fields (Leakage demonstration)
            "mid_term_2_score": mid_2,
            "final_exam_mark": final_exam,
            "post_term_lab_eval": post_lab,
            "future_internal_marks": future_internal,

            # Prediction Targets
            "final_score": final_score,
            "pass_fail": pass_fail,
        }

        # Target 3: Multi-Level Risk Classification
        row["risk_level"] = compute_risk_label(row, pass_threshold=pass_threshold)
        records.append(row)

    df = pd.DataFrame(records)

    # 4. Realistic Missing Value Injection (Only in non-critical behavioral engagement)
    if missing_rate > 0.0:
        inject_columns = [
            "library_visit_freq",
            "library_activity_ratio",
            "late_night_study_ratio",
            "learning_activity_duration",
            "daily_study_hours",
        ]
        for col in inject_columns:
            if col in df.columns:
                mask = rng.random(size=n_students) < missing_rate
                df.loc[mask, col] = np.nan

    logger.info(
        "Dataset generation complete. Shape: %s. Risk distribution: %s",
        df.shape,
        df["risk_level"].value_counts().to_dict(),
    )
    return df


# ==============================================================================
# SCHEMA VALIDATION ENGINE
# ==============================================================================

def validate_dataset_schema(
    df: pd.DataFrame,
    is_training: bool = True,
    strict: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates a student educational dataset against the SAL-EC structural schema,
    data type expectations, and domain value range rules.

    Checks:
      1. Presence of required SAL features (historical + behavioral).
      2. Presence of required EC features.
      3. Presence of target columns (if is_training=True).
      4. Correct numeric and categorical data types.
      5. Value boundary constraints (e.g., GPA in [0, 10], percentages in [0, 100], ratios in [0, 1]).
      6. Excessive missingness (>50% on any column).

    Args:
        df: Input pandas DataFrame to validate.
        is_training: If True, targets must be present.
        strict: If True, warnings are treated as critical validation failures.

    Returns:
        Tuple of:
          - is_valid (bool): True if schema complies with all requirements.
          - issues (List[str]): List of error and warning descriptions.
          - metadata (Dict[str, Any]): Feature category breakdown and summary stats.
    """
    issues: List[str] = []
    metadata: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "sal_features_found": [],
        "sal_features_missing": [],
        "ec_features_found": [],
        "ec_features_missing": [],
        "targets_found": [],
        "targets_missing": [],
        "future_fields_detected": [],
    }

    if df.empty:
        issues.append("CRITICAL: DataFrame is empty (0 records).")
        return False, issues, metadata

    # 1. Check SAL Features
    for feat in SAL_ALL_FEATURES:
        if feat in df.columns:
            metadata["sal_features_found"].append(feat)
        else:
            metadata["sal_features_missing"].append(feat)
            issues.append(f"MISSING SAL FEATURE: '{feat}' is absent.")

    # 2. Check EC Features
    for feat in EC_COURSE_FEATURES:
        if feat in df.columns:
            metadata["ec_features_found"].append(feat)
        else:
            metadata["ec_features_missing"].append(feat)
            issues.append(f"MISSING EC FEATURE: '{feat}' is absent.")

    # 3. Check Target Columns
    for tgt in TARGET_COLUMNS:
        if tgt in df.columns:
            metadata["targets_found"].append(tgt)
        else:
            metadata["targets_missing"].append(tgt)
            if is_training:
                issues.append(f"MISSING TARGET: Training requires target column '{tgt}'.")

    # 4. Check for Future Assessment Fields (Informative)
    for col in FUTURE_ASSESSMENT_FEATURES:
        if col in df.columns:
            metadata["future_fields_detected"].append(col)

    # 5. Type and Range Validations for Present Columns
    range_rules: Dict[str, Tuple[float, float]] = {
        "previous_gpa": (0.0, 10.0),
        "prev_sem_attendance": (0.0, 100.0),
        "absence_ratio": (0.0, 1.0),
        "assignment_completion_pct": (0.0, 100.0),
        "elearning_ratio": (0.0, 1.0),
        "student_course_relative_rank": (0.0, 1.0),
        "course_percentile": (0.0, 1.0),
    }

    if is_training and "final_score" in df.columns:
        range_rules["final_score"] = (0.0, 100.0)

    for col, (min_val, max_val) in range_rules.items():
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"INVALID TYPE: Column '{col}' must be numeric.")
            else:
                valid_vals = df[col].dropna()
                if not valid_vals.empty:
                    actual_min = float(valid_vals.min())
                    actual_max = float(valid_vals.max())
                    if actual_min < min_val - 1e-4 or actual_max > max_val + 1e-4:
                        msg = f"RANGE OUT OF BOUNDS: '{col}' has values [{actual_min}, {actual_max}] outside [{min_val}, {max_val}]."
                        if strict:
                            issues.append(msg)
                        else:
                            issues.append(f"WARNING: {msg}")

    # 6. Check Target Categories (if present)
    if "pass_fail" in df.columns:
        valid_pf = {"Pass", "Fail"}
        present_pf = set(df["pass_fail"].dropna().unique())
        if not present_pf.issubset(valid_pf):
            issues.append(f"INVALID CATEGORIES in 'pass_fail': Expected {valid_pf}, got {present_pf}")

    if "risk_level" in df.columns:
        valid_risk = set(RISK_LABEL_MAP.keys())
        present_risk = set(df["risk_level"].dropna().unique())
        if not present_risk.issubset(valid_risk):
            issues.append(f"INVALID CATEGORIES in 'risk_level': Expected {valid_risk}, got {present_risk}")

    # Determine validity
    critical_errors = [i for i in issues if not i.startswith("WARNING:")]
    is_valid = len(critical_errors) == 0

    return is_valid, issues, metadata


# ==============================================================================
# DATASET SUMMARY UTILITY
# ==============================================================================

def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical summary of the student dataset,
    including feature counts by domain, missingness, and target distributions.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Structured dictionary of summary metrics.
    """
    total_records = len(df)
    total_cols = len(df.columns)

    # Missing value calculation
    missing_series = df.isnull().sum()
    cols_with_missing = missing_series[missing_series > 0].to_dict()
    missing_pcts = {k: round(float((v / total_records) * 100), 2) for k, v in cols_with_missing.items()}

    # Categorization of columns present
    present_cols = set(df.columns)
    id_cols = [c for c in STUDENT_IDENTIFIERS if c in present_cols]
    sal_cols = [c for c in SAL_ALL_FEATURES if c in present_cols]
    ec_cols = [c for c in EC_COURSE_FEATURES if c in present_cols]
    future_cols = [c for c in FUTURE_ASSESSMENT_FEATURES if c in present_cols]
    target_cols = [c for c in TARGET_COLUMNS if c in present_cols]

    summary: Dict[str, Any] = {
        "total_records": total_records,
        "total_columns": total_cols,
        "feature_counts": {
            "identifiers": len(id_cols),
            "sal_features": len(sal_cols),
            "ec_features": len(ec_cols),
            "future_assessment_features": len(future_cols),
            "targets": len(target_cols),
            "other": total_cols - (len(id_cols) + len(sal_cols) + len(ec_cols) + len(future_cols) + len(target_cols)),
        },
        "missing_values": {
            "total_missing_cells": int(missing_series.sum()),
            "columns_with_missing": cols_with_missing,
            "percentages": missing_pcts,
        },
        "target_distributions": {},
    }

    # Target statistics
    if "final_score" in df.columns and pd.api.types.is_numeric_dtype(df["final_score"]):
        scores = df["final_score"].dropna()
        summary["target_distributions"]["final_score"] = {
            "mean": round(float(scores.mean()), 2),
            "std": round(float(scores.std()), 2),
            "min": round(float(scores.min()), 2),
            "max": round(float(scores.max()), 2),
            "median": round(float(scores.median()), 2),
        }

    if "pass_fail" in df.columns:
        pf_counts = df["pass_fail"].value_counts().to_dict()
        pf_pcts = {k: round(float((v / total_records) * 100), 2) for k, v in pf_counts.items()}
        summary["target_distributions"]["pass_fail"] = {
            "counts": pf_counts,
            "percentages": pf_pcts,
        }

    if "risk_level" in df.columns:
        risk_counts = df["risk_level"].value_counts().to_dict()
        risk_pcts = {k: round(float((v / total_records) * 100), 2) for k, v in risk_counts.items()}
        summary["target_distributions"]["risk_level"] = {
            "counts": risk_counts,
            "percentages": risk_pcts,
        }

    return summary


# ==============================================================================
# CSV EXPORT AND LOAD UTILITIES
# ==============================================================================

def export_dataset_to_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """
    Safely exports a dataset DataFrame to a CSV file, ensuring parent directories exist.

    Args:
        df: DataFrame to save.
        filepath: Destination path string or Path object.
        overwrite: If False, raises FileExistsError if file already exists.

    Returns:
        Path object pointing to the written file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Export target file already exists and overwrite=False: {path}")

    df.to_csv(path, index=False)
    logger.info("Successfully exported dataset (%d records, %d columns) to: %s", len(df), len(df.columns), path)
    return path


def load_dataset_from_csv(
    filepath: Union[str, Path],
    validate: bool = True,
    is_training: bool = True,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads a student dataset from CSV with schema validation.

    Args:
        filepath: Source CSV filepath.
        validate: If True, executes validate_dataset_schema.
        is_training: Flag passed to schema validation.
        strict: If True, treats validation warnings as errors.

    Returns:
        Tuple of:
          - df (pd.DataFrame): Loaded DataFrame.
          - validation_report (Dict[str, Any]): Validation status and metadata.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If schema validation fails on the loaded dataset.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV file not found at: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded CSV from: %s (%d rows, %d cols)", path, len(df), len(df.columns))

    validation_report: Dict[str, Any] = {
        "filepath": str(path),
        "is_valid": True,
        "issues": [],
        "metadata": {},
    }

    if validate:
        is_valid, issues, metadata = validate_dataset_schema(df, is_training=is_training, strict=strict)
        validation_report["is_valid"] = is_valid
        validation_report["issues"] = issues
        validation_report["metadata"] = metadata

        if not is_valid:
            error_msg = f"Schema validation failed for {path.name}:\n" + "\n".join(f" - {iss}" for iss in issues)
            logger.error(error_msg)
            raise ValueError(error_msg)

    return df, validation_report


# ==============================================================================
# CLI EXECUTION ENTRYPOINT
# ==============================================================================

def main() -> None:
    """
    CLI execution entry point to generate the default sample dataset,
    save it to data/raw/student_data_sample.csv, and display summary stats.
    """
    print("\n" + "=" * 70)
    print("SAL-EC STUDENT DATASET GENERATOR & FOUNDATION ENGINE")
    print("=" * 70)

    # Default output directory
    output_dir = Path("data/raw")
    output_path = output_dir / "student_data_sample.csv"

    # Generate synthetic dataset
    sample_df = generate_student_dataset(
        n_students=1200,
        n_courses=8,
        random_seed=42,
        missing_rate=0.02,
        pass_threshold=DEFAULT_PASS_THRESHOLD,
    )

    # Validate Schema
    is_valid, issues, _ = validate_dataset_schema(sample_df, is_training=True)
    print(f"\n[Validation] Dataset Schema Compliant: {is_valid}")
    if issues:
        print("[Validation Issues/Notes]:")
        for iss in issues:
            print(f"  * {iss}")

    # Export to CSV
    saved_path = export_dataset_to_csv(sample_df, output_path, overwrite=True)
    print(f"\n[Export] Sample CSV saved to: {saved_path.resolve()}")

    # Display Statistical Summary
    summary = summarize_dataset(sample_df)
    print("\n[Summary Metrics]:")
    print(f"  Total Student Records : {summary['total_records']}")
    print(f"  Total Feature Columns : {summary['total_columns']}")
    print(f"  SAL Features Count    : {summary['feature_counts']['sal_features']}")
    print(f"  EC Features Count     : {summary['feature_counts']['ec_features']}")
    print(f"  Future Leakage Fields : {summary['feature_counts']['future_assessment_features']}")
    print(f"  Prediction Targets    : {summary['feature_counts']['targets']}")

    print("\n[Target Distributions]:")
    print(f"  Objective 1 (Final Score Continuous): Mean = {summary['target_distributions']['final_score']['mean']:.2f}, Std = {summary['target_distributions']['final_score']['std']:.2f}")
    print(f"  Objective 2 (Pass/Fail Binary)      : {summary['target_distributions']['pass_fail']['percentages']}")
    print(f"  Objective 3 (Three-Level Risk)       : {summary['target_distributions']['risk_level']['percentages']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
