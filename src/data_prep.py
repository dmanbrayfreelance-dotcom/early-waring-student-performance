"""
Module: src.data_prep
Description: Data loading, Qin et al. (2023) IEEE research-inspired feature engineering,
             strict early-warning leakage guards, and leak-free preprocessing pipelines
             for the UCI Student Performance benchmark dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_RAW_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    UCI_BASE_NUMERIC_COLS,
    UCI_CATEGORICAL_COLS,
    UCI_DOWNSTREAM_GRADE_COLS,
    UCI_EARLY_GRADE_COLS,
    UCI_ENGINEERED_COLS,
    UCI_MAT_DATASET,
    UCI_PASS_THRESHOLD,
    UCI_POR_DATASET,
    UCI_RISK_MAPPING,
    UCI_RISK_TIERS,
    UCI_SUBJECTIVE_BEHAVIORAL_COLS,
)
from src.utils import get_logger

logger = get_logger("DATA_PREP")


# ==============================================================================
# 1. UCI DATASET LOADER
# ==============================================================================

def load_uci_data(
    subject: str = "mat",
    raw_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load the UCI Student Performance dataset (Mathematics or Portuguese).

    Parameters
    ----------
    subject : str
        'mat' for student-mat.csv, 'por' for student-por.csv, or 'both' for merged.
    raw_dir : Optional[Union[str, Path]]
        Directory containing the CSV files. Defaults to DATA_RAW_DIR.

    Returns
    -------
    pd.DataFrame
        Loaded raw student dataframe.
    """
    directory = Path(raw_dir) if raw_dir is not None else DATA_RAW_DIR

    if subject == "mat":
        file_path = directory / "student-mat.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"UCI Mathematics dataset not found at {file_path}")
        df = pd.read_csv(file_path, sep=";")
        df["subject"] = "Mathematics"
    elif subject == "por":
        file_path = directory / "student-por.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"UCI Portuguese dataset not found at {file_path}")
        df = pd.read_csv(file_path, sep=";")
        df["subject"] = "Portuguese"
    elif subject == "both":
        mat_path = directory / "student-mat.csv"
        por_path = directory / "student-por.csv"
        if not mat_path.exists() or not por_path.exists():
            raise FileNotFoundError("Both student-mat.csv and student-por.csv must exist in raw_dir.")
        df_mat = pd.read_csv(mat_path, sep=";")
        df_mat["subject"] = "Mathematics"
        df_por = pd.read_csv(por_path, sep=";")
        df_por["subject"] = "Portuguese"
        df = pd.concat([df_mat, df_por], ignore_index=True)
    else:
        raise ValueError(f"Unknown subject '{subject}'. Expected 'mat', 'por', or 'both'.")

    # Add a synthetic unique student identifier if not present
    if "student_id" not in df.columns:
        prefix = "MAT" if subject == "mat" else ("POR" if subject == "por" else "STU")
        df.insert(0, "student_id", [f"{prefix}-{i+1:04d}" for i in range(len(df))])

    logger.info("Loaded UCI %s dataset: %d records, %d columns.", subject, len(df), df.shape[1])
    return df


# ==============================================================================
# 2. QIN ET AL. (2023) FEATURE ENGINEERING
# ==============================================================================

def engineer_features_qin_et_al(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs interaction features blending subjective student/behavioral signals
    with objective educational context, as adapted from Qin et al. (IEEE Access 2023).

    Engineered features:
      - study_to_absence_ratio: Study investment vs absenteeism.
      - relative_g1_cohort: Student standing relative to cohort G1 baseline.
      - relative_absence_cohort: Student absenteeism relative to cohort average.
      - parental_edu_avg: Joint parental educational capital (Medu + Fedu) / 2.
      - alcohol_workday_ratio: Workday vs weekend alcohol consumption ratio.
      - total_alcohol: Cumulative weekly alcohol consumption.
      - social_study_ratio: Free time and going out vs focused study time.
      - academic_strain: Compound failure & absence pressure normalized by study time.
    """
    df_out = df.copy()

    # 1. Study to Absence Ratio
    df_out["study_to_absence_ratio"] = df_out["studytime"] / (df_out["absences"] + 1.0)

    # 2. Relative G1 Cohort Standing (Z-Score relative to school or dataset)
    if "school" in df_out.columns and len(df_out) > 1:
        school_means = df_out.groupby("school")["G1"].transform("mean")
        school_stds = df_out.groupby("school")["G1"].transform("std").fillna(1.0).replace(0, 1.0)
        df_out["relative_g1_cohort"] = (df_out["G1"] - school_means) / (school_stds + 1e-5)
    else:
        g1_std = df_out["G1"].std() if len(df_out) > 1 else 1.0
        if pd.isna(g1_std) or g1_std == 0:
            g1_std = 1.0
        df_out["relative_g1_cohort"] = (df_out["G1"] - df_out["G1"].mean()) / (g1_std + 1e-5)
    df_out["relative_g1_cohort"] = df_out["relative_g1_cohort"].fillna(0.0)

    # 3. Relative Absence Cohort Standing
    abs_mean = df_out["absences"].mean()
    abs_std = df_out["absences"].std() if len(df_out) > 1 else 1.0
    if pd.isna(abs_std) or abs_std == 0:
        abs_std = 1.0
    df_out["relative_absence_cohort"] = (df_out["absences"] - abs_mean) / (abs_std + 1e-5)
    df_out["relative_absence_cohort"] = df_out["relative_absence_cohort"].fillna(0.0)

    # 4. Joint Parental Educational Capital
    df_out["parental_edu_avg"] = (df_out["Medu"] + df_out["Fedu"]) / 2.0

    # 5. Alcohol Workday vs Weekend Ratio
    df_out["alcohol_workday_ratio"] = df_out["Dalc"] / (df_out["Walc"] + 1e-4)

    # 6. Cumulative Alcohol
    df_out["total_alcohol"] = df_out["Dalc"] + df_out["Walc"]

    # 7. Social to Study Ratio
    df_out["social_study_ratio"] = (df_out["goout"] + df_out["freetime"]) / (df_out["studytime"] * 2.0 + 1.0)

    # 8. Academic Strain Index
    df_out["academic_strain"] = (df_out["failures"] * 3.0 + df_out["absences"] / 5.0) / (df_out["studytime"] + 1.0)

    logger.debug("Engineered Qin et al. (2023) subjective-objective feature interactions.")
    return df_out


# ==============================================================================
# 3. MULTI-TASK TARGET GENERATION & RISK TIERS
# ==============================================================================

def create_targets(
    df: pd.DataFrame,
    pass_threshold: float = UCI_PASS_THRESHOLD,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generates three distinct prediction targets from final grade G3:
      1. Continuous Final Score (0 - 20)
      2. Binary Classification: Pass (1) / Fail (0)
      3. Three-Tier Academic Risk:
         - High Risk (2): Final score < 10 (or severe distress)
         - Medium Risk / Attention (1): Borderline (10 <= score < 12) or failing attendance/backlogs
         - Low Risk / Normal (0): Score >= 12 with stable academic profile
    """
    if "G3" not in df.columns:
        raise KeyError("Column 'G3' is required to generate academic prediction targets.")

    # Target 1: Continuous Score
    y_score = df["G3"].astype(float)

    # Target 2: Binary Pass / Fail
    y_binary = (df["G3"] >= pass_threshold).astype(int)

    # Target 3: Multi-Tier Risk Classification
    # IEEE Qin et al. concept: identify at-risk students proactively
    risk_labels = []
    for _, row in df.iterrows():
        g3 = row["G3"]
        g1 = row["G1"] if "G1" in row else 10
        failures = row["failures"] if "failures" in row else 0
        absences = row["absences"] if "absences" in row else 0

        if g3 < pass_threshold:
            risk_labels.append("High Risk")
        elif (pass_threshold <= g3 < pass_threshold + 2.0) or failures > 0 or absences >= 12:
            risk_labels.append("Medium Risk")
        else:
            risk_labels.append("Low Risk")

    y_risk = pd.Series([UCI_RISK_MAPPING[label] for label in risk_labels], index=df.index, name="risk_tier")

    return y_score, y_binary, y_risk


# ==============================================================================
# 4. LEAK-FREE FEATURE VIEW BUILDING
# ==============================================================================

def build_feature_views(
    df: pd.DataFrame,
    mode: str = "early_warning",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Constructs the feature matrix X enforcing strict leakage boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw and engineered features.
    mode : str
        'early_warning': Strictly excludes downstream marks ('G2' and 'G3').
        'full_feature': Includes 'G2' alongside 'G1' for comparative benchmarking.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    cat_cols : List[str]
        Categorical feature column names.
    num_cols : List[str]
        Numeric feature column names.
    """
    # Columns to always exclude from feature matrix
    always_exclude = {"student_id", "G3", "final_score", "pass_fail", "risk_tier", "subject"}

    if mode == "early_warning":
        # CRITICAL LEAKAGE GUARD: Strictly drop downstream/later-term marks
        drop_cols = always_exclude | {"G2"}
    elif mode == "full_feature":
        drop_cols = always_exclude
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'early_warning' or 'full_feature'.")

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Verify no downstream leakage in early_warning mode
    if mode == "early_warning":
        assert "G2" not in feature_cols, "Data Leakage Alert: G2 detected in early_warning features!"
        assert "G3" not in feature_cols, "Data Leakage Alert: G3 detected in early_warning features!"

    X = df[feature_cols].copy()

    # Identify categorical and numeric columns robustly across pandas versions
    cat_cols = [
        c for c in feature_cols
        if c in UCI_CATEGORICAL_COLS
        or pd.api.types.is_string_dtype(X[c])
        or isinstance(X[c].dtype, pd.CategoricalDtype)
        or X[c].dtype == "object"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    logger.info(
        "Built feature view [mode=%s]: %d features (%d numeric, %d categorical).",
        mode,
        len(feature_cols),
        len(num_cols),
        len(cat_cols),
    )
    return X, cat_cols, num_cols


# ==============================================================================
# 5. PREPROCESSING PIPELINE
# ==============================================================================

def get_preprocessing_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    """
    Builds a leak-free ColumnTransformer pipeline.
    Standardizes numeric features and One-Hot Encodes categorical features.
    Must only be fitted on X_train.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ==============================================================================
# 6. END-TO-END DATASET PREPARATION
# ==============================================================================

def prepare_dataset(
    subject: str = "mat",
    mode: str = "early_warning",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    use_smote: bool = False,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    End-to-end dataset preparation and leak-free splitting.

    Returns a dictionary containing:
      - X_train, X_test (transformed numpy arrays)
      - X_train_raw, X_test_raw (untransformed pandas DataFrames)
      - y_train, y_test (dict with 'score', 'binary', 'risk')
      - preprocessor (fitted ColumnTransformer)
      - feature_names (list of transformed feature names)
      - cat_cols, num_cols (raw feature groupings)
      - raw_df (complete raw DataFrame with student_id)
    """
    # 1. Load data
    raw_df = load_uci_data(subject=subject, raw_dir=raw_dir)

    # 2. Engineer features
    df_feat = engineer_features_qin_et_al(raw_df)

    # 3. Create targets
    y_score, y_binary, y_risk = create_targets(df_feat)

    # 4. Build feature view
    X, cat_cols, num_cols = build_feature_views(df_feat, mode=mode)

    # 5. Train/Test Split (Stratified on risk tier to preserve minority classes)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_risk,
    )

    X_train_raw = X.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X.iloc[test_idx].reset_index(drop=True)

    y_train = {
        "score": y_score.iloc[train_idx].values,
        "binary": y_binary.iloc[train_idx].values,
        "risk": y_risk.iloc[train_idx].values,
    }
    y_test = {
        "score": y_score.iloc[test_idx].values,
        "binary": y_binary.iloc[test_idx].values,
        "risk": y_risk.iloc[test_idx].values,
    }

    test_student_df = raw_df.iloc[test_idx].copy().reset_index(drop=True)

    # 6. Fit preprocessor ONLY on X_train_raw (No Data Leakage)
    preprocessor = get_preprocessing_pipeline(cat_cols, num_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_feature_names = []
    if hasattr(preprocessor.named_transformers_["cat"], "get_feature_names_out"):
        cat_feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names

    # Optional SMOTE resampling on X_train for imbalanced risk tiers
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_smote_risk, y_train_risk_smote = smote.fit_resample(X_train, y_train["risk"])
            y_train["risk_smote"] = y_train_risk_smote
            y_train["X_train_smote_risk"] = X_train_smote_risk
            logger.info("Applied SMOTE resampling to training split for risk tiers.")
        except Exception as e:
            logger.warning("SMOTE resampling skipped or failed: %s", str(e))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "raw_df": raw_df,
        "test_student_df": test_student_df,
    }
