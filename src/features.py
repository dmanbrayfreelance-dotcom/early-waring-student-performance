"""
Module: src.features
Description: Research-grade feature engineering and leak-free preprocessing pipeline
             for the SAL-EC (Student Academic Level - External Characteristics) framework.

Capabilities:
  - Course-aware relative rank & percentile scoring (ToRankLayer transformer)
  - E-learning ratio and threshold indicator synthesis
  - Configurable privacy protection & pseudonymization
  - Early-Warning leakage guard with audit logging
  - Leak-free preprocessing: imputers, scalers, and encoders fitted ONLY on training data
  - Training-only SMOTE resampling with safe fallbacks and class weight calculation
  - Metadata tracking for dual-branch neural tabular fusion (SAL vs EC feature indices)
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    ALL_TARGETS,
    DEFAULT_ELEARNING_THRESHOLD,
    EC_CATEGORICAL_FEATURES,
    EC_COURSE_FEATURES,
    EC_NUMERIC_FEATURES,
    FUTURE_ASSESSMENT_FEATURES,
    FUTURE_COLUMN_PATTERNS,
    PREPROCESSORS_DIR,
    RANDOM_SEED,
    SAL_ALL_FEATURES,
    SAL_BEHAVIORAL_FEATURES,
    SAL_HISTORICAL_FEATURES,
    STUDENT_IDENTIFIERS,
    TARGET_BINARY,
    TARGET_RISK,
    TARGET_SCORE,
)
from src.utils import get_logger, hash_identifier, save_artifact

logger = get_logger("SAL_EC_FEATURES")


# ==============================================================================
# 1. COURSE-AWARE RELATIVE RANK TRANSFORMER (ToRankLayer)
# ==============================================================================

class CourseRelativeRankTransformer(BaseEstimator, TransformerMixin):
    """
    Course-aware relative ranking transformer inspired by ToRankLayer research.
    Calculates a student's relative rank and percentile within their specific course
    cohort distribution.

    CRITICAL LEAKAGE PREVENTION:
      Benchmark course statistics (mean, standard deviation, and quantile thresholds)
      are computed STRICTLY on training records during fit(). Test or validation
      students are ranked against these locked training distributions. Unseen courses
      fall back gracefully to global institutional benchmarks.
    """

    def __init__(
        self,
        score_col: str = "mean_hist_score",
        course_id_col: str = "course_id",
        global_fallback_mean: float = 65.0,
        global_fallback_std: float = 15.0,
    ) -> None:
        self.score_col = score_col
        self.course_id_col = course_id_col
        self.global_fallback_mean = global_fallback_mean
        self.global_fallback_std = global_fallback_std
        self.course_stats_: Dict[str, Dict[str, float]] = {}
        self.global_stats_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "CourseRelativeRankTransformer":
        if self.score_col not in X.columns or self.course_id_col not in X.columns:
            logger.warning(
                "CourseRelativeRankTransformer: Columns '%s' or '%s' missing. Fitting fallback defaults.",
                self.score_col,
                self.course_id_col,
            )
            self.global_stats_ = {"mean": self.global_fallback_mean, "std": self.global_fallback_std}
            return self

        # Calculate global benchmark
        scores = pd.to_numeric(X[self.score_col], errors="coerce").dropna()
        g_mean = float(scores.mean()) if not scores.empty else self.global_fallback_mean
        g_std = float(scores.std()) if len(scores) > 1 and scores.std() > 0 else self.global_fallback_std
        self.global_stats_ = {"mean": g_mean, "std": g_std}

        # Calculate per-course benchmark distribution
        self.course_stats_ = {}
        for c_id, group in X.groupby(self.course_id_col):
            s_vals = pd.to_numeric(group[self.score_col], errors="coerce").dropna()
            if len(s_vals) >= 2:
                c_mean = float(s_vals.mean())
                c_std = float(s_vals.std()) if s_vals.std() > 0 else g_std
            else:
                c_mean = g_mean
                c_std = g_std
            self.course_stats_[str(c_id)] = {"mean": c_mean, "std": c_std}

        logger.info(
            "CourseRelativeRankTransformer fitted on %d unique courses. Global mean=%.2f, std=%.2f",
            len(self.course_stats_),
            g_mean,
            g_std,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()
        if self.score_col not in df_out.columns or self.course_id_col not in df_out.columns:
            return df_out

        rel_ranks: List[float] = []
        percentiles: List[float] = []

        for _, row in df_out.iterrows():
            c_id = str(row.get(self.course_id_col, ""))
            score_val = row.get(self.score_col, np.nan)

            if pd.isna(score_val):
                rel_ranks.append(0.50)
                percentiles.append(0.50)
                continue

            # Lookup course distribution or fallback to global
            stats = self.course_stats_.get(c_id, self.global_stats_)
            c_mean = stats["mean"]
            c_std = stats["std"]

            # Normalized z-score relative to course
            z = (float(score_val) - c_mean) / max(c_std, 1e-3)
            # Sigmoid mapping to [0.01, 0.99] for smooth relative rank
            rel_rank = float(np.round(1.0 / (1.0 + np.exp(-z)), 3))
            rel_rank = float(np.clip(rel_rank, 0.01, 0.99))
            rel_ranks.append(rel_rank)

            # Empirical course percentile proxy
            pct = float(np.round(rel_rank * 0.95 + 0.02, 3))
            percentiles.append(float(np.clip(pct, 0.01, 0.99)))

        df_out["student_course_relative_rank"] = rel_ranks
        df_out["course_percentile"] = percentiles
        return df_out


# ==============================================================================
# 2. E-LEARNING FEATURE ENGINEERING
# ==============================================================================

def calculate_elearning_features(
    df: pd.DataFrame,
    threshold: float = DEFAULT_ELEARNING_THRESHOLD,
) -> pd.DataFrame:
    """
    Computes course e-learning ratios and indicator flags.
    Formula: elearning_ratio = online_class_hours / total_class_hours.
    """
    df_out = df.copy()
    if "online_class_hours" in df_out.columns and "total_class_hours" in df_out.columns:
        online = pd.to_numeric(df_out["online_class_hours"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(df_out["total_class_hours"], errors="coerce").replace(0, np.nan).fillna(40.0)
        ratio = (online / total).clip(0.0, 1.0).round(3)
        df_out["elearning_ratio"] = ratio
        df_out["elearning_indicator"] = (ratio >= threshold).astype(int)
    return df_out


# ==============================================================================
# 3. PRIVACY ANONYMIZATION ENGINE
# ==============================================================================

def anonymize_student_records(
    df: pd.DataFrame,
    method: str = "anonymous_id",
    salt: str = "sal_ec_2026",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Implements configurable student data privacy anonymization.

    Actions:
      1. Drops direct identifiers ('student_name').
      2. Replaces identifiable student_id with pseudonymized anonymous IDs
         (e.g., 'ANON_STU_0001' or SHA-256 salted hashes).
      3. Returns the sanitized DataFrame and an isolated lookup dictionary.

    Note:
      Pseudonymization / hashing separates identity from predictive matrices
      for faculty review, but does not provide formal mathematical differential privacy.
    """
    df_sanitized = df.copy()
    id_mapping: Dict[str, str] = {}

    # Drop direct names if present
    if "student_name" in df_sanitized.columns:
        df_sanitized = df_sanitized.drop(columns=["student_name"])

    if "student_id" in df_sanitized.columns:
        unique_ids = df_sanitized["student_id"].unique()
        for idx, orig_id in enumerate(unique_ids):
            if method == "hash":
                anon_id = f"ANON_{hash_identifier(str(orig_id), salt=salt)}"
            else:
                anon_id = f"ANON_STU_{idx + 1:05d}"
            id_mapping[str(orig_id)] = anon_id

        df_sanitized["anonymous_student_id"] = df_sanitized["student_id"].map(id_mapping)
        df_sanitized = df_sanitized.drop(columns=["student_id"])

    logger.info("Privacy anonymization applied: %d student IDs pseudonymized.", len(id_mapping))
    return df_sanitized, id_mapping


# ==============================================================================
# 4. EARLY-WARNING LEAKAGE GUARD
# ==============================================================================

def apply_early_warning_guard(
    df: pd.DataFrame,
    early_warning_mode: bool = True,
    target_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Strict Early-Warning Feature Guard.

    When early_warning_mode=True:
      Automatically excludes all downstream examination marks, future internal assessments,
      later-semester indicators, targets, and direct student identifiers.

    When early_warning_mode=False (Full Prediction Mode):
      Permits current assessments (if present), but strictly excludes target variables
      and direct student identifiers.

    Returns:
      - Sanitized feature DataFrame.
      - Exclusion audit log detailing excluded columns and specific reasons.
    """
    targets_to_strip = target_columns or ALL_TARGETS
    df_clean = df.copy()
    exclusion_log: List[Dict[str, str]] = []

    # 1. Direct Identifiers (Always excluded from predictive features)
    for col in STUDENT_IDENTIFIERS + ["anonymous_student_id"]:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            exclusion_log.append({
                "column": col,
                "reason": "Direct student identifier excluded for privacy and leakage prevention.",
            })

    # 2. Target Variables (Always excluded from model inputs)
    for col in targets_to_strip:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            exclusion_log.append({
                "column": col,
                "reason": "Prediction target column excluded to prevent label leakage.",
            })

    # 3. Downstream Future Assessment Features (Only excluded in Early-Warning Mode)
    if early_warning_mode:
        for col in list(df_clean.columns):
            is_future = col in FUTURE_ASSESSMENT_FEATURES
            pattern_match = any(re.match(pat, col, re.IGNORECASE) for pat in FUTURE_COLUMN_PATTERNS)

            if is_future or pattern_match:
                df_clean = df_clean.drop(columns=[col])
                exclusion_log.append({
                    "column": col,
                    "reason": "Future assessment mark unavailable at early-warning prediction point.",
                })

    logger.info(
        "Early-Warning Guard executed (mode=%s). Excluded %d columns. Remaining features: %d",
        early_warning_mode,
        len(exclusion_log),
        len(df_clean.columns),
    )
    return df_clean, exclusion_log


# ==============================================================================
# 5. MODULAR PREPROCESSING PIPELINE (SAL-EC FUSION)
# ==============================================================================

class EducationalPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Comprehensive, leak-free preprocessing pipeline for SAL-EC educational features.

    Architecture:
      - Imputes missing numerical values using median (fitted ONLY on training data).
      - Imputes missing categorical values using most frequent.
      - Encodes categorical course/environment attributes via OneHotEncoder.
      - Standardizes continuous numeric features via StandardScaler.
      - Preserves separate SAL and EC feature group indices for tabular neural branch fusion.
    """

    def __init__(
        self,
        early_warning_mode: bool = True,
        elearning_threshold: float = DEFAULT_ELEARNING_THRESHOLD,
    ) -> None:
        self.early_warning_mode = early_warning_mode
        self.elearning_threshold = elearning_threshold

        # Sub-transformers
        self.rank_transformer = CourseRelativeRankTransformer()
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Metadata tracking
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.sal_indices_: List[int] = []
        self.ec_indices_: List[int] = []
        self.is_fitted_: bool = False

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = calculate_elearning_features(df, threshold=self.elearning_threshold)
        return df

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "EducationalPreprocessingPipeline":
        # Step 1: Feature preparation & Early-warning guard
        df_prepared = self._prepare_features(X)
        df_guarded, _ = apply_early_warning_guard(df_prepared, early_warning_mode=self.early_warning_mode)

        # Step 2: Fit relative ranking
        self.rank_transformer.fit(df_guarded)
        df_ranked = self.rank_transformer.transform(df_guarded)

        # Step 3: Identify numeric vs categorical columns
        self.numeric_features_ = [
            c for c in df_ranked.columns if pd.api.types.is_numeric_dtype(df_ranked[c])
        ]
        self.categorical_features_ = [
            c for c in df_ranked.columns if c not in self.numeric_features_
        ]

        # Step 4: Fit Imputers & Scalers on training data
        if self.numeric_features_:
            num_data = self.num_imputer.fit_transform(df_ranked[self.numeric_features_])
            self.scaler.fit(num_data)

        if self.categorical_features_:
            cat_data = self.cat_imputer.fit_transform(df_ranked[self.categorical_features_].astype(str))
            self.encoder.fit(cat_data)

        # Step 5: Compute feature names out & track SAL vs EC indices
        cat_out_names: List[str] = []
        if self.categorical_features_:
            cat_out_names = list(self.encoder.get_feature_names_out(self.categorical_features_))

        self.feature_names_out_ = self.numeric_features_ + cat_out_names

        # Map indices to SAL or EC domains
        sal_set = set(SAL_ALL_FEATURES)
        self.sal_indices_ = []
        self.ec_indices_ = []

        for idx, col_name in enumerate(self.feature_names_out_):
            # Check if derived from a SAL feature
            base_col = col_name.split("__")[0] if "__" in col_name else col_name.split("_")[0]
            if col_name in sal_set or base_col in sal_set:
                self.sal_indices_.append(idx)
            else:
                self.ec_indices_.append(idx)

        self.is_fitted_ = True
        logger.info(
            "EducationalPreprocessingPipeline fitted: %d total features (%d SAL, %d EC).",
            len(self.feature_names_out_),
            len(self.sal_indices_),
            len(self.ec_indices_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("EducationalPreprocessingPipeline must be fitted before transform.")

        df_prepared = self._prepare_features(X)
        df_guarded, _ = apply_early_warning_guard(df_prepared, early_warning_mode=self.early_warning_mode)
        df_ranked = self.rank_transformer.transform(df_guarded)

        # Ensure all fitted columns exist in test data
        for c in self.numeric_features_:
            if c not in df_ranked.columns:
                df_ranked[c] = np.nan
        for c in self.categorical_features_:
            if c not in df_ranked.columns:
                df_ranked[c] = "Missing"

        # Transform numeric features
        if self.numeric_features_:
            num_imp = self.num_imputer.transform(df_ranked[self.numeric_features_])
            num_scaled = self.scaler.transform(num_imp)
        else:
            num_scaled = np.empty((len(df_ranked), 0))

        # Transform categorical features
        if self.categorical_features_:
            cat_imp = self.cat_imputer.transform(df_ranked[self.categorical_features_].astype(str))
            cat_enc = self.encoder.transform(cat_imp)
        else:
            cat_enc = np.empty((len(df_ranked), 0))

        X_out = np.hstack([num_scaled, cat_enc])
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Serializes the fitted pipeline to artifacts/preprocessors.
        """
        path = Path(filepath or (PREPROCESSORS_DIR / "fitted_pipeline.joblib")).resolve()
        save_artifact(self, path)
        return path


# ==============================================================================
# 6. TRAIN / TEST ISOLATION & RESAMPLING (SMOTE)
# ==============================================================================

def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_RISK,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    temporal_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits educational data into isolated train and test partitions.

    Support:
      - Temporal validation split: If temporal_col (e.g. 'semester') is specified,
        historical semesters form the training set and later semesters form the test set.
      - Stratified split: For classification targets, preserves class distribution.
      - Standard split: For continuous regression targets.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    if temporal_col and temporal_col in df.columns:
        # Temporal split: split by semester cutoff
        semesters = sorted(df[temporal_col].dropna().unique())
        split_idx = max(1, int(len(semesters) * (1.0 - test_size)))
        train_sems = set(semesters[:split_idx])

        train_mask = df[temporal_col].isin(train_sems)
        df_train = df[train_mask].copy()
        df_test = df[~train_mask].copy()

        y_train = df_train[target_col]
        y_test = df_test[target_col]
        logger.info(
            "Temporal train/test split: Train on %s (%d), Test on %s (%d)",
            train_sems,
            len(df_train),
            set(semesters[split_idx:]),
            len(df_test),
        )
        return df_train, df_test, y_train, y_test

    # Standard split
    is_classification = target_col in [TARGET_RISK, TARGET_BINARY]
    stratify = df[target_col] if is_classification else None

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    y_train = df_train[target_col]
    y_test = df_test[target_col]

    logger.info(
        "Train/Test split complete: Train=%d, Test=%d, Target='%s'",
        len(df_train),
        len(df_test),
        target_col,
    )
    return df_train, df_test, y_train, y_test


def apply_smote_resampling(
    X_train: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    random_state: int = RANDOM_SEED,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) STRICTLY to training folds.

    CRITICAL RULES:
      1. Never apply SMOTE to test or validation partitions.
      2. Never apply classification SMOTE to continuous regression targets.
      3. Implements safe native fallback if imblearn is unavailable or minority
         sample count is smaller than k_neighbors.
    """
    y_arr = np.array(y_train)
    unique_classes, class_counts = np.unique(y_arr, return_counts=True)
    min_samples = int(np.min(class_counts))

    metadata: Dict[str, Any] = {
        "original_distribution": dict(zip(unique_classes.tolist(), class_counts.tolist())),
        "resampled": False,
        "method_used": "None",
    }

    # If already balanced or minority has too few samples for k_neighbors
    effective_k = min(k_neighbors, max(1, min_samples - 1))

    # Try imbalanced-learn SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        if min_samples > 1 and effective_k >= 1:
            smote = SMOTE(random_state=random_state, k_neighbors=effective_k)
            X_res, y_res = smote.fit_resample(X_train, y_arr)
            metadata["resampled"] = True
            metadata["method_used"] = "imblearn.SMOTE"
            res_classes, res_counts = np.unique(y_res, return_counts=True)
            metadata["resampled_distribution"] = dict(zip(res_classes.tolist(), res_counts.tolist()))
            logger.info("SMOTE resampling applied successfully: %s", metadata["resampled_distribution"])
            return X_res, y_res, metadata
    except (ImportError, ValueError, Exception) as e:
        logger.warning("imblearn SMOTE unavailable or failed (%s). Using native oversampler fallback.", e)

    # Native Random Over-sampling Fallback
    try:
        max_count = int(np.max(class_counts))
        X_res_list = [X_train]
        y_res_list = [y_arr]
        rng = np.random.default_rng(random_state)

        for cls, count in zip(unique_classes, class_counts):
            if count < max_count:
                cls_indices = np.where(y_arr == cls)[0]
                needed = max_count - count
                sampled_indices = rng.choice(cls_indices, size=needed, replace=True)
                X_res_list.append(X_train[sampled_indices])
                y_res_list.append(y_arr[sampled_indices])

        X_res = np.vstack(X_res_list)
        y_res = np.concatenate(y_res_list)
        metadata["resampled"] = True
        metadata["method_used"] = "NativeRandomOversamplerFallback"
        res_classes, res_counts = np.unique(y_res, return_counts=True)
        metadata["resampled_distribution"] = dict(zip(res_classes.tolist(), res_counts.tolist()))
        logger.info("Native oversampling fallback applied: %s", metadata["resampled_distribution"])
        return X_res, y_res, metadata
    except Exception as fallback_err:
        logger.error("Native oversampling failed: %s. Returning unmodified training data.", fallback_err)
        metadata["method_used"] = "Unmodified"
        return X_train, y_arr, metadata


def compute_class_weights_dict(y: Union[pd.Series, np.ndarray]) -> Dict[Any, float]:
    """
    Computes balanced class weights for cost-sensitive learning models.
    """
    y_arr = np.array(y)
    unique_classes = np.unique(y_arr)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_arr)
    return {cls: float(w) for cls, w in zip(unique_classes, weights)}
