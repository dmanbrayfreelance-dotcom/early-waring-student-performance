"""
Module: src.models
Description: Research-grade Machine Learning, Deep Learning, and Stacking Ensemble
             prediction models for the SAL-EC educational early-warning framework.

Prediction Tasks:
  - Task A: Continuous Final Score Regression (0 - 100)
  - Task B: Binary Academic Classification (Pass / Fail)
  - Task C: Multi-Level Academic Risk Classification (Normal, Warning 1, Warning 2)

Architectures:
  - Linear / Logistic Regression
  - Random Forest (Bagging)
  - Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost with fallbacks)
  - Support Vector Machines (SVM)
  - PyTorch SAL-EC Dual-Branch Tabular Neural Network with Multi-Class Focal Loss
  - Stacking Ensembles with Out-of-Fold cross-validated meta-learners
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from src.config import (
    BINARY_MAPPING,
    MODELS_DIR,
    RANDOM_SEED,
    RISK_MAPPING,
    TARGET_BINARY,
    TARGET_RISK,
    TARGET_SCORE,
)
from src.utils import get_logger, save_artifact

logger = get_logger("SAL_EC_MODELS")


# ==============================================================================
# 1. OPTIONAL GRADIENT BOOSTING DEPENDENCIES
# ==============================================================================

HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

HAS_CATBOOST = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    pass

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    pass

HAS_TABNET = False
try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    HAS_TABNET = True
except ImportError:
    pass


# ==============================================================================
# 2. MULTI-CLASS FOCAL LOSS & PYTORCH SAL-EC NETWORK
# ==============================================================================

if HAS_TORCH:
    class FocalLoss(nn.Module):
        """
        Multi-Class Focal Loss for addressing severe academic risk class imbalance.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Downweights well-classified normal students and forces the network
        to concentrate on hard minority at-risk students (Warning 2).
        """

        def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[Union[List[float], torch.Tensor]] = None,
            reduction: str = "mean",
        ) -> None:
            super().__init__()
            self.gamma = gamma
            self.reduction = reduction
            if alpha is not None:
                if isinstance(alpha, list):
                    alpha = torch.tensor(alpha, dtype=torch.float32)
                self.register_buffer("alpha", alpha)
            else:
                self.alpha = None

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_term = (1.0 - pt) ** self.gamma
            loss = focal_term * ce_loss

            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    class SAL_EC_DualBranchNet(nn.Module):
        """
        PyTorch Dual-Branch Tabular Neural Architecture for SAL-EC Fusion.
        Maintains separate sub-networks for SAL behavioral features and EC
        course/context features, followed by non-linear fusion and output heads.
        """

        def __init__(
            self,
            sal_dim: int,
            ec_dim: int,
            num_classes: int = 3,
            task: str = "multiclass",
            hidden_dim: int = 64,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.task = task
            self.sal_dim = sal_dim
            self.ec_dim = ec_dim

            # SAL Feature Branch
            self.sal_branch = nn.Sequential(
                nn.Linear(sal_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
            )

            # EC Feature Branch
            self.ec_branch = nn.Sequential(
                nn.Linear(ec_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
            )

            # Fusion Layer
            fusion_in = (hidden_dim // 2) + (hidden_dim // 2)
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Task Output Head
            if task == "regression":
                self.head = nn.Linear(hidden_dim, 1)
            elif task == "binary":
                self.head = nn.Linear(hidden_dim, 2)
            else:  # multiclass
                self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x_sal: torch.Tensor, x_ec: torch.Tensor) -> torch.Tensor:
            sal_rep = self.sal_branch(x_sal)
            ec_rep = self.ec_branch(x_ec)
            fused = torch.cat([sal_rep, ec_rep], dim=1)
            rep = self.fusion(fused)
            out = self.head(rep)
            return out


class PyTorchTabularEstimator:
    """
    Scikit-learn compatible wrapper for the PyTorch SAL-EC Dual-Branch architecture,
    with automatic CPU/GPU placement and Focal Loss integration.
    Falls back gracefully to scikit-learn MLP if PyTorch is unavailable.
    """

    def __init__(
        self,
        task: str = "multiclass",
        sal_indices: Optional[List[int]] = None,
        ec_indices: Optional[List[int]] = None,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 0.003,
        focal_gamma: float = 2.0,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.task = task
        self.sal_indices = sal_indices
        self.ec_indices = ec_indices
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.focal_gamma = focal_gamma
        self.random_state = random_state
        self.model_: Any = None
        self.classes_: np.ndarray = np.array([])
        self.is_torch_ = HAS_TORCH

    def fit(self, X: np.ndarray, y: Union[np.ndarray, pd.Series]) -> "PyTorchTabularEstimator":
        y_arr = np.array(y)

        # Scikit-Learn MLP Fallback if PyTorch is absent
        if not self.is_torch_:
            logger.info("PyTorch not installed. Using Scikit-Learn MLP fallback.")
            if self.task == "regression":
                self.model_ = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    early_stopping=True,
                    random_state=self.random_state,
                )
            else:
                self.model_ = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    early_stopping=True,
                    random_state=self.random_state,
                )
                self.classes_ = np.unique(y_arr)
            self.model_.fit(X, y_arr)
            return self

        # Setup feature slices for SAL and EC branches
        n_features = X.shape[1]
        sal_idx = self.sal_indices if self.sal_indices else list(range(n_features // 2))
        ec_idx = self.ec_indices if self.ec_indices else list(range(n_features // 2, n_features))

        x_sal = torch.tensor(X[:, sal_idx], dtype=torch.float32)
        x_ec = torch.tensor(X[:, ec_idx], dtype=torch.float32)

        if self.task == "regression":
            y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
            num_classes = 1
            criterion = nn.MSELoss()
        else:
            self.classes_ = np.unique(y_arr)
            num_classes = len(self.classes_)
            # Convert class labels to 0-indexed integers
            label_map = {cls: i for i, cls in enumerate(self.classes_)}
            mapped_y = np.array([label_map[item] for item in y_arr])
            y_tensor = torch.tensor(mapped_y, dtype=torch.long)

            # Compute class weights for Focal Loss
            _, counts = np.unique(mapped_y, return_counts=True)
            weights = (1.0 / (counts + 1e-4))
            weights = weights / weights.sum() * num_classes
            criterion = FocalLoss(gamma=self.focal_gamma, alpha=weights.tolist())

        dataset = TensorDataset(x_sal, x_ec, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)

        self.model_ = SAL_EC_DualBranchNet(
            sal_dim=len(sal_idx),
            ec_dim=len(ec_idx),
            num_classes=num_classes,
            task=self.task,
        )

        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_sal, batch_ec, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_sal, batch_ec)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_torch_:
            return self.model_.predict(X)

        proba = self.predict_proba(X)
        if self.task == "regression":
            return proba.squeeze()
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_torch_:
            if hasattr(self.model_, "predict_proba"):
                return self.model_.predict_proba(X)
            return self.model_.predict(X)

        n_features = X.shape[1]
        sal_idx = self.sal_indices if self.sal_indices else list(range(n_features // 2))
        ec_idx = self.ec_indices if self.ec_indices else list(range(n_features // 2, n_features))

        x_sal = torch.tensor(X[:, sal_idx], dtype=torch.float32)
        x_ec = torch.tensor(X[:, ec_idx], dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x_sal, x_ec)
            if self.task == "regression":
                return out.numpy().squeeze()
            probs = nn.functional.softmax(out, dim=1).numpy()
            return probs


# ==============================================================================
# 3. TABNET ESTIMATOR (NATIVE OR INTERPRETABLE TABULAR FALLBACK)
# ==============================================================================

class TabNetEstimator:
    """
    Wrapper for Google Research TabNet: Attentive Interpretable Tabular Learning.
    Supports native TabNetClassifier / TabNetRegressor when installed,
    with an optimized neural MLP fallback and clear diagnostic messaging
    for environments without pre-compiled TabNet binaries.
    """

    def __init__(
        self,
        task: str = "binary",  # "binary", "multiclass", or "regression"
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.3,
        max_epochs: int = 40,
        batch_size: int = 64,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.task = task
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: Any = None
        self.classes_: np.ndarray = np.array([])
        self.is_tabnet_ = HAS_TABNET

    def fit(self, X: np.ndarray, y: Union[np.ndarray, pd.Series]) -> "TabNetEstimator":
        y_arr = np.array(y)
        if not self.is_tabnet_:
            logger.info("pytorch-tabnet is not installed. Utilizing optimized neural MLP fallback for TabNet.")
            if self.task == "regression":
                self.model_ = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    early_stopping=True,
                    random_state=self.random_state,
                )
            else:
                self.model_ = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    early_stopping=True,
                    random_state=self.random_state,
                )
                self.classes_ = np.unique(y_arr)
            self.model_.fit(X, y_arr)
            return self

        # Native pytorch-tabnet fitting
        if self.task == "regression":
            self.model_ = TabNetRegressor(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                seed=self.random_state,
                verbose=0,
            )
            y_in = y_arr.reshape(-1, 1).astype(np.float32)
        else:
            self.classes_ = np.unique(y_arr)
            self.model_ = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                seed=self.random_state,
                verbose=0,
            )
            y_in = y_arr.astype(int)

        self.model_.fit(
            X.astype(np.float32),
            y_in,
            max_epochs=self.max_epochs,
            batch_size=min(self.batch_size, len(X)),
            drop_last=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_tabnet_:
            return self.model_.predict(X)
        preds = self.model_.predict(X.astype(np.float32))
        if self.task == "regression":
            return preds.squeeze()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_tabnet_:
            if hasattr(self.model_, "predict_proba"):
                return self.model_.predict_proba(X)
            return self.model_.predict(X)
        return self.model_.predict_proba(X.astype(np.float32))


# ==============================================================================
# 4. MODEL FACTORY BUILDERS
# ==============================================================================

def get_regression_models(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Constructs the model registry for Task A: Continuous Final Score Prediction.
    """
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        ),
        "Support Vector Regressor": SVR(kernel="rbf", C=1.0),
        "TabNet Regressor": TabNetEstimator(task="regression", random_state=random_state),
    }

    if HAS_XGBOOST:
        models["XGBoost Regressor"] = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1
        )
    else:
        models["XGBoost Regressor"] = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=random_state
        )

    if HAS_LIGHTGBM:
        models["LightGBM Regressor"] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, random_state=random_state, verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        models["CatBoost Regressor"] = cb.CatBoostRegressor(
            iterations=100, depth=6, random_seed=random_state, verbose=False
        )

    # Base estimators for Stacking Regressor
    stack_base = [
        ("lr", models["Linear Regression"]),
        ("rf", models["Random Forest Regressor"]),
        ("svr", models["Support Vector Regressor"]),
        ("gbr", models["XGBoost Regressor"]),
    ]
    models["Stacking Regressor"] = StackingRegressor(
        estimators=stack_base,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )

    return models


def get_binary_classification_models(
    random_state: int = RANDOM_SEED,
    class_weight: str = "balanced",
) -> Dict[str, Any]:
    """
    Constructs the model registry for Task B: Binary Academic Classification (Pass/Fail).
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=random_state
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight=class_weight, random_state=random_state, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, class_weight=class_weight, random_state=random_state
        ),
        "TabNet Classifier": TabNetEstimator(task="binary", random_state=random_state),
    }

    if HAS_XGBOOST:
        models["XGBoost Classifier"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1
        )
    else:
        models["XGBoost Classifier"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        )

    if HAS_LIGHTGBM:
        models["LightGBM Classifier"] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, class_weight=class_weight, random_state=random_state, verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        models["CatBoost Classifier"] = cb.CatBoostClassifier(
            iterations=100, depth=6, random_seed=random_state, verbose=False
        )

    # PyTorch Neural Network
    models["PyTorch Neural Network"] = PyTorchTabularEstimator(
        task="binary", epochs=30, random_state=random_state
    )

    # Stacking Classifier
    stack_base = [
        ("lr", models["Logistic Regression"]),
        ("rf", models["Random Forest Classifier"]),
        ("svm", models["Support Vector Machine"]),
        ("xgb", models["XGBoost Classifier"]),
    ]
    models["Stacking Classifier"] = StackingClassifier(
        estimators=stack_base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=5,
        n_jobs=-1,
    )

    return models


def get_multiclass_risk_models(
    random_state: int = RANDOM_SEED,
    class_weight: str = "balanced",
    sal_indices: Optional[List[int]] = None,
    ec_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Constructs the model registry for Task C: Three-Level Academic Risk Classification
    (Normal, Warning 1, Warning 2 / Low Risk, Medium Risk, High Risk).
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=random_state
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=120, max_depth=12, class_weight=class_weight, random_state=random_state, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, class_weight=class_weight, random_state=random_state
        ),
        "TabNet Classifier": TabNetEstimator(task="multiclass", random_state=random_state),
    }

    if HAS_XGBOOST:
        models["XGBoost Classifier"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1
        )
    else:
        models["XGBoost Classifier"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        )

    if HAS_LIGHTGBM:
        models["LightGBM Classifier"] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, class_weight=class_weight, random_state=random_state, verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        models["CatBoost Classifier"] = cb.CatBoostClassifier(
            iterations=100, depth=6, random_seed=random_state, verbose=False
        )

    # PyTorch SAL-EC Dual Branch Network with Focal Loss
    models["PyTorch Neural Network"] = PyTorchTabularEstimator(
        task="multiclass",
        sal_indices=sal_indices,
        ec_indices=ec_indices,
        epochs=35,
        focal_gamma=2.0,
        random_state=random_state,
    )

    # Stacking Classifier
    stack_base = [
        ("lr", models["Logistic Regression"]),
        ("rf", models["Random Forest Classifier"]),
        ("svm", models["Support Vector Machine"]),
        ("xgb", models["XGBoost Classifier"]),
    ]
    models["Stacking Classifier"] = StackingClassifier(
        estimators=stack_base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=5,
        n_jobs=-1,
    )

    return models


# ==============================================================================
# 5. TRAINING & PERSISTENCE ORCHESTRATOR
# ==============================================================================

def train_and_save_model(
    model: Any,
    model_name: str,
    task: str,
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    save: bool = True,
    mode: Optional[str] = None,
) -> Any:
    """
    Fits an individual model and persists the artifact.
    Supports optional mode prefix (e.g. early_warning, full_feature).
    """
    logger.info("Training %s for task '%s' [mode=%s] on %d samples...", model_name, task, mode or "standard", len(X_train))
    model.fit(X_train, y_train)

    if save:
        clean_name = model_name.lower().replace(" ", "_")
        prefix = f"{mode}_" if mode else ""
        filename = f"{prefix}{task}_{clean_name}.joblib"
        path = MODELS_DIR / filename
        save_artifact(model, path)

    return model
