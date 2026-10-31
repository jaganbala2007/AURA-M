"""
AURA-M Remaining Useful Life (RUL) Modeling Engine (Innovation 2B)
Implements Random Forest (Baseline) and XGBoost (Advanced) prognostic regressors.
Evaluates prediction quality, computes feature importance, and handles model persistence.
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.rul.evaluator import evaluate_predictions

logger = logging.getLogger("AURA-M.RULModels")

class BaseRULModel:
    """Abstract base class for RUL prognostic models."""

    def __init__(self, model_name: str, model_type: str, params: Dict[str, Any]):
        self.model_name = model_name
        self.model_type = model_type
        self.params = params
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRULModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> Dict[str, float]:
        raise NotImplementedError


class RandomForestRULModel(BaseRULModel):
    """Random Forest Prognostic Regressor (Robust Baseline)."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 14,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
            "n_jobs": -1
        }
        super().__init__(model_name="RandomForest_Baseline", model_type="random_forest", params=params)
        self.model = RandomForestRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRULModel":
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples with {X.shape[1]} features...")
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")
        if isinstance(X, pd.DataFrame):
            vals = X[self.feature_names].values
        else:
            vals = np.array(X)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
        preds = self.model.predict(vals)
        return np.clip(preds, 0.0, None)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        sorted_pairs = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 5) for k, v in sorted_pairs}


class XGBoostRULModel(BaseRULModel):
    """XGBoost Prognostic Regressor (High Performance Gradient Boosting)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1
        }
        super().__init__(model_name="XGBoost_Advanced", model_type="xgboost", params=params)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRULModel":
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples with {X.shape[1]} features...")
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")
        if isinstance(X, pd.DataFrame):
            vals = X[self.feature_names].values
        else:
            vals = np.array(X)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
        preds = self.model.predict(vals)
        return np.clip(preds, 0.0, None)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        sorted_pairs = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 5) for k, v in sorted_pairs}


def save_rul_model(model: BaseRULModel, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Saves RUL model artifact and metadata to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    
    if metadata:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    logger.info(f"Model {model.model_name} saved to {path}")


def load_rul_model(filepath: str) -> BaseRULModel:
    """Loads RUL model artifact from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded {model.model_name} ({model.model_type}) from {filepath}")
    return model
