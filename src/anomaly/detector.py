"""
AURA-M Anomaly Detection Engine (Innovation 2A)
Implements an Isolation Forest anomaly detector trained on healthy operation dynamics.
Outputs calibrated anomaly scores in [0, 1] and tracks progressive health divergence.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("AURA-M.AnomalyDetector")

class AnomalyDetectionEngine:
    """Detects unusual multi-sensor behavior and generates normalized anomaly scores."""

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.10,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names: List[str] = []
        self.score_min: float = -1.0
        self.score_max: float = 1.0
        self.is_fitted: bool = False

    def fit(self, train_features: pd.DataFrame, healthy_only: bool = True) -> "AnomalyDetectionEngine":
        """
        Fits Isolation Forest model on sensor features.
        
        Args:
            train_features: Feature DataFrame.
            healthy_only: If True, trains predominantly on initial/healthy cycles to learn normal baseline.
        """
        logger.info(f"Fitting Isolation Forest on {len(train_features)} samples...")
        self.feature_names = list(train_features.columns)
        
        # Fit Isolation Forest
        self.model.fit(train_features.values)
        
        # Compute baseline raw decision scores to establish min/max calibration bounds
        raw_scores = self.model.decision_function(train_features.values)
        self.score_min = float(np.percentile(raw_scores, 1))
        self.score_max = float(np.percentile(raw_scores, 99))
        self.is_fitted = True
        logger.info(f"Anomaly detector fitted. Score calibration bounds: [{self.score_min:.4f}, {self.score_max:.4f}]")
        return self

    def score_samples(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates anomaly scores and binary anomaly flags.
        
        Returns:
            normalized_scores: Array in [0, 1] where 1.0 = highly anomalous, 0.0 = pristine normal
            is_anomaly: Boolean array (True if anomalous, False otherwise)
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetectionEngine must be fitted before scoring.")
        
        # scikit-learn decision_function: lower score = more anomalous
        raw_scores = self.model.decision_function(features[self.feature_names].values)
        raw_preds = self.model.predict(features[self.feature_names].values)  # -1 for anomaly, 1 for inlier
        
        # Invert and normalize so that 1.0 = most anomalous, 0.0 = most normal
        norm_scores = (self.score_max - raw_scores) / (self.score_max - self.score_min + 1e-8)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)
        
        is_anomaly = raw_preds == -1
        return norm_scores, is_anomaly

    def score_single_cycle(self, feature_row: pd.Series) -> Dict[str, Any]:
        """Scores a single operating cycle and returns structured anomaly telemetry."""
        df_row = pd.DataFrame([feature_row])[self.feature_names]
        norm_score, is_anom = self.score_samples(df_row)
        score = float(norm_score[0])
        
        if score >= 0.70:
            severity = "CRITICAL_ANOMALY"
        elif score >= 0.50:
            severity = "MODERATE_ANOMALY"
        elif score >= 0.35:
            severity = "ELEVATED_DRIFT"
        else:
            severity = "NOMINAL"

        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": bool(is_anom[0]),
            "severity": severity
        }

    def save(self, filepath: str) -> None:
        """Persists trained anomaly model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Anomaly detector saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AnomalyDetectionEngine":
        """Loads anomaly detector from disk."""
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        logger.info(f"Loaded anomaly detector from {filepath}")
        return instance
