"""
AURA-M Concept & Covariate Drift Monitoring Engine (Technology Module 7)
Uses statistical two-sample Kolmogorov-Smirnov (KS) tests and Wasserstein distance
to detect telemetry distribution drift and prediction shifts without uncontrolled retraining.
"""

import datetime
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

logger = logging.getLogger("AURA-M.DriftMonitor")

class DriftDetectionEngine:
    """Monitors incoming sensor streams against baseline distributions for covariate and prediction drift."""

    def __init__(self, p_value_threshold: float = 0.05, wasserstein_threshold: float = 0.15):
        self.p_value_threshold = p_value_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_rul_preds: np.ndarray = np.array([])
        self.is_initialized = False

    def set_reference_distributions(
        self,
        reference_features: pd.DataFrame,
        reference_rul: Optional[np.ndarray] = None
    ) -> None:
        """Establishes reference empirical distributions from verified training/baseline data."""
        for col in reference_features.columns:
            vals = reference_features[col].dropna().values
            self.reference_data[col] = vals

        if reference_rul is not None:
            self.reference_rul_preds = np.array(reference_rul)

        self.is_initialized = True
        logger.info(f"Drift monitor calibrated against {len(self.reference_data)} reference feature channels.")

    def detect_feature_drift(
        self,
        current_window_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executes two-sample KS-test and Wasserstein distance against baseline for each channel.
        
        Returns:
            Dict containing overall drift status, drifted feature list, and recommended remediation.
        """
        if not self.is_initialized:
            raise RuntimeError("Drift monitor must be initialized with reference data.")

        drifted_features = []
        metrics_by_feature = {}

        for col in current_window_features.columns:
            if col not in self.reference_data:
                continue

            ref_vals = self.reference_data[col]
            curr_vals = current_window_features[col].dropna().values
            if len(curr_vals) < 20:
                continue

            # Two-sample Kolmogorov-Smirnov test (H0: both samples come from same distribution)
            ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
            # Wasserstein (Earth Mover's) Distance
            w_dist = float(wasserstein_distance(ref_vals, curr_vals))

            is_drift = bool(p_val < self.p_value_threshold and w_dist > self.wasserstein_threshold)
            
            metrics_by_feature[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "wasserstein_distance": round(w_dist, 4),
                "is_drift": is_drift
            }

            if is_drift:
                drifted_features.append({
                    "feature": col,
                    "p_value": round(float(p_val), 4),
                    "wasserstein_dist": round(w_dist, 4)
                })

        overall_drift_detected = len(drifted_features) >= 3

        if overall_drift_detected:
            drift_status = "COVARIATE_DRIFT_DETECTED"
            recommended_response = (
                f"Significant distribution shift observed in {len(drifted_features)} feature channels. "
                "Schedule a controlled validation batch. Do not retrain automatically without human sign-off."
            )
        else:
            drift_status = "STABLE"
            recommended_response = "Sensor distributions conform to calibrated operational envelopes."

        return {
            "drift_status": drift_status,
            "overall_drift_detected": overall_drift_detected,
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recommended_response": recommended_response
        }

    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, Any]:
        """Detects population shifts in model RUL outputs."""
        if len(self.reference_rul_preds) == 0:
            return {"drift_status": "NO_REFERENCE", "is_drift": False}

        ks_stat, p_val = ks_2samp(self.reference_rul_preds, current_predictions)
        w_dist = float(wasserstein_distance(self.reference_rul_preds, current_predictions))

        is_drift = bool(p_val < self.p_value_threshold and w_dist > 15.0)
        return {
            "prediction_drift": is_drift,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "wasserstein_distance": round(w_dist, 2),
            "status": "PREDICTION_DISTRIBUTION_SHIFT" if is_drift else "PREDICTION_DISTRIBUTION_STABLE"
        }
