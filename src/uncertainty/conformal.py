"""
AURA-M Confidence-Aware Prediction & Uncertainty Engine (Innovation 3)
Implements Split Conformal Prediction and Calibrated Residual Quantiles to generate
mathematically rigorous prediction intervals [RUL_lower, RUL_upper] with finite-sample coverage guarantees.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("AURA-M.Uncertainty")

class ConformalPredictor:
    """
    Confidence-Aware Prediction Interval Engine via Split Conformal Prediction.
    Provides finite-sample coverage guarantees without parametric distributional assumptions.
    """

    def __init__(self, alpha: float = 0.10, high_uncertainty_threshold: float = 45.0):
        """
        Args:
            alpha: Significance level (alpha=0.10 gives 90% prediction intervals).
            high_uncertainty_threshold: Interval width in cycles above which human review is recommended.
        """
        self.alpha = alpha
        self.confidence_level = 1.0 - alpha
        self.high_uncertainty_threshold = high_uncertainty_threshold
        self.conformal_quantile: float = 0.0
        self.residual_errors: np.ndarray = np.array([])
        self.is_calibrated: bool = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor":
        """
        Calibrates prediction intervals using non-conformity scores on a calibration dataset
        composed of independent engine trajectories.
        
        Non-conformity score: R_i = |y_true_i - y_pred_i|
        """
        y_t = np.array(y_true, dtype=np.float64)
        y_p = np.array(y_pred, dtype=np.float64)
        n = len(y_t)
        if n < 10:
            raise ValueError(f"Calibration set too small ({n} samples). At least 10 required.")

        # Compute absolute non-conformity scores
        self.residual_errors = np.abs(y_t - y_p)
        
        # Conformal quantile calculation with finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        q_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.conformal_quantile = float(np.quantile(self.residual_errors, q_level, method="higher"))
        self.is_calibrated = True

        logger.info(
            f"Conformal calibration completed across {n} samples: "
            f"Coverage Target={self.confidence_level * 100:.1f}%, Quantile Width Q={self.conformal_quantile:.2f} cycles"
        )
        return self

    def predict_interval(self, y_pred: float) -> Dict[str, Any]:
        """
        Generates calibrated prediction interval for a single RUL point prediction.
        
        Returns:
            Dict containing:
                - rul_point: Expected point prediction
                - rul_lower: Lower confidence bound
                - rul_upper: Upper confidence bound
                - interval_width: Width in operating cycles
                - uncertainty_category: HIGH_CONFIDENCE, MODERATE_CONFIDENCE, HIGH_UNCERTAINTY
                - human_review_required: Boolean flag
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated before generating intervals.")

        lower_bound = float(np.clip(y_pred - self.conformal_quantile, 0.0, None))
        upper_bound = float(y_pred + self.conformal_quantile)
        interval_width = float(upper_bound - lower_bound)

        # Categorize uncertainty
        if interval_width <= 25.0:
            category = "HIGH_CONFIDENCE"
            human_review = False
        elif interval_width <= self.high_uncertainty_threshold:
            category = "MODERATE_CONFIDENCE"
            human_review = False
        else:
            category = "HIGH_UNCERTAINTY"
            human_review = True

        return {
            "rul_point": round(float(y_pred), 1),
            "rul_lower": round(lower_bound, 1),
            "rul_upper": round(upper_bound, 1),
            "interval_width": round(interval_width, 1),
            "confidence_level": f"{int(self.confidence_level * 100)}%",
            "uncertainty_category": category,
            "human_review_required": human_review
        }

    def predict_batch_intervals(self, y_preds: np.ndarray) -> pd.DataFrame:
        """Batch interval prediction for evaluation."""
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated.")

        y_p = np.array(y_preds, dtype=np.float64)
        lowers = np.clip(y_p - self.conformal_quantile, 0.0, None)
        uppers = y_p + self.conformal_quantile
        widths = uppers - lowers

        categories = np.where(widths <= 25.0, "HIGH_CONFIDENCE",
                     np.where(widths <= self.high_uncertainty_threshold, "MODERATE_CONFIDENCE", "HIGH_UNCERTAINTY"))
        human_review = widths > self.high_uncertainty_threshold

        return pd.DataFrame({
            "rul_point": np.round(y_p, 1),
            "rul_lower": np.round(lowers, 1),
            "rul_upper": np.round(uppers, 1),
            "interval_width": np.round(widths, 1),
            "uncertainty_category": categories,
            "human_review_required": human_review
        })

    def save(self, filepath: str) -> None:
        """Persists calibrated conformal predictor."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Conformal predictor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ConformalPredictor":
        """Loads calibrated conformal predictor."""
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        logger.info(f"Loaded conformal predictor from {filepath}")
        return instance
