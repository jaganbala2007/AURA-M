"""
AURA-M Prognostic Evaluation Metrics & NASA Scoring Module
Implements standard regression metrics (MAE, RMSE, R2) and the official NASA C-MAPSS
asymmetric prognostic scoring metric.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def nasa_cmapss_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA C-MAPSS asymmetric evaluation score.
    
    In prognostics, an overestimation (predicting the engine will survive longer than it
    actually does) is far more dangerous than an underestimation (triggering conservative
    inspection early). Thus, late predictions are penalized exponentially harder (divisor 10)
    than early predictions (divisor 13).
    
    Formula:
        d = y_pred - y_true
        s_i = exp(-d_i / 13) - 1   if d_i < 0 (early prediction)
        s_i = exp( d_i / 10) - 1   if d_i >= 0 (late prediction)
        Score = sum(s_i)
    """
    d = np.array(y_pred, dtype=np.float64) - np.array(y_true, dtype=np.float64)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates prognostic model predictions across multiple standard and competition metrics.
    
    Returns:
        Dict containing MAE, RMSE, R2, and NASA Score.
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.clip(np.array(y_pred, dtype=np.float64), 0.0, None)
    
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    nasa_score = nasa_cmapss_scoring_function(y_t, y_p)
    
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 2)
    }
