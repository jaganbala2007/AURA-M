"""
AURA-M Fleet Risk Prioritization Engine (Innovation 6)
Computes multi-factor risk scores across all connected machines in the fleet.
Combines RUL proximity, degradation rate, anomaly severity, and prediction uncertainty.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("AURA-M.FleetPrioritizer")

class FleetRiskPrioritizer:
    """Ranks fleet assets by operational failure risk using a transparent multi-factor formula."""

    DEFAULT_WEIGHTS = {
        "rul_proximity": 0.40,
        "degradation_severity": 0.25,
        "anomaly_severity": 0.20,
        "uncertainty_risk": 0.15
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # Normalize weights to ensure sum = 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_asset_priority_score(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes composite priority risk score [0, 100] for a single asset.
        
        Formula:
            S_rul   = 100 * max(0, 1 - RUL / 125)
            S_deg   = 100 * (1 - Health_Index)
            S_anom  = 100 * Anomaly_Score
            S_unc   = 100 * min(1.0, Interval_Width / 60)
            Priority_Score = w1*S_rul + w2*S_deg + w3*S_anom + w4*S_unc
        """
        rul = float(telemetry.get("predicted_rul", 125.0))
        health_idx = float(telemetry.get("health_index", 1.0))
        anomaly_score = float(telemetry.get("anomaly_score", 0.0))
        width = float(telemetry.get("interval_width", 20.0))

        s_rul = float(np.clip(100.0 * (1.0 - (rul / 125.0)), 0.0, 100.0))
        s_deg = float(np.clip(100.0 * (1.0 - health_idx), 0.0, 100.0))
        s_anom = float(np.clip(100.0 * anomaly_score, 0.0, 100.0))
        s_unc = float(np.clip(100.0 * (width / 60.0), 0.0, 100.0))

        score = (
            self.weights["rul_proximity"] * s_rul +
            self.weights["degradation_severity"] * s_deg +
            self.weights["anomaly_severity"] * s_anom +
            self.weights["uncertainty_risk"] * s_unc
        )
        score = float(np.clip(score, 0.0, 100.0))

        if score >= 75.0:
            band = "CRITICAL"
        elif score >= 55.0:
            band = "HIGH"
        elif score >= 35.0:
            band = "MEDIUM"
        else:
            band = "LOW"

        return {
            "engine_id": int(telemetry["engine_id"]),
            "cycle": int(telemetry["cycle"]),
            "priority_score": round(score, 1),
            "risk_band": band,
            "component_scores": {
                "rul_proximity_score": round(s_rul, 1),
                "degradation_score": round(s_deg, 1),
                "anomaly_score": round(s_anom, 1),
                "uncertainty_risk_score": round(s_unc, 1)
            },
            "predicted_rul": round(rul, 1),
            "health_index": round(health_idx, 3),
            "anomaly_score_raw": round(anomaly_score, 3),
            "interval_width": round(width, 1)
        }

    def rank_fleet(self, fleet_telemetries: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Ranks all engines in the fleet by descending risk priority.
        
        Returns:
            DataFrame sorted with highest-risk engines at the top.
        """
        records = [self.compute_asset_priority_score(t) for t in fleet_telemetries]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        
        df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        df["fleet_rank"] = range(1, len(df) + 1)
        return df
