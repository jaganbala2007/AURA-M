"""
AURA-M Fusion Prediction Engine
Integrates preprocessing, feature extraction, degradation fingerprinting, anomaly scoring,
RUL point prediction, and conformal uncertainty into a unified inference output.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.engineer import FeatureEngineer
from src.degradation.fingerprint import DegradationFingerprintEngine
from src.anomaly.detector import AnomalyDetectionEngine
from src.rul.models import BaseRULModel
from src.uncertainty.conformal import ConformalPredictor

logger = logging.getLogger("AURA-M.FusionEngine")

class FusionInferenceEngine:
    """End-to-end inference orchestrator for real-time engine condition assessment."""

    def __init__(
        self,
        preprocessor: TimeSeriesPreprocessor,
        feature_engineer: FeatureEngineer,
        fingerprint_engine: DegradationFingerprintEngine,
        anomaly_engine: AnomalyDetectionEngine,
        rul_model: BaseRULModel,
        uncertainty_engine: ConformalPredictor,
        model_version: str = "v1.0.0-PROD"
    ):
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.fingerprint_engine = fingerprint_engine
        self.anomaly_engine = anomaly_engine
        self.rul_model = rul_model
        self.uncertainty_engine = uncertainty_engine
        self.model_version = model_version

    def analyze_engine(
        self,
        engine_df: pd.DataFrame,
        current_cycle: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes complete multi-model inference pipeline on an engine's operational history.
        
        Args:
            engine_df: Raw or normalized telemetry up to current observation for one engine.
            current_cycle: Cycle to evaluate (defaults to last available cycle).
            
        Returns:
            Unified intelligence payload including degradation fingerprint, anomaly telemetry,
            RUL prediction, conformal intervals, and decision inputs.
        """
        if engine_df.empty:
            raise ValueError("Engine DataFrame is empty.")

        # 1. Compute Degradation Fingerprint (from raw/standardized values)
        fingerprint = self.fingerprint_engine.compute_engine_fingerprint(
            engine_df=engine_df,
            current_cycle=current_cycle
        )
        eval_cycle = fingerprint["cycle"]

        # Filter up to evaluated cycle
        sub_df = engine_df[engine_df["cycle"] <= eval_cycle].copy()

        # 2. Preprocess (Scale Informative Sensors)
        scaled_df = self.preprocessor.transform(sub_df)

        # 3. Engineer Features
        feat_df = self.feature_engineer.generate_features(scaled_df)
        last_row_feat = feat_df.iloc[-1:]

        # 4. Anomaly Scoring
        anomaly_res = self.anomaly_engine.score_single_cycle(last_row_feat.iloc[0])

        # 5. RUL Prediction
        rul_pred = float(self.rul_model.predict(last_row_feat)[0])

        # 6. Prediction Uncertainty (Conformal Interval)
        uncertainty_res = self.uncertainty_engine.predict_interval(rul_pred)

        # Build unified intelligence payload
        payload = {
            "engine_id": fingerprint["engine_id"],
            "cycle": eval_cycle,
            "model_version": self.model_version,
            "model_type": self.rul_model.model_type,
            "degradation_fingerprint": fingerprint,
            "anomaly": anomaly_res,
            "rul": uncertainty_res,
            # Quick summary fields for agents and tables
            "health_index": fingerprint["health_index"],
            "health_status": fingerprint["health_status"],
            "predicted_rul": uncertainty_res["rul_point"],
            "rul_lower": uncertainty_res["rul_lower"],
            "rul_upper": uncertainty_res["rul_upper"],
            "interval_width": uncertainty_res["interval_width"],
            "anomaly_score": anomaly_res["anomaly_score"],
            "is_anomaly": anomaly_res["is_anomaly"],
            "uncertainty_category": uncertainty_res["uncertainty_category"],
            "human_review_required": uncertainty_res["human_review_required"]
        }
        return payload
