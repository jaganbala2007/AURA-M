"""
Unit tests for preprocessor, feature generation, anomaly detection, RUL models, and uncertainty.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.loader import CMAPSSDataLoader
from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.engineer import FeatureEngineer
from src.anomaly.detector import AnomalyDetectionEngine
from src.rul.evaluator import nasa_cmapss_scoring_function, evaluate_predictions
from src.uncertainty.conformal import ConformalPredictor

def test_preprocessor_and_feature_generation():
    loader = CMAPSSDataLoader(data_dir="data/raw")
    train_df, _, _ = loader.load_subset("FD001", clip_rul=125)
    
    # Take 2 engines for fast test
    sample_df = train_df[train_df["engine_id"].isin([1, 2])].copy()
    
    preproc = TimeSeriesPreprocessor()
    scaled = preproc.fit_transform(sample_df)
    
    feat_eng = FeatureEngineer()
    feat_df = feat_eng.generate_features(scaled)
    
    feature_cols = feat_eng.get_feature_columns(feat_df)
    assert len(feature_cols) > 20
    assert "s_2_roll_mean_5" in feat_df.columns
    assert "s_2_roc" in feat_df.columns
    assert "s_2_drift" in feat_df.columns
    assert not feat_df[feature_cols].isnull().any().any()

def test_nasa_scoring_asymmetry():
    # Scenario: Early prediction (-5 cycles) vs Late prediction (+5 cycles)
    y_true = np.array([50.0])
    early_pred = np.array([45.0])  # d = -5
    late_pred = np.array([55.0])   # d = +5
    
    score_early = nasa_cmapss_scoring_function(y_true, early_pred)
    score_late = nasa_cmapss_scoring_function(y_true, late_pred)
    
    # In NASA C-MAPSS prognostics, late prediction must be penalized higher
    assert score_late > score_early
    assert score_early == pytest.approx(np.exp(5.0 / 13.0) - 1.0, rel=1e-3)
    assert score_late == pytest.approx(np.exp(5.0 / 10.0) - 1.0, rel=1e-3)

def test_conformal_uncertainty_bounds():
    conformal = ConformalPredictor(alpha=0.10, high_uncertainty_threshold=45.0)
    
    # Synthetic calibration residuals with known quantile
    y_true = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10] * 10)
    y_pred = y_true + np.random.normal(0, 5, size=len(y_true))
    
    conformal.calibrate(y_true, y_pred)
    assert conformal.is_calibrated is True
    assert conformal.conformal_quantile > 0.0
    
    res = conformal.predict_interval(50.0)
    assert res["rul_lower"] <= 50.0
    assert res["rul_upper"] >= 50.0
    assert res["interval_width"] == pytest.approx(res["rul_upper"] - res["rul_lower"], abs=0.2)
    assert res["uncertainty_category"] in ["HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", "HIGH_UNCERTAINTY"]

def test_anomaly_detector_scoring():
    detector = AnomalyDetectionEngine(n_estimators=50, contamination=0.10, random_state=42)
    
    # Synthetic normal data
    X_train = pd.DataFrame(np.random.normal(0, 1, size=(200, 10)), columns=[f"f_{i}" for i in range(10)])
    detector.fit(X_train)
    
    # Normal sample vs extreme anomalous perturbation
    normal_sample = pd.DataFrame(np.random.normal(0, 1, size=(1, 10)), columns=[f"f_{i}" for i in range(10)])
    extreme_anomaly = pd.DataFrame(np.ones((1, 10)) * 25.0, columns=[f"f_{i}" for i in range(10)])
    
    score_norm, _ = detector.score_samples(normal_sample)
    score_anom, _ = detector.score_samples(extreme_anomaly)
    
    assert 0.0 <= score_norm[0] <= 1.0
    assert 0.0 <= score_anom[0] <= 1.0
    assert score_anom[0] > score_norm[0]
