"""
AURA-M Model Evaluation & Benchmarking Script
Loads trained models, computes detailed prognostic and anomaly metrics,
and generates comprehensive comparison tables and reports.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import CMAPSSDataLoader
from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.engineer import FeatureEngineer
from src.rul.models import load_rul_model
from src.rul.evaluator import evaluate_predictions
from src.uncertainty.conformal import ConformalPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.Evaluator")

def main():
    logger.info("==================================================================")
    logger.info("          AURA-M PROGNOSTIC BENCHMARK & EVALUATION AUDIT          ")
    logger.info("==================================================================")

    loader = CMAPSSDataLoader(data_dir="data/raw")
    _, test_df, test_rul = loader.load_subset(subset="FD001", clip_rul=125)

    models_dir = Path("models/trained")
    rf_model_path = models_dir / "rul_model_v1.0.0-BASELINE-RF.pkl"
    xgb_model_path = models_dir / "rul_model_v1.0.0-PROD-XGB.pkl"

    if not rf_model_path.exists() or not xgb_model_path.exists():
        logger.error("Trained models not found. Run scripts/train.py first.")
        return 1

    rf_model = load_rul_model(str(rf_model_path))
    xgb_model = load_rul_model(str(xgb_model_path))

    preproc = TimeSeriesPreprocessor.load(str(models_dir / "preprocessor_v1.0.0-BASELINE-RF.pkl"))
    feat_eng = FeatureEngineer()

    # Preprocess & feature engineer test set
    test_scaled = preproc.transform(test_df)
    test_feat = feat_eng.generate_features(test_scaled)
    feature_cols = feat_eng.get_feature_columns(test_feat)

    # Evaluate on final cycle of all 100 test engines
    test_last = test_feat.groupby("engine_id").last().reset_index()
    X_test_last = test_last[feature_cols]
    true_y = np.array([test_rul.loc[eng] for eng in test_last["engine_id"]])

    # Predictions
    rf_preds = rf_model.predict(X_test_last)
    xgb_preds = xgb_model.predict(X_test_last)

    rf_eval = evaluate_predictions(true_y, rf_preds)
    xgb_eval = evaluate_predictions(true_y, xgb_preds)

    # Load Conformal Uncertainty Predictor
    unc_rf = ConformalPredictor.load(str(models_dir / "uncertainty_conformal_v1.0.0-BASELINE-RF.pkl"))
    rf_intervals = unc_rf.predict_batch_intervals(rf_preds)

    # Calculate actual empirical coverage on test set
    coverage = np.mean((true_y >= rf_intervals["rul_lower"].values) & (true_y <= rf_intervals["rul_upper"].values)) * 100.0

    print("\n" + "="*70)
    print("                     OFFICIAL TEST SET RESULTS (FD001)           ")
    print("="*70)
    print(f"{'Metric':<25} | {'Random Forest (Baseline)':<22} | {'XGBoost (Advanced)':<20}")
    print("-"*70)
    print(f"{'RMSE (cycles)':<25} | {rf_eval['rmse']:<22.2f} | {xgb_eval['rmse']:<20.2f}")
    print(f"{'MAE (cycles)':<25} | {rf_eval['mae']:<22.2f} | {xgb_eval['mae']:<20.2f}")
    print(f"{'R² Score':<25} | {rf_eval['r2']:<22.3f} | {xgb_eval['r2']:<20.3f}")
    print(f"{'NASA Asymmetric Score':<25} | {rf_eval['nasa_score']:<22.1f} | {xgb_eval['nasa_score']:<20.1f}")
    print(f"{'Conformal Interval Width':<25} | {unc_rf.conformal_quantile * 2:<22.1f} | {unc_rf.conformal_quantile * 2:<20.1f}")
    print(f"{'Empirical 90% Coverage':<25} | {coverage:<22.1f}% | {'--':<20}")
    print("="*70)
    print("Metric Interpretations:")
    print("  * RMSE / MAE: Mean prediction deviation in operating cycles until breakdown.")
    print("  * NASA Score: Exponential penalty function where late predictions (unplanned failure)")
    print("                are penalized exponentially higher than early maintenance alerts.")
    print("  * Conformal Coverage: Mathematically guaranteed finite-sample confidence bounds.")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
