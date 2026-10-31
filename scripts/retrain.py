"""
AURA-M Controlled Retraining Demonstration Script
Simulates receiving a new batch of machine operational data, trains a candidate model,
validates performance against the current production gate, and executes controlled promotion or rejection.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import CMAPSSDataLoader
from src.mlops.pipeline import MLOpsPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.RetrainingDemo")

def main():
    logger.info("==================================================================")
    logger.info("       AURA-M CONTROLLED RETRAINING & MODEL PROMOTION DEMO        ")
    logger.info("==================================================================")

    loader = CMAPSSDataLoader(data_dir="data/raw")
    train_df, test_df, test_rul = loader.load_subset(subset="FD001", clip_rul=125)

    pipeline = MLOpsPipeline(
        experiment_name="AURA-M_Reliability_Platform",
        tracking_uri="sqlite:///data/mlflow.db",
        models_dir="models/trained",
        db_path="data/auram.db"
    )

    logger.info("Triggering controlled retraining pipeline with candidate XGBoost model...")
    result = pipeline.execute_controlled_retraining(
        candidate_data=train_df,
        candidate_test_df=test_df,
        candidate_test_rul=test_rul,
        candidate_model_type="xgboost"
    )

    print("\n" + "="*60)
    print("                 RETRAINING GATEKEEPER DECISION                ")
    print("="*60)
    print(f"Run ID                   : {result['run_id']}")
    print(f"Active Production Model  : {result['current_production_version']}")
    print(f"Active Production RMSE   : {result['current_production_rmse']:.2f} cycles")
    print(f"Candidate Model Version  : {result['candidate_version']}")
    print(f"Candidate Model RMSE     : {result['candidate_metrics']['rmse']:.2f} cycles")
    print(f"Candidate NASA Score     : {result['candidate_metrics']['nasa_score']:.2f}")
    print(f"Promotion Status         : {'PROMOTED' if result['promoted'] else 'REJECTED'}")
    print(f"Gatekeeper Reason        : {result['decision_reason']}")
    print("="*60 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
