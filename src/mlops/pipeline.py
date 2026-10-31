"""
AURA-M MLOps & Controlled Retraining Module (Part 5)
Orchestrates MLflow experiment tracking, candidate model training, evaluation,
and controlled model promotion gates.
"""

import os
import json
import uuid
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from src.data.loader import CMAPSSDataLoader
from src.data.validator import DataValidator
from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.engineer import FeatureEngineer
from src.anomaly.detector import AnomalyDetectionEngine
from src.rul.models import RandomForestRULModel, XGBoostRULModel, BaseRULModel, save_rul_model
from src.uncertainty.conformal import ConformalPredictor
from src.rul.evaluator import evaluate_predictions
from src.database.db import DatabaseManager

logger = logging.getLogger("AURA-M.MLOps")

class MLOpsPipeline:
    """Enterprise MLOps pipeline for tracking experiments and enforcing controlled retraining gates."""

    def __init__(
        self,
        experiment_name: str = "AURA-M_Reliability_Platform",
        tracking_uri: str = "sqlite:///data/mlflow.db",
        models_dir: str = "models/trained",
        db_path: str = "data/auram.db"
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(db_path=db_path)

        # Configure MLflow
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow initialization warning (continuing with local file tracking): {e}")

    def train_and_register_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        test_rul: pd.Series,
        model_type: str = "xgboost",
        version: str = "v1.0.0-PROD",
        is_candidate: bool = False
    ) -> Tuple[BaseRULModel, AnomalyDetectionEngine, ConformalPredictor, Dict[str, Any]]:
        """
        Executes end-to-end training:
        1. Preprocessing (Leakage-free fitting)
        2. Feature Engineering
        3. Anomaly Detection (Isolation Forest)
        4. RUL Regression (Random Forest or XGBoost)
        5. Conformal Uncertainty Calibration (on holdout validation engines)
        6. Official Test Set Evaluation (MAE, RMSE, NASA Score)
        7. MLflow Experiment Logging
        """
        run_name = f"AURA-M_{model_type}_{version}"
        logger.info(f"Starting MLflow experiment run: {run_name} (Candidate={is_candidate})...")

        # -------------------------------------------------------------
        # STEP 1: Strict Engine-Level Train / Validation Splitting
        # (Split 80 engines for train, 20 engines for conformal calibration)
        # -------------------------------------------------------------
        all_engines = sorted(train_df["engine_id"].unique())
        np.random.seed(42)
        val_engine_count = 20
        val_engines = sorted(list(np.random.choice(all_engines, size=val_engine_count, replace=False)))
        train_engines = [e for e in all_engines if e not in val_engines]

        train_split = train_df[train_df["engine_id"].isin(train_engines)].copy()
        val_split = train_df[train_df["engine_id"].isin(val_engines)].copy()

        logger.info(f"Data Split: {len(train_engines)} engines for training, {len(val_engines)} engines for calibration.")

        # -------------------------------------------------------------
        # STEP 2: Preprocessing (Fit exclusively on training split)
        # -------------------------------------------------------------
        preprocessor = TimeSeriesPreprocessor()
        train_scaled = preprocessor.fit_transform(train_split)
        val_scaled = preprocessor.transform(val_split)
        test_scaled = preprocessor.transform(test_df)

        preproc_path = self.models_dir / f"preprocessor_{version}.pkl"
        preprocessor.save(str(preproc_path))

        # -------------------------------------------------------------
        # STEP 3: Feature Engineering
        # -------------------------------------------------------------
        feat_eng = FeatureEngineer()
        train_feat = feat_eng.generate_features(train_scaled)
        val_feat = feat_eng.generate_features(val_scaled)
        test_feat = feat_eng.generate_features(test_scaled)

        feature_cols = feat_eng.get_feature_columns(train_feat)
        X_train = train_feat[feature_cols]
        y_train = train_feat["RUL_target"]

        X_val = val_feat[feature_cols]
        y_val = val_feat["RUL_target"]

        # -------------------------------------------------------------
        # STEP 4: Train Anomaly Detection Engine (Isolation Forest)
        # -------------------------------------------------------------
        # Train on healthy cycles (RUL > 100) or all initial cycles
        healthy_mask = y_train >= 80.0
        X_train_healthy = X_train[healthy_mask] if healthy_mask.sum() > 500 else X_train
        
        anomaly_engine = AnomalyDetectionEngine(n_estimators=150, contamination=0.10, random_state=42)
        anomaly_engine.fit(X_train_healthy)
        anomaly_path = self.models_dir / f"anomaly_detector_{version}.pkl"
        anomaly_engine.save(str(anomaly_path))

        # -------------------------------------------------------------
        # STEP 5: Train RUL Model
        # -------------------------------------------------------------
        if model_type == "random_forest":
            rul_model = RandomForestRULModel(n_estimators=150, max_depth=14, random_state=42)
        elif model_type == "xgboost":
            rul_model = XGBoostRULModel(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        rul_model.fit(X_train, y_train)

        # -------------------------------------------------------------
        # STEP 6: Conformal Uncertainty Calibration (on Validation Engines)
        # -------------------------------------------------------------
        val_preds = rul_model.predict(X_val)
        uncertainty_engine = ConformalPredictor(alpha=0.10, high_uncertainty_threshold=45.0)
        uncertainty_engine.calibrate(y_val.values, val_preds)
        uncertainty_path = self.models_dir / f"uncertainty_conformal_{version}.pkl"
        uncertainty_engine.save(str(uncertainty_path))

        # -------------------------------------------------------------
        # STEP 7: Official Test Set Evaluation (Evaluating last cycle of each test engine)
        # -------------------------------------------------------------
        # In C-MAPSS, test set RUL target corresponds to the very last observed cycle of each engine
        test_last_cycles = test_feat.groupby("engine_id").last().reset_index()
        X_test_last = test_last_cycles[feature_cols]
        test_preds = rul_model.predict(X_test_last)

        # True ground truth RUL from test_rul series
        true_y_test = [test_rul.loc[eng] for eng in test_last_cycles["engine_id"]]
        metrics = evaluate_predictions(np.array(true_y_test), test_preds)
        metrics["conformal_quantile_q"] = round(uncertainty_engine.conformal_quantile, 2)
        metrics["features_count"] = len(feature_cols)

        # Save model artifact
        model_path = self.models_dir / f"rul_model_{version}.pkl"
        metadata = {
            "version": version,
            "model_type": model_type,
            "metrics": metrics,
            "feature_columns": feature_cols,
            "training_engines": len(train_engines),
            "calibration_engines": len(val_engines),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        save_rul_model(rul_model, str(model_path), metadata)

        # -------------------------------------------------------------
        # STEP 8: MLflow Logging
        # -------------------------------------------------------------
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model_type": model_type,
                    "version": version,
                    "is_candidate": is_candidate,
                    "num_features": len(feature_cols),
                    "train_engines": len(train_engines),
                    "calibration_engines": len(val_engines),
                    "clip_rul": 125
                })
                mlflow.log_params(rul_model.params)
                mlflow.log_metrics(metrics)
                mlflow.set_tag("pipeline", "AURA-M_Closed_Loop")
                mlflow.set_tag("status", "STAGING" if is_candidate else "PRODUCTION")
        except Exception as e:
            logger.warning(f"MLflow run logging warning: {e}")

        # Register in application database
        with self.db.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO model_versions (model_name, version, model_type, mae, rmse, nasa_score, status, artifacts_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                rul_model.model_name, version, model_type,
                metrics["mae"], metrics["rmse"], metrics["nasa_score"],
                "STAGING" if is_candidate else "PRODUCTION",
                str(model_path)
            ))
            conn.commit()

        logger.info(
            f"Pipeline [{version}] evaluation on official NASA test set: "
            f"RMSE={metrics['rmse']}, MAE={metrics['mae']}, NASA Score={metrics['nasa_score']}"
        )
        return rul_model, anomaly_engine, uncertainty_engine, metrics

    def execute_controlled_retraining(
        self,
        candidate_data: pd.DataFrame,
        candidate_test_df: pd.DataFrame,
        candidate_test_rul: pd.Series,
        candidate_model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Executes a controlled retraining cycle:
        1. Validates incoming training batch.
        2. Fetches currently deployed PRODUCTION model metrics.
        3. Trains candidate model.
        4. Applies controlled promotion gate:
           Candidate must not degrade RMSE by > 0.5 cycles and must be demonstrably valid.
        5. Promotes or rejects with complete audit trail.
        """
        run_id = f"RETRAIN-{str(uuid.uuid4())[:8].upper()}"
        logger.info(f"Initiating controlled retraining run: {run_id}")

        # 1. Validate data
        val_report = DataValidator.validate_dataset(candidate_data, is_train=True)
        if not val_report["is_valid"]:
            reason = f"Retraining rejected: Data validation failed ({val_report['errors']})"
            logger.error(reason)
            return {"promoted": False, "reason": reason, "run_id": run_id}

        # 2. Get current active production model
        with self.db.get_connection() as conn:
            prod_row = conn.execute("SELECT * FROM model_versions WHERE status = 'PRODUCTION' ORDER BY registered_at DESC LIMIT 1;").fetchone()
        
        current_prod_version = prod_row["version"] if prod_row else "NONE"
        current_prod_rmse = float(prod_row["rmse"]) if prod_row and prod_row["rmse"] else 999.0

        # 3. Train candidate model
        cand_version = f"v1.{int(datetime.datetime.now().timestamp()) % 10000}-CANDIDATE"
        cand_model, cand_anom, cand_unc, cand_metrics = self.train_and_register_pipeline(
            train_df=candidate_data,
            test_df=candidate_test_df,
            test_rul=candidate_test_rul,
            model_type=candidate_model_type,
            version=cand_version,
            is_candidate=True
        )

        cand_rmse = cand_metrics["rmse"]

        # 4. Controlled Promotion Gate
        # Condition: Candidate RMSE must be strictly <= current production RMSE (or within tight tolerance <= prod_rmse * 1.02)
        promoted = False
        if prod_row is None:
            promoted = True
            decision_reason = "Promoted as initial baseline production model."
        elif cand_rmse <= current_prod_rmse:
            promoted = True
            decision_reason = f"Promoted: Candidate RMSE ({cand_rmse:.2f}) outperformed current production RMSE ({current_prod_rmse:.2f})."
        else:
            promoted = False
            decision_reason = f"Rejected: Candidate RMSE ({cand_rmse:.2f}) failed to outperform production model ({current_prod_rmse:.2f}). Retaining {current_prod_version}."

        # 5. Update Database status
        with self.db.get_connection() as conn:
            if promoted:
                # Archive old production model
                conn.execute("UPDATE model_versions SET status = 'ARCHIVED' WHERE status = 'PRODUCTION';")
                # Promote candidate
                conn.execute("UPDATE model_versions SET status = 'PRODUCTION' WHERE version = ?;", (cand_version,))
            else:
                conn.execute("UPDATE model_versions SET status = 'REJECTED' WHERE version = ?;", (cand_version,))

            # Record retraining run
            conn.execute("""
            INSERT INTO retraining_runs (run_id, dataset_subset, old_model_version, candidate_model_version, old_rmse, candidate_rmse, promoted, decision_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (run_id, "FD001", current_prod_version, cand_version, current_prod_rmse, cand_rmse, 1 if promoted else 0, decision_reason))

            # Audit log
            conn.execute("""
            INSERT INTO audit_logs (event_type, entity_type, entity_id, action, details_json, operator)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                "MODEL_RETRAINING_EVALUATION", "model", cand_version,
                f"Candidate model {'PROMOTED' if promoted else 'REJECTED'}",
                json.dumps({"candidate_metrics": cand_metrics, "decision_reason": decision_reason}),
                "MLOPS_GATEKEEPER"
            ))
            conn.commit()

        logger.info(f"Retraining Run {run_id} Result: Promoted={promoted}. Reason: {decision_reason}")
        return {
            "run_id": run_id,
            "promoted": promoted,
            "candidate_version": cand_version,
            "candidate_metrics": cand_metrics,
            "current_production_version": current_prod_version,
            "current_production_rmse": current_prod_rmse,
            "decision_reason": decision_reason
        }
