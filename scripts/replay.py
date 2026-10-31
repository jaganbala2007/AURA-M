"""
AURA-M Simulated Streaming Replay Script
Demonstrates cycle-by-cycle simulated telemetry streaming from the NASA C-MAPSS dataset.
Shows progressive degradation fingerprinting, anomaly scoring, RUL prediction, and triage response.
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import CMAPSSDataLoader
from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.engineer import FeatureEngineer
from src.degradation.fingerprint import DegradationFingerprintEngine
from src.anomaly.detector import AnomalyDetectionEngine
from src.rul.models import load_rul_model
from src.uncertainty.conformal import ConformalPredictor
from src.fusion.engine import FusionInferenceEngine
from src.agent.triage import AutonomousTriageAgent
from src.workorders.manager import WorkOrderManager
from src.database.db import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.Replay")

def main():
    parser = argparse.ArgumentParser(description="AURA-M Simulated Streaming Replay")
    parser.add_argument("--engine-id", type=int, default=34, help="Engine Unit ID (1-100)")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds per cycle")
    parser.add_argument("--start-cycle", type=int, default=1, help="Starting cycle")
    args = parser.parse_args()

    logger.info(f"Initiating simulated streaming replay for Engine #{args.engine_id} (NASA C-MAPSS FD001)...")

    loader = CMAPSSDataLoader(data_dir="data/raw")
    _, test_df, test_rul = loader.load_subset(subset="FD001", clip_rul=125)

    eng_data = test_df[test_df["engine_id"] == args.engine_id].sort_values("cycle").reset_index(drop=True)
    if eng_data.empty:
        logger.error(f"Engine #{args.engine_id} not found in test dataset.")
        return 1

    models_dir = Path("models/trained")
    version = "v1.0.0-BASELINE-RF"
    preproc = TimeSeriesPreprocessor.load(str(models_dir / f"preprocessor_{version}.pkl"))
    feat_eng = FeatureEngineer()
    fingerprint_engine = DegradationFingerprintEngine()
    anom_engine = AnomalyDetectionEngine.load(str(models_dir / f"anomaly_detector_{version}.pkl"))
    rul_model = load_rul_model(str(models_dir / f"rul_model_{version}.pkl"))
    unc_engine = ConformalPredictor.load(str(models_dir / f"uncertainty_conformal_{version}.pkl"))

    fusion_engine = FusionInferenceEngine(
        preprocessor=preproc,
        feature_engineer=feat_eng,
        fingerprint_engine=fingerprint_engine,
        anomaly_engine=anom_engine,
        rul_model=rul_model,
        uncertainty_engine=unc_engine,
        model_version=version
    )
    triage_agent = AutonomousTriageAgent()
    db = DatabaseManager(db_path="data/auram.db")

    total_cycles = len(eng_data)
    true_final_rul = test_rul.loc[args.engine_id]
    print(f"\n[STREAM] Streaming Engine #{args.engine_id}: {total_cycles} observed cycles | True End-of-Test RUL: {true_final_rul} cycles")
    print("-" * 105)
    print(f"{'Cycle':<7} | {'Health Idx':<10} | {'Status':<16} | {'Anomaly':<8} | {'Pred RUL':<9} | {'90% Conf Interval':<19} | {'Triage Action':<22}")
    print("-" * 105)

    # Step through cycles (subsampling if long for demonstration)
    step = 5 if total_cycles > 50 else 1
    for c_idx in range(args.start_cycle - 1, total_cycles, step):
        current_cycle = int(eng_data["cycle"].iloc[c_idx])
        sub_df = eng_data.iloc[:c_idx + 1]

        payload = fusion_engine.analyze_engine(sub_df, current_cycle=current_cycle)
        active_wo = db.get_active_work_order(args.engine_id)
        decision = triage_agent.evaluate(payload, active_work_order=active_wo)

        if decision["should_create_work_order"]:
            wo = WorkOrderManager.build_work_order(payload, decision)
            db.insert_work_order(wo)

        conf_str = f"[{payload['rul_lower']:.1f}, {payload['rul_upper']:.1f}]"
        print(
            f"#{current_cycle:<6} | "
            f"{payload['health_index']:<10.2f} | "
            f"{payload['health_status']:<16} | "
            f"{payload['anomaly_score']:<8.2f} | "
            f"{payload['predicted_rul']:<9.1f} | "
            f"{conf_str:<19} | "
            f"{decision['decision']:<22}"
        )
        time.sleep(args.delay)

    print("-" * 105)
    print("[STREAM COMPLETE] Telemetry stream finished. Inspection and maintenance records updated.\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
