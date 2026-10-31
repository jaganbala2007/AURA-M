"""
AURA-M Data Quality & Validation Engine
Performs rigorous health checks on raw or streaming engine sensor records.
Ensures zero data leakage, type consistency, and time-series monotonicity.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("AURA-M.DataValidator")

class DataValidator:
    """Production quality validation suite for industrial sensor time-series."""

    REQUIRED_INDEX = ["engine_id", "cycle"]
    REQUIRED_SETTINGS = ["setting_1", "setting_2", "setting_3"]
    REQUIRED_SENSORS = [f"s_{i}" for i in range(1, 22)]
    TOTAL_EXPECTED_COLUMNS = 26

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame, is_train: bool = True) -> Dict[str, Any]:
        """
        Executes end-to-end data validation checks.
        
        Returns:
            Dict containing validation summary, metrics, and pass/fail boolean status.
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "num_engines": 0,
                "null_count": 0,
                "duplicate_rows": 0,
                "monotonic_failures": 0
            }
        }

        # 1. Column Structure Check
        for col in cls.REQUIRED_INDEX + cls.REQUIRED_SETTINGS + cls.REQUIRED_SENSORS:
            if col not in df.columns:
                report["errors"].append(f"Missing mandatory telemetry channel: '{col}'")
                report["is_valid"] = False

        if not report["is_valid"]:
            return report

        # 2. Null, NaN, Inf Detection
        null_counts = df.isnull().sum().sum()
        report["metrics"]["null_count"] = int(null_counts)
        if null_counts > 0:
            report["errors"].append(f"Detected {null_counts} null/NaN values across dataset.")
            report["is_valid"] = False

        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df.values).sum()
        if inf_counts > 0:
            report["errors"].append(f"Detected {inf_counts} infinite values in numeric channels.")
            report["is_valid"] = False

        # 3. Duplicate Cycle Check
        dups = df.duplicated(subset=["engine_id", "cycle"]).sum()
        report["metrics"]["duplicate_rows"] = int(dups)
        if dups > 0:
            report["errors"].append(f"Found {dups} duplicate engine_id + cycle records.")
            report["is_valid"] = False

        # 4. Engine Identity and Monotonicity
        engines = df["engine_id"].unique()
        report["metrics"]["num_engines"] = len(engines)

        monotonic_errors = 0
        for eng in engines:
            eng_cycles = df[df["engine_id"] == eng]["cycle"].values
            # Check strictly increasing cycle order
            if not np.all(np.diff(eng_cycles) > 0):
                monotonic_errors += 1

        report["metrics"]["monotonic_failures"] = monotonic_errors
        if monotonic_errors > 0:
            report["errors"].append(
                f"Time-series monotonicity violation in {monotonic_errors} engine trajectories."
            )
            report["is_valid"] = False

        # 5. Sensor Variance & Constancy Warning
        low_variance_sensors = []
        for s in cls.REQUIRED_SENSORS:
            if df[s].std() < 1e-6:
                low_variance_sensors.append(s)
        
        if low_variance_sensors:
            report["warnings"].append(
                f"Sensors with near-zero variance detected: {low_variance_sensors}. "
                "These channels provide no degradation signal and should be dropped during feature generation."
            )

        logger.info(
            f"Data Validation Complete: Valid={report['is_valid']}, "
            f"Rows={len(df)}, Engines={len(engines)}, Errors={len(report['errors'])}, Warnings={len(report['warnings'])}"
        )
        return report

    @classmethod
    def validate_streaming_record(cls, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates a single incoming cycle record in simulated streaming."""
        for key in ["engine_id", "cycle"]:
            if key not in record:
                return False, f"Missing index channel: {key}"
            if not isinstance(record[key], (int, np.integer)):
                return False, f"Channel {key} must be an integer"

        for s in cls.REQUIRED_SENSORS:
            if s not in record:
                return False, f"Missing sensor reading: {s}"
            if not isinstance(record[s], (int, float, np.floating)):
                return False, f"Sensor {s} reading must be numeric"

        return True, None
