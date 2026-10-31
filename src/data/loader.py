"""
AURA-M Data Loading and Ingestion Module
Handles loading, validation, and schema enforcement for the NASA C-MAPSS dataset.
Supports FD001 through FD004.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger("AURA-M.DataLoader")

INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors with zero/near-zero variance in FD001 (single operating condition)
CONSTANT_SENSORS_FD001 = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
INFORMATIVE_SENSORS_FD001 = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS_FD001]

class CMAPSSDataLoader:
    """Robust data loader for NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_subset(
        self, subset: str = "FD001", clip_rul: Optional[int] = 125
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Loads training, test, and ground-truth RUL for the specified subset.
        
        Args:
            subset: One of 'FD001', 'FD002', 'FD003', 'FD004'
            clip_rul: Upper cap for piece-wise linear RUL target (standard in prognostics literature: 125 cycles).
                      None for unclipped linear RUL.
        
        Returns:
            train_df: Full run-to-failure telemetry with calculated true RUL and clipped RUL
            test_df: Censored operational telemetry up to current observation
            true_test_rul: Series of true ground-truth RUL for each test engine at last cycle
        """
        train_file = self.data_dir / f"train_{subset}.txt"
        test_file = self.data_dir / f"test_{subset}.txt"
        rul_file = self.data_dir / f"RUL_{subset}.txt"

        if not train_file.exists():
            raise FileNotFoundError(f"Missing train file: {train_file.resolve()}")
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test file: {test_file.resolve()}")
        if not rul_file.exists():
            raise FileNotFoundError(f"Missing RUL file: {rul_file.resolve()}")

        logger.info(f"Loading C-MAPSS subset {subset}...")
        
        # Load raw space-delimited text
        train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
        test_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])

        # Compute Ground Truth RUL for Training Data (Run-to-failure: RUL = max_cycle - current_cycle)
        max_cycle_per_engine = train_df.groupby("engine_id")["cycle"].max().reset_index()
        max_cycle_per_engine.rename(columns={"cycle": "max_cycle"}, inplace=True)
        train_df = train_df.merge(max_cycle_per_engine, on="engine_id", how="left")
        train_df["RUL_true"] = train_df["max_cycle"] - train_df["cycle"]
        
        # Piecewise-linear RUL target capping (standard industrial benchmark practice)
        if clip_rul is not None:
            train_df["RUL_target"] = train_df["RUL_true"].clip(upper=clip_rul)
        else:
            train_df["RUL_target"] = train_df["RUL_true"]

        # True RUL series for test engines (indexed by 1-based engine_id)
        test_rul.index = range(1, len(test_rul) + 1)
        test_rul.index.name = "engine_id"

        logger.info(
            f"Subset {subset} loaded successfully: "
            f"Train={len(train_df)} rows ({train_df['engine_id'].nunique()} engines), "
            f"Test={len(test_df)} rows ({test_df['engine_id'].nunique()} engines)"
        )
        return train_df, test_df, test_rul["true_rul"]

    @staticmethod
    def get_column_definitions() -> Dict[str, str]:
        """Provides human-readable telemetry schema documentation."""
        return {
            "engine_id": "Engine Unit Number (Fleet Identifier)",
            "cycle": "Operating Time Cycles completed",
            "setting_1": "Altitude / Operational Setting 1",
            "setting_2": "Mach Number / Operational Setting 2",
            "setting_3": "TRA (Throttle Resolver Angle) Setting 3",
            "s_1": "Fan Inlet Temperature (T2) [°R]",
            "s_2": "LPC Outlet Temperature (T24) [°R]",
            "s_3": "HPC Outlet Temperature (T30) [°R]",
            "s_4": "LPT Outlet Temperature (T50) [°R]",
            "s_5": "Fan Inlet Pressure (P2) [psia]",
            "s_6": "Bypass Duct Pressure (P15) [psia]",
            "s_7": "HPC Outlet Pressure (P30) [psia]",
            "s_8": "Physical Fan Speed (Nf) [rpm]",
            "s_9": "Physical Core Speed (Nc) [rpm]",
            "s_10": "Engine Pressure Ratio (epr)",
            "s_11": "HPC Static Pressure (Ps30) [psia]",
            "s_12": "Ratio of Fuel Flow to Ps30 (phi) [pps/psi]",
            "s_13": "Corrected Fan Speed (NRf) [rpm]",
            "s_14": "Corrected Core Speed (NRc) [rpm]",
            "s_15": "Bypass Ratio (BPR)",
            "s_16": "Burner Fuel-Air Ratio (farB)",
            "s_17": "Bleed Enthalpy (htBleed)",
            "s_18": "Demanded Fan Speed (Nf_dmd) [rpm]",
            "s_19": "Demanded Corrected Fan Speed (PCNfR_dmd) [rpm]",
            "s_20": "HPT Coolant Bleed (W31) [lbm/s]",
            "s_21": "LPT Coolant Bleed (W32) [lbm/s]"
        }
