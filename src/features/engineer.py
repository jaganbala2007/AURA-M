"""
AURA-M Feature Engineering Engine
Computes rolling window dynamics, rate of change, baseline drift, and degradation indicators.
Preserves engine boundary integrity (never leaks cross-engine information).
"""

import logging
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("AURA-M.FeatureEngineer")

class FeatureEngineer:
    """Engineers temporal degradation signals from multivariate sensor time-series."""

    def __init__(
        self,
        informative_sensors: Optional[List[str]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        self.sensors = informative_sensors or [
            "s_2", "s_3", "s_4", "s_6", "s_7", "s_8", "s_9",
            "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
        ]
        self.rolling_windows = rolling_windows or [5, 15]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling statistics, differential trends, and baseline drift.
        Groups strictly by engine_id to prevent cross-engine contamination.
        """
        logger.info(f"Generating temporal features across {len(self.sensors)} sensors for {df['engine_id'].nunique()} engines...")
        
        # Sort values chronologically per engine
        df_sorted = df.sort_values(by=["engine_id", "cycle"]).copy()

        engineered_dfs = []
        for engine_id, group in df_sorted.groupby("engine_id"):
            eng_data = group.copy()
            
            # 1. Baseline estimation from initial pristine cycles (cycles 1 to 5)
            initial_baseline = eng_data.iloc[:min(5, len(eng_data))][self.sensors].mean()

            for s in self.sensors:
                # 2. Rate of change (1-cycle difference)
                eng_data[f"{s}_roc"] = eng_data[s].diff().fillna(0.0)

                # 3. Baseline drift (deviation from healthy initial condition)
                eng_data[f"{s}_drift"] = eng_data[s] - initial_baseline[s]

                # 4. Rolling statistics for each window
                for w in self.rolling_windows:
                    eng_data[f"{s}_roll_mean_{w}"] = (
                        eng_data[s].rolling(window=w, min_periods=1).mean()
                    )
                    eng_data[f"{s}_roll_std_{w}"] = (
                        eng_data[s].rolling(window=w, min_periods=1).std().fillna(0.0)
                    )

            engineered_dfs.append(eng_data)

        out_df = pd.concat(engineered_dfs, ignore_index=True)
        logger.info(f"Feature engineering complete. Total feature columns: {len(out_df.columns)}")
        return out_df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of feature column names used for ML modeling (excluding metadata & targets)."""
        exclude = ["engine_id", "cycle", "max_cycle", "RUL_true", "RUL_target", "setting_1", "setting_2", "setting_3"]
        # Also exclude raw constant sensors
        exclude += ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
        
        feature_cols = [c for c in df.columns if c not in exclude]
        return feature_cols
