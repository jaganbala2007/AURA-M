"""
AURA-M Preprocessing Module
Strictly prevents data leakage by fitting normalizers and transformers on training engine trajectories only.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger("AURA-M.Preprocessor")

class TimeSeriesPreprocessor:
    """Industrial sensor preprocessor preserving engine identity and time ordering."""

    def __init__(
        self,
        informative_sensors: Optional[List[str]] = None,
        scaler_type: str = "minmax"
    ):
        self.informative_sensors = informative_sensors or [
            "s_2", "s_3", "s_4", "s_6", "s_7", "s_8", "s_9",
            "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
        ]
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type == "minmax" else StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "TimeSeriesPreprocessor":
        """Fit scaler exclusively on training dataset."""
        logger.info(f"Fitting {self.scaler_type} scaler on {len(self.informative_sensors)} informative sensors...")
        self.scaler.fit(train_df[self.informative_sensors])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to dataset without data leakage."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform can be invoked.")
        
        df_out = df.copy()
        scaled_values = self.scaler.transform(df[self.informative_sensors])
        
        # Replace sensor values with scaled versions
        for idx, col in enumerate(self.informative_sensors):
            df_out[col] = scaled_values[:, idx]
            
        return df_out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training dataset."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: str) -> None:
        """Persists fitted scaler artifact."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Fitted preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TimeSeriesPreprocessor":
        """Loads fitted preprocessor from disk."""
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        logger.info(f"Loaded preprocessor from {filepath}")
        return instance
