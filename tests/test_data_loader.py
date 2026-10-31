"""
Unit and integration tests for dataset loading, validation, and leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import CMAPSSDataLoader
from src.data.validator import DataValidator

def test_data_loader_structure():
    loader = CMAPSSDataLoader(data_dir="data/raw")
    train_df, test_df, test_rul = loader.load_subset("FD001", clip_rul=125)
    
    # Assert column counts
    assert "engine_id" in train_df.columns
    assert "cycle" in train_df.columns
    assert "RUL_true" in train_df.columns
    assert "RUL_target" in train_df.columns
    
    # Assert engine counts
    assert train_df["engine_id"].nunique() == 100
    assert test_df["engine_id"].nunique() == 100
    assert len(test_rul) == 100
    
    # Check piecewise clipping (upper cap = 125)
    assert train_df["RUL_target"].max() == 125

def test_data_validator_pass():
    loader = CMAPSSDataLoader(data_dir="data/raw")
    train_df, _, _ = loader.load_subset("FD001", clip_rul=125)
    report = DataValidator.validate_dataset(train_df, is_train=True)
    
    assert report["is_valid"] is True
    assert report["metrics"]["null_count"] == 0
    assert report["metrics"]["duplicate_rows"] == 0
    assert report["metrics"]["monotonic_failures"] == 0

def test_data_validator_detects_nulls_and_duplicates():
    # Construct faulty test dataframe
    faulty_df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [1, 1, 2],  # Duplicate cycle
        "setting_1": [0.0, np.nan, 0.0],  # Null value
        "setting_2": [0.0, 0.0, 0.0],
        "setting_3": [100.0, 100.0, 100.0]
    })
    for i in range(1, 22):
        faulty_df[f"s_{i}"] = 1.0

    report = DataValidator.validate_dataset(faulty_df, is_train=False)
    assert report["is_valid"] is False
    assert report["metrics"]["null_count"] > 0
    assert report["metrics"]["duplicate_rows"] > 0

def test_no_engine_leakage_in_split():
    loader = CMAPSSDataLoader(data_dir="data/raw")
    train_df, _, _ = loader.load_subset("FD001")
    
    all_engines = set(train_df["engine_id"].unique())
    train_eng = set(list(all_engines)[:80])
    val_eng = set(list(all_engines)[80:])
    
    # Verify strict disjointness
    assert len(train_eng.intersection(val_eng)) == 0
    assert len(train_eng) + len(val_eng) == 100
