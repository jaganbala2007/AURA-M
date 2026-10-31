"""
Unit and integration tests for SQLite database operations and FastAPI backend endpoints.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.database.db import DatabaseManager
from src.api.app import app

TEST_DB_PATH = "data/test_auram.db"

@pytest.fixture(scope="module")
def test_db():
    p = Path(TEST_DB_PATH)
    if p.exists():
        p.unlink()
    db = DatabaseManager(db_path=TEST_DB_PATH)
    yield db
    if p.exists():
        p.unlink()

def test_database_crud(test_db):
    # 1. Upsert engine
    test_db.upsert_engine(engine_id=99, subset="FD001", total_cycles=150, last_cycle=150, health_status="HEALTHY", rul=85.0, anomaly_score=0.2)
    engine = test_db.get_engine(99)
    assert engine is not None
    assert engine["engine_id"] == 99
    assert engine["current_rul"] == 85.0
    
    # 2. Record prediction
    payload = {
        "engine_id": 99,
        "cycle": 150,
        "predicted_rul": 85.0,
        "rul_lower": 65.0,
        "rul_upper": 105.0,
        "interval_width": 40.0,
        "anomaly_score": 0.22,
        "uncertainty_category": "MODERATE_CONFIDENCE",
        "human_review_required": False,
        "model_version": "v1.0.0-TEST",
        "health_index": 0.78,
        "health_status": "HEALTHY",
        "degradation_fingerprint": {"degradation_rate_per_cycle": 0.005}
    }
    test_db.record_prediction(payload)
    
    # 3. Work Order Insert & Status Update
    wo = {
        "work_order_id": "WO-099-0150-TEST",
        "engine_id": 99,
        "cycle": 150,
        "priority": "MEDIUM",
        "reason": "RULE_SCHEDULED_INSPECTION",
        "recommended_action": "Borescope inspection",
        "status": "OPEN",
        "evidence_report": {},
        "model_version": "v1.0.0-TEST",
        "created_at": "2026-10-31T00:00:00Z",
        "updated_at": "2026-10-31T00:00:00Z"
    }
    test_db.insert_work_order(wo)
    active_wo = test_db.get_active_work_order(99)
    assert active_wo is not None
    assert active_wo["work_order_id"] == "WO-099-0150-TEST"
    
    # Transition to CLOSED
    success = test_db.update_work_order_status("WO-099-0150-TEST", "CLOSED", notes="Inspection completed, no cracks found.")
    assert success is True
    assert test_db.get_active_work_order(99) is None

def test_fastapi_endpoints():
    client = TestClient(app)
    
    # Health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "HEALTHY"
    
    # Engines listing
    res_engines = client.get("/api/v1/engines")
    assert res_engines.status_code == 200
    assert "engines" in res_engines.json()
    
    # Work orders listing
    res_wo = client.get("/api/v1/work-orders")
    assert res_wo.status_code == 200
    assert "work_orders" in res_wo.json()
