"""
Unit tests for Autonomous Triage Agent, policy rule triggers, deduplication, and work orders.
"""

import pytest
from src.agent.triage import AutonomousTriageAgent
from src.workorders.manager import WorkOrderManager

@pytest.fixture
def agent():
    return AutonomousTriageAgent()

def test_triage_normal_monitoring(agent):
    payload = {
        "engine_id": 1,
        "cycle": 20,
        "predicted_rul": 115.0,
        "interval_width": 20.0,
        "human_review_required": False,
        "anomaly_score": 0.15,
        "health_index": 0.95,
        "health_status": "HEALTHY"
    }
    decision = agent.evaluate(payload, active_work_order=None)
    assert decision["decision"] == "NORMAL_MONITORING"
    assert decision["severity"] == "LOW"
    assert decision["should_create_work_order"] is False

def test_triage_scheduled_inspection(agent):
    payload = {
        "engine_id": 2,
        "cycle": 80,
        "predicted_rul": 75.0,
        "interval_width": 22.0,
        "human_review_required": False,
        "anomaly_score": 0.35,
        "health_index": 0.65,
        "health_status": "MODERATE_WEAR"
    }
    decision = agent.evaluate(payload, active_work_order=None)
    assert decision["decision"] == "SCHEDULE_INSPECTION"
    assert decision["severity"] == "MEDIUM"
    assert decision["should_create_work_order"] is True

def test_triage_critical_maintenance(agent):
    payload = {
        "engine_id": 3,
        "cycle": 180,
        "predicted_rul": 12.0,
        "interval_width": 20.0,
        "human_review_required": False,
        "anomaly_score": 0.88,
        "health_index": 0.15,
        "health_status": "CRITICAL_DEGRADATION"
    }
    decision = agent.evaluate(payload, active_work_order=None)
    assert decision["decision"] == "CRITICAL_MAINTENANCE"
    assert decision["severity"] == "CRITICAL"
    assert decision["should_create_work_order"] is True

def test_triage_human_verification_on_high_uncertainty(agent):
    payload = {
        "engine_id": 4,
        "cycle": 50,
        "predicted_rul": 35.0,
        "interval_width": 55.0,  # Exceeds max policy interval (45.0)
        "human_review_required": True,
        "anomaly_score": 0.40,
        "health_index": 0.55,
        "health_status": "MODERATE_WEAR"
    }
    decision = agent.evaluate(payload, active_work_order=None)
    assert decision["decision"] == "HUMAN_VERIFICATION"
    assert "UNCERTAINTY" in decision["triggered_rule"]

def test_work_order_deduplication(agent):
    payload = {
        "engine_id": 5,
        "cycle": 90,
        "predicted_rul": 65.0,
        "interval_width": 20.0,
        "human_review_required": False,
        "anomaly_score": 0.40,
        "health_index": 0.60,
        "health_status": "MODERATE_WEAR"
    }
    # Simulate existing active work order
    active_wo = {
        "work_order_id": "WO-005-0085-TEST",
        "priority": "MEDIUM",
        "status": "OPEN"
    }
    decision = agent.evaluate(payload, active_work_order=active_wo)
    
    # Must suppress duplicate work order creation
    assert decision["should_create_work_order"] is False
    assert "SUPPRESSED" in decision["rationale"]

def test_work_order_escalation(agent):
    payload = {
        "engine_id": 6,
        "cycle": 150,
        "predicted_rul": 15.0,  # Now critical
        "interval_width": 20.0,
        "human_review_required": False,
        "anomaly_score": 0.85,
        "health_index": 0.18,
        "health_status": "CRITICAL_DEGRADATION"
    }
    # Previous order was only MEDIUM priority
    active_wo = {
        "work_order_id": "WO-006-0120-TEST",
        "priority": "MEDIUM",
        "status": "OPEN"
    }
    decision = agent.evaluate(payload, active_work_order=active_wo)
    
    # Must flag escalation
    assert decision["work_order_escalated"] is True
    assert "ESCALATION" in decision["rationale"]
