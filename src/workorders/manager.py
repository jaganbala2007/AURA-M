"""
AURA-M Maintenance Work Order Service & Evidence Report Generator (Innovation 5)
Manages the complete lifecycle of maintenance work orders and generates explainable,
evidence-backed maintenance passports for industrial assets.
"""

import uuid
import datetime
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger("AURA-M.WorkOrderManager")

class WorkOrderManager:
    """Enterprise work order lifecycle manager with comprehensive evidence reporting."""

    STATUS_OPEN = "OPEN"
    STATUS_IN_PROGRESS = "IN_PROGRESS"
    STATUS_CLOSED = "CLOSED"
    STATUS_ESCALATED = "ESCALATED"

    @classmethod
    def generate_evidence_report(
        cls,
        telemetry_payload: Dict[str, Any],
        agent_decision: Dict[str, Any],
        work_order_id: str
    ) -> Dict[str, Any]:
        """
        Assembles a self-contained, auditable Evidence Report supporting maintenance actions.
        """
        fingerprint = telemetry_payload.get("degradation_fingerprint", {})
        top_deviating = fingerprint.get("dominant_deviating_sensors", [])

        evidence = {
            "work_order_id": work_order_id,
            "engine_id": int(telemetry_payload["engine_id"]),
            "cycle_evaluated": int(telemetry_payload["cycle"]),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_version": telemetry_payload.get("model_version", "v1.0.0-PROD"),
            "model_type": telemetry_payload.get("model_type", "xgboost"),
            "prognostics": {
                "predicted_rul": float(telemetry_payload["predicted_rul"]),
                "rul_lower_bound": float(telemetry_payload["rul_lower"]),
                "rul_upper_bound": float(telemetry_payload["rul_upper"]),
                "interval_width_cycles": float(telemetry_payload["interval_width"]),
                "confidence_level": telemetry_payload.get("rul", {}).get("confidence_level", "90%"),
                "uncertainty_category": telemetry_payload["uncertainty_category"]
            },
            "condition_monitoring": {
                "health_index": float(telemetry_payload["health_index"]),
                "health_status": telemetry_payload["health_status"],
                "anomaly_score": float(telemetry_payload["anomaly_score"]),
                "is_anomalous": bool(telemetry_payload["is_anomaly"]),
                "degradation_rate_per_cycle": float(fingerprint.get("degradation_rate_per_cycle", 0.0)),
                "leading_deviating_sensors": top_deviating
            },
            "decision_intelligence": {
                "decision": agent_decision["decision"],
                "severity": agent_decision["severity"],
                "rule_triggered": agent_decision["triggered_rule"],
                "policy_version": agent_decision.get("policy_version", "2026.1"),
                "rationale": agent_decision["rationale"],
                "recommended_action": agent_decision["recommended_action"]
            },
            "audit_note": "Generated autonomously by AURA-M Decision Engine. Real-world physical dispatch requires verified integration."
        }
        return evidence

    @classmethod
    def build_work_order(
        cls,
        telemetry_payload: Dict[str, Any],
        agent_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Constructs a new work order record with evidence attachment."""
        work_order_id = f"WO-{telemetry_payload['engine_id']:03d}-{telemetry_payload['cycle']:04d}-{str(uuid.uuid4())[:6].upper()}"
        evidence = cls.generate_evidence_report(telemetry_payload, agent_decision, work_order_id)
        
        return {
            "work_order_id": work_order_id,
            "engine_id": int(telemetry_payload["engine_id"]),
            "cycle": int(telemetry_payload["cycle"]),
            "priority": agent_decision["severity"],
            "reason": agent_decision["triggered_rule"],
            "recommended_action": agent_decision["recommended_action"],
            "status": cls.STATUS_OPEN,
            "evidence_report": evidence,
            "model_version": telemetry_payload.get("model_version", "v1.0.0-PROD"),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "closing_notes": None
        }
