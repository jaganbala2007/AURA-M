"""
AURA-M Autonomous Maintenance Triage Agent (Innovation 4)
A deterministic, explainable, non-LLM decision engine.
Evaluates multi-model prognostic evidence, prevents duplicate open work orders,
and routes high-uncertainty cases to human verification.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger("AURA-M.TriageAgent")

class AutonomousTriageAgent:
    """Non-LLM deterministic decision engine implementing configurable triage policies."""

    def __init__(self, thresholds_path: str = "config/thresholds.yaml"):
        self.thresholds_path = Path(thresholds_path)
        self.config = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, Any]:
        """Loads policy thresholds from YAML configuration."""
        if self.thresholds_path.exists():
            with open(self.thresholds_path, "r") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("triage_agent", {})
        logger.warning(f"Thresholds file not found at {self.thresholds_path}. Using fallback defaults.")
        return {
            "policy_version": "2026.1-FALLBACK",
            "rul_thresholds": {
                "normal_monitoring_min": 100.0,
                "scheduled_inspection_min": 50.0,
                "high_priority_maintenance_min": 20.0,
                "critical_maintenance_max": 20.0
            },
            "anomaly_triggers": {
                "anomaly_score_threshold": 0.65,
                "rapid_degradation_slope": -0.85
            },
            "uncertainty_policy": {
                "max_acceptable_interval_width": 45.0,
                "confidence_alpha": 0.10
            },
            "work_order_rules": {
                "deduplication_enabled": True,
                "auto_escalation": True
            }
        }

    def evaluate(
        self,
        telemetry_payload: Dict[str, Any],
        active_work_order: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes policy rules on fused prognostic intelligence.
        
        Args:
            telemetry_payload: Output from FusionInferenceEngine
            active_work_order: Existing unresolved work order for engine, if any
            
        Returns:
            Dict containing action decision, triggered rule, severity, rationale,
            and work order creation directive.
        """
        engine_id = telemetry_payload["engine_id"]
        cycle = telemetry_payload["cycle"]
        rul_point = telemetry_payload["predicted_rul"]
        interval_width = telemetry_payload["interval_width"]
        human_review = telemetry_payload["human_review_required"]
        anomaly_score = telemetry_payload["anomaly_score"]
        health_index = telemetry_payload["health_index"]
        
        rul_cfg = self.config.get("rul_thresholds", {})
        norm_min = rul_cfg.get("normal_monitoring_min", 100.0)
        sched_min = rul_cfg.get("scheduled_inspection_min", 50.0)
        high_min = rul_cfg.get("high_priority_maintenance_min", 20.0)
        crit_max = rul_cfg.get("critical_maintenance_max", 20.0)
        
        anom_thresh = self.config.get("anomaly_triggers", {}).get("anomaly_score_threshold", 0.65)
        max_interval = self.config.get("uncertainty_policy", {}).get("max_acceptable_interval_width", 45.0)

        # -------------------------------------------------------------
        # RULE 0: UNCERTAINTY ESCALATION / HUMAN VERIFICATION
        # If model epistemic uncertainty is excessive, do not automate action blindly.
        # -------------------------------------------------------------
        if human_review or interval_width > max_interval:
            decision = "HUMAN_VERIFICATION"
            triggered_rule = "RULE_UNCERTAINTY_EXCESSIVE"
            severity = "HIGH"
            rationale = (
                f"Prediction interval width ({interval_width:.1f} cycles) exceeds maximum acceptable policy "
                f"threshold ({max_interval:.1f} cycles). Automated work order suspended pending human verification."
            )
            action_code = "ROUTE_HUMAN_REVIEW"
            recommended_action = "Certified reliability specialist must review raw sensor logs and verify calibration."

        # -------------------------------------------------------------
        # RULE 1: CRITICAL MAINTENANCE
        # Predicted RUL < 20 cycles OR extremely collapsed Health Index (< 0.20)
        # -------------------------------------------------------------
        elif rul_point < crit_max or health_index < 0.20:
            decision = "CRITICAL_MAINTENANCE"
            triggered_rule = "RULE_CRITICAL_RUL_LIMIT"
            severity = "CRITICAL"
            rationale = (
                f"Predicted Remaining Useful Life ({rul_point:.1f} cycles) is below critical threshold ({crit_max:.1f} cycles) "
                f"with Health Index={health_index:.2f}. Immediate servicing required to prevent uncontained failure."
            )
            action_code = "DISPATCH_CRITICAL_MAINTENANCE"
            recommended_action = "Ground asset and schedule immediate hot-section / HPC maintenance overhaul."

        # -------------------------------------------------------------
        # RULE 2: HIGH PRIORITY MAINTENANCE
        # 20 <= RUL < 50 cycles OR elevated anomaly score (>= 0.65) with degrading HI
        # -------------------------------------------------------------
        elif (high_min <= rul_point < sched_min) or (anomaly_score >= anom_thresh and health_index < 0.50):
            decision = "HIGH_PRIORITY_MAINTENANCE"
            triggered_rule = "RULE_HIGH_PRIORITY_DEGRADATION"
            severity = "HIGH"
            rationale = (
                f"Predicted RUL ({rul_point:.1f} cycles) is in the 20-50 cycle window, supported by "
                f"anomaly score {anomaly_score:.2f} and Health Index {health_index:.2f}."
            )
            action_code = "DISPATCH_HIGH_PRIORITY"
            recommended_action = "Procure replacement compressor modules and schedule service within next 10 operating cycles."

        # -------------------------------------------------------------
        # RULE 3: SCHEDULED INSPECTION
        # 50 <= RUL <= 100 cycles OR notable early sensor drift
        # -------------------------------------------------------------
        elif (sched_min <= rul_point <= norm_min) or (anomaly_score >= 0.50):
            decision = "SCHEDULE_INSPECTION"
            triggered_rule = "RULE_SCHEDULED_INSPECTION"
            severity = "MEDIUM"
            rationale = (
                f"Predicted RUL ({rul_point:.1f} cycles) indicates onset of measurable degradation. "
                f"Routine preventive inspection recommended before accelerated wear."
            )
            action_code = "RAISE_INSPECTION_ORDER"
            recommended_action = "Perform routine borescope inspection and vibration baseline validation."

        # -------------------------------------------------------------
        # RULE 4: NORMAL MONITORING
        # RUL > 100 cycles and nominal anomaly scores
        # -------------------------------------------------------------
        else:
            decision = "NORMAL_MONITORING"
            triggered_rule = "RULE_NOMINAL_OPERATION"
            severity = "LOW"
            rationale = (
                f"Engine operating within nominal envelope (Predicted RUL: {rul_point:.1f} cycles, "
                f"Anomaly Score: {anomaly_score:.2f}, Health Index: {health_index:.2f})."
            )
            action_code = "CONTINUE_MONITORING"
            recommended_action = "Maintain continuous high-frequency telemetry streaming."

        # -------------------------------------------------------------
        # WORK ORDER DUPLICATION & ESCALATION GUARD
        # -------------------------------------------------------------
        should_create_work_order = False
        work_order_escalated = False

        if decision in ["SCHEDULE_INSPECTION", "HIGH_PRIORITY_MAINTENANCE", "CRITICAL_MAINTENANCE", "HUMAN_VERIFICATION"]:
            if active_work_order is None:
                # No active unresolved work order exists -> Create new work order
                should_create_work_order = True
            else:
                # Existing active work order exists: check if severity escalated
                existing_priority = active_work_order.get("priority", "MEDIUM")
                severity_ranks = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
                if severity_ranks.get(severity, 1) > severity_ranks.get(existing_priority, 1):
                    should_create_work_order = False
                    work_order_escalated = True
                    rationale += f" [ESCALATION: Existing active order #{active_work_order.get('work_order_id')} escalated from {existing_priority} to {severity}]"
                else:
                    should_create_work_order = False
                    rationale += f" [SUPPRESSED: Active work order #{active_work_order.get('work_order_id')} ({active_work_order.get('status')}) already exists. Duplicate order prevented.]"

        agent_result = {
            "engine_id": engine_id,
            "cycle": cycle,
            "decision": decision,
            "triggered_rule": triggered_rule,
            "severity": severity,
            "action_code": action_code,
            "recommended_action": recommended_action,
            "rationale": rationale,
            "should_create_work_order": should_create_work_order,
            "work_order_escalated": work_order_escalated,
            "policy_version": self.config.get("policy_version", "2026.1")
        }
        return agent_result
