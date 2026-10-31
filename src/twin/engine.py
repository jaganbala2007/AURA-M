"""
AURA-M AI Digital Twin Operations Engine (Technology Module 2)
Maintains virtual twin health states, subsystem wear telemetry, thermodynamic status,
and synchronizes cycle-by-cycle sensor updates with physical-to-virtual boundaries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("AURA-M.DigitalTwin")

class EngineDigitalTwin:
    """
    Virtual Engine Digital Twin maintaining live component-level degradation states.
    Models key aerothermodynamic turbofan sections:
    - Fan & Inlet
    - Low Pressure Compressor (LPC)
    - High Pressure Compressor (HPC)
    - Combustor / Burner
    - High Pressure Turbine (HPT)
    - Low Pressure Turbine (LPT) & Bypass
    """

    SUBSYSTEM_SENSORS = {
        "fan_inlet": ["s_1", "s_8", "s_13"],
        "lpc": ["s_2"],
        "hpc": ["s_3", "s_7", "s_11"],
        "combustor": ["s_4", "s_12", "s_16"],
        "hpt": ["s_9", "s_14", "s_20"],
        "lpt_bypass": ["s_15", "s_17", "s_21"]
    }

    def __init__(self, engine_id: int):
        self.engine_id = engine_id
        self.current_cycle = 0
        self.health_index = 1.0
        self.health_status = "HEALTHY"
        self.predicted_rul = 125.0
        self.rul_interval = [105.0, 145.0]
        self.anomaly_score = 0.0
        self.active_work_order: Optional[Dict[str, Any]] = None
        self.subsystem_health = {
            "fan_inlet": 1.0,
            "lpc": 1.0,
            "hpc": 1.0,
            "combustor": 1.0,
            "hpt": 1.0,
            "lpt_bypass": 1.0
        }
        self.history: List[Dict[str, Any]] = []

    def update_cycle_telemetry(
        self,
        cycle: int,
        sensor_values: Dict[str, float],
        prediction_payload: Dict[str, Any],
        active_wo: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingests a new cycle event, recalculates subsystem health, and synchronizes twin state.
        """
        self.current_cycle = int(cycle)
        self.health_index = float(prediction_payload.get("health_index", 1.0))
        self.health_status = str(prediction_payload.get("health_status", "HEALTHY"))
        self.predicted_rul = float(prediction_payload.get("predicted_rul", 125.0))
        self.rul_interval = [
            float(prediction_payload.get("rul_lower", 105.0)),
            float(prediction_payload.get("rul_upper", 145.0))
        ]
        self.anomaly_score = float(prediction_payload.get("anomaly_score", 0.0))
        self.active_work_order = active_wo

        # Compute Subsystem Degradation Indices
        fingerprint = prediction_payload.get("degradation_fingerprint", {})
        dev_vec = fingerprint.get("deviation_vector", {})

        for sub, sensors in self.SUBSYSTEM_SENSORS.items():
            devs = [abs(dev_vec.get(s, 0.0)) for s in sensors if s in dev_vec]
            mean_dev = float(np.mean(devs)) if devs else 0.0
            # Convert deviation z-score into subsystem health percentage (0.0 to 1.0)
            sub_hi = float(np.clip(1.0 - (mean_dev / 5.0), 0.05, 1.0))
            self.subsystem_health[sub] = round(sub_hi, 3)

        twin_snapshot = self.get_state_snapshot()
        self.history.append(twin_snapshot)
        if len(self.history) > 100:
            self.history.pop(0)

        return twin_snapshot

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Returns structured JSON representation of the Digital Twin state."""
        return {
            "engine_id": self.engine_id,
            "cycle": self.current_cycle,
            "health_index": round(self.health_index, 3),
            "health_status": self.health_status,
            "predicted_rul": round(self.predicted_rul, 1),
            "prediction_interval": [round(self.rul_interval[0], 1), round(self.rul_interval[1], 1)],
            "interval_width": round(self.rul_interval[1] - self.rul_interval[0], 1),
            "anomaly_score": round(self.anomaly_score, 3),
            "subsystems": self.subsystem_health,
            "active_work_order": self.active_work_order,
            "fidelity_declaration": {
                "type": "AI-Driven Empirical Digital Twin",
                "physical_basis": "Empirical run-to-failure multivariate degradation modeling",
                "disclaimer": "Virtual twin reflects data-driven state estimations from simulated streaming; not a direct fluid-dynamics PDE solver."
            }
        }
