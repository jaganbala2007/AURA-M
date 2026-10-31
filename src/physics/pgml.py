"""
AURA-M Physics-Guided Machine Learning (PGML) Engine (Technology Module 3)
Extracts aerothermodynamic cycle features, enforces physical monotonicity constraints,
detects thermodynamic law violations, and benchmarks PGML against pure data-driven models.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("AURA-M.PGML")

class PhysicsGuidedEngine:
    """Computes aerothermodynamic turbofan features and detects physical boundary violations."""

    @classmethod
    def compute_physics_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates physically motivated thermodynamic relations based on gas turbine principles:
        1. Compressor Temperature Ratio (CTR): T30 / T24 (s_3 / s_2)
        2. Turbine Temperature Differential (Delta T_turb): T30 - T50 (s_3 - s_4)
        3. Core Aerodynamic Loading: Ps30 / Nc (s_11 / s_9)
        4. Bypass Ratio to Core Speed: BPR / Nc (s_15 / s_9)
        5. Combustor Enthalpy Parameter: (s_12 * s_11) / (s_17 + 1e-4)
        6. Turbine Coolant Bleed Ratio: W31 / (W32 + 1e-4) (s_20 / s_21)
        """
        df_out = df.copy()

        # 1. Compressor Temperature Ratio (T30 / T24)
        df_out["pg_ctr"] = df_out["s_3"] / (df_out["s_2"] + 1e-6)

        # 2. Turbine Temperature Drop (Expansion across HPT/LPT)
        df_out["pg_delta_t_turb"] = df_out["s_3"] - df_out["s_4"]

        # 3. Core Loading Ratio (Ps30 / Nc)
        df_out["pg_core_loading"] = df_out["s_11"] / (df_out["s_9"] + 1e-6)

        # 4. Bypass to Fan Speed Ratio
        df_out["pg_bypass_fan_ratio"] = df_out["s_15"] / (df_out["s_8"] + 1e-6)

        # 5. Combustor Fuel-Enthalpy Index
        df_out["pg_fuel_enthalpy_index"] = (df_out["s_12"] * df_out["s_11"]) / (df_out["s_17"] + 1e-4)

        # 6. Coolant Bleed Ratio (HPT to LPT bleed flow)
        df_out["pg_coolant_ratio"] = df_out["s_20"] / (df_out["s_21"] + 1e-4)

        return df_out

    @classmethod
    def check_physics_violations(cls, row: pd.Series) -> List[str]:
        """
        Evaluates physical consistency against thermodynamic laws.
        Returns list of violated physical principles.
        """
        violations = []

        # Law 1: HPC temperature T30 must exceed LPC temperature T24
        if row.get("s_3", 0.0) < row.get("s_2", 0.0):
            violations.append(
                f"Thermodynamic Inversion: HPC outlet temp T30 ({row.get('s_3'):.1f} °R) "
                f"< LPC outlet temp T24 ({row.get('s_2'):.1f} °R)"
            )

        # Law 2: HPC outlet pressure P30 must exceed bypass pressure P15
        if row.get("s_7", 0.0) < row.get("s_6", 0.0):
            violations.append(
                f"Pressure Gradient Violation: HPC outlet pressure P30 ({row.get('s_7'):.1f} psia) "
                f"< Bypass duct pressure P15 ({row.get('s_6'):.1f} psia)"
            )

        # Law 3: Non-negativity of operational speeds and ratios
        for s in ["s_8", "s_9", "s_12", "s_15"]:
            if row.get(s, 0.0) <= 0.0:
                violations.append(f"Non-physical negative or zero telemetry channel: {s} = {row.get(s)}")

        return violations

    @classmethod
    def apply_monotonic_degradation_filter(cls, raw_health_series: pd.Series) -> pd.Series:
        """
        Enforces physical monotonic wear constraint:
        Without physical maintenance replacement, cumulative wear cannot decrease over time.
        Implements cumulative minimum filter on Health Index: HI(t) <= HI(t-1).
        """
        return raw_health_series.cummin()
