# AURA-M Telemetry & Sensor Data Dictionary

## 1. NASA C-MAPSS FD001 Benchmark Schema

The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset provides cycle-by-cycle operational readings from simulated 90,000-lb thrust turbofan engines. Each record corresponds to one flight cycle.

### Primary Benchmark Subset (FD001)
- **Operational Conditions**: Sea Level (Mach 0.0, Altitude 0 ft, Sea Level Temperature).
- **Fault Mode**: High-Pressure Compressor (HPC) degradation.
- **Engines in Training Set**: 100 units (run to failure, 20,631 total cycles).
- **Engines in Test Set**: 100 units (truncated before failure, 13,096 total cycles).
- **True Ground Truth**: Provided in `RUL_FD001.txt` (100 scalar remaining cycles).

---

## 2. Sensor Channel Mapping & Informational Utility

| Sensor Index | Variable Name | Physical Description | Units | Variance Status | Retained in AURA-M | Physical Diagnostic Relevance |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **s1** | `T2` | Total temperature at fan inlet | °R | Zero Variance | Dropped | Constant ambient sea-level inlet temperature (518.67 °R). |
| **s2** | `T24` | Total temperature at LPC outlet | °R | Dynamic | **Retained** | Sensitive to booster compressor compression ratio. |
| **s3** | `T30` | Total temperature at HPC outlet | °R | Dynamic | **Retained** | Critical indicator of HPC thermal efficiency degradation. |
| **s4** | `T50` | Total temperature at LPT outlet | °R | Dynamic | **Retained** | Primary exhaust gas temperature (EGT) wear metric. |
| **s5** | `P2` | Pressure at fan inlet | psia | Zero Variance | Dropped | Constant ambient inlet pressure (14.62 psia). |
| **s6** | `P15` | Total pressure in bypass duct | psia | Zero Variance | Dropped | Uninformative under single-condition sea-level operation. |
| **s7** | `P30` | Total pressure at HPC outlet | psia | Dynamic | **Retained** | Direct measure of HPC compression degradation. |
| **s8** | `Nf` | Physical fan speed | rpm | Dynamic | **Retained** | Low-pressure spool rotational speed. |
| **s9** | `Nc` | Physical core speed | rpm | Dynamic | **Retained** | High-pressure spool rotational speed; increases as HPC fouls. |
| **s10** | `epr` | Engine pressure ratio (P50/P2) | -- | Zero Variance | Dropped | Constant baseline value in FD001. |
| **s11** | `Ps30` | Static pressure at HPC outlet | psia | Dynamic | **Retained** | Strongly correlated with compressor blade erosion. |
| **s12** | `phi` | Ratio of fuel flow to Ps30 | pps/psi | Dynamic | **Retained** | Combustor fuel scheduling efficiency indicator. |
| **s13** | `NRf` | Corrected fan speed | rpm | Dynamic | **Retained** | Thermally normalized low-pressure spool speed. |
| **s14** | `NRc` | Corrected core speed | rpm | Dynamic | **Retained** | Thermally normalized high-pressure spool speed. |
| **s15** | `BPR` | Bypass ratio | -- | Dynamic | **Retained** | Airflow bypass efficiency across the engine core. |
| **s16** | `farB` | Burner fuel-air ratio | -- | Zero Variance | Dropped | Controlled at constant stoichiometry in FD001. |
| **s17** | `htBleed` | Bleed enthalpy | -- | Dynamic | **Retained** | Thermodynamic energy extracted for aircraft systems. |
| **s18** | `Nf_dmd` | Demanded fan speed | rpm | Zero Variance | Dropped | Constant commanded value (2388 rpm). |
| **s19** | `PCNfR_dmd` | Demanded corrected fan speed | rpm | Zero Variance | Dropped | Constant commanded value (100.0 rpm). |
| **s20** | `W31` | HPT coolant bleed | lbm/s | Dynamic | **Retained** | High-Pressure Turbine cooling flow demand. |
| **s21** | `W32` | LPT coolant bleed | lbm/s | Dynamic | **Retained** | Low-Pressure Turbine cooling flow demand. |

---

## 3. Operational Settings

| Setting Index | Variable Name | Description | Range in FD001 | Retained |
| :---: | :---: | :--- | :---: | :---: |
| **op_setting_1** | Altitude / Mach proxy | Commanded flight profile throttle | [-0.0087, +0.0087] | **Retained** |
| **op_setting_2** | Ambient condition | Environmental sea-level variance | [-0.0006, +0.0006] | **Retained** |
| **op_setting_3** | Sea-level flag | Fixed reference flag (100.0) | Constant | Dropped |

---

## 4. Derived & Engineered Feature Channels

AURA-M expands the 14 retained dynamic sensors into **105 to 119 feature dimensions** via temporal feature extraction:

1. **Rolling Averages (`_roll_mean_5`, `_roll_mean_15`)**:
   - Captures low-pass smoothed trends, eliminating cycle-to-cycle flight noise.
2. **Rolling Volatility (`_roll_std_5`, `_roll_std_15`)**:
   - Captures high-frequency mechanical vibration and thermal instability preceding component failure.
3. **Rate of Change (`_roc_5`)**:
   - Measures wear acceleration: $\frac{s_k(t) - s_k(t-5)}{5}$.
4. **Baseline Drift (`_drift`)**:
   - Absolute displacement from the engine's initial healthy state: $s_k(t) - \mu_{k, base}$.
5. **Physics-Guided Domain Features**:
   - `CTR`: Compressor Temperature Ratio ($T_{30} / T_{24}$).
   - `Turbine_DeltaT`: Turbine Temperature Drop ($T_{30} - T_{50}$).
   - `Core_Load`: Ratio of physical core speed to compressor static pressure ($N_c / P_{s30}$).
   - `Monotonic_Filter`: Exponential moving average ensuring non-decreasing thermodynamic wear vectors.
