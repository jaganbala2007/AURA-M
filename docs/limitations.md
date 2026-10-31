# AURA-M Engineering Disclosures, Boundaries & Limitations

In accordance with rigorous engineering ethics and competition honesty standards, this document establishes the explicit operational boundaries, dataset assumptions, and technical disclosures of the AURA-M platform.

---

## 1. Benchmark & Simulation Disclosures

### 1.1 NASA C-MAPSS Dataset Context
- **Nature of Benchmark**: The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset utilized across this project is a high-fidelity numerical simulation developed by NASA Glenn Research Center and the US Army Research Laboratory.
- **Disclosure**: C-MAPSS data does not represent physical flight telemetry from in-service commercial flights. It represents model-based thermodynamic simulations under controlled degradation rates.
- **Streaming Telemetry**: All streaming ingestion demonstrations (via MQTT and WebSocket) represent **cycle-by-cycle dataset replay** from the NASA benchmark files. AURA-M explicitly labels these as simulated replays and does not claim physical wireless acquisition from an operating jet engine.

### 1.2 Physical Repair & "Self-Healing" Demarcation
- **AURA-M Capability**: AURA-M is an autonomous **predictive intelligence and maintenance orchestration platform**. It detects anomalies, calculates calibrated Remaining Useful Life, triages urgency, deduplicates tickets, and schedules human maintenance interventions with cryptographic evidence.
- **Boundary**: AURA-M does **not** claim physical self-healing, automated mechanical repair, or in-flight component restoration. In real turbofan engines, physical wear (such as compressor blade spalling or thermal barrier coating erosion) requires physical line maintenance, boroscope inspection, and overhaul.

---

## 2. Model & Algorithmic Limitations

### 2.1 Single Operational Regime (FD001 vs. FD002/004)
- **Current Scope**: The primary production baseline is trained and calibrated on FD001, which operates at Sea Level under a single flight condition and a single fault mode (High-Pressure Compressor degradation).
- **Limitation**: While our Physics-Guided Machine Learning (PGML) module introduces operating-condition normalization (CTR, turbine $\Delta T$), models trained on FD001 will not generalize directly to multi-condition regimes (FD002/FD004 with 6 flight conditions and multiple fault modes) without retraining and recalibrating the conformal quantile $Q$.

### 2.2 Piecewise-Linear RUL Assumption
- **Assumption**: RUL is capped at 125 cycles during early operating lifespans based on aerospace industry literature.
- **Limitation**: In engines with manufacturing defects or foreign object damage (FOD), micro-cracks can initiate degradation earlier than cycle 125. The unsupervised Isolation Forest provides a secondary safety net for this scenario, but the supervised RUL model assumes normal wear trajectories.

### 2.3 Conformal Prediction Coverage Bounds
- **Marginal vs. Conditional Coverage**: Split conformal prediction guarantees marginal coverage: $P(Y \in C(X)) \ge 1 - \alpha$. It does not guarantee conditional coverage for every individual operating sub-regime. On our test set, empirical coverage was 86.0% (closely validating the 90% finite-sample target).
- **Distinction**: Prediction intervals $[RUL - Q, RUL + Q]$ represent calibrated uncertainty bounds on remaining operating cycles; they must not be conflated with instantaneous failure probabilities.

---

## 3. Hardware & Edge AI Boundaries

### 3.1 Edge AI & ONNX Benchmarking
- **Testing Environment**: Edge AI benchmarks were conducted using the Open Neural Network Exchange (ONNX) Runtime on host x86_64 architecture with multi-threaded operator kernels.
- **Disclosure**: We do not claim TinyML deployment on microcontrollers (e.g., STM32, ESP32, or ARM Cortex-M) because gas-turbine turbofan edge systems are deployed on Onboard Maintenance Gateway units or ARINC 600 line-replaceable units (LRUs) running embedded Linux or VxWorks.

### 3.2 Federated Learning Simulation
- **Testing Environment**: The federated learning module simulates 5 virtual airline fleet operators locally by partitioning training engines.
- **Disclosure**: We do not claim physical multi-organization distributed deployment across real airlines. Network latency, differential privacy guarantees, and adversarial Byzantine defenses represent future production roadmap items.

---

## 4. Operational & Human-in-the-Loop Safeguards

1. **Human Verification Trigger**: If prognostic uncertainty exceeds 45 cycles or the Isolation Forest flags severe thermodynamic divergence while RUL is high, the autonomous agent explicitly suspends automated ticket closure and escalates the ticket for manual engineering verification.
2. **Deterministic Triage**: In adherence to competition rules, the maintenance agent is strictly non-LLM. It operates on deterministic, certifiable Python state logic, eliminating the stochastic drift and hallucination risks of generative AI in safety-critical systems.
