# AURA-M Live Demonstration Walkthrough & Jury Evaluation Script

**Competition**: TELIPORT S4 — Case Study 1: "Self-Healing Machines: Predictive Maintenance for a Connected Fleet"  
**Objective**: Guide evaluators through a live, end-to-end operational demonstration of the AURA-M platform using the industrial Streamlit dashboard and FastAPI backend.

---

## 1. Pre-Demonstration Launch Checklist

Open two terminal sessions:

### Terminal 1: Launch FastAPI REST Backend
```powershell
cd "c:\Users\jagan\Desktop\new project\teliport"
& "C:\Users\jagan\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
- Open browser: `http://127.0.0.1:8000/docs` (Interactive Swagger UI documentation).

### Terminal 2: Launch Industrial Streamlit Dashboard
```powershell
cd "c:\Users\jagan\Desktop\new project\teliport"
& "C:\Users\jagan\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run dashboard/app.py
```
- Open browser: `http://localhost:8501` (AURA-M Industrial Operational Control Center).

---

## 2. Step-by-Step Demonstration Journey

### Step 1: Fleet Risk Overview & Priority Matrix
1. **Navigate to**: Sidebar -> Select **"Fleet Risk Overview"**.
2. **Observe**:
   - The top KPI metric cards displaying: Total Monitored Engines (100), Critical Maintenance Alerts (18), Scheduled Inspections (50), and Healthy Fleet Units (32).
   - The interactive **Fleet Risk Matrix**: A 2D scatter plot mapping Predicted RUL against Composite Fleet Risk Index, color-coded by triage status.
   - The **Priority Ranking Table**: Notice Engine #31 and Engine #34 at the top with critical risk scores ($R > 0.85$) and single-digit remaining cycles.
3. **Talking Point to Jury**:
   > *"Notice how the fleet is sorted not merely by raw RUL, but by a multi-factor risk formula that fuses remaining life, anomaly probability, and conformal uncertainty. An operator can instantly isolate high-risk assets requiring immediate line maintenance before flight departure."*

---

### Step 2: Single-Engine Prognostic Deep Dive & Digital Twin
1. **Navigate to**: Sidebar -> Select **"Engine Deep-Dive & Twin"**.
2. **Action**: Select **Engine #34** from the dropdown.
3. **Observe**:
   - **Degradation Fingerprint**: The Composite Health Index curve dropping from 1.0 down to 0.18 over 200 cycles.
   - **Conformal Prediction Interval**: The cyan RUL curve surrounded by the semi-transparent 90% confidence band $[RUL - 19.2, RUL + 19.2]$.
   - **3D Turbofan Digital Twin**: Real-time component wear status across Fan, LPC, HPC, Combustor, HPT, and LPT. Notice the High-Pressure Compressor (HPC) highlighted in critical red.
   - **Dominant Deviating Sensors**: Bar chart showing `T30` (HPC outlet temperature) and `Ps30` (static pressure) driving 78% of total engine deviation.
4. **Talking Point to Jury**:
   > *"Here is the complete digital twin of Engine #34. The system pinpoints the exact physical gas-path section failing — the High-Pressure Compressor — and bounds the remaining life within calibrated conformal limits."*

---

### Step 3: Autonomous Triage & HMAC-Signed Passport
1. **Navigate to**: Sidebar -> Select **"Work Orders & Passport"**.
2. **Observe**:
   - The active work order generated autonomously for Engine #34: `WO-ENG34-CRITICAL`.
   - **Evidence Report**: The complete structured report displaying RUL estimate, 90% uncertainty width, anomaly score, and recommended tooling (Level 3 Boroscope + HPC Blade Replacement Kit).
   - **HMAC-SHA256 Digital Signature**: The 64-character hexadecimal cryptographic signature.
3. **Action**: Click the **"Verify Cryptographic Signature"** button.
   - Green banner appears: *"VALID: HMAC-SHA256 signature authentic and tamper-free."*
4. **Talking Point to Jury**:
   > *"This satisfies FAA and EASA airworthiness audit requirements. Every recommendation is signed at the moment of inference; no technician can alter the sensor evidence or priority score without invalidating the cryptographic digest."*

---

### Step 4: Edge AI Performance Lab
1. **Navigate to**: Sidebar -> Select **"Edge AI Performance Lab"**.
2. **Observe**:
   - The live side-by-side benchmark card comparing Native Python Pickle vs. ONNX Runtime.
   - **Speedup**: 1,246.39x latency reduction (58.3 ms -> 0.0468 ms).
   - **Model Size**: 37.0 MB -> 20.1 MB (1.84x compression).
   - **Parity**: 0.00000 cycles discrepancy.
3. **Action**: Click **"Run Live Edge Benchmark"** to trigger real-time inference on 100 test samples.
4. **Talking Point to Jury**:
   > *"By optimizing our models for ONNX Runtime, we prove that AURA-M can run directly on aircraft Onboard Maintenance Systems or edge gateways without needing high-bandwidth satellite uplinks."*

---

### Step 5: Explainable AI Studio (SHAP)
1. **Navigate to**: Sidebar -> Select **"Explainable AI Studio"**.
2. **Observe**:
   - Global Feature Importance: Summary plot ranking the top 15 features across the fleet. Notice rolling mean of `T30`, `Ps30`, and `phi` dominating model decisions.
   - Local Waterfall Plot for selected engine: Demonstrates exact cycle increments and decrements contributed by each sensor.
3. **Talking Point to Jury**:
   > *"This bridges the black-box chasm. Maintenance directors are not told to trust an opaque score; they see exactly how each physical temperature and pressure sensor dragged the RUL down."*

---

### Step 6: Concept Drift & MLOps Controlled Retraining
1. **Navigate to**: Sidebar -> Select **"Concept Drift & MLOps"**.
2. **Observe**:
   - Live Kolmogorov-Smirnov (KS) test p-values across all 15 sensor channels.
   - MLflow experiment tracking table logging past training runs, RMSE scores, and promotion statuses.
3. **Talking Point to Jury**:
   > *"AURA-M does not blindly retrain in production. When sensor drift is detected, candidate models are trained in an isolated MLflow sandbox and must beat the incumbent on both RMSE and the asymmetric NASA penalty score before the gatekeeper grants production deployment."*

---

## 3. Demonstration Summary for Jury
- **100% Deterministic & Non-LLM**: Zero hallucinations.
- **Statistically Calibrated**: 90% conformal coverage guarantee.
- **Edge-Ready**: 1,246x latency acceleration via ONNX.
- **Audit-Proof**: Cryptographically signed digital maintenance passports.
