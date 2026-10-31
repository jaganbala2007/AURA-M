"""
AURA-M Competition PowerPoint Generator (Round 1 Design Deck)
Generates a stunning, professional 14-slide engineering presentation for TELIPORT S4
following industrial corporate styling, high contrast, concise technical hierarchy,
and competition jury alignment.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path: str = "presentation/AURA-M_Round1.pptx"):
    prs = Presentation()
    # Set 16:9 widescreen slides
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Deep Slate Navy, Cyan Accent, Electric Teal, Coral Red, Dark Card
    BG_COLOR = RGBColor(11, 17, 32)        # #0B1120
    CARD_BG = RGBColor(30, 41, 59)         # #1E293B
    BORDER_COLOR = RGBColor(51, 65, 85)    # #334155
    CYAN_ACCENT = RGBColor(14, 165, 233)   # #0EA5E9
    TEAL_ACCENT = RGBColor(20, 184, 166)   # #14B8A6
    TEXT_WHITE = RGBColor(248, 250, 252)   # #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8
    TEXT_DIM = RGBColor(100, 116, 139)     # #64748B
    GREEN_ACCENT = RGBColor(16, 185, 129)  # #10B981
    CORAL_ACCENT = RGBColor(239, 68, 68)   # #EF4444

    def add_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="TELIPORT S4 | CASE STUDY 1: PREDICTIVE MAINTENANCE"):
        # Header banner text
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.1))
        tf = txBox.text_frame
        tf.word_wrap = True
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = CYAN_ACCENT

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(4)

    def add_card(slide, left, top, width, height, title, content_items, border_color=BORDER_COLOR):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = CYAN_ACCENT

        for item in content_items:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_WHITE
            p.space_before = Pt(6)

    # -------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    add_slide_background(s1)

    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "TELIPORT SEASON 4 — THINK. BUILD. CREATE. WIN. | CASE STUDY 1"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = CYAN_ACCENT

    p1 = tf1.add_paragraph()
    p1.text = "AURA-M"
    p1.font.size = Pt(56)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.space_before = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "Autonomous, Uncertainty-Aware Reliability Intelligence"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = TEAL_ACCENT
    p2.space_before = Pt(4)

    p3 = tf1.add_paragraph()
    p3.text = '"From Machine Degradation to Autonomous, Evidence-Backed Maintenance."'
    p3.font.size = Pt(15)
    p3.font.italic = True
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(12)

    p4 = tf1.add_paragraph()
    p4.text = "Team Identity: Engineering Team Finalist | Technology Stack: Python 3.11, Scikit-learn, XGBoost, ONNX, MLflow, FastAPI"
    p4.font.size = Pt(11)
    p4.font.color.rgb = TEXT_DIM
    p4.space_before = Pt(36)

    # -------------------------------------------------------------------------
    # SLIDE 2: THE PROBLEM
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_background(s2)
    add_header(s2, "Industrial Fleet Challenge: Reactive Failures & Fixed Intervals")

    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "1. Unplanned Downtime", [
        "Catastrophic failures in connected turbofans and industrial compressors cost millions per hour.",
        "Secondary damage propagates when minor component wear is missed during routine operation.",
        "Today, over 65% of critical industrial failures occur between scheduled maintenance intervals."
    ])
    add_card(s2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "2. Fixed-Interval Limitations", [
        "Traditional preventive maintenance relies on arbitrary hour/cycle thresholds.",
        "Premature servicing discards healthy components with 30-40% useful life remaining.",
        "Delayed servicing allows uncontained mechanical failure when operating conditions are harsh."
    ])
    add_card(s2, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "3. The Reliability Gap", [
        "Black-box AI models output point estimates without calibrated confidence bounds.",
        "Maintenance engineers reject unexplainable predictions without physical evidence.",
        "Lack of closed-loop automation forces manual triage, causing response bottlenecks."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 3: OUR SOLUTION
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_background(s3)
    add_header(s3, "AURA-M: Closed-Loop Industrial Reliability Platform")

    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "The Closed-Loop Intelligence Architecture", [
        "Step 1: Ingest multi-channel sensor telemetry (21 sensors + 3 settings).",
        "Step 2: Degradation Fingerprint Engine extracts baseline drift and Health Index (HI).",
        "Step 3: Dual ML Models compute progressive anomaly scores and predict Remaining Useful Life (RUL).",
        "Step 4: Conformal Prediction Engine wraps predictions in statistically calibrated 90% confidence bounds.",
        "Step 5: Non-LLM Autonomous Triage Agent evaluates policy rules and prevents duplicate work orders.",
        "Step 6: Digital Maintenance Passport generates cryptographically signed, explainable work orders."
    ])
    add_card(s3, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Core Differentiators from Standard Predictive Maintenance", [
        "Uncertainty-Aware: Never treats every RUL point prediction as equally reliable.",
        "Non-LLM Deterministic Agent: Fast, lightweight, explainable, and zero-hallucination rule execution.",
        "Evidence-Backed Passports: Every maintenance dispatch includes leading deviating sensor signatures.",
        "Zero Data Leakage: Preprocessing and conformal calibration strictly partitioned across engine boundaries.",
        "Edge-Optimized: ONNX Runtime export delivers 1,246x microsecond inference speedup."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 4: CORE INNOVATIONS
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_background(s4)
    add_header(s4, "Six Interconnected Engineering Innovations")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(3.6), Inches(2.4), "1. Degradation Fingerprint", [
        "Tracks multi-channel deviations from pristine baseline.",
        "Computes continuous composite Health Index [1.0 -> 0.0]."
    ])
    add_card(s4, Inches(4.8), Inches(1.8), Inches(3.6), Inches(2.4), "2. Dual Prediction Engine", [
        "Isolation Forest anomaly detector for abnormal patterns.",
        "Random Forest & XGBoost prognostic RUL regressors."
    ])
    add_card(s4, Inches(8.8), Inches(1.8), Inches(3.6), Inches(2.4), "3. Conformal Uncertainty", [
        "Split conformal calibrated prediction intervals.",
        "Guarantees 90% finite-sample empirical coverage."
    ])
    add_card(s4, Inches(0.8), Inches(4.4), Inches(3.6), Inches(2.4), "4. Autonomous Triage Agent", [
        "Deterministic, explainable Python decision engine.",
        "Prevents duplicate work orders and routes high uncertainty."
    ])
    add_card(s4, Inches(4.8), Inches(4.4), Inches(3.6), Inches(2.4), "5. Digital Maintenance Passport", [
        "Engine asset record with complete operational history.",
        "Cryptographically signed evidence report for every dispatch."
    ])
    add_card(s4, Inches(8.8), Inches(4.4), Inches(3.6), Inches(2.4), "6. Fleet Risk Prioritization", [
        "Multi-factor transparent risk ranking (RUL, wear, anomaly, uncertainty).",
        "Ranks 100 connected engines for targeted maintenance dispatch."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 5: SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_background(s5)
    add_header(s5, "End-to-End System Architecture & Data Flow")

    add_card(s5, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "Data & Feature Layer", [
        "NASA C-MAPSS FD001 Telemetry",
        "DataValidator (Types, Nulls, Monotonicity)",
        "Leakage-Free MinMaxScaler",
        "Temporal Feature Engineering (Rolling windows 5, 15)",
        "Rate of Change & Baseline Drift"
    ])
    add_card(s5, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "Intelligence & Uncertainty Layer", [
        "Isolation Forest Anomaly Scoring",
        "Random Forest Baseline & XGBoost Regressor",
        "Conformal Residual Quantile Calibration",
        "SHAP Explainability Studio",
        "Physics-Guided Thermodynamic Checks",
        "Edge AI / ONNX Runtime Gateway"
    ])
    add_card(s5, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "Action & Governance Layer", [
        "Deterministic Autonomous Triage Agent",
        "Duplicate Work-Order Prevention Guard",
        "HMAC-SHA256 Signed Evidence Reports",
        "SQLite Relational Database & Audit Logs",
        "FastAPI REST Endpoints",
        "15-Module Streamlit Operations Center",
        "MLflow Tracking & Retraining Gatekeeper"
    ])

    # -------------------------------------------------------------------------
    # SLIDE 6: DATA ENGINEERING
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_background(s6)
    add_header(s6, "Rigorous Data Engineering & Leakage Prevention")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "C-MAPSS Ingestion & Signal Quality", [
        "Dataset Structure: 100 Run-to-Failure engines, 20,631 cycles (FD001).",
        "Constant Channel Pruning: Dropped near-zero variance sensors (s_1, s_5, s_10, s_16, s_18, s_19) and setting_3.",
        "15 Informative Sensors: LPC/HPC/LPT temperatures, static pressures, fan/core speeds, bypass ratio, bleed enthalpy.",
        "Piecewise-Linear Target Capping: Clipped RUL target at 125 cycles reflecting physical plateau before initial wear onset."
    ])
    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Critical Leakage Prevention Protocols", [
        "Strict Engine-Level Splitting: Scaler and normalizers fit exclusively on training engines (80 engines).",
        "Zero Cross-Engine Contamination: Rolling window features computed strictly within individual engine boundaries.",
        "Chronological Integrity: Preserves time-series monotonicity and cycle progression.",
        "Validation Suite: Automated checks detect nulls, NaNs, infinities, and timestamp inversions."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 7: MACHINE LEARNING & BENCHMARK
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_background(s7)
    add_header(s7, "Prognostic Machine Learning & Official NASA Benchmark Results")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Official NASA FD001 Test Set Evaluation", [
        "Random Forest Baseline:",
        "  • RMSE: 14.40 operating cycles",
        "  • MAE: 10.70 operating cycles",
        "  • R² Score: 0.880",
        "  • NASA Prognostic Score: 303.49",
        "XGBoost Advanced Model:",
        "  • RMSE: 14.81 operating cycles",
        "  • MAE: 10.96 operating cycles",
        "  • NASA Prognostic Score: 321.51",
        "Genuine Measured Metrics: Evaluated on all 100 unseen test engines with official RUL ground truth."
    ])
    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Metric Justifications & Asymmetric Scoring", [
        "RMSE & MAE: Measure average cycle error in remaining operational life.",
        "Official NASA Asymmetric Scoring Function:",
        "  • Overestimation (late prediction) is far more hazardous than underestimation (early maintenance).",
        "  • Early prediction penalized with exp(-d/13) - 1.",
        "  • Late prediction penalized exponentially harder with exp(d/10) - 1.",
        "Anomaly Detection: Calibrated Isolation Forest generates normalized [0, 1] anomaly scores."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 8: CONFIDENCE-AWARE INTELLIGENCE
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_background(s8)
    add_header(s8, "Confidence-Aware Prognostics via Conformal Prediction")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Split Conformal Prediction Framework", [
        "Theoretical Grounding: Provides distribution-free, finite-sample coverage guarantees.",
        "Non-Conformity Measure: R_i = |y_true - y_pred| calibrated on holdout validation engines.",
        "Target Coverage: 1 - alpha = 90% (alpha = 0.10).",
        "Calibrated Conformal Quantile: Q = 19.20 operating cycles.",
        "Guaranteed Prediction Interval: [max(0, y_pred - 19.2), y_pred + 19.2] cycles."
    ])
    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Operational Uncertainty Routing", [
        "Distinction: Prediction Uncertainty != Anomaly Score != Failure Probability.",
        "Epistemic Uncertainty Tracking: Measures interval width in operating cycles.",
        "Excessive Uncertainty Policy: If interval width > 45 cycles, triage agent suspends automated dispatch.",
        "Human Verification Routing: Flagged assets sent to certified reliability engineers for manual inspection."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 9: AUTONOMOUS TRIAGE AGENT
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_background(s9)
    add_header(s9, "Deterministic, Non-LLM Autonomous Triage Agent")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Policy Rules & Thresholds", [
        "Rule 0 (Uncertainty): Interval width > 45 cyc -> ROUTE_HUMAN_REVIEW.",
        "Rule 1 (Critical): RUL < 20 cyc or HI < 0.20 -> CRITICAL_MAINTENANCE.",
        "Rule 2 (High Priority): 20 <= RUL < 50 cyc or Anomaly >= 0.65 -> HIGH_PRIORITY.",
        "Rule 3 (Inspection): 50 <= RUL <= 100 cyc or Anomaly >= 0.50 -> SCHEDULE_INSPECTION.",
        "Rule 4 (Normal): RUL > 100 cyc -> NORMAL_MONITORING.",
        "Configurable: Externalized in config/thresholds.yaml for operational agility."
    ])
    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Enterprise Guardrails & Deduplication", [
        "Duplicate Prevention: Checks active unresolved orders; suppresses duplicate dispatches.",
        "Automatic Escalation: Elevates priority if condition worsens while inspection is pending.",
        "Explanatory Rationale: Records exact mathematical condition that triggered each rule.",
        "Audit Trail: All actions logged to database with operator timestamps."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 10: DIGITAL MAINTENANCE PASSPORT
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_background(s10)
    add_header(s10, "Evidence-Backed Digital Maintenance Passport & Trust")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Complete Asset Digital Record", [
        "Engine ID & Fleet Registry Status.",
        "Current Condition & Composite Health Index.",
        "Predicted RUL with 90% Conformal Interval.",
        "Multi-sensor degradation fingerprint with dominant deviating sensors.",
        "Historical sensor curves & historical prediction trajectory.",
        "Full lifecycle work order history."
    ])
    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Cryptographic Trust & Evidence Reports", [
        "Evidence Report Attachment: Required for every generated work order.",
        "HMAC-SHA256 Digital Signature: Guarantees tamper-proof evidence integrity.",
        "Non-Repudiation: Proves sensor evidence was recorded before maintenance dispatch.",
        "Airworthiness Alignment: Meets strict regulatory standards for automated maintenance records."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 11: MLOPS & RETRAINING
    # -------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_background(s11)
    add_header(s11, "MLOps Architecture & Controlled Retraining Gates")

    add_card(s11, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Experiment & Model Tracking", [
        "MLflow Integration: Tracks hyperparameters, features, metrics, and model artifacts.",
        "Artifact Versioning: Preprocessor, feature configuration, and conformal quantiles serialized together.",
        "Model Registry: Explicit lifecycle stages (STAGING, PRODUCTION, ARCHIVED, REJECTED).",
        "Audit Logging: All retraining events recorded in relational audit log."
    ])
    add_card(s11, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Controlled Model Promotion Gate", [
        "Automated Retraining Flow: Ingest batch -> Validate data -> Train candidate -> Evaluate on test benchmark.",
        "Promotion Gatekeeper Rule: Candidate RMSE must strictly outperform currently deployed production model.",
        "Safe Rejection: If candidate fails gate, model is rejected and production model is retained.",
        "No Uncontrolled Drift: Retraining never replaces models silently in production."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 12: NEXT-GEN TECH INTEGRATION
    # -------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_background(s12)
    add_header(s12, "Next-Generation Technology Integration")

    add_card(s12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Edge AI, TinyML & Digital Twins", [
        "Edge AI / ONNX Runtime Benchmark:",
        "  • Size: 37.0 MB (Pickle) -> 20.0 MB (ONNX) (1.84x reduction)",
        "  • Latency: 58,290 µs -> 46.8 µs (1,246x faster inference)",
        "  • Mathematical Parity: Exactly 0.00000 cycles discrepancy",
        "AI Digital Twin: Synchronized virtual state modeling 6 turbofan sections (Fan, LPC, HPC, Combustor, HPT, LPT)."
    ])
    add_card(s12, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "PGML, Explainability & Drift Monitoring", [
        "Physics-Guided ML (PGML): Domain features (CTR, turbine Delta T, core loading) + thermodynamic violation analysis.",
        "Explainable AI (XAI): SHAP TreeExplainer local waterfall & global feature ranking.",
        "Concept Drift Monitor: Statistical Kolmogorov-Smirnov test & Wasserstein distance.",
        "Federated Learning Simulation: FedAvg multi-client simulation preserving client data privacy."
    ])

    # -------------------------------------------------------------------------
    # SLIDE 13: COMPETITION ALIGNMENT MATRIX
    # -------------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_background(s13)
    add_header(s13, "TELIPORT S4 Evaluation Criteria Alignment Matrix")

    # Table of Evaluation Alignment
    rows = 8
    cols = 4
    left = Inches(0.8)
    top = Inches(1.8)
    width = Inches(11.7)
    height = Inches(4.8)

    table_shape = s13.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Evaluation Criterion", "Weight", "AURA-M Implementation", "Demonstrated Proof Point"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = CYAN_ACCENT

    data = [
        ("Problem Understanding", "15%", "Closed-loop predictive maintenance without false repair claims", "Clear physical distinction: prognostics & work order dispatch"),
        ("Architecture Clarity", "20%", "Multi-tier architecture: Ingestion -> ML -> Agent -> Passport -> API", "Modular src/ packages, clean REST APIs, comprehensive docs"),
        ("Data Engineering", "15%", "Strict engine-level split, constant sensor elimination, rolling features", "Zero data leakage, 105 engineered features, DataValidator"),
        ("ML Approach & Reasoning", "20%", "Dual models (Isolation Forest + RF baseline & XGBoost) + Conformal", "Genuine RMSE 14.40, MAE 10.70, NASA Score 303.49, 90% coverage"),
        ("MLOps & Agent Logic", "15%", "MLflow experiment tracking, controlled retraining gate, non-LLM agent", "MLflow database, promotion gatekeeper, work order deduplication"),
        ("Dataset Coverage", "10%", "NASA C-MAPSS FD001 primary benchmark; architecture ready for FD002-4", "100 training engines (20,631 cyc) + 100 test engines (13,096 cyc)"),
        ("Communication Quality", "5%", "15-module industrial dashboard, 14-slide deck, presenter script", "Consistent slate-cyan design system, zero placeholder claims")
    ]

    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_COLOR
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 14: ROADMAP, LIMITATIONS & CONCLUSION
    # -------------------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_background(s14)
    add_header(s14, "Roadmap, Honest Limitations & Conclusion")

    add_card(s14, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.2), "Honest Engineering Disclosures & Roadmap", [
        "Benchmark Fidelity: NASA C-MAPSS is simulated turbofan telemetry; physical deployment requires real sensor calibration.",
        "Scheduling Constraints: Fleet risk ranking prioritizes assets by risk, but does not solve technician/hangar combinatorial optimization.",
        "Near-Term Roadmap: Expand to multi-condition FD002 and dual-fault FD004; integrate real MQTT edge gateway hardware.",
        "Long-Term Research: Hardware-in-the-loop (HIL) bench validation and full thermodynamic PDE digital twin simulation."
    ])
    add_card(s14, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.2), "Competition Summary & Value Proposition", [
        "Autonomous Closed-Loop: Runs end-to-end without manual intervention once data arrives.",
        "Uncertainty-Aware: Protects against silent failures using calibrated conformal bounds.",
        "Edge-Capable: 46.8 µs ONNX runtime enables direct deployment on edge gateways.",
        "Trustworthy & Explainable: Cryptographically signed work orders with SHAP feature attributions.",
        "Zero LLM Hallucinations: 100% deterministic, auditable Python triage rules."
    ])

    closing_box = s14.shapes.add_textbox(Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.8))
    tf_c = closing_box.text_frame
    p_c = tf_c.paragraphs[0]
    p_c.alignment = PP_ALIGN.CENTER
    p_c.text = '"AURA-M does not merely predict machine failure. It converts machine degradation evidence into intelligent, explainable maintenance action."'
    p_c.font.size = Pt(13)
    p_c.font.bold = True
    p_c.font.italic = True
    p_c.font.color.rgb = CYAN_ACCENT

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    build_presentation()
