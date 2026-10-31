"""
AURA-M FastAPI Backend REST Service (Part 6)
Provides secure, high-performance industrial APIs for telemetry ingestion,
RUL prognostics, autonomous triage, work order management, fleet ranking, and MLOps.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Path as FPath, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database.db import DatabaseManager
from src.agent.triage import AutonomousTriageAgent
from src.workorders.manager import WorkOrderManager
from src.fleet.prioritizer import FleetRiskPrioritizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.API")

app = FastAPI(
    title="AURA-M Industrial Reliability Intelligence Platform API",
    description="Autonomous, Uncertainty-Aware Reliability Intelligence for Connected Industrial Assets.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabaseManager()
triage_agent = AutonomousTriageAgent()
fleet_prioritizer = FleetRiskPrioritizer()

# ----------------- Pydantic Models -----------------

class WorkOrderStatusUpdate(BaseModel):
    new_status: str = Field(..., description="Target status: OPEN, IN_PROGRESS, CLOSED, ESCALATED")
    closing_notes: Optional[str] = Field(None, description="Resolution notes or technician remarks")
    operator: str = Field("OPERATOR", description="Name or ID of user acknowledging/closing")

class TriageEvaluationRequest(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    rul_lower: float
    rul_upper: float
    interval_width: float
    anomaly_score: float
    health_index: float
    health_status: str = "HEALTHY"
    human_review_required: bool = False
    is_anomaly: bool = False

# ----------------- Endpoints -----------------

@app.get("/", tags=["System Health"])
def root():
    """AURA-M API root — returns platform identity and navigation links."""
    return {
        "platform": "AURA-M",
        "full_name": "Autonomous, Uncertainty-Aware Reliability Intelligence",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "fleet_engines": "/api/v1/engines",
            "fleet_risk_ranking": "/api/v1/fleet/risk-ranking",
            "triage_evaluate": "/api/v1/triage/evaluate",
            "work_orders": "/api/v1/work-orders",
            "models": "/api/v1/models",
            "audit_logs": "/api/v1/audit-logs"
        }
    }

@app.get("/health", tags=["System Health"])
def get_system_health():
    """Returns real-time service health, database status, and model availability."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            engine_count = cursor.execute("SELECT count(*) FROM engines;").fetchone()[0]
            wo_count = cursor.execute("SELECT count(*) FROM work_orders WHERE status != 'CLOSED';").fetchone()[0]
            prod_model = cursor.execute("SELECT version, model_type, rmse FROM model_versions WHERE status = 'PRODUCTION' LIMIT 1;").fetchone()
            
        return {
            "status": "HEALTHY",
            "database": "CONNECTED",
            "registered_engines": engine_count,
            "open_work_orders": wo_count,
            "production_model": dict(prod_model) if prod_model else "UNINITIALIZED",
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failure: {e}")
        return {"status": "DEGRADED", "error": str(e)}

@app.get("/api/v1/engines", tags=["Fleet Management"])
def list_engines(status: Optional[str] = Query(None, description="Filter by health status")):
    """Retrieves all registered industrial engines in the fleet."""
    engines = db.list_engines()
    if status:
        engines = [e for e in engines if e.get("current_health_status") == status.upper()]
    return {"total": len(engines), "engines": engines}

@app.get("/api/v1/engines/{engine_id}", tags=["Fleet Management"])
def get_engine_digital_passport(engine_id: int = FPath(..., description="Engine ID (1-100)")):
    """Retrieves complete Evidence-Backed Digital Maintenance Passport for an engine."""
    engine = db.get_engine(engine_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine #{engine_id} not found.")

    with db.get_connection() as conn:
        latest_pred = conn.execute("""
        SELECT * FROM predictions WHERE engine_id = ? ORDER BY cycle DESC LIMIT 1;
        """, (engine_id,)).fetchone()

        latest_fp = conn.execute("""
        SELECT * FROM degradation_fingerprints WHERE engine_id = ? ORDER BY cycle DESC LIMIT 1;
        """, (engine_id,)).fetchone()

        work_orders = conn.execute("""
        SELECT * FROM work_orders WHERE engine_id = ? ORDER BY created_at DESC;
        """, (engine_id,)).fetchall()

    fp_dict = dict(latest_fp) if latest_fp else {}
    if fp_dict.get("dominant_sensors_json"):
        fp_dict["dominant_sensors"] = json.loads(fp_dict["dominant_sensors_json"])
    if fp_dict.get("deviation_vector_json"):
        fp_dict["deviation_vector"] = json.loads(fp_dict["deviation_vector_json"])

    return {
        "engine": engine,
        "latest_prediction": dict(latest_pred) if latest_pred else None,
        "degradation_fingerprint": fp_dict,
        "work_orders": [dict(w) for w in work_orders]
    }

@app.post("/api/v1/triage/evaluate", tags=["Autonomous Triage"])
def evaluate_triage_decision(request: TriageEvaluationRequest):
    """Executes deterministic rule-based triage evaluation on prognostic evidence."""
    active_wo = db.get_active_work_order(request.engine_id)
    decision = triage_agent.evaluate(request.model_dump(), active_work_order=active_wo)
    
    # Auto-generate work order if decision mandates and no duplicate exists
    created_wo = None
    if decision.get("should_create_work_order"):
        wo_record = WorkOrderManager.build_work_order(request.model_dump(), decision)
        db.insert_work_order(wo_record)
        created_wo = wo_record

    return {
        "decision": decision,
        "work_order_created": created_wo is not None,
        "work_order": created_wo
    }

@app.get("/api/v1/fleet/risk-ranking", tags=["Fleet Management"])
def get_fleet_risk_ranking():
    """Ranks all engines in the fleet by transparent multi-factor risk score."""
    engines = db.list_engines()
    telemetries = []
    for eng in engines:
        telemetries.append({
            "engine_id": eng["engine_id"],
            "cycle": eng.get("last_cycle_observed", 1),
            "predicted_rul": eng.get("current_rul", 125.0),
            "health_index": 0.85 if eng.get("current_health_status") == "HEALTHY" else (0.4 if eng.get("current_health_status") == "MODERATE_WEAR" else 0.15),
            "anomaly_score": eng.get("anomaly_score", 0.0),
            "interval_width": 24.0
        })
    ranked_df = fleet_prioritizer.rank_fleet(telemetries)
    return {"total": len(ranked_df), "rankings": ranked_df.to_dict(orient="records")}

@app.get("/api/v1/work-orders", tags=["Work Orders"])
def list_work_orders(status: Optional[str] = Query(None, description="OPEN, IN_PROGRESS, CLOSED, ESCALATED")):
    """Lists maintenance work orders with optional status filter."""
    orders = db.list_work_orders(status=status)
    return {"total": len(orders), "work_orders": orders}

@app.patch("/api/v1/work-orders/{work_order_id}/status", tags=["Work Orders"])
def update_work_order_status(
    work_order_id: str,
    update: WorkOrderStatusUpdate
):
    """Updates work order lifecycle status (e.g. acknowledge, mark in-progress, or close)."""
    valid_statuses = ["OPEN", "IN_PROGRESS", "CLOSED", "ESCALATED"]
    if update.new_status.upper() not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")

    success = db.update_work_order_status(
        work_order_id=work_order_id,
        new_status=update.new_status.upper(),
        notes=update.closing_notes,
        operator=update.operator
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Work order #{work_order_id} not found.")

    return {"status": "SUCCESS", "work_order_id": work_order_id, "new_status": update.new_status.upper()}

@app.get("/api/v1/models", tags=["MLOps"])
def list_models():
    """Lists registered model versions and deployment statuses."""
    with db.get_connection() as conn:
        rows = conn.execute("SELECT * FROM model_versions ORDER BY registered_at DESC;").fetchall()
        return {"models": [dict(r) for r in rows]}

@app.get("/api/v1/audit-logs", tags=["System Health"])
def get_audit_logs(limit: int = 50):
    """Retrieves audit trail of system events, work orders, and triage decisions."""
    logs = db.get_audit_logs(limit=limit)
    return {"total": len(logs), "audit_logs": logs}
