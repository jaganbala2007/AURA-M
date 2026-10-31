"""
AURA-M Database Architecture & Persistence Layer
Implements a self-contained SQLite relational schema for engines, telemetry,
fingerprints, predictions, work orders, agent decisions, MLOps versions, and audit logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

logger = logging.getLogger("AURA-M.Database")

DEFAULT_DB_PATH = "data/auram.db"

class DatabaseManager:
    """Manages SQLite relational database lifecycle, migrations, and queries."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with dict-like row factory and foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Engines Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS engines (
                engine_id INTEGER PRIMARY KEY,
                subset TEXT DEFAULT 'FD001',
                total_cycles INTEGER DEFAULT 0,
                last_cycle_observed INTEGER DEFAULT 0,
                current_health_status TEXT DEFAULT 'HEALTHY',
                current_rul REAL DEFAULT 125.0,
                anomaly_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Sensor Readings (Telemetry Stream)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine_id INTEGER,
                cycle INTEGER,
                setting_1 REAL,
                setting_2 REAL,
                setting_3 REAL,
                s_1 REAL, s_2 REAL, s_3 REAL, s_4 REAL, s_5 REAL,
                s_6 REAL, s_7 REAL, s_8 REAL, s_9 REAL, s_10 REAL,
                s_11 REAL, s_12 REAL, s_13 REAL, s_14 REAL, s_15 REAL,
                s_16 REAL, s_17 REAL, s_18 REAL, s_19 REAL, s_20 REAL, s_21 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(engine_id, cycle),
                FOREIGN KEY(engine_id) REFERENCES engines(engine_id)
            );
            """)

            # 3. Degradation Fingerprints
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS degradation_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine_id INTEGER,
                cycle INTEGER,
                health_index REAL,
                health_status TEXT,
                degradation_rate REAL,
                dominant_sensors_json TEXT,
                deviation_vector_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(engine_id) REFERENCES engines(engine_id)
            );
            """)

            # 4. Predictions & Uncertainty
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine_id INTEGER,
                cycle INTEGER,
                rul_point REAL,
                rul_lower REAL,
                rul_upper REAL,
                interval_width REAL,
                anomaly_score REAL,
                uncertainty_category TEXT,
                human_review_required INTEGER,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(engine_id) REFERENCES engines(engine_id)
            );
            """)

            # 5. Maintenance Work Orders
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_order_id TEXT UNIQUE NOT NULL,
                engine_id INTEGER,
                cycle INTEGER,
                priority TEXT,
                reason TEXT,
                recommended_action TEXT,
                status TEXT DEFAULT 'OPEN',
                evidence_report_json TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closing_notes TEXT,
                FOREIGN KEY(engine_id) REFERENCES engines(engine_id)
            );
            """)

            # 6. Agent Triage Decisions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine_id INTEGER,
                cycle INTEGER,
                decision TEXT,
                triggered_rule TEXT,
                severity TEXT,
                rationale TEXT,
                action_code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(engine_id) REFERENCES engines(engine_id)
            );
            """)

            # 7. Model Versions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                mae REAL,
                rmse REAL,
                nasa_score REAL,
                status TEXT DEFAULT 'STAGING', -- STAGING, PRODUCTION, ARCHIVED
                artifacts_path TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 8. Retraining Runs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                dataset_subset TEXT,
                old_model_version TEXT,
                candidate_model_version TEXT,
                old_rmse REAL,
                candidate_rmse REAL,
                promoted INTEGER,
                decision_reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 9. Audit Logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details_json TEXT,
                operator TEXT DEFAULT 'AURA-M_AGENT',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Indexes for high performance querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_eng ON sensor_readings(engine_id, cycle);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_wo_eng ON work_orders(engine_id, status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_eng ON predictions(engine_id, cycle);")
            
            conn.commit()
            logger.info(f"Database schema initialized at {self.db_path}")

    # ------------------ CRUD Helpers ------------------

    def upsert_engine(
        self,
        engine_id: int,
        subset: str = "FD001",
        total_cycles: int = 0,
        last_cycle: int = 0,
        health_status: str = "HEALTHY",
        rul: float = 125.0,
        anomaly_score: float = 0.0
    ) -> None:
        """Upsert engine record."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO engines (engine_id, subset, total_cycles, last_cycle_observed, current_health_status, current_rul, anomaly_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id) DO UPDATE SET
                total_cycles=excluded.total_cycles,
                last_cycle_observed=excluded.last_cycle_observed,
                current_health_status=excluded.current_health_status,
                current_rul=excluded.current_rul,
                anomaly_score=excluded.anomaly_score,
                updated_at=excluded.updated_at;
            """, (int(engine_id), str(subset), int(total_cycles), int(last_cycle), str(health_status), float(rul), float(anomaly_score), str(now), str(now)))
            conn.commit()

    def record_prediction(self, payload: Dict[str, Any]) -> None:
        """Saves unified prediction and fingerprint telemetry."""
        eng_id = int(payload["engine_id"])
        cycle = int(payload["cycle"])
        fp = payload.get("degradation_fingerprint", {})
        
        with self.get_connection() as conn:
            # Insert fingerprint
            conn.execute("""
            INSERT INTO degradation_fingerprints (engine_id, cycle, health_index, health_status, degradation_rate, dominant_sensors_json, deviation_vector_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                int(eng_id), int(cycle),
                float(payload.get("health_index", 1.0)),
                str(payload.get("health_status", "HEALTHY")),
                float(fp.get("degradation_rate_per_cycle", 0.0)),
                json.dumps(fp.get("dominant_deviating_sensors", [])),
                json.dumps(fp.get("deviation_vector", {}))
            ))

            # Insert prediction
            conn.execute("""
            INSERT INTO predictions (engine_id, cycle, rul_point, rul_lower, rul_upper, interval_width, anomaly_score, uncertainty_category, human_review_required, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                int(eng_id), int(cycle),
                float(payload["predicted_rul"]),
                float(payload["rul_lower"]),
                float(payload["rul_upper"]),
                float(payload["interval_width"]),
                float(payload["anomaly_score"]),
                str(payload["uncertainty_category"]),
                1 if payload["human_review_required"] else 0,
                str(payload.get("model_version", "v1.0.0-PROD"))
            ))

            # Update engine overview
            conn.execute("""
            UPDATE engines SET
                last_cycle_observed = ?,
                current_health_status = ?,
                current_rul = ?,
                anomaly_score = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE engine_id = ?;
            """, (int(cycle), str(payload.get("health_status", "HEALTHY")), float(payload["predicted_rul"]), float(payload["anomaly_score"]), int(eng_id)))

            conn.commit()

    def get_active_work_order(self, engine_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves active (OPEN or IN_PROGRESS or ESCALATED) work order for engine, if any."""
        with self.get_connection() as conn:
            row = conn.execute("""
            SELECT * FROM work_orders 
            WHERE engine_id = ? AND status IN ('OPEN', 'IN_PROGRESS', 'ESCALATED')
            ORDER BY created_at DESC LIMIT 1;
            """, (engine_id,)).fetchone()
            if row:
                d = dict(row)
                if d.get("evidence_report_json"):
                    d["evidence_report"] = json.loads(d["evidence_report_json"])
                return d
            return None

    def insert_work_order(self, wo: Dict[str, Any]) -> None:
        """Inserts a new work order."""
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO work_orders (work_order_id, engine_id, cycle, priority, reason, recommended_action, status, evidence_report_json, model_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                str(wo["work_order_id"]),
                int(wo["engine_id"]),
                int(wo["cycle"]),
                str(wo["priority"]),
                str(wo["reason"]),
                str(wo["recommended_action"]),
                str(wo["status"]),
                json.dumps(wo.get("evidence_report", {})),
                str(wo.get("model_version", "v1.0.0-PROD")),
                str(wo.get("created_at")),
                str(wo.get("updated_at"))
            ))
            
            # Log audit trail
            conn.execute("""
            INSERT INTO audit_logs (event_type, entity_type, entity_id, action, details_json, operator)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                "WORK_ORDER_CREATED", "work_order", wo["work_order_id"],
                f"Created {wo['priority']} work order for Engine #{wo['engine_id']}",
                json.dumps({"reason": wo["reason"], "predicted_rul": wo.get("evidence_report", {}).get("prognostics", {}).get("predicted_rul")}),
                "AURA-M_AGENT"
            ))
            conn.commit()

    def update_work_order_status(self, work_order_id: str, new_status: str, notes: Optional[str] = None, operator: str = "OPERATOR") -> bool:
        """Updates work order lifecycle status."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute("""
            UPDATE work_orders SET status = ?, closing_notes = ?, updated_at = ?
            WHERE work_order_id = ?;
            """, (new_status, notes, now, work_order_id))
            
            conn.execute("""
            INSERT INTO audit_logs (event_type, entity_type, entity_id, action, details_json, operator)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                "WORK_ORDER_STATUS_CHANGED", "work_order", work_order_id,
                f"Status transitioned to {new_status}",
                json.dumps({"new_status": new_status, "notes": notes}),
                operator
            ))
            conn.commit()
            return cursor.rowcount > 0

    def list_work_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists work orders with optional status filter."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute("SELECT * FROM work_orders WHERE status = ? ORDER BY created_at DESC;", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM work_orders ORDER BY created_at DESC;").fetchall()
            
            res = []
            for r in rows:
                d = dict(r)
                if d.get("evidence_report_json"):
                    try:
                        d["evidence_report"] = json.loads(d["evidence_report_json"])
                    except Exception:
                        pass
                res.append(d)
            return res

    def list_engines(self) -> List[Dict[str, Any]]:
        """Lists all registered engines."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM engines ORDER BY engine_id ASC;").fetchall()
            return [dict(r) for r in rows]

    def get_engine(self, engine_id: int) -> Optional[Dict[str, Any]]:
        """Gets single engine summary."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM engines WHERE engine_id = ?;", (engine_id,)).fetchone()
            return dict(row) if row else None

    def log_audit(self, event_type: str, entity_type: str, entity_id: str, action: str, details: Dict[str, Any], operator: str = "SYSTEM") -> None:
        """Records system audit event."""
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO audit_logs (event_type, entity_type, entity_id, action, details_json, operator)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (event_type, entity_type, entity_id, action, json.dumps(details), operator))
            conn.commit()

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Gets recent audit events."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?;", (limit,)).fetchall()
            return [dict(r) for r in rows]
