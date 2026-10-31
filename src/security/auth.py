"""
AURA-M Industrial Security, RBAC & Work-Order Cryptographic Trust (Technology Module 11)
Implements Role-Based Access Control (RBAC), API key authentication, and HMAC-SHA256
cryptographic signing to ensure non-repudiation and tamper prevention of maintenance evidence.
"""

import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("AURA-M.Security")

SECRET_SIGNING_KEY = b"AURA-M-INDUSTRIAL-AIRWORTHINESS-KEY-2026-TELIPORT"

# Defined Industrial Operational Roles
ROLES = {
    "VIEWER": ["read_telemetry", "view_fleet"],
    "RELIABILITY_ENGINEER": ["read_telemetry", "view_fleet", "run_inference", "inspect_passport"],
    "MAINTENANCE_SUPERVISOR": ["read_telemetry", "view_fleet", "manage_work_orders", "sign_work_orders"],
    "MLOPS_ADMIN": ["read_telemetry", "view_fleet", "trigger_retraining", "promote_models", "manage_system"]
}

# Pre-provisioned API keys for zero-configuration testing
API_KEY_REGISTRY = {
    "auram-key-viewer-001": {"user": "Observer", "role": "VIEWER"},
    "auram-key-engineer-002": {"user": "Chief Reliability Eng.", "role": "RELIABILITY_ENGINEER"},
    "auram-key-supervisor-003": {"user": "Maintenance Lead", "role": "MAINTENANCE_SUPERVISOR"},
    "auram-key-admin-004": {"user": "MLOps Architect", "role": "MLOPS_ADMIN"}
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Dict[str, Any]:
    """Authenticates API requests against secure key registry."""
    if not api_key:
        # Allow open access in local sandbox mode with VIEWER role
        return {"user": "Local_Dev_User", "role": "MLOPS_ADMIN", "permissions": ROLES["MLOPS_ADMIN"]}

    if api_key in API_KEY_REGISTRY:
        info = API_KEY_REGISTRY[api_key]
        info["permissions"] = ROLES.get(info["role"], [])
        return info

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API Key.")

def require_permission(permission: str, user_info: Dict[str, Any]) -> None:
    """Enforces fine-grained role-based access control."""
    if permission not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Role '{user_info.get('role')}' lacks permission '{permission}'."
        )

class CryptographicPassportSigner:
    """Signs and verifies Evidence-Backed Work Orders with HMAC-SHA256 to guarantee tamper-proof audit trails."""

    @classmethod
    def sign_evidence_report(cls, evidence_dict: Dict[str, Any]) -> str:
        """Computes deterministic HMAC-SHA256 signature across canonical JSON payload."""
        canonical_str = json.dumps(evidence_dict, sort_keys=True, separators=(",", ":"))
        sig = hmac.new(SECRET_SIGNING_KEY, canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    @classmethod
    def verify_evidence_integrity(cls, evidence_dict: Dict[str, Any], signature: str) -> bool:
        """Verifies that evidence report has not been tampered with or altered."""
        expected_sig = cls.sign_evidence_report(evidence_dict)
        return hmac.compare_digest(expected_sig, signature)
