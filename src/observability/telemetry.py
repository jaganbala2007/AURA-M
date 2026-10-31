"""
AURA-M Observability & Operational Metrics Engine (Technology Module 12)
Provides structured JSON telemetry, API/Inference latency tracking, and pipeline health telemetry.
"""

import time
import datetime
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("AURA-M.Observability")

class SystemMetricsCollector:
    """Collects real-time operational telemetry for system observability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.start_time = time.time()
            cls._instance.inference_latencies_ms: List[float] = []
            cls._instance.api_latencies_ms: List[float] = []
            cls._instance.total_inferences = 0
            cls._instance.total_api_calls = 0
            cls._instance.error_count = 0
        return cls._instance

    def record_inference(self, duration_seconds: float) -> None:
        self.total_inferences += 1
        self.inference_latencies_ms.append(duration_seconds * 1000.0)
        if len(self.inference_latencies_ms) > 1000:
            self.inference_latencies_ms.pop(0)

    def record_api_call(self, duration_seconds: float) -> None:
        self.total_api_calls += 1
        self.api_latencies_ms.append(duration_seconds * 1000.0)
        if len(self.api_latencies_ms) > 1000:
            self.api_latencies_ms.pop(0)

    def record_error(self) -> None:
        self.error_count += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calculates latency percentiles, throughput, and uptime."""
        uptime_seconds = time.time() - self.start_time
        
        inf_p50 = float(np.percentile(self.inference_latencies_ms, 50)) if self.inference_latencies_ms else 0.0
        inf_p95 = float(np.percentile(self.inference_latencies_ms, 95)) if self.inference_latencies_ms else 0.0
        inf_p99 = float(np.percentile(self.inference_latencies_ms, 99)) if self.inference_latencies_ms else 0.0

        api_p50 = float(np.percentile(self.api_latencies_ms, 50)) if self.api_latencies_ms else 0.0
        api_p95 = float(np.percentile(self.api_latencies_ms, 95)) if self.api_latencies_ms else 0.0

        throughput = (self.total_inferences / uptime_seconds) if uptime_seconds > 0 else 0.0

        return {
            "uptime_seconds": round(uptime_seconds, 1),
            "total_inferences": self.total_inferences,
            "total_api_requests": self.total_api_calls,
            "error_count": self.error_count,
            "inference_throughput_per_sec": round(throughput, 2),
            "inference_latency_ms": {
                "p50": round(inf_p50, 2),
                "p95": round(inf_p95, 2),
                "p99": round(inf_p99, 2)
            },
            "api_latency_ms": {
                "p50": round(api_p50, 2),
                "p95": round(api_p95, 2)
            },
            "health_status": "OPTIMAL" if self.error_count == 0 else "WARNING"
        }
