"""
AURA-M Edge AI & TinyML Optimization Engine (Technology Module 1)
Exports trained prognostic models to ONNX format, performs offline edge inference,
measures latency benchmarks, and ensures graceful fallback.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger("AURA-M.EdgeAI")

class EdgeModelOptimizer:
    """Exports and serves optimized ONNX models for edge reliability gateways."""

    def __init__(self, model_dir: str = "models/trained", onnx_dir: str = "models/onnx"):
        self.model_dir = Path(model_dir)
        self.onnx_dir = Path(onnx_dir)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.ort_sessions: Dict[str, ort.InferenceSession] = {}

    def export_to_onnx(
        self,
        sklearn_model: Any,
        num_features: int,
        onnx_filename: str = "rul_model_edge.onnx"
    ) -> Path:
        """
        Converts a trained scikit-learn regressor (Random Forest) to optimized ONNX format.
        """
        out_path = self.onnx_dir / onnx_filename
        logger.info(f"Exporting model to ONNX format at {out_path} ({num_features} input features)...")
        
        initial_type = [("float_input", FloatTensorType([None, num_features]))]
        onnx_proto = convert_sklearn(sklearn_model, initial_types=initial_type, target_opset=15)
        
        with open(out_path, "wb") as f:
            f.write(onnx_proto.SerializeToString())
            
        file_size_kb = out_path.stat().st_size / 1024.0
        logger.info(f"ONNX export successful: {out_path.name} ({file_size_kb:.1f} KB)")
        return out_path

    def get_session(self, onnx_filename: str = "rul_model_edge.onnx") -> Optional[ort.InferenceSession]:
        """Loads and caches an ONNX Runtime inference session."""
        if onnx_filename in self.ort_sessions:
            return self.ort_sessions[onnx_filename]

        onnx_path = self.onnx_dir / onnx_filename
        if not onnx_path.exists():
            logger.warning(f"ONNX model file not found at {onnx_path}")
            return None

        # Create session with minimal thread overhead for deterministic edge execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.ort_sessions[onnx_filename] = session
        return session

    def predict_edge(
        self,
        feature_values: np.ndarray,
        onnx_filename: str = "rul_model_edge.onnx",
        fallback_model: Optional[Any] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Executes edge inference via ONNX Runtime with graceful fallback to native Python model.
        
        Returns:
            predictions: np.ndarray
            runtime_used: 'ONNX_RUNTIME' or 'PYTHON_FALLBACK'
        """
        session = self.get_session(onnx_filename)
        
        if session is not None:
            try:
                input_name = session.get_inputs()[0].name
                inp = np.array(feature_values, dtype=np.float32)
                if inp.ndim == 1:
                    inp = inp.reshape(1, -1)
                ort_out = session.run(None, {input_name: inp})[0]
                preds = np.clip(ort_out.flatten(), 0.0, None)
                return preds, "ONNX_RUNTIME"
            except Exception as e:
                logger.error(f"ONNX edge inference failed, falling back to native runtime: {e}")

        # Fallback to standard python model
        if fallback_model is not None:
            preds = fallback_model.predict(feature_values)
            return preds, "PYTHON_FALLBACK"

        raise RuntimeError("Both ONNX edge session and fallback model are unavailable.")

    def benchmark_edge_performance(
        self,
        native_model: Any,
        onnx_filename: str,
        sample_input: np.ndarray,
        iterations: int = 500
    ) -> Dict[str, Any]:
        """
        Rigorous edge performance comparison: size, latency (mean, median, P95, P99), and parity.
        """
        session = self.get_session(onnx_filename)
        if session is None:
            raise FileNotFoundError(f"ONNX model {onnx_filename} not found.")

        input_name = session.get_inputs()[0].name
        x_32 = np.array(sample_input, dtype=np.float32).reshape(1, -1)
        x_native = np.array(sample_input).reshape(1, -1)

        # 1. Warm-up
        for _ in range(20):
            session.run(None, {input_name: x_32})
            if hasattr(native_model, "predict"):
                native_model.predict(x_native)

        # 2. Benchmark ONNX Latency
        onnx_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            pred_onnx = session.run(None, {input_name: x_32})[0]
            t1 = time.perf_counter()
            onnx_times.append((t1 - t0) * 1e6)  # microseconds

        # 3. Benchmark Native Python Latency
        native_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            pred_native = native_model.predict(x_native)
            t1 = time.perf_counter()
            native_times.append((t1 - t0) * 1e6)

        # 4. Parity / Consistency check (L_inf norm difference)
        diff = float(np.max(np.abs(pred_native.flatten() - pred_onnx.flatten())))

        onnx_file_path = self.onnx_dir / onnx_filename
        onnx_size_kb = onnx_file_path.stat().st_size / 1024.0

        return {
            "onnx_model_file": onnx_filename,
            "onnx_size_kb": round(onnx_size_kb, 1),
            "iterations": iterations,
            "latency_native_us": {
                "mean": round(float(np.mean(native_times)), 1),
                "median": round(float(np.median(native_times)), 1),
                "p95": round(float(np.percentile(native_times, 95)), 1),
                "p99": round(float(np.percentile(native_times, 99)), 1)
            },
            "latency_onnx_us": {
                "mean": round(float(np.mean(onnx_times)), 1),
                "median": round(float(np.median(onnx_times)), 1),
                "p95": round(float(np.percentile(onnx_times, 95)), 1),
                "p99": round(float(np.percentile(onnx_times, 99)), 1)
            },
            "speedup_factor": round(float(np.mean(native_times)) / max(float(np.mean(onnx_times)), 1e-4), 2),
            "max_prediction_disparity_cycles": round(diff, 4),
            "hardware_platform": "x86_64 CPU (Edge Gateway Architecture Emulation)"
        }
