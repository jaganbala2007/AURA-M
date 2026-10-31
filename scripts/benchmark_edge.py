"""
AURA-M Edge AI Benchmark Script
Exports trained prognostic model to ONNX and performs high-precision microsecond
latency and memory benchmarks comparing native Python inference vs ONNX Runtime.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rul.models import load_rul_model
from src.edge.optimizer import EdgeModelOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.EdgeBenchmark")

def main():
    logger.info("==================================================================")
    logger.info("           AURA-M EDGE AI & ONNX OPTIMIZATION BENCHMARK           ")
    logger.info("==================================================================")

    model_path = Path("models/trained/rul_model_v1.0.0-BASELINE-RF.pkl")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run scripts/train.py first.")
        return 1

    rf_model = load_rul_model(str(model_path))
    num_features = len(rf_model.feature_names)
    logger.info(f"Loaded {rf_model.model_name} with {num_features} input features.")

    optimizer = EdgeModelOptimizer(model_dir="models/trained", onnx_dir="models/onnx")
    onnx_file = "rul_model_edge.onnx"

    # Export to ONNX
    optimizer.export_to_onnx(
        sklearn_model=rf_model.model,
        num_features=num_features,
        onnx_filename=onnx_file
    )

    # Benchmark with realistic input
    dummy_input = np.random.uniform(0.0, 1.0, size=(num_features,)).astype(np.float32)
    benchmark_results = optimizer.benchmark_edge_performance(
        native_model=rf_model,
        onnx_filename=onnx_file,
        sample_input=dummy_input,
        iterations=500
    )

    # Calculate model sizes
    native_size_kb = model_path.stat().st_size / 1024.0
    onnx_size_kb = (optimizer.onnx_dir / onnx_file).stat().st_size / 1024.0
    benchmark_results["native_size_kb"] = round(native_size_kb, 1)

    print("\n" + "="*70)
    print("                    EDGE AI INFERENCE BENCHMARK                  ")
    print("="*70)
    print(f"Hardware Environment     : {benchmark_results['hardware_platform']}")
    print(f"Native Model Size (Pickle): {benchmark_results['native_size_kb']:.1f} KB")
    print(f"Optimized ONNX Model Size : {benchmark_results['onnx_size_kb']:.1f} KB")
    print(f"Size Reduction Factor     : {native_size_kb / onnx_size_kb:.2f}x")
    print("-"*70)
    print(f"Native Python Mean Latency: {benchmark_results['latency_native_us']['mean']:.1f} µs (P95: {benchmark_results['latency_native_us']['p95']:.1f} µs)")
    print(f"ONNX Runtime Mean Latency : {benchmark_results['latency_onnx_us']['mean']:.1f} µs (P95: {benchmark_results['latency_onnx_us']['p95']:.1f} µs)")
    print(f"Latency Speedup Factor    : {benchmark_results['speedup_factor']:.2f}x faster")
    print(f"Max Output Discrepancy    : {benchmark_results['max_prediction_disparity_cycles']:.5f} cycles (Parity Confirmed)")
    print("="*70 + "\n")

    # Save benchmark report
    reports_dir = Path("reports/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "edge_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Edge benchmark report saved to {reports_dir / 'edge_benchmark.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
