"""
Comprehensive Latency Benchmarking Script.

Measures:
  - Stage 1 Retrieval Latency (Two-Tower user embedding + FAISS search on 62,423 items)
  - Stage 2 Ranking Latency (Wide & Deep on 100 candidate items)
  - Stage 2 Ranking Latency (Wide-only on 100 candidate items)
  - End-to-End Two-Stage Pipeline Latency via get_recommendations
  - System hardware specifications (CPU, RAM, OS, Python version)

Outputs detailed statistics: Mean, P50, P90, P95, P99, Min, Max.
Saves benchmark report to data/experiments/latency_benchmark.json.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["REDIS_ENABLED"] = "false"  # measure pure compute latency without cache hits

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import EXPERIMENT_LOG_DIR
from src.serving.api import manager, get_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def get_hardware_info() -> Dict[str, Any]:
    """Capture environment and hardware specs using standard library."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 2)
            avail_ram_gb = round(stat.ullAvailPhys / (1024 ** 3), 2)
    except Exception:
        pass

    cpu_count = os.cpu_count() or 1
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "cpu_count_logical": cpu_count,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": avail_ram_gb,
        "python_version": platform.python_version(),
    }


def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of latencies in milliseconds."""
    arr = np.array(latencies_ms)
    return {
        "count": len(arr),
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def run_benchmark(iterations: int = 100, warmup: int = 10):
    logger.info("Initializing serving manager and loading ML artifacts...")
    manager.initialize()

    # Select warm users from vocabulary
    learned_users = sorted(list(manager.user_vocab_set), key=lambda x: int(x) if x.isdigit() else 0)
    if len(learned_users) < iterations + warmup:
        sample_users = (learned_users * ((iterations + warmup) // len(learned_users) + 1))[: iterations + warmup]
    else:
        sample_users = learned_users[: iterations + warmup]

    logger.info(f"Using {len(sample_users)} users for benchmark ({warmup} warmup, {iterations} timed).")

    # Warmup
    for uid in sample_users[:warmup]:
        _ = get_recommendations(user_id=uid, k=10, variant="wide_and_deep")
        _ = get_recommendations(user_id=uid, k=10, variant="wide_only")

    eval_users = sample_users[warmup : warmup + iterations]
    logger.info(f"Starting {len(eval_users)} benchmark iterations...")

    # Stage 1: Retrieval Latency (User embedding + FAISS Search k=100)
    retrieval_emb_latencies = []
    faiss_search_latencies = []
    total_retrieval_latencies = []
    user_embeddings = []
    retrieved_candidate_ids = []

    for uid in eval_users:
        t0 = time.perf_counter()
        emb = manager.two_tower.get_user_embedding(uid)
        t1 = time.perf_counter()
        cands = manager.faiss_manager.search(query_embedding=emb, top_k=100)
        t2 = time.perf_counter()

        retrieval_emb_latencies.append((t1 - t0) * 1000.0)
        faiss_search_latencies.append((t2 - t1) * 1000.0)
        total_retrieval_latencies.append((t2 - t0) * 1000.0)
        user_embeddings.append(emb)
        retrieved_candidate_ids.append([str(c[0]) for c in cands])

    # Stage 2: Feature Transformation Latency (for 100 candidates per user)
    feat_transform_latencies = []
    ranking_batches = []

    for uid, cands in zip(eval_users, retrieved_candidate_ids):
        t0 = time.perf_counter()
        feats = manager.feature_pipeline.transform_candidates(uid, cands)
        t1 = time.perf_counter()
        feat_transform_latencies.append((t1 - t0) * 1000.0)
        ranking_batches.append(feats)

    # Stage 2: Ranking Latency - Wide Only
    wide_only_latencies = []
    for feats in ranking_batches:
        t0 = time.perf_counter()
        _ = manager.wide_model.predict_scores(feats)
        t1 = time.perf_counter()
        wide_only_latencies.append((t1 - t0) * 1000.0)

    # Stage 2: Ranking Latency - Wide & Deep
    wide_and_deep_latencies = []
    for feats in ranking_batches:
        t0 = time.perf_counter()
        _ = manager.wd_model.predict_scores(feats)
        t1 = time.perf_counter()
        wide_and_deep_latencies.append((t1 - t0) * 1000.0)

    # End-to-End Recommendations through get_recommendations
    e2e_wide_latencies = []
    for uid in eval_users:
        t0 = time.perf_counter()
        _ = get_recommendations(user_id=uid, k=10, variant="wide_only")
        t1 = time.perf_counter()
        e2e_wide_latencies.append((t1 - t0) * 1000.0)

    e2e_wd_latencies = []
    for uid in eval_users:
        t0 = time.perf_counter()
        _ = get_recommendations(user_id=uid, k=10, variant="wide_and_deep")
        t1 = time.perf_counter()
        e2e_wd_latencies.append((t1 - t0) * 1000.0)

    hw_info = get_hardware_info()
    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_environment": hw_info,
        "sample_count": len(eval_users),
        "stage_1_user_embedding": compute_percentiles(retrieval_emb_latencies),
        "stage_1_faiss_search_k100": compute_percentiles(faiss_search_latencies),
        "stage_1_total_retrieval": compute_percentiles(total_retrieval_latencies),
        "stage_2_feature_transform_100_items": compute_percentiles(feat_transform_latencies),
        "stage_2_wide_only_inference": compute_percentiles(wide_only_latencies),
        "stage_2_wide_and_deep_inference": compute_percentiles(wide_and_deep_latencies),
        "stage_2_total_wide_only": compute_percentiles([f + w for f, w in zip(feat_transform_latencies, wide_only_latencies)]),
        "stage_2_total_wide_and_deep": compute_percentiles([f + wd for f, wd in zip(feat_transform_latencies, wide_and_deep_latencies)]),
        "end_to_end_pipeline_wide_only": compute_percentiles(e2e_wide_latencies),
        "end_to_end_pipeline_wide_and_deep": compute_percentiles(e2e_wd_latencies),
    }

    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXPERIMENT_LOG_DIR / "latency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Latency benchmark saved to {out_file}")

    print("\n" + "=" * 95)
    print("LATENCY BENCHMARK RESULTS (MovieLens 25M, 62,423 Catalog Items, N=100 requests)")
    print("=" * 95)
    print(f"Hardware: {hw_info['cpu']} ({hw_info['cpu_count_logical']} logical cores) | RAM: {hw_info['total_ram_gb']} GB")
    print(f"OS: {hw_info['os']} | Python: {hw_info['python_version']}")
    print("-" * 95)
    header = f"{'Pipeline Component':<42} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}"
    print(header)
    print("-" * 95)

    components = [
        ("1. Stage 1: Two-Tower User Embedding", results["stage_1_user_embedding"]),
        ("2. Stage 1: FAISS CPU Index Search (k=100)", results["stage_1_faiss_search_k100"]),
        ("3. Stage 1: Total Retrieval (Emb + FAISS)", results["stage_1_total_retrieval"]),
        ("4. Stage 2: Feature Transformation (100 items)", results["stage_2_feature_transform_100_items"]),
        ("5. Stage 2: Wide-only Model Inference", results["stage_2_wide_only_inference"]),
        ("6. Stage 2: Wide & Deep Model Inference", results["stage_2_wide_and_deep_inference"]),
        ("7. Stage 2: Total Ranking (Wide-only)", results["stage_2_total_wide_only"]),
        ("8. Stage 2: Total Ranking (Wide & Deep)", results["stage_2_total_wide_and_deep"]),
        ("9. Full End-to-End Pipeline (Wide-only)", results["end_to_end_pipeline_wide_only"]),
        ("10. Full End-to-End Pipeline (Wide & Deep)", results["end_to_end_pipeline_wide_and_deep"]),
    ]

    for name, stats in components:
        print(f"{name:<42} | {stats['mean_ms']:<10.2f} | {stats['p50_ms']:<10.2f} | {stats['p95_ms']:<10.2f} | {stats['p99_ms']:<10.2f}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_benchmark(iterations=100, warmup=10)
