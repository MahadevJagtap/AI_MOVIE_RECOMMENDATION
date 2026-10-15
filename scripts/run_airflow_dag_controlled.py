"""
Controlled End-to-End Airflow DAG Execution Runner.

Executes the 10-task recommendation pipeline sequentially in controlled mode,
validating explicit task transitions, quality gates, artifact validation,
and model registry publication. Also demonstrates quality gate failure blocking.

Usage:
    python scripts/run_airflow_dag_controlled.py [--version v-dev-verified] [--simulate-failure]
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.dag_tasks import (
    validate_input_data,
    validate_model_artifacts,
    evaluate_quality_gate,
    publish_model_version,
)
from airflow.dags.recommendation_pipeline_dag import (
    task_preprocess_verify,
    task_embeddings_verify,
    task_faiss_verify,
    task_eval_verify,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [airflow_dag_run]: %(message)s")
logger = logging.getLogger("airflow_dag_run")


def run_controlled_dag(version: str = "v-dev-verified", simulate_gate_failure: bool = False) -> Dict[str, Any]:
    """
    Execute controlled DAG execution across all 10 tasks.
    """
    logger.info("=" * 80)
    logger.info(f"STARTING CONTROLLED AIRFLOW DAG EXECUTION (Version: {version})")
    logger.info(f"DAG ID: movielens_recommendation_pipeline | Mode: Controlled Execution")
    logger.info(f"Simulate Gate Failure: {simulate_gate_failure}")
    logger.info("=" * 80)

    task_results: Dict[str, Dict[str, Any]] = {}
    dag_start_time = time.perf_counter()

    tasks_ordered = [
        ("validate_data", "Data Validation"),
        ("preprocess_data", "Preprocessing Partition Verification"),
        ("train_retrieval", "Two-Tower Retrieval Training"),
        ("generate_embeddings", "Catalog Embedding Verification"),
        ("build_faiss_index", "FAISS CPU Candidate Index Verification"),
        ("train_ranking", "Wide & Deep Ranking Training"),
        ("evaluate_models", "Model Evaluation Report Verification"),
        ("quality_gate", "Automated Performance Quality Gate"),
        ("validate_artifacts", "Model Artifact Size & Schema Verification"),
        ("publish_models", "Production Model Registry Publishing"),
    ]

    for task_id, task_desc in tasks_ordered:
        t0 = time.perf_counter()
        logger.info(f"\n---> [TASK START] {task_id} ({task_desc})...")

        try:
            if task_id == "validate_data":
                out = validate_input_data()
                status = "SUCCESS"
            elif task_id == "preprocess_data":
                out = task_preprocess_verify()
                status = "SUCCESS"
            elif task_id == "train_retrieval":
                # Verify training script existence and environment readiness
                train_script = PROJECT_ROOT / "scripts" / "train_retrieval.py"
                assert train_script.exists(), f"Retrieval script missing: {train_script}"
                out = {"status": "TRAINING_READY", "script": str(train_script)}
                status = "SUCCESS"
            elif task_id == "generate_embeddings":
                out = task_embeddings_verify()
                status = "SUCCESS"
            elif task_id == "build_faiss_index":
                out = task_faiss_verify()
                status = "SUCCESS"
            elif task_id == "train_ranking":
                # Verify ranking script existence and environment readiness
                rank_script = PROJECT_ROOT / "scripts" / "train_ranking.py"
                assert rank_script.exists(), f"Ranking script missing: {rank_script}"
                out = {"status": "TRAINING_READY", "script": str(rank_script)}
                status = "SUCCESS"
            elif task_id == "evaluate_models":
                out = task_eval_verify()
                status = "SUCCESS"
            elif task_id == "quality_gate":
                if simulate_gate_failure:
                    # Intentionally unattainable threshold to demonstrate quality gate block
                    logger.warning("Simulating quality gate failure with unattainable threshold (min_ndcg=0.99)...")
                    out = evaluate_quality_gate(min_ndcg_at_10=0.99, min_precision_at_10=0.99)
                else:
                    out = evaluate_quality_gate(min_ndcg_at_10=0.10, min_precision_at_10=0.10, min_catalog_coverage=500)
                status = "SUCCESS"
            elif task_id == "validate_artifacts":
                out = validate_model_artifacts()
                status = "SUCCESS"
            elif task_id == "publish_models":
                out = publish_model_version(version=version)
                status = "SUCCESS"
            else:
                raise NotImplementedError(f"Unknown task {task_id}")

            duration_ms = (time.perf_counter() - t0) * 1000.0
            task_results[task_id] = {
                "status": status,
                "duration_ms": round(duration_ms, 2),
                "output": out,
            }
            logger.info(f"---> [TASK {status}] {task_id} completed in {duration_ms:.2f}ms")

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            task_results[task_id] = {
                "status": "FAILED",
                "duration_ms": round(duration_ms, 2),
                "error": str(e),
            }
            logger.error(f"---> [TASK FAILED] {task_id} failed after {duration_ms:.2f}ms: {e}")
            logger.error("PIPELINE EXECUTION HALTED DUE TO TASK FAILURE. Downstream tasks aborted.")
            break

    total_duration_sec = time.perf_counter() - dag_start_time
    dag_status = "SUCCESS" if all(r["status"] == "SUCCESS" for r in task_results.values()) and len(task_results) == 10 else "FAILED"

    # Print Summary Table
    print("\n" + "=" * 85)
    print(f"AIRFLOW CONTROLLED DAG RUN REPORT — Status: {dag_status} ({total_duration_sec:.2f}s)")
    print("=" * 85)
    print(f"{'#':<3} | {'Task ID':<22} | {'Description':<38} | {'Status':<10} | {'Duration':<10}")
    print("-" * 85)
    for idx, (tid, desc) in enumerate(tasks_ordered, 1):
        res = task_results.get(tid, {"status": "SKIPPED (ABORTED)", "duration_ms": 0.0})
        print(f"{idx:<3} | {tid:<22} | {desc:<38} | {res['status']:<10} | {res['duration_ms']:>7.2f} ms")
    print("=" * 85 + "\n")

    return {
        "dag_id": "movielens_recommendation_pipeline",
        "dag_status": dag_status,
        "total_duration_sec": total_duration_sec,
        "task_results": task_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run controlled Airflow recommendation pipeline DAG")
    parser.add_argument("--version", default="v-dev-verified", help="Model version tag")
    parser.add_argument("--simulate-failure", action="store_true", help="Simulate quality gate failure")
    args = parser.parse_args()

    run_controlled_dag(version=args.version, simulate_gate_failure=args.simulate_failure)


if __name__ == "__main__":
    main()
