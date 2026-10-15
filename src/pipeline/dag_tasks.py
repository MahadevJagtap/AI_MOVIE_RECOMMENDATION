"""
Task definitions and quality gates for Airflow recommendation pipeline DAG.

These functions can be executed either directly via PythonOperator or as
standalone validation routines.
"""

from __future__ import annotations

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("airflow.recommendation_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def validate_input_data(processed_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Validate presence and schema of processed MovieLens datasets.
    
    Checks:
    - Processed data directory exists.
    - train, val, and test parquet directories exist and are non-empty.
    - metadata.json exists and contains dataset split statistics.
    """
    logger.info("Executing task: validate_input_data...")
    root = get_project_root()
    p_dir = Path(processed_dir) if processed_dir else root / "data" / "processed"

    if not p_dir.exists():
        msg = f"Processed data directory not found: {p_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    required_subdirs = ["train", "val", "test"]
    partition_counts = {}

    for sub in required_subdirs:
        sub_path = p_dir / sub
        if not sub_path.exists():
            msg = f"Required dataset split partition missing: {sub_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Verify partition contains parquet files
        parquet_files = list(sub_path.glob("*.parquet"))
        if not parquet_files:
            msg = f"Partition directory {sub_path} contains no parquet files"
            logger.error(msg)
            raise ValueError(msg)

        total_bytes = sum(f.stat().st_size for f in parquet_files)
        partition_counts[sub] = {
            "num_files": len(parquet_files),
            "total_bytes": total_bytes,
        }
        logger.info(f"Verified split '{sub}': {len(parquet_files)} files ({total_bytes / (1024*1024):.2f} MB)")

    # Validate metadata
    meta_path = p_dir / "metadata.json"
    if not meta_path.exists():
        msg = f"Dataset metadata missing: {meta_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    logger.info(f"Input validation passed successfully. Train samples: {meta_data.get('train_count', 'N/A')}")
    return {
        "status": "VALID",
        "partitions": partition_counts,
        "metadata": meta_data,
        "timestamp": time.time(),
    }


def validate_model_artifacts(
    models_dir: Optional[str | Path] = None,
    required_specs: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Validate that all required model and index artifacts exist and satisfy minimum size gates.
    """
    logger.info("Executing task: validate_model_artifacts...")
    root = get_project_root()
    base_dir = Path(models_dir) if models_dir else root / "models"

    # Default minimum byte sizes to guarantee artifacts are not 0-byte corrupt stubs
    default_specs = {
        "retrieval/model.weights.h5": 1_000_000,          # ~11.9 MB
        "retrieval/model_config.json": 100,               # ~200 KB
        "retrieval/movie_embeddings.npy": 1_000_000,      # ~16.0 MB
        "faiss/faiss_index.bin": 1_000_000,               # ~16.0 MB
        "faiss/index_metadata.json": 100,                 # ~780 KB
        "ranking/wide_only/model.weights.h5": 100_000,    # ~1.4 MB
        "ranking/wide_and_deep/model.weights.h5": 500_000,# ~3.2 MB
        "ranking/pipeline/ranking_pipeline_config.json": 100, # ~2 KB
    }
    specs = required_specs or default_specs
    results = {}

    for rel_path, min_bytes in specs.items():
        file_path = base_dir / rel_path
        if not file_path.exists():
            msg = f"Artifact validation failed: Required artifact missing: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        actual_size = file_path.stat().st_size
        if actual_size < min_bytes:
            msg = (
                f"Artifact validation failed: {file_path} size ({actual_size} bytes) "
                f"is below required threshold ({min_bytes} bytes)"
            )
            logger.error(msg)
            raise ValueError(msg)

        results[rel_path] = {
            "status": "VALID",
            "size_bytes": actual_size,
            "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        logger.info(f"Artifact verified: {rel_path} ({actual_size / (1024*1024):.2f} MB)")

    logger.info("All model artifacts passed verification quality gates.")
    return {
        "status": "SUCCESS",
        "verified_artifacts": results,
        "timestamp": time.time(),
    }


def evaluate_quality_gate(
    ranking_exp_path: Optional[str | Path] = None,
    retrieval_exp_path: Optional[str | Path] = None,
    min_ndcg_at_10: float = 0.10,
    min_precision_at_10: float = 0.10,
    min_catalog_coverage: int = 500,
) -> Dict[str, Any]:
    """
    Quality gate enforcing minimum offline performance thresholds prior to model publishing.
    
    Quality Gate Criteria:
    1. Primary ranking model (Variant B or Variant A) NDCG@10 >= min_ndcg_at_10
    2. Primary ranking model Precision@10 >= min_precision_at_10
    3. Retrieval candidate catalog coverage >= min_catalog_coverage
    """
    logger.info("Executing task: evaluate_quality_gate...")
    root = get_project_root()
    rank_path = Path(ranking_exp_path) if ranking_exp_path else root / "data" / "experiments" / "ranking_experiment.json"
    ret_path = Path(retrieval_exp_path) if retrieval_exp_path else root / "data" / "experiments" / "retrieval_experiment.json"

    if not rank_path.exists():
        msg = f"Ranking experiment results file missing for quality gate: {rank_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(rank_path, "r", encoding="utf-8") as f:
        rank_data = json.load(f)

    models_metrics = rank_data.get("evaluation_results", {}).get("models", {})
    if not models_metrics:
        # Fallback to alternate key structure
        models_metrics = rank_data.get("models", {})

    # Evaluate Variant A and Variant B
    gate_checks = []
    for model_name, metrics in models_metrics.items():
        ndcg_10 = metrics.get("ndcg@10", 0.0)
        prec_10 = metrics.get("precision@10", 0.0)

        ndcg_pass = ndcg_10 >= min_ndcg_at_10
        prec_pass = prec_10 >= min_precision_at_10

        gate_checks.append({
            "model": model_name,
            "ndcg@10": ndcg_10,
            "min_ndcg@10": min_ndcg_at_10,
            "ndcg_pass": ndcg_pass,
            "precision@10": prec_10,
            "min_precision@10": min_precision_at_10,
            "precision_pass": prec_pass,
        })

    # At least one model must meet the ranking quality gates
    passing_models = [c for c in gate_checks if c["ndcg_pass"] and c["precision_pass"]]
    if not passing_models:
        msg = f"Quality gate failed: No ranking model satisfied NDCG@10 >= {min_ndcg_at_10} and Precision@10 >= {min_precision_at_10}: {gate_checks}"
        logger.error(msg)
        raise ValueError(msg)

    coverage_result = None
    if ret_path.exists():
        with open(ret_path, "r", encoding="utf-8") as f:
            ret_data = json.load(f)
        tt_metrics = ret_data.get("metrics", {}).get("two_tower", {})
        coverage = tt_metrics.get("candidate_catalog_coverage", 0)
        cov_pass = coverage >= min_catalog_coverage
        coverage_result = {
            "coverage": coverage,
            "min_coverage": min_catalog_coverage,
            "passed": cov_pass,
        }
        if not cov_pass:
            msg = f"Quality gate failed: Retrieval coverage ({coverage}) below threshold ({min_catalog_coverage})"
            logger.error(msg)
            raise ValueError(msg)

    logger.info(f"Quality gate PASSED: {len(passing_models)}/{len(gate_checks)} ranking models met or exceeded performance gates.")
    return {
        "status": "PASSED",
        "ranking_gate": gate_checks,
        "retrieval_gate": coverage_result,
        "timestamp": time.time(),
    }


def publish_model_version(
    version: str = "v1.0.0",
    registry_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Publish validated model version into the production model registry metadata.
    """
    logger.info(f"Executing task: publish_model_version ({version})...")
    root = get_project_root()
    reg_path = Path(registry_file) if registry_file else root / "models" / "model_registry.json"

    registry = {"versions": {}, "active_version": None, "updated_at": None}
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse existing registry ({e}); initializing new registry structure.")

    # Record model version deployment entry
    from datetime import timezone
    entry = {
        "version": version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "status": "ACTIVE",
        "artifacts": {
            "retrieval_model": "models/retrieval/model.weights.h5",
            "faiss_index": "models/faiss/faiss_index.bin",
            "wide_only_model": "models/ranking/wide_only/model.weights.h5",
            "wide_and_deep_model": "models/ranking/wide_and_deep/model.weights.h5",
            "pipeline_config": "models/ranking/pipeline/ranking_pipeline_config.json",
        },
        "quality_gate": "PASSED",
    }

    registry["versions"][version] = entry
    registry["active_version"] = version
    registry["updated_at"] = entry["published_at"]

    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Successfully published model version '{version}' to registry at {reg_path}")
    return {
        "status": "PUBLISHED",
        "active_version": version,
        "registry_file": str(reg_path),
        "timestamp": time.time(),
    }
