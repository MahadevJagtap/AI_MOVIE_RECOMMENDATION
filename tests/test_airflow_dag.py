"""
Unit and integration tests for Apache Airflow DAG and quality gates.

Tests:
1. Airflow DAG loading, structure, and cycle detection.
2. Explicit task upstream/downstream dependency wiring.
3. Task retry and default configuration parameters.
4. Data validation quality gate task.
5. Model artifact size and existence quality gate.
6. Performance metric quality gate (NDCG, Precision, Coverage).
7. Model version publishing into the production model registry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

from src.pipeline.dag_tasks import (
    validate_input_data,
    validate_model_artifacts,
    evaluate_quality_gate,
    publish_model_version,
    get_project_root,
)

AIRFLOW_PYTHON = Path(r"D:\setup\airflow-venv\Scripts\python.exe")
AIRFLOW_HOME = Path(r"D:\setup\recommendation-engine\airflow")


class TestAirflowDAGStructure:
    """Test Airflow DAG graph definition, structure, and dependencies."""

    @pytest.mark.skipif(not AIRFLOW_PYTHON.exists(), reason="Airflow isolated venv not found")
    def test_airflow_dag_import_and_integrity(self):
        """Verify DAG parses cleanly in the isolated Airflow environment with no cycles."""
        validation_code = """
import sys
from pathlib import Path
root = Path(r'D:\\setup\\recommendation-engine')
sys.path.insert(0, str(root))

from airflow.dags.recommendation_pipeline_dag import recommendation_pipeline_dag

dag = recommendation_pipeline_dag
assert dag is not None, "DAG instance is None"
assert dag.dag_id == "movielens_recommendation_pipeline", f"Unexpected DAG ID: {dag.dag_id}"
assert len(dag.tasks) == 10, f"Expected 10 tasks, found {len(dag.tasks)}"

task_ids = {t.task_id for t in dag.tasks}
expected_tasks = {
    "validate_data",
    "preprocess_data",
    "train_retrieval",
    "generate_embeddings",
    "build_faiss_index",
    "train_ranking",
    "evaluate_models",
    "quality_gate",
    "validate_artifacts",
    "publish_models",
}
assert task_ids == expected_tasks, f"Task mismatch: {task_ids ^ expected_tasks}"

# Check cycle-free topology (topological sort)
tasks_dict = {t.task_id: t for t in dag.tasks}
visited = set()
path = set()

def dfs(task_id):
    visited.add(task_id)
    path.add(task_id)
    for downstream in tasks_dict[task_id].downstream_list:
        down_id = downstream.task_id
        if down_id in path:
            raise ValueError(f"Cycle detected between {task_id} and {down_id}")
        if down_id not in visited:
            dfs(down_id)
    path.remove(task_id)

for tid in task_ids:
    if tid not in visited:
        dfs(tid)

print("DAG_INTEGRITY_SUCCESS")
"""
        env = os.environ.copy()
        env["AIRFLOW_HOME"] = str(AIRFLOW_HOME)
        proc = subprocess.run(
            [str(AIRFLOW_PYTHON), "-c", validation_code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, f"Airflow DAG validation failed:\n{proc.stderr}\n{proc.stdout}"
        assert "DAG_INTEGRITY_SUCCESS" in proc.stdout

    @pytest.mark.skipif(not AIRFLOW_PYTHON.exists(), reason="Airflow isolated venv not found")
    def test_task_dependencies_and_retries(self):
        """Verify strict ordering and retry configurations on all tasks."""
        check_code = """
import sys
from pathlib import Path
root = Path(r'D:\\setup\\recommendation-engine')
sys.path.insert(0, str(root))

from airflow.dags.recommendation_pipeline_dag import recommendation_pipeline_dag

dag = recommendation_pipeline_dag
task_map = {t.task_id: t for t in dag.tasks}

# Check default retries
for t in dag.tasks:
    assert t.retries == 2, f"Task {t.task_id} retries={t.retries}, expected 2"

# Check sequential ordering
expected_chain = [
    "validate_data",
    "preprocess_data",
    "train_retrieval",
    "generate_embeddings",
    "build_faiss_index",
    "train_ranking",
    "evaluate_models",
    "quality_gate",
    "validate_artifacts",
    "publish_models",
]

for i in range(len(expected_chain) - 1):
    upstream_id = expected_chain[i]
    downstream_id = expected_chain[i + 1]
    downstream_tasks = [d.task_id for d in task_map[upstream_id].downstream_list]
    assert downstream_id in downstream_tasks, f"Missing dependency: {upstream_id} -> {downstream_id}"

print("DEPENDENCIES_VERIFIED")
"""
        env = os.environ.copy()
        env["AIRFLOW_HOME"] = str(AIRFLOW_HOME)
        proc = subprocess.run(
            [str(AIRFLOW_PYTHON), "-c", check_code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, f"Dependency check failed:\n{proc.stderr}\n{proc.stdout}"
        assert "DEPENDENCIES_VERIFIED" in proc.stdout


class TestPipelineTasks:
    """Test individual pipeline task logic, quality gates, and failure modes."""

    def test_validate_input_data_success(self):
        """Verify input data validation succeeds on the current processed dataset."""
        result = validate_input_data()
        assert result["status"] == "VALID"
        assert "train" in result["partitions"]
        assert "val" in result["partitions"]
        assert "test" in result["partitions"]

    def test_validate_input_data_missing_dir(self):
        """Verify input validation raises FileNotFoundError on missing path."""
        with pytest.raises(FileNotFoundError):
            validate_input_data(processed_dir="data/non_existent_dir_12345")

    def test_validate_model_artifacts_success(self):
        """Verify all existing production artifacts satisfy size constraints."""
        result = validate_model_artifacts()
        assert result["status"] == "SUCCESS"
        assert "retrieval/model.weights.h5" in result["verified_artifacts"]
        assert "faiss/faiss_index.bin" in result["verified_artifacts"]
        assert "ranking/wide_and_deep/model.weights.h5" in result["verified_artifacts"]

    def test_validate_model_artifacts_missing_artifact(self):
        """Verify artifact validation fails if a required model is absent."""
        fake_specs = {"non_existent_model/weights.h5": 1000}
        with pytest.raises(FileNotFoundError):
            validate_model_artifacts(required_specs=fake_specs)

    def test_validate_model_artifacts_undersized_artifact(self):
        """Verify artifact validation fails if an artifact is suspiciously undersized."""
        # Current wide_only is ~300KB; setting required to 100MB should fail gate
        unrealistic_specs = {"ranking/wide_only/model.weights.h5": 100_000_000}
        with pytest.raises(ValueError, match="is below required threshold"):
            validate_model_artifacts(required_specs=unrealistic_specs)

    def test_evaluate_quality_gate_success(self):
        """Verify quality gate passes with existing experiment results."""
        result = evaluate_quality_gate(
            min_ndcg_at_10=0.10,
            min_precision_at_10=0.10,
            min_catalog_coverage=500,
        )
        assert result["status"] == "PASSED"
        assert len(result["ranking_gate"]) > 0

    def test_evaluate_quality_gate_failure_on_high_threshold(self):
        """Verify quality gate blocks publishing when metrics do not meet criteria."""
        with pytest.raises(ValueError, match="Quality gate failed"):
            evaluate_quality_gate(
                min_ndcg_at_10=0.99,  # Unattainable threshold
                min_precision_at_10=0.99,
            )

    def test_publish_model_version(self, tmp_path):
        """Verify atomic model version registration in the registry."""
        registry_file = tmp_path / "test_registry.json"
        res = publish_model_version(version="v2.0.0", registry_file=registry_file)
        assert res["status"] == "PUBLISHED"
        assert res["active_version"] == "v2.0.0"

        with open(registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["active_version"] == "v2.0.0"
        assert "v2.0.0" in data["versions"]
        assert data["versions"]["v2.0.0"]["status"] == "ACTIVE"
