"""
Unit and integration tests for NVIDIA Triton Inference Server compatibility and client.

Tests:
1. Triton Model Repository structure, SavedModel exports, and config.pbtxt schemas.
2. TritonInferenceClient offline behavior, timeout handling, and local fallback delegation.
3. Live Triton HTTP v2 inference requests for user_tower and ranking_model.
4. Tensor shape, data type, and numerical value parity between Triton and local TensorFlow.
5. FastAPI integration with Triton health reporting and fallback recommendation serving.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from configs.config import (
    TRITON_MODEL_REPOSITORY,
    RETRIEVAL_MODEL_DIR,
    RANKING_MODEL_DIR,
    PROJECT_ROOT,
)
from src.retrieval.two_tower import TwoTowerModel
from src.ranking.wide_and_deep import WideAndDeepModel
from src.serving.triton_client import TritonInferenceClient
from src.serving.api import app


class TestTritonModelRepository:
    """Validate Triton Model Repository directory structure and config files."""

    def test_user_tower_repository_structure(self):
        """Verify user_tower SavedModel export and config.pbtxt specification."""
        repo = Path(TRITON_MODEL_REPOSITORY)
        ut_dir = repo / "user_tower"
        assert ut_dir.exists(), f"user_tower directory missing in {repo}"

        config_path = ut_dir / "config.pbtxt"
        assert config_path.exists(), "user_tower config.pbtxt missing"
        cfg_text = config_path.read_text(encoding="utf-8")
        assert 'name: "user_tower"' in cfg_text
        assert 'platform: "tensorflow_savedmodel"' in cfg_text
        assert 'name: "user_id"' in cfg_text
        assert "TYPE_STRING" in cfg_text
        assert 'name: "user_embedding"' in cfg_text
        assert "TYPE_FP32" in cfg_text
        assert "dims: [ 64 ]" in cfg_text

        savedmodel_pb = ut_dir / "1" / "model.savedmodel" / "saved_model.pb"
        assert savedmodel_pb.exists(), f"user_tower saved_model.pb missing at {savedmodel_pb}"

    def test_ranking_model_repository_structure(self):
        """Verify ranking_model SavedModel export and config.pbtxt specification."""
        repo = Path(TRITON_MODEL_REPOSITORY)
        rk_dir = repo / "ranking_model"
        assert rk_dir.exists(), f"ranking_model directory missing in {repo}"

        config_path = rk_dir / "config.pbtxt"
        assert config_path.exists(), "ranking_model config.pbtxt missing"
        cfg_text = config_path.read_text(encoding="utf-8")
        assert 'name: "ranking_model"' in cfg_text
        assert 'platform: "tensorflow_savedmodel"' in cfg_text
        assert 'name: "userId"' in cfg_text
        assert 'name: "movieId"' in cfg_text
        assert 'name: "genres"' in cfg_text
        assert 'name: "activity_pop_cross"' in cfg_text
        assert 'name: "user_movie_rating_cross"' in cfg_text
        assert 'name: "dense_features"' in cfg_text
        assert 'name: "probabilities"' in cfg_text
        assert "dims: [ 1 ]" in cfg_text

        savedmodel_pb = rk_dir / "1" / "model.savedmodel" / "saved_model.pb"
        assert savedmodel_pb.exists(), f"ranking_model saved_model.pb missing at {savedmodel_pb}"


class TestTritonInferenceClientUnit:
    """Test client connection logic, offline handling, and fallback behavior."""

    def test_client_offline_availability(self):
        """Verify is_available() returns False when server is offline without raising."""
        client = TritonInferenceClient(url="127.0.0.1:9999", enabled=True, connection_timeout=0.2)
        assert client.is_available(force_check=True) is False

    def test_infer_user_embedding_offline_raises(self):
        """Verify strict infer_user_embedding raises ConnectionError when offline."""
        client = TritonInferenceClient(url="127.0.0.1:9999", enabled=True, connection_timeout=0.2)
        with pytest.raises(ConnectionError, match="not available"):
            client.infer_user_embedding(["198"])

    def test_user_embedding_fallback_to_local(self):
        """Verify infer_user_embedding_with_fallback delegates to local model when offline."""
        client = TritonInferenceClient(url="127.0.0.1:9999", enabled=True, connection_timeout=0.2)
        retrieval_dir = Path(RETRIEVAL_MODEL_DIR)
        two_tower = TwoTowerModel.load_artifacts(retrieval_dir)

        embs, source = client.infer_user_embedding_with_fallback(
            user_ids=["198", "662"],
            local_user_tower=two_tower.user_tower,
        )
        assert source == "local_tensorflow"
        assert embs.shape == (2, 64)
        assert embs.dtype == np.float32

    def test_ranking_scores_fallback_to_local(self):
        """Verify infer_ranking_scores_with_fallback delegates to local model when offline."""
        client = TritonInferenceClient(url="127.0.0.1:9999", enabled=True, connection_timeout=0.2)
        wd_dir = Path(RANKING_MODEL_DIR) / "wide_and_deep"
        local_wd = WideAndDeepModel.load_artifacts(wd_dir)

        features = {
            "userId": np.array(["198", "662"], dtype=object),
            "movieId": np.array(["1", "356"], dtype=object),
            "genres": np.array(["Adventure|Animation", "Comedy|Drama"], dtype=object),
            "activity_pop_cross": np.array([0, 1], dtype=np.int32),
            "user_movie_rating_cross": np.array([2, 3], dtype=np.int32),
            "dense_features": np.zeros((2, 5), dtype=np.float32),
        }
        scores, source = client.infer_ranking_scores_with_fallback(
            features=features,
            local_ranking_model=local_wd,
        )
        assert source == "local_tensorflow"
        assert scores.shape == (2,)
        assert scores.dtype == np.float32


class TestTritonLiveInferenceParity:
    """Test real HTTP inference against Triton server and verify numerical parity."""

    @pytest.fixture(scope="class")
    @classmethod
    def triton_server_process(cls):
        """Start Triton HTTP v2 protocol server on port 8002 for isolated testing."""
        port = 8002
        cmd = [
            sys.executable,
            "-m", "src.serving.triton_server_runner",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        env = os.environ.copy()
        env["TF_USE_LEGACY_KERAS"] = "1"
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)

        client = TritonInferenceClient(url=f"127.0.0.1:{port}", enabled=True, connection_timeout=1.0)
        ready = False
        for _ in range(15):
            time.sleep(1.0)
            if client.is_available(force_check=True):
                ready = True
                break

        if not ready:
            proc.kill()
            pytest.skip(f"Could not start Triton server on port {port}")

        yield f"127.0.0.1:{port}"

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_live_user_tower_inference_and_parity(self, triton_server_process):
        """Verify live user_tower inference returns correct shape, dtype, and identical numbers."""
        client = TritonInferenceClient(url=triton_server_process, enabled=True)
        assert client.is_available(force_check=True) is True

        test_uids = ["198", "662", "805", "924"]

        # Triton inference
        triton_embs = client.infer_user_embedding(test_uids)
        assert triton_embs.shape == (4, 64)
        assert triton_embs.dtype == np.float32

        # Local inference
        two_tower = TwoTowerModel.load_artifacts(Path(RETRIEVAL_MODEL_DIR))
        local_embs = two_tower.user_tower(tf.constant(test_uids, dtype=tf.string)).numpy()

        # Strict numerical parity check
        assert np.allclose(triton_embs, local_embs, atol=1e-5)

    def test_live_ranking_model_inference_and_parity(self, triton_server_process):
        """Verify live ranking_model inference returns correct shape, dtype, and identical numbers."""
        client = TritonInferenceClient(url=triton_server_process, enabled=True)
        assert client.is_available(force_check=True) is True

        features = {
            "userId": np.array(["198", "198", "662", "662"], dtype=object),
            "movieId": np.array(["1", "356", "296", "593"], dtype=object),
            "genres": np.array([
                "Adventure|Animation|Children",
                "Comedy|Drama|Romance",
                "Crime|Drama",
                "Crime|Horror|Thriller",
            ], dtype=object),
            "activity_pop_cross": np.array([0, 1, 2, 3], dtype=np.int32),
            "user_movie_rating_cross": np.array([1, 2, 0, 3], dtype=np.int32),
            "dense_features": np.array([
                [0.1, -0.2, 0.5, 1.2, 0.8],
                [0.4, 0.1, -0.4, 0.2, -0.1],
                [-0.3, 0.8, 0.1, -0.5, 0.3],
                [0.2, -0.1, 0.7, 0.3, 0.0],
            ], dtype=np.float32),
        }

        # Triton inference
        triton_scores = client.infer_ranking_scores(features)
        assert triton_scores.shape == (4,)
        assert triton_scores.dtype == np.float32

        # Local inference
        local_wd = WideAndDeepModel.load_artifacts(Path(RANKING_MODEL_DIR) / "wide_and_deep")
        local_scores = local_wd.predict_scores(features)

        # Strict numerical parity check
        assert np.allclose(triton_scores, local_scores, atol=1e-5)


class TestFastAPITritonIntegration:
    """Test FastAPI application with Triton client availability and fallback."""

    def test_health_endpoint_reports_triton_status(self):
        """Verify /health reports triton_server status in models_loaded dictionary."""
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "triton_server" in data["models_loaded"]
            assert isinstance(data["models_loaded"]["triton_server"], bool)

    def test_recommendation_serving_unaffected_when_triton_offline(self):
        """Verify /recommend serves valid recommendations using local models when Triton is offline."""
        with TestClient(app) as client:
            resp = client.get("/recommend/198?k=5&variant=wide_and_deep")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 5
            assert data["variant"] == "wide_and_deep"
            assert data["is_cold_start"] is False
