"""
Unit and Integration Tests for CineMatch Memory Optimization, Model Reuse, and Fallback Behavior.

Tests cover:
  1. TwoTowerModel instance reuse in RankingFeaturePipeline (no duplicate load).
  2. Dataframe memory release in RankingFeaturePipeline after cache population.
  3. Compact train_consumed data representation (int32 numpy arrays).
  4. /health endpoint reports accurate models_loaded, serving_mode, and status.
  5. /recommend endpoint in lightweight mode (source='two_tower_retrieval', no ranking models).
  6. Graceful popularity fallback when user is cold-start or when models are unavailable.
  7. API response contracts match frontend expectations in all modes.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from configs.config import PROCESSED_DATA_DIR, RANKING_MODEL_DIR, RETRIEVAL_MODEL_DIR
from src.features.ranking_features import RankingFeaturePipeline
from src.retrieval.two_tower import TwoTowerModel
from src.serving.api import app, manager, ModelManager


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# 1. Model Reuse & Memory Lifecycle Tests
# -----------------------------------------------------------------------------
class TestModelReuseAndMemoryLifecycle:
    """Verify that models and data structures do not duplicate memory."""

    def test_pipeline_reuses_existing_two_tower_instance(self):
        """RankingFeaturePipeline.load must reuse an injected TwoTowerModel rather than reloading from disk."""
        mock_two_tower = MagicMock(spec=TwoTowerModel)
        mock_user_tower = MagicMock()
        # Mock user_tower forward pass returning (N, 64) tensor
        dummy_embs = tf_tensor = MagicMock()
        dummy_embs.numpy.return_value = np.zeros((10, 64), dtype=np.float32)
        mock_user_tower.return_value = dummy_embs
        mock_two_tower.user_tower = mock_user_tower

        pipeline_dir = RANKING_MODEL_DIR / "pipeline"
        if not pipeline_dir.exists():
            pytest.skip("Pipeline configuration artifacts not present on disk.")

        with patch("src.retrieval.two_tower.TwoTowerModel.load_artifacts") as mock_load:
            pipeline = RankingFeaturePipeline.load(
                save_dir=pipeline_dir,
                processed_data_dir=PROCESSED_DATA_DIR,
                retrieval_model_dir=RETRIEVAL_MODEL_DIR,
                two_tower_model=mock_two_tower,
            )
            # TwoTowerModel.load_artifacts should NOT be called when instance is provided
            mock_load.assert_not_called()
            # The mock user_tower should have been called to compute embeddings
            assert mock_user_tower.called

    def test_pipeline_frees_dataframes_after_init(self):
        """RankingFeaturePipeline must set user_features_df and movie_features_df to None to free memory."""
        pipeline_dir = RANKING_MODEL_DIR / "pipeline"
        if not pipeline_dir.exists():
            pytest.skip("Pipeline configuration artifacts not present.")

        pipeline = RankingFeaturePipeline.load(
            save_dir=pipeline_dir,
            processed_data_dir=PROCESSED_DATA_DIR,
            retrieval_model_dir=RETRIEVAL_MODEL_DIR,
            two_tower_model=manager.two_tower,
        )
        assert pipeline.user_features_df is None
        assert pipeline.movie_features_df is None
        # Lookup caches must remain populated
        assert len(pipeline._user_stats_cache) > 0
        assert len(pipeline._movie_stats_cache) > 0

    def test_train_consumed_stores_compact_numpy_arrays(self, client):
        """Train consumed mapping should hold numpy arrays of int32, not heavy Python sets."""
        if not manager.train_consumed:
            pytest.skip("Manager train_consumed is empty.")

        for uid, movie_ids in list(manager.train_consumed.items())[:20]:
            assert isinstance(movie_ids, np.ndarray)
            assert movie_ids.dtype == np.int32
            # On-the-fly set conversion works as expected
            s = set(movie_ids)
            assert isinstance(s, set)


# -----------------------------------------------------------------------------
# 2. Health Endpoint Tests
# -----------------------------------------------------------------------------
class TestHealthEndpointVerification:
    """Validate /health endpoint contract, model detection, and serving mode."""

    def test_health_reports_serving_mode(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "serving_mode" in data
        assert data["serving_mode"] in ("full", "lightweight")
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"]["two_tower"], bool)
        assert isinstance(data["models_loaded"]["faiss_index"], bool)
        assert isinstance(data["models_loaded"]["popularity"], bool)

    def test_health_reports_degraded_if_core_missing(self, client):
        """If core FAISS index is missing, health should report degraded."""
        original_faiss = manager.faiss_manager
        try:
            manager.faiss_manager = None
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["models_loaded"]["faiss_index"] is False
        finally:
            manager.faiss_manager = original_faiss


# -----------------------------------------------------------------------------
# 3. Recommendation Endpoint & Fallback Behavior Tests
# -----------------------------------------------------------------------------
class TestRecommendationAndFallbackBehavior:
    """Validate recommendation endpoint in various model availability modes."""

    def test_known_user_recommendations(self, client):
        """Warm user (e.g. '198') should return valid recommendations with scores and metadata."""
        resp = client.get("/recommend/198?k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "198"
        assert len(data["recommendations"]) == 5
        assert data["is_cold_start"] is False

        first = data["recommendations"][0]
        assert "movie_id" in first
        assert "title" in first
        assert "genres" in first
        assert "score" in first
        assert "rank" in first
        assert first["rank"] == 1
        assert first["source"] in ("two_stage_ranking", "two_tower_retrieval")

    def test_cold_start_fallback(self, client):
        """Cold-start user (e.g. '999999') must fall back to popularity baseline."""
        resp = client.get("/recommend/999999?k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_cold_start"] is True
        assert len(data["recommendations"]) == 5
        for rec in data["recommendations"]:
            assert rec["source"] == "popularity_fallback"

    def test_lightweight_mode_recommendation_attribution(self, client):
        """When ranking models are omitted/disabled, recommendations must be sourced as 'two_tower_retrieval'."""
        with patch("src.serving.api.LIGHTWEIGHT_MODE", True):
            resp = client.get("/recommend/198?k=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["variant"] == "two_tower_retrieval"
            for rec in data["recommendations"]:
                assert rec["source"] == "two_tower_retrieval"
                assert 0.0 <= rec["score"] <= 1.0 or -1.0 <= rec["score"] <= 1.0

    def test_graceful_fallback_when_retrieval_fails(self, client):
        """If retrieval fails unexpectedly, API must gracefully fall back to popularity without throwing HTTP 500."""
        with patch.object(manager.faiss_manager, "search", side_effect=RuntimeError("Simulated FAISS error")):
            resp = client.get("/recommend/198?k=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_cold_start"] is True
            assert len(data["recommendations"]) == 5
            for rec in data["recommendations"]:
                assert rec["source"] == "popularity_fallback"
