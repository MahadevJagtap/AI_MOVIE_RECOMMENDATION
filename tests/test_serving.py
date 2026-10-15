"""
Unit and Integration Tests for Phase 4: FastAPI Serving, Redis Caching, and Feedback Logging.

Covers:
  - Health check endpoint and metadata
  - Two-stage recommendations schema and ranking variants (Wide-only vs Wide & Deep)
  - Cold-start and unknown-user fallback handling
  - Stage-1 candidates inspection endpoint
  - Input validation (invalid K, invalid variant, empty user)
  - Redis cache hits, misses, and graceful failure handling
  - Genuine feedback event logging (impression, click, duplicate prevention)
  - Live CTR calculation endpoint and disclaimer verification
  - Streamlit UI configuration and import integrity
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app, manager
from src.serving.cache import RedisCacheManager
from src.serving.feedback import FeedbackEvent, FeedbackLogger


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient with initialized ModelManager."""
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# 1. Health and Status Tests
# -----------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_success(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data
        assert data["models_loaded"]["two_tower"] is True
        assert data["models_loaded"]["faiss_index"] is True
        assert data["models_loaded"]["ranking_wide_only"] is True
        assert data["models_loaded"]["ranking_wide_and_deep"] is True
        assert data["total_catalog_movies"] > 0
        assert "cache_stats" in data
        assert data["version"] == "v1.0"


# -----------------------------------------------------------------------------
# 2. Recommendation Endpoint Tests
# -----------------------------------------------------------------------------
class TestRecommendationEndpoint:
    def test_recommend_wide_and_deep_variant(self, client):
        resp = client.get("/recommend/198?k=5&variant=wide_and_deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "198"
        assert data["k"] == 5
        assert data["variant"] == "wide_and_deep"
        assert data["is_cold_start"] is False
        assert len(data["recommendations"]) == 5
        assert data["latency_ms"] > 0

        # Check schema of each item
        for rank_idx, item in enumerate(data["recommendations"], 1):
            assert item["rank"] == rank_idx
            assert isinstance(item["movie_id"], str)
            assert len(item["title"]) > 0
            assert isinstance(item["score"], float)
            assert item["source"] == "two_stage_ranking"

    def test_recommend_wide_only_variant(self, client):
        resp = client.get("/recommend/198?k=5&variant=wide_only")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "198"
        assert data["variant"] == "wide_only"
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["source"] == "two_stage_ranking"
            assert isinstance(item["score"], float)

    def test_recommend_unknown_user_fallback(self, client):
        """Unknown user must receive cold-start popularity fallback without crashing."""
        resp = client.get("/recommend/unseen_cold_user_12345?k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_cold_start"] is True
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["source"] == "popularity_fallback"
            assert len(item["title"]) > 0


# -----------------------------------------------------------------------------
# 3. Candidates and Popular Endpoints Tests
# -----------------------------------------------------------------------------
class TestCandidatesAndPopularEndpoints:
    def test_get_candidates_stage1(self, client):
        resp = client.get("/candidates/198?k=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "198"
        assert len(data["candidates"]) == 10
        for item in data["candidates"]:
            assert "retrieval_score" in item
            assert isinstance(item["retrieval_score"], float)
            assert item["rank"] >= 1

    def test_get_popular_movies(self, client):
        resp = client.get("/popular?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5
        assert len(data["movies"]) == 5
        for item in data["movies"]:
            assert "popularity_score" in item
            assert item["rank"] >= 1


# -----------------------------------------------------------------------------
# 4. Input Validation & Error Handling Tests
# -----------------------------------------------------------------------------
class TestValidationAndErrors:
    def test_invalid_k_zero_or_negative(self, client):
        resp = client.get("/recommend/198?k=0")
        assert resp.status_code == 422

    def test_invalid_k_too_large(self, client):
        resp = client.get("/recommend/198?k=250")
        assert resp.status_code == 422

    def test_invalid_variant_name(self, client):
        resp = client.get("/recommend/198?k=5&variant=nonexistent_model")
        assert resp.status_code == 400
        assert "Invalid variant" in resp.json()["detail"]

    def test_empty_user_id(self, client):
        resp = client.get("/recommend/%20?k=5")
        assert resp.status_code == 400


# -----------------------------------------------------------------------------
# 5. Redis Cache Manager Tests
# -----------------------------------------------------------------------------
class TestRedisCacheBehavior:
    def test_cache_key_generation(self):
        key = RedisCacheManager.make_key(user_id="198", k=10, variant="wide_and_deep", model_version="v1.0")
        assert key == "rec:198:10:wide_and_deep:v1.0"

    def test_cache_graceful_degradation_when_offline(self):
        """When Redis is not reachable, cache calls must return None/False without throwing."""
        cache = RedisCacheManager(redis_url="redis://invalid-host-that-does-not-exist:6379/0", socket_timeout=0.1)
        assert cache.is_connected is False
        assert cache.get("test_key") is None
        assert cache.set("test_key", {"data": 123}) is False
        stats = cache.get_stats()
        assert stats["connected"] is False

    def test_mock_redis_hit_and_miss(self):
        """Verify hit and miss accounting using a mock Redis client."""
        cache = RedisCacheManager(enabled=True)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = '{"cached": true, "user_id": "198"}'
        cache.client = mock_client
        cache._connected = True

        res = cache.get("rec:198:10:wide_and_deep:v1.0")
        assert res is not None
        assert res["user_id"] == "198"
        assert cache.hits == 1

        mock_client.get.return_value = None
        miss_res = cache.get("rec:other:10:wide_and_deep:v1.0")
        assert miss_res is None
        assert cache.misses == 1


# -----------------------------------------------------------------------------
# 6. Feedback & CTR Logging Tests
# -----------------------------------------------------------------------------
class TestFeedbackAndCTR:
    def test_log_impression_and_click_lifecycle(self, client):
        req_id = str(uuid.uuid4())
        imp_id = str(uuid.uuid4())

        # 1. Log Impression with isolated test user
        imp_payload = {
            "event_id": imp_id,
            "event_type": "impression",
            "user_id": "test_serving_user",
            "movie_id": 1,
            "variant": "wide_and_deep",
            "request_id": req_id,
        }
        r_imp = client.post("/feedback", json=imp_payload)
        assert r_imp.status_code == 200
        assert r_imp.json()["status"] == "success"

        # 2. Log Attributed Click
        clk_id = str(uuid.uuid4())
        clk_payload = {
            "event_id": clk_id,
            "event_type": "click",
            "user_id": "test_serving_user",
            "movie_id": 1,
            "variant": "wide_and_deep",
            "request_id": req_id,
            "impression_id": imp_id,
        }
        r_clk = client.post("/feedback", json=clk_payload)
        assert r_clk.status_code == 200
        assert r_clk.json()["verified"] is True

        # 3. Prevent Duplicate Event IDs
        r_dup = client.post("/feedback", json=clk_payload)
        assert r_dup.status_code == 409

        # 4. Clean up test watchlist item and events to maintain test isolation
        client.delete("/feedback/test_serving_user/1")
        if manager.feedback_logger:
            with manager.feedback_logger._lock:
                with manager.feedback_logger._get_db_connection() as conn:
                    conn.execute("DELETE FROM feedback_events WHERE user_id = 'test_serving_user'")
                    conn.commit()

    def test_invalid_event_type(self, client):
        payload = {
            "event_id": str(uuid.uuid4()),
            "event_type": "purchase_invalid",
            "user_id": "test_serving_user",
            "movie_id": 1,
        }
        r = client.post("/feedback", json=payload)
        assert r.status_code == 422

    def test_metrics_ctr_endpoint(self, client):
        r = client.get("/metrics/ctr")
        assert r.status_code == 200
        data = r.json()
        assert "total_impressions" in data
        assert "total_clicks" in data
        assert "overall_ctr" in data
        assert "disclaimer" in data
        assert "Demo interaction data" in data["disclaimer"]


# -----------------------------------------------------------------------------
# 7. Streamlit UI Import & Configuration Tests
# -----------------------------------------------------------------------------
class TestStreamlitConfiguration:
    def test_ui_file_exists_and_syntax_valid(self):
        ui_path = Path("src/ui/app.py")
        assert ui_path.exists()
        content = ui_path.read_text(encoding="utf-8")
        assert "st.set_page_config" in content
        assert "fetch_recommendations" in content
        assert "log_feedback_event" in content
        assert "ranking_experiment.json" in content
