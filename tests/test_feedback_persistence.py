"""
Unit and integration tests for SQLite feedback persistence, watchlist management,
metadata resolution, score interpretations, and cold-start verification.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.serving.feedback import FeedbackEvent, FeedbackLogger
from src.serving.api import app, manager


@pytest.fixture
def temp_feedback_dir():
    """Create and tear down a temporary feedback directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_feedback_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestFeedbackPersistence:
    """Test suite for SQLite feedback logging and user watchlist persistence."""

    def test_sqlite_tables_created(self, temp_feedback_dir):
        """Verify feedback.db is created with required schema tables."""
        logger = FeedbackLogger(log_dir=temp_feedback_dir)
        db_file = temp_feedback_dir / "feedback.db"
        assert db_file.exists(), "feedback.db must be created"

        with logger._get_db_connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "feedback_events" in tables
            assert "user_watchlist" in tables

    def test_log_impression_and_click_persistence(self, temp_feedback_dir):
        """Verify impression and click events persist to SQLite and populate user_watchlist."""
        logger = FeedbackLogger(log_dir=temp_feedback_dir)

        # 1. Log impression
        imp_id = str(uuid.uuid4())
        imp_event = FeedbackEvent(
            event_id=imp_id,
            event_type="impression",
            user_id="test_user_42",
            movie_id="318",
            variant="wide_and_deep",
            request_id="req-123",
            model_version="v1.0",
        )
        res_imp = logger.log_event(imp_event)
        assert res_imp["status"] == "success"
        assert res_imp["persisted"] is True

        # 2. Log click attributed to impression
        clk_id = str(uuid.uuid4())
        clk_event = FeedbackEvent(
            event_id=clk_id,
            event_type="click",
            user_id="test_user_42",
            movie_id="318",
            variant="wide_and_deep",
            request_id="req-123",
            impression_id=imp_id,
            model_version="v1.0",
        )
        res_clk = logger.log_event(clk_event)
        assert res_clk["status"] == "success"
        assert res_clk["persisted"] is True
        assert res_clk["verified"] is True

        # 3. Verify watchlist contains movie 318
        watchlist = logger.get_user_watchlist("test_user_42")
        assert len(watchlist) == 1
        assert watchlist[0]["movie_id"] == "318"
        assert watchlist[0]["event_id"] == clk_id

    def test_duplicate_event_prevention(self, temp_feedback_dir):
        """Verify duplicate event IDs are rejected and raise ValueError."""
        logger = FeedbackLogger(log_dir=temp_feedback_dir)
        eid = str(uuid.uuid4())
        event = FeedbackEvent(
            event_id=eid,
            event_type="impression",
            user_id="user_100",
            movie_id="296",
        )
        logger.log_event(event)

        # Attempt duplicate insertion
        with pytest.raises(ValueError, match="Duplicate event_id"):
            logger.log_event(event)

    def test_remove_watchlist_item(self, temp_feedback_dir):
        """Verify removing a movie from watchlist updates SQLite correctly."""
        logger = FeedbackLogger(log_dir=temp_feedback_dir)
        clk_id = str(uuid.uuid4())
        clk_event = FeedbackEvent(
            event_id=clk_id,
            event_type="click",
            user_id="user_toggle",
            movie_id="356",
        )
        logger.log_event(clk_event)

        assert len(logger.get_user_watchlist("user_toggle")) == 1

        # Remove movie
        removed = logger.remove_watchlist_item("user_toggle", "356")
        assert removed is True
        assert len(logger.get_user_watchlist("user_toggle")) == 0

        # Second removal should return False
        removed_again = logger.remove_watchlist_item("user_toggle", "356")
        assert removed_again is False

    def test_persistence_survives_backend_restart(self, temp_feedback_dir):
        """Simulate backend restart by creating a new FeedbackLogger pointing to existing directory."""
        # Session 1: Log events
        logger1 = FeedbackLogger(log_dir=temp_feedback_dir)
        clk_id = str(uuid.uuid4())
        logger1.log_event(
            FeedbackEvent(
                event_id=clk_id,
                event_type="click",
                user_id="user_survive",
                movie_id="593",
            )
        )
        assert len(logger1.get_user_watchlist("user_survive")) == 1

        # Session 2: Fresh instance (simulating restart)
        logger2 = FeedbackLogger(log_dir=temp_feedback_dir)
        watchlist2 = logger2.get_user_watchlist("user_survive")
        assert len(watchlist2) == 1
        assert watchlist2[0]["movie_id"] == "593"
        assert watchlist2[0]["event_id"] == clk_id

        # Verify duplicate prevention also survives restart
        with pytest.raises(ValueError, match="Duplicate event_id"):
            logger2.log_event(
                FeedbackEvent(
                    event_id=clk_id,
                    event_type="click",
                    user_id="user_survive",
                    movie_id="593",
                )
            )


class TestServingApiVerification:
    """Test suite for FastAPI endpoints with real model manager state."""

    @classmethod
    def setup_class(cls):
        manager.initialize()
        cls.client = TestClient(app)

    def test_user_1_cold_start_behavior(self):
        """Verify user 1 (not in sampled training cohort) returns cold-start popularity baseline."""
        res = self.client.get("/recommend/1?k=5")
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "1"
        assert data["is_cold_start"] is True
        assert len(data["recommendations"]) == 5
        # All items must have source 'popularity_fallback'
        for item in data["recommendations"]:
            assert item["source"] == "popularity_fallback"
            # Popularity score is rating volume * avg stars (> 100.0)
            assert item["score"] > 0.0

    def test_user_198_personalized_ranking(self):
        """Verify known user 198 returns personalized two-stage neural ranking."""
        res = self.client.get("/recommend/198?k=5")
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "198"
        assert data["is_cold_start"] is False
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["source"] == "two_stage_ranking"
            # Score is sigmoid probability in [0, 1]
            assert 0.0 <= item["score"] <= 1.0

    def test_movie_metadata_resolution_no_generic_movie_318(self):
        """Verify movie 318 is returned with actual title 'Shawshank Redemption, The (1994)'."""
        res = self.client.get("/popular?limit=10")
        assert res.status_code == 200
        data = res.json()
        movies = data["movies"]
        # Find movie 318
        m318 = next((m for m in movies if m["movie_id"] == "318"), None)
        assert m318 is not None, "Movie 318 should be in top popular movies"
        assert "Shawshank Redemption" in m318["title"], f"Expected real title, got '{m318['title']}'"
        assert not m318["title"].startswith("Movie 318"), "Generic placeholder 'Movie 318' must be replaced"

    def test_feedback_watchlist_api_lifecycle(self):
        """Test complete API lifecycle: save movie, retrieve watchlist, remove movie."""
        test_uid = f"test_api_user_{uuid.uuid4().hex[:6]}"
        test_mid = "2571"  # The Matrix (1999)

        # 1. Post click feedback
        ev_id = str(uuid.uuid4())
        res = self.client.post(
            "/feedback",
            json={
                "event_id": ev_id,
                "event_type": "click",
                "user_id": test_uid,
                "movie_id": test_mid,
                "variant": "wide_and_deep",
            },
        )
        assert res.status_code == 200
        assert res.json()["status"] == "success"

        # 2. Get watchlist for user
        res_wl = self.client.get(f"/feedback/{test_uid}")
        assert res_wl.status_code == 200
        wl_data = res_wl.json()
        assert wl_data["user_id"] == test_uid
        assert test_mid in wl_data["saved_movie_ids"]
        assert len(wl_data["watchlist"]) == 1
        assert "Matrix" in wl_data["watchlist"][0]["title"]

        # 3. Delete from watchlist
        res_del = self.client.delete(f"/feedback/{test_uid}/{test_mid}")
        assert res_del.status_code == 200
        assert res_del.json()["removed"] is True

        # 4. Verify watchlist is now empty
        res_wl2 = self.client.get(f"/feedback/{test_uid}")
        assert res_wl2.status_code == 200
        assert test_mid not in res_wl2.json()["saved_movie_ids"]
        assert len(res_wl2.json()["watchlist"]) == 0

        # 5. Clean up test event record to maintain zero database contamination
        if manager.feedback_logger:
            with manager.feedback_logger._lock:
                with manager.feedback_logger._get_db_connection() as conn:
                    conn.execute("DELETE FROM feedback_events WHERE user_id = ?", (test_uid,))
                    conn.commit()
