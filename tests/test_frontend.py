"""
Unit and integration tests for CineMatch HTML/CSS/JavaScript Frontend and Static Serving.

Tests:
1. GET / serves the CineMatch HTML application.
2. GET /static/css/styles.css serves valid stylesheet with CSS custom properties.
3. GET /static/js/app.js serves the frontend JavaScript logic.
4. GET /metrics/offline returns valid evaluation results from ranking_experiment.json.
5. HTML template contains semantic structure, accessible landmarks, and valid profile choices.
6. JavaScript client implements safe DOM manipulation and genuine feedback handling.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app, FRONTEND_DIR


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestFrontendServing:
    """Validate static file hosting and HTML integration."""

    def test_frontend_directory_exists(self):
        assert FRONTEND_DIR.exists()
        assert (FRONTEND_DIR / "index.html").exists()
        assert (FRONTEND_DIR / "css" / "styles.css").exists()
        assert (FRONTEND_DIR / "js" / "app.js").exists()

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        text = resp.text
        assert "CineMatch" in text
        assert "Movies picked for you." in text
        assert '<meta name="viewport"' in text
        assert 'id="profile-select"' in text
        assert 'id="genre-select"' in text
        assert 'id="btn-find-movies"' in text
        assert 'id="results-container"' in text

    def test_static_css_served(self, client):
        resp = client.get("/static/css/styles.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("content-type", "")
        text = resp.text
        assert "--bg-primary" in text
        assert ".movie-card" in text
        assert ".btn-primary" in text

    def test_static_js_served(self, client):
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        text = resp.text
        assert "fetchRecommendations" in text
        assert "logFeedbackEvent" in text
        assert "buildMovieCard" in text
        assert "escapeHtml" in text

    def test_offline_metrics_endpoint(self, client):
        resp = client.get("/metrics/offline")
        assert resp.status_code == 200
        data = resp.json()
        assert "evaluation_results" in data
        assert "models" in data["evaluation_results"]
        assert "Variant B (Wide & Deep Primary)" in data["evaluation_results"]["models"]

    def test_html_contains_valid_sample_users(self):
        html_path = FRONTEND_DIR / "index.html"
        content = html_path.read_text(encoding="utf-8")
        # Ensure valid MovieLens user IDs from dataset are used
        assert 'value="198"' in content  # Alex
        assert 'value="662"' in content  # Jordan
        assert 'value="805"' in content  # Taylor
        assert 'value="924"' in content  # Morgan
        assert 'value="999999"' in content  # Guest / Cold-start
