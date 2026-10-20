"""
Unit and Integration Tests for OMDb Movie Poster Integration (Phase 4 Extension).

Tests cover:
  1. Title and year normalization for diverse MovieLens titles.
  2. Successful OMDb response with valid poster URL.
  3. Movie with no available poster (Poster: "N/A" or not found).
  4. Invalid API key or OMDb error response.
  5. Timeout or network failure graceful handling.
  6. Duplicate requests hitting memory and SQLite persistent cache (no duplicate API calls).
  7. Correct matching using movie title and year fallback.
  8. End-to-end: poster retrieval failure without breaking /recommend endpoint.
  9. Dedicated /poster/{movie_id} and /posters/batch endpoints.
  10. Frontend static assets contain poster handling, aspect ratio, and fallback logic.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app, manager
from src.serving.posters import MoviePosterService, clean_movielens_title


# -----------------------------------------------------------------------------
# 1. Title Normalization Tests
# -----------------------------------------------------------------------------
class TestTitleNormalization:
    def test_standard_title_with_year(self):
        title, year = clean_movielens_title("Shawshank Redemption, The (1994)")
        assert title == "The Shawshank Redemption"
        assert year == "1994"

    def test_article_the_suffix(self):
        title, year = clean_movielens_title("Matrix, The (1999)")
        assert title == "The Matrix"
        assert year == "1999"

    def test_article_a_suffix(self):
        title, year = clean_movielens_title("Beautiful Mind, A (2001)")
        assert title == "A Beautiful Mind"
        assert year == "2001"

    def test_subtitles_and_sequels(self):
        title, year = clean_movielens_title("Godfather: Part II, The (1974)")
        assert title == "The Godfather: Part II"
        assert year == "1974"

    def test_foreign_title_in_parentheses(self):
        title, year = clean_movielens_title("City of God (Cidade de Deus) (2002)")
        assert title == "City of God"
        assert year == "2002"

    def test_aka_in_title(self):
        title, year = clean_movielens_title("Léon: The Professional (a.k.a. The Professional) (1994)")
        assert "Léon: The Professional" in title
        assert year == "1994"

    def test_empty_or_none_title(self):
        assert clean_movielens_title("") == ("", "")
        assert clean_movielens_title(None) == ("", "")


# -----------------------------------------------------------------------------
# 2. MoviePosterService Unit Tests (Mocked OMDb)
# -----------------------------------------------------------------------------
class TestMoviePosterService:
    @pytest.fixture
    def temp_cache_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def poster_service(self, temp_cache_dir):
        # Service initialized with mock key and isolated temp cache directory
        return MoviePosterService(
            api_key="mock_test_key",
            timeout=2.0,
            cache_dir=temp_cache_dir,
            links_data={"318": "tt0111161", "296": "tt0110912"},
        )

    def test_successful_omdb_response(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "Response": "True",
                "Title": "The Shawshank Redemption",
                "Year": "1994",
                "Poster": "https://m.media-amazon.com/images/M/shawshank.jpg",
                "imdbID": "tt0111161",
            },
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_response) as mock_get:
            url = poster_service.get_poster("318", title="Shawshank Redemption, The (1994)")
            assert url == "https://m.media-amazon.com/images/M/shawshank.jpg"
            assert mock_get.call_count == 1
            # Verify IMDb ID was passed in params
            called_params = mock_get.call_args[1]["params"]
            assert called_params["i"] == "tt0111161"
            assert called_params["apikey"] == "mock_test_key"

    def test_movie_with_no_available_poster_na(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "Response": "True",
                "Title": "Indie Short",
                "Poster": "N/A",
                "imdbID": "tt9999999",
            },
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_response):
            url = poster_service.get_poster("999999", title="Indie Short (2020)")
            assert url is None

    def test_movie_not_found_response(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={"Response": "False", "Error": "Movie not found!"},
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_response):
            url = poster_service.get_poster("888888", title="Nonexistent Movie (2099)")
            assert url is None

    def test_invalid_api_key_error(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={"Response": "False", "Error": "Invalid API key!"},
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_response):
            url = poster_service.get_poster("318", title="Shawshank Redemption, The (1994)")
            assert url is None

    def test_timeout_and_network_failure(self, poster_service):
        with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("Read timed out")):
            url = poster_service.get_poster("318", title="Shawshank Redemption, The (1994)")
            assert url is None

        with patch.object(httpx.Client, "get", side_effect=httpx.ConnectError("Network unreachable")):
            url = poster_service.get_poster("296", title="Pulp Fiction (1994)")
            assert url is None

    def test_caching_prevents_duplicate_api_calls(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "Response": "True",
                "Title": "Pulp Fiction",
                "Poster": "https://m.media-amazon.com/images/M/pulp.jpg",
                "imdbID": "tt0110912",
            },
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_response) as mock_get:
            # First lookup: calls OMDb
            url1 = poster_service.get_poster("296", title="Pulp Fiction (1994)")
            assert url1 == "https://m.media-amazon.com/images/M/pulp.jpg"
            assert mock_get.call_count == 1

            # Second lookup: in-memory cache hit (no API call)
            url2 = poster_service.get_poster("296")
            assert url2 == "https://m.media-amazon.com/images/M/pulp.jpg"
            assert mock_get.call_count == 1

            # Third lookup: new service instance reading from same SQLite DB
            service_fresh = MoviePosterService(
                api_key="mock_test_key",
                cache_dir=poster_service.cache_dir,
                links_data={"296": "tt0110912"},
            )
            url3 = service_fresh.get_poster("296")
            assert url3 == "https://m.media-amazon.com/images/M/pulp.jpg"
            assert mock_get.call_count == 1

    def test_batch_lookup_with_mixed_cache(self, poster_service):
        mock_response = httpx.Response(
            status_code=200,
            json={
                "Response": "True",
                "Title": "Forrest Gump",
                "Poster": "https://m.media-amazon.com/images/M/gump.jpg",
            },
            request=httpx.Request("GET", "https://www.omdbapi.com/"),
        )

        # Preload one item into cache
        poster_service._memory_cache["318"] = "https://cached-url.jpg"

        items = [
            {"movie_id": "318", "title": "Shawshank Redemption, The (1994)"},
            {"movie_id": "356", "title": "Forrest Gump (1994)"},
        ]

        with patch.object(httpx.Client, "get", return_value=mock_response) as mock_get:
            results = poster_service.get_posters_batch(items)
            assert results["318"] == "https://cached-url.jpg"
            assert results["356"] == "https://m.media-amazon.com/images/M/gump.jpg"
            assert mock_get.call_count == 1


# -----------------------------------------------------------------------------
# 3. Integration Tests with FastAPI TestClient
# -----------------------------------------------------------------------------
class TestServingPosterIntegration:
    @pytest.fixture(scope="module")
    def client(self):
        with TestClient(app) as test_client:
            yield test_client

    def test_recommendation_includes_poster_url_field(self, client):
        resp = client.get("/recommend/198?k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) == 3
        for item in data["recommendations"]:
            assert "poster_url" in item
            # poster_url is either None or valid URL string
            assert item["poster_url"] is None or item["poster_url"].startswith("http")

    def test_recommendation_never_fails_if_poster_service_raises(self, client):
        # Patch poster_service to raise an unexpected error
        with patch.object(manager.poster_service, "get_posters_batch", side_effect=RuntimeError("Simulated OMDb crash")):
            resp = client.get("/recommend/198?k=3")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 3

    def test_popular_endpoint_includes_poster_url(self, client):
        resp = client.get("/popular?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["movies"]) == 3
        for item in data["movies"]:
            assert "poster_url" in item

    def test_single_poster_endpoint(self, client):
        resp = client.get("/poster/318?title=The%20Shawshank%20Redemption")
        assert resp.status_code == 200
        data = resp.json()
        assert data["movie_id"] == "318"
        assert "poster_url" in data

    def test_batch_poster_endpoint(self, client):
        resp = client.post("/posters/batch", json={"movie_ids": ["318", "296"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "posters" in data
        assert "318" in data["posters"]
        assert "296" in data["posters"]


# -----------------------------------------------------------------------------
# 4. Frontend Asset Tests for Poster Integration
# -----------------------------------------------------------------------------
class TestFrontendPosterAssets:
    def test_frontend_css_contains_poster_styles(self):
        css_path = Path("frontend/css/styles.css")
        assert css_path.exists()
        css_text = css_path.read_text(encoding="utf-8")
        assert "aspect-ratio: 2 / 3" in css_text
        assert ".card-poster-img" in css_text
        assert ".card-poster-loader" in css_text
        assert ".card-poster-fallback" in css_text

    def test_frontend_js_contains_poster_handling(self):
        js_path = Path("frontend/js/app.js")
        assert js_path.exists()
        js_text = js_path.read_text(encoding="utf-8")
        assert "createCardPoster" in js_text
        assert "fetchPosterLazy" in js_text
        assert "posterCache" in js_text
        assert "img.onerror" in js_text
