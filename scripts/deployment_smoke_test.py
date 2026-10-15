"""
CineMatch Deployment Smoke Test.
Validates all live endpoints required for Render production demo:
  - System health and model readiness (/health)
  - Static frontend serving (/, /static/css/styles.css, /static/js/app.js)
  - Two-Stage Personalized Recommendations (/recommend/{user_id})
  - Ranking Variants (wide_and_deep vs wide_only)
  - Cold-Start Fallback (/recommend/1, /popular)
  - Stage 1 Candidate Inspection (/candidates/{user_id})
  - Feedback and Watchlist APIs (/feedback, /feedback/{user_id})
  - Live CTR Metrics (/metrics/ctr)
  - Offline Experiment Benchmarks (/metrics/offline)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import uuid

BASE_URL = os.environ.get("DEPLOYMENT_BASE_URL", "http://127.0.0.1:8000")

def req(path, method="GET", data=None):
    url = BASE_URL + path
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=10) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()
        if "application/json" in content_type:
            return resp.status, json.loads(raw.decode("utf-8"))
        return resp.status, raw.decode("utf-8")

def run_smoke_tests():
    print(f"=== Running CineMatch Deployment Smoke Tests on {BASE_URL} ===")
    passed = 0
    total = 0

    def test(name, fn):
        nonlocal passed, total
        total += 1
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # 1. Health & Model Readiness
    def check_health():
        status, data = req("/health")
        assert status == 200, f"Expected 200, got {status}"
        assert data["status"] in ("healthy", "degraded")
        assert data["models_loaded"]["two_tower"] is True, "Two-Tower not loaded"
        assert data["models_loaded"]["faiss_index"] is True, "FAISS not loaded"
        assert data["models_loaded"]["ranking_wide_only"] is True, "Wide-only not loaded"
        assert data["models_loaded"]["ranking_wide_and_deep"] is True, "Wide & Deep not loaded"
        assert data["total_catalog_movies"] > 60000, f"Low catalog size: {data['total_catalog_movies']}"
    test("GET /health (all ML models loaded & 62k+ catalog movies)", check_health)

    # 2. Frontend HTML
    def check_root_html():
        status, text = req("/")
        assert status == 200
        assert "CineMatch" in text, "Missing CineMatch in HTML"
        assert "Movies picked for you." in text, "Missing hero title"
        assert "id=\"profile-select\"" in text, "Missing profile selector"
    test("GET / (Frontend root HTML served)", check_root_html)

    # 3. Static CSS
    def check_css():
        status, text = req("/static/css/styles.css")
        assert status == 200
        assert "--bg-primary" in text, "Missing CSS variables"
    test("GET /static/css/styles.css (Static CSS stylesheet served)", check_css)

    # 4. Static JS
    def check_js():
        status, text = req("/static/js/app.js")
        assert status == 200
        assert "fetchRecommendations" in text, "Missing JS functions"
    test("GET /static/js/app.js (Frontend client logic served)", check_js)

    # 5. Personalized Recommendations (Wide & Deep)
    def check_rec_wd():
        status, data = req("/recommend/198?k=5&variant=wide_and_deep")
        assert status == 200
        assert data["user_id"] == "198"
        assert data["variant"] == "wide_and_deep"
        assert data["is_cold_start"] is False
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["source"] == "two_stage_ranking"
            assert not item["title"].startswith("Movie "), "Placeholder title detected"
            assert 0.0 <= item["score"] <= 1.0, f"Invalid score: {item['score']}"
    test("GET /recommend/198?k=5&variant=wide_and_deep (Personalized Neural Match)", check_rec_wd)

    # 6. Personalized Recommendations (Wide-only)
    def check_rec_wide():
        status, data = req("/recommend/198?k=5&variant=wide_only")
        assert status == 200
        assert data["user_id"] == "198"
        assert data["variant"] == "wide_only"
        assert len(data["recommendations"]) == 5
    test("GET /recommend/198?k=5&variant=wide_only (Personalized Wide Variant)", check_rec_wide)

    # 7. Cold-Start Fallback (User 1)
    def check_cold_start():
        status, data = req("/recommend/1?k=5")
        assert status == 200
        assert data["user_id"] == "1"
        assert data["is_cold_start"] is True
        assert len(data["recommendations"]) == 5
        for item in data["recommendations"]:
            assert item["source"] == "popularity_fallback"
            assert item["score"] > 100.0, "Expected raw popularity volume score"
    test("GET /recommend/1 (Cold-Start Popularity Fallback for Unobserved User)", check_cold_start)

    # 8. Stage 1 Candidates Inspection
    def check_candidates():
        status, data = req("/candidates/198?k=10")
        assert status == 200
        assert data["user_id"] == "198"
        assert len(data["candidates"]) == 10
        assert data["candidates"][0]["retrieval_score"] != 0.0
    test("GET /candidates/198?k=10 (Stage 1 FAISS Vector Search Inspection)", check_candidates)

    # 9. Popular Movies Endpoint
    def check_popular():
        status, data = req("/popular?limit=5")
        assert status == 200
        assert len(data["movies"]) == 5
        assert "Shawshank" in data["movies"][0]["title"]
    test("GET /popular?limit=5 (Global Catalog Popularity Endpoint)", check_popular)

    # 10. Feedback & Watchlist Lifecycle
    def check_feedback_lifecycle():
        t_user = f"smoke_user_{uuid.uuid4().hex[:6]}"
        t_movie = "318"
        
        # Add to watchlist
        st_p, res_p = req("/feedback", method="POST", data={
            "event_id": str(uuid.uuid4()),
            "event_type": "click",
            "user_id": t_user,
            "movie_id": t_movie,
            "variant": "wide_and_deep"
        })
        assert st_p == 200 and res_p["persisted"] is True

        # Query watchlist
        st_g, res_g = req(f"/feedback/{t_user}")
        assert st_g == 200 and t_movie in res_g["saved_movie_ids"]
        assert any("Shawshank" in m["title"] for m in res_g["watchlist"])

        # Delete
        st_d, res_d = req(f"/feedback/{t_user}/{t_movie}", method="DELETE")
        assert st_d == 200 and res_d["removed"] is True

        # Clean DB
        st_g2, res_g2 = req(f"/feedback/{t_user}")
        assert len(res_g2["watchlist"]) == 0

        # Purge event from SQLite to ensure zero test footprint
        try:
            import sqlite3
            conn = sqlite3.connect("data/feedback/feedback.db")
            conn.execute("DELETE FROM feedback_events WHERE user_id = ?", (t_user,))
            conn.commit()
            conn.close()
        except Exception:
            pass
    test("POST & GET & DELETE /feedback (Watchlist Storage & Toggle Lifecycle)", check_feedback_lifecycle)

    # 11. Live CTR Metrics
    def check_ctr():
        status, data = req("/metrics/ctr")
        assert status == 200
        assert "total_impressions" in data
        assert "total_clicks" in data
        assert "overall_ctr" in data
        assert "disclaimer" in data
    test("GET /metrics/ctr (Live Demo Interaction CTR Calculation)", check_ctr)

    # 12. Offline Benchmarks
    def check_offline():
        status, data = req("/metrics/offline")
        assert status == 200
        assert "evaluation_results" in data
        assert "models" in data["evaluation_results"]
    test("GET /metrics/offline (Offline Evaluation Benchmarks from JSON)", check_offline)

    print(f"\n=== Summary: {passed}/{total} Deployment Smoke Tests Passed ===")
    return passed == total

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
