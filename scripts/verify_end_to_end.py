"""
End-to-End System Verification Script.

Tests the full recommendation pipeline:
MovieLens processed data -> Two-Tower user embedding -> FAISS candidate retrieval
-> selected ranking model (Wide & Deep / Wide-only) -> filtering/fallback
-> Redis caching -> FastAPI responses -> feedback event logging.

Usage:
    python scripts/verify_end_to_end.py
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import time
import uuid
import json
import logging
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.api import app, manager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [e2e_verify]: %(message)s")
logger = logging.getLogger("e2e_verify")


def verify_full_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING END-TO-END PIPELINE VERIFICATION")
    logger.info("=" * 80)

    results = {}

    with TestClient(app) as client:
        # 1. Health Endpoint
        logger.info("\n---> [TEST 1/5] Checking /health endpoint...")
        t0 = time.perf_counter()
        health_resp = client.get("/health")
        h_latency = (time.perf_counter() - t0) * 1000
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        logger.info(f"Health Status: {health_data['status']} | Latency: {h_latency:.2f}ms")
        logger.info(f"Models Loaded: {health_data['models_loaded']}")
        logger.info(f"Catalog Movies: {health_data['total_catalog_movies']}")
        assert health_data["total_catalog_movies"] == 62423
        assert health_data["models_loaded"]["two_tower"] is True
        assert health_data["models_loaded"]["faiss_index"] is True
        assert health_data["models_loaded"]["ranking_wide_and_deep"] is True
        assert health_data["models_loaded"]["ranking_wide_only"] is True
        results["health"] = {"status": "PASS", "latency_ms": round(h_latency, 2)}

        # 2. Warm User Recommendations (Wide & Deep vs Wide-Only)
        logger.info("\n---> [TEST 2/5] Testing Warm User (User 198) with Wide & Deep and Wide-only...")
        test_uid = "198"

        # Variant B: Wide & Deep Primary
        t0 = time.perf_counter()
        wd_resp = client.get(f"/recommend/{test_uid}?k=10&variant=wide_and_deep")
        wd_latency = (time.perf_counter() - t0) * 1000
        assert wd_resp.status_code == 200, f"Wide & Deep request failed: {wd_resp.text}"
        wd_data = wd_resp.json()
        assert len(wd_data["recommendations"]) == 10
        assert wd_data["variant"] == "wide_and_deep"
        assert wd_data["is_cold_start"] is False
        assert wd_data["recommendations"][0]["source"] == "two_stage_ranking"
        # Verify scores are sorted descending
        wd_scores = [r["score"] for r in wd_data["recommendations"]]
        assert wd_scores == sorted(wd_scores, reverse=True)
        logger.info(f"Wide & Deep returned 10 items | Server Latency: {wd_data['latency_ms']}ms | Client Latency: {wd_latency:.2f}ms")
        logger.info(f"Top 3 W&D recommendations: {[(r['title'], r['score']) for r in wd_data['recommendations'][:3]]}")

        # Variant A: Wide-only Baseline
        t0 = time.perf_counter()
        wo_resp = client.get(f"/recommend/{test_uid}?k=10&variant=wide_only")
        wo_latency = (time.perf_counter() - t0) * 1000
        assert wo_resp.status_code == 200, f"Wide-only request failed: {wo_resp.text}"
        wo_data = wo_resp.json()
        assert len(wo_data["recommendations"]) == 10
        assert wo_data["variant"] == "wide_only"
        assert wo_data["is_cold_start"] is False
        wo_scores = [r["score"] for r in wo_data["recommendations"]]
        assert wo_scores == sorted(wo_scores, reverse=True)
        logger.info(f"Wide-only returned 10 items | Server Latency: {wo_data['latency_ms']}ms | Client Latency: {wo_latency:.2f}ms")
        logger.info(f"Top 3 Wide recommendations: {[(r['title'], r['score']) for r in wo_data['recommendations'][:3]]}")

        results["warm_user"] = {
            "status": "PASS",
            "wide_and_deep_latency_ms": round(wd_latency, 2),
            "wide_only_latency_ms": round(wo_latency, 2),
        }

        # 3. Stage 1 Candidates Inspection
        logger.info("\n---> [TEST 3/5] Inspecting Stage 1 Retrieval Candidates for User 198...")
        cand_resp = client.get(f"/candidates/{test_uid}?k=50")
        assert cand_resp.status_code == 200
        cand_data = cand_resp.json()
        assert len(cand_data["candidates"]) == 50
        assert cand_data["is_cold_start"] is False
        logger.info(f"Retrieved 50 Stage-1 candidates | Latency: {cand_data['retrieval_latency_ms']:.2f}ms")
        results["candidates"] = {"status": "PASS", "latency_ms": cand_data["retrieval_latency_ms"]}

        # 4. Cold-Start Unknown User Fallback
        logger.info("\n---> [TEST 4/5] Testing Cold-Start User (User 999999)...")
        cold_uid = "999999"
        t0 = time.perf_counter()
        cold_resp = client.get(f"/recommend/{cold_uid}?k=10")
        cold_latency = (time.perf_counter() - t0) * 1000
        assert cold_resp.status_code == 200
        cold_data = cold_resp.json()
        assert cold_data["is_cold_start"] is True
        assert len(cold_data["recommendations"]) == 10
        assert cold_data["recommendations"][0]["source"] == "popularity_fallback"
        logger.info(f"Cold-Start user successfully served via Popularity Recommender in {cold_latency:.2f}ms")
        logger.info(f"Top 3 Popular items: {[(r['title'], r['score']) for r in cold_data['recommendations'][:3]]}")
        results["cold_start"] = {"status": "PASS", "latency_ms": round(cold_latency, 2)}

        # 5. Feedback Logging & Live CTR Reporting
        logger.info("\n---> [TEST 5/5] Testing Feedback Logging, Attribution, and CTR...")
        session_id = str(uuid.uuid4())
        rec_ids = [r["movie_id"] for r in wd_data["recommendations"][:5]]

        # Log impression
        imp_event_id = str(uuid.uuid4())
        imp_payload = {
            "event_id": imp_event_id,
            "session_id": session_id,
            "user_id": test_uid,
            "event_type": "impression",
            "movie_id": rec_ids[0],
            "movie_ids": rec_ids,
            "position": 1,
            "ranking_variant": "wide_and_deep",
            "model_version": "v1.0",
        }
        imp_resp = client.post("/feedback", json=imp_payload)
        assert imp_resp.status_code in (200, 201), f"Impression logging failed: {imp_resp.text}"

        # Deduplication check: duplicate event_id should be rejected with 409
        dup_resp = client.post("/feedback", json=imp_payload)
        assert dup_resp.status_code == 409, f"Duplicate check failed: {dup_resp.status_code}"
        logger.info("Event deduplication confirmed: duplicate event_id rejected with HTTP 409 Conflict.")

        # Log attributed click
        click_event_id = str(uuid.uuid4())
        click_payload = {
            "event_id": click_event_id,
            "session_id": session_id,
            "user_id": test_uid,
            "event_type": "click",
            "movie_id": rec_ids[0],
            "position": 1,
            "ranking_variant": "wide_and_deep",
            "model_version": "v1.0",
        }
        click_resp = client.post("/feedback", json=click_payload)
        assert click_resp.status_code in (200, 201), f"Click logging failed: {click_resp.text}"

        # Fetch CTR metrics
        ctr_resp = client.get("/metrics/ctr")
        assert ctr_resp.status_code == 200
        ctr_data = ctr_resp.json()
        assert ctr_data["total_impressions"] > 0
        assert "wide_and_deep" in ctr_data["by_variant"]
        wd_metrics = ctr_data["by_variant"]["wide_and_deep"]
        logger.info(f"CTR Metrics: Impressions={ctr_data['total_impressions']} | Clicks={ctr_data['total_clicks']} | W&D Impressions={wd_metrics['impressions']} | Clicks={wd_metrics['clicks']} | CTR={wd_metrics['ctr']:.4f}")
        results["feedback"] = {"status": "PASS", "ctr_data": wd_metrics}

    print("\n" + "=" * 80)
    print("END-TO-END RECOMMENDATION ENGINE PIPELINE VERIFICATION PASSED")
    print("=" * 80)
    print(f"{'Verification Stage':<35} | {'Status':<10} | {'Details':<30}")
    print("-" * 80)
    print(f"{'1. Health Endpoint & Models':<35} | {'PASS':<10} | {'62,423 catalog items loaded':<30}")
    print(f"{'2. Warm User Two-Stage (W&D)':<35} | {'PASS':<10} | {f'{wd_latency:.2f} ms latency (10 items)':<30}")
    print(f"{'3. Warm User Two-Stage (Wide)':<35} | {'PASS':<10} | {f'{wo_latency:.2f} ms latency (10 items)':<30}")
    print(f"{'4. Stage-1 Candidates Inspection':<35} | {'PASS':<10} | {'50 FAISS vector candidates':<30}")
    print(f"{'5. Cold-Start Unknown User Fallback':<35} | {'PASS':<10} | {'PopularityRecommender active':<30}")
    print(f"{'6. Feedback Logging & Attribution':<35} | {'PASS':<10} | {'409 deduplication verified':<30}")
    ctr_val = wd_metrics['ctr']
    print(f"{'7. Live Demo CTR Reporting':<35} | {'PASS':<10} | {f'Recorded CTR: {ctr_val:.4f}':<30}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    verify_full_pipeline()
