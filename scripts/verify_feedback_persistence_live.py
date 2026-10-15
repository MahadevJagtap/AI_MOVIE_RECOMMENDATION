"""
Dedicated verification script for feedback persistence across client refresh and backend restart.
User: test_restart_verifier
Movie: 318 (The Shawshank Redemption)
"""

import json
import sqlite3
import sys
import time
import urllib.request
import uuid

BASE_URL = "http://127.0.0.1:8000"
USER_ID = "test_restart_verifier"
MOVIE_ID = "318"

def req(method, path, data=None):
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def phase1_pre_restart():
    print(f"=== Phase 1: Pre-Restart Verification for user '{USER_ID}', movie '{MOVIE_ID}' ===")
    
    # 1. Clean prior state if any
    try:
        req("DELETE", f"/feedback/{USER_ID}/{MOVIE_ID}")
    except Exception:
        pass

    # 2. Add movie to watchlist via POST /feedback
    ev_id = str(uuid.uuid4())
    post_res = req("POST", "/feedback", {
        "event_id": ev_id,
        "event_type": "click",
        "user_id": USER_ID,
        "movie_id": MOVIE_ID,
        "variant": "wide_and_deep",
        "model_version": "v1.0"
    })
    print("1. POST /feedback:", post_res)
    assert post_res["status"] == "success"
    assert post_res["persisted"] is True

    # 3. Retrieve via GET /feedback/{user_id}
    get1 = req("GET", f"/feedback/{USER_ID}")
    print("2. GET /feedback/{user_id}:", get1)
    assert MOVIE_ID in get1["saved_movie_ids"]
    assert any("Shawshank Redemption" in item["title"] for item in get1["watchlist"])

    # 4. Direct SQLite inspection
    conn = sqlite3.connect("data/feedback/feedback.db")
    cur = conn.cursor()
    cur.execute("SELECT user_id, movie_id, timestamp, event_id FROM user_watchlist WHERE user_id = ?", (USER_ID,))
    row = cur.fetchone()
    conn.close()
    print("3. Direct SQLite verification:", row)
    assert row is not None
    assert row[0] == USER_ID and row[1] == MOVIE_ID
    print(">>> Phase 1 Pre-Restart Verification PASSED!\n")

def phase2_post_restart():
    print(f"=== Phase 2: Post-Restart Verification for user '{USER_ID}' ===")

    # 1. Check health
    health = req("GET", "/health")
    print("1. Fresh backend /health:", health["status"], "Total movies:", health["total_catalog_movies"])
    assert health["status"] in ("healthy", "degraded")

    # 2. Retrieve watchlist on new process (simulating restart persistence)
    get2 = req("GET", f"/feedback/{USER_ID}")
    print("2. GET /feedback/{user_id} on fresh backend:", get2)
    assert MOVIE_ID in get2["saved_movie_ids"], "Item MUST persist across backend restart!"
    assert any("Shawshank Redemption" in item["title"] for item in get2["watchlist"])
    print(">>> Persistence across backend restart confirmed!")

    # 3. Delete item from watchlist
    del_res = req("DELETE", f"/feedback/{USER_ID}/{MOVIE_ID}")
    print("3. DELETE /feedback/{user_id}/{movie_id}:", del_res)
    assert del_res["removed"] is True

    # 4. Verify watchlist is now empty
    get3 = req("GET", f"/feedback/{USER_ID}")
    print("4. GET /feedback/{user_id} (after delete):", get3)
    assert MOVIE_ID not in get3["saved_movie_ids"]
    assert len(get3["watchlist"]) == 0

    # 5. Clean up event record from SQLite
    conn = sqlite3.connect("data/feedback/feedback.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM feedback_events WHERE user_id = ?", (USER_ID,))
    cur.execute("DELETE FROM user_watchlist WHERE user_id = ?", (USER_ID,))
    conn.commit()
    conn.close()
    print(">>> Test event and watchlist cleaned. Phase 2 Post-Restart Verification PASSED!\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        phase2_post_restart()
    else:
        phase1_pre_restart()
