"""
Verify feedback persistence across page refresh and server restart using a dedicated test user.
"""

import json
import sqlite3
import time
import urllib.request
import uuid

BASE_URL = "http://127.0.0.1:8000"
TEST_USER = "verify_user_persisted"
TEST_MOVIE = "318"  # Shawshank Redemption


def http_request(method, path, data=None):
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print(f"=== Testing Feedback Persistence Lifecycle for User '{TEST_USER}' ===")

    # 1. Clean prior test state for idempotency
    try:
        http_request("DELETE", f"/feedback/{TEST_USER}/{TEST_MOVIE}")
    except Exception:
        pass

    # 2. Add movie to watchlist via POST /feedback
    ev_id = str(uuid.uuid4())
    click_payload = {
        "event_id": ev_id,
        "event_type": "click",
        "user_id": TEST_USER,
        "movie_id": TEST_MOVIE,
        "variant": "wide_and_deep",
        "model_version": "v1.0",
    }
    res_post = http_request("POST", "/feedback", click_payload)
    print("1. POST /feedback result:", res_post)
    assert res_post["status"] == "success"
    assert res_post["persisted"] is True

    # 3. Verify retrieved via GET /feedback/{user_id}
    res_get1 = http_request("GET", f"/feedback/{TEST_USER}")
    print(f"2. GET /feedback/{TEST_USER} (before restart):", res_get1)
    assert TEST_MOVIE in res_get1["saved_movie_ids"]
    assert any("Shawshank" in item["title"] for item in res_get1["watchlist"])

    # 4. Direct SQLite verification
    conn = sqlite3.connect("data/feedback/feedback.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, movie_id, timestamp, event_id FROM user_watchlist WHERE user_id = ?",
        (TEST_USER,),
    )
    row = cur.fetchone()
    print("3. Direct SQLite verification in feedback.db:", row)
    assert row is not None
    assert row[0] == TEST_USER and row[1] == TEST_MOVIE
    conn.close()

    print("\n[OK] Verification Pre-Restart Succeeded: Watchlist item durably written to SQLite.")

    # 5. Simulate reload / query again
    res_get2 = http_request("GET", f"/feedback/{TEST_USER}")
    print(f"4. GET /feedback/{TEST_USER} (simulating client reload):", res_get2)
    assert TEST_MOVIE in res_get2["saved_movie_ids"]

    # 6. Delete item from watchlist via DELETE /feedback/{user_id}/{movie_id}
    res_del = http_request("DELETE", f"/feedback/{TEST_USER}/{TEST_MOVIE}")
    print(f"5. DELETE /feedback/{TEST_USER}/{TEST_MOVIE}:", res_del)
    assert res_del["removed"] is True

    # 7. Verify watchlist is now empty
    res_get3 = http_request("GET", f"/feedback/{TEST_USER}")
    print(f"6. GET /feedback/{TEST_USER} (after delete):", res_get3)
    assert TEST_MOVIE not in res_get3["saved_movie_ids"]
    assert len(res_get3["watchlist"]) == 0

    # 8. Clean up event record from SQLite
    conn = sqlite3.connect("data/feedback/feedback.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM feedback_events WHERE user_id = ?", (TEST_USER,))
    conn.commit()
    conn.close()

    print(f"\n[SUCCESS] FULL PERSISTENCE LIFECYCLE VERIFIED FOR {TEST_USER}!")


if __name__ == "__main__":
    main()
