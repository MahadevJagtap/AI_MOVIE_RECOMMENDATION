"""
Targeted sanitization script:
Remove synthetic test records (test_*, audit_*, verify_*, and test-injected movie 1 for 198)
while preserving 100% of legitimate user interactions and watchlists.
"""

import json
from pathlib import Path
import sqlite3

def main():
    db_path = Path("data/feedback/feedback.db")
    if not db_path.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Clean test entries from user_watchlist
    cur.execute("DELETE FROM user_watchlist WHERE user_id LIKE 'test_%' OR user_id LIKE 'audit_%' OR user_id LIKE 'verify_%'")
    deleted_wl_test = cur.rowcount

    # Clean the test-generated movie 1 for 198
    cur.execute("DELETE FROM user_watchlist WHERE user_id = '198' AND movie_id = '1'")
    deleted_wl_198_1 = cur.rowcount

    # 2. Clean test entries from feedback_events
    cur.execute("DELETE FROM feedback_events WHERE user_id LIKE 'test_%' OR user_id LIKE 'audit_%' OR user_id LIKE 'verify_%'")
    deleted_events_test = cur.rowcount

    cur.execute("DELETE FROM feedback_events WHERE user_id = '198' AND movie_id = '1'")
    deleted_events_198_1 = cur.rowcount

    conn.commit()

    print(f"Watchlist: deleted {deleted_wl_test} synthetic test rows, {deleted_wl_198_1} test item rows for 198.")
    print(f"Feedback events: deleted {deleted_events_test} synthetic test rows, {deleted_events_198_1} test item rows for 198.")

    # 3. Clean events.jsonl
    jsonl = Path("data/feedback/events.jsonl")
    if jsonl.exists():
        preserved_lines = []
        removed_count = 0
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                uid = str(d.get("user_id", "")).lower()
                mid = str(d.get("movie_id", ""))
                if uid.startswith("test_") or uid.startswith("audit_") or uid.startswith("verify_") or (uid == "198" and mid == "1"):
                    removed_count += 1
                else:
                    preserved_lines.append(line)
        with open(jsonl, "w", encoding="utf-8") as f:
            f.writelines(preserved_lines)
        print(f"events.jsonl: removed {removed_count} test lines, preserved {len(preserved_lines)} real lines.")

    print("\n=== Remaining Watchlist Users (All Genuine) ===")
    for row in cur.execute("SELECT user_id, count(*) FROM user_watchlist GROUP BY user_id ORDER BY CAST(user_id AS INTEGER)").fetchall():
        print(f"User {row[0]}: {row[1]} saved movies")

    conn.close()

if __name__ == "__main__":
    main()
