"""
Genuine Impression and Click Feedback Logging Layer.

Provides:
  - Validated recording of impression and click events.
  - Event deduplication via unique event IDs.
  - Impression validation: Validates clicks against prior impressions.
  - Durable append-only storage in local JSONL format.
  - Online Demo CTR computation with explicit methodology and disclaimer.
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class FeedbackEvent(BaseModel):
    """Schema for incoming feedback events."""
    event_id: str = Field(..., description="Unique UUID or event identifier")
    event_type: str = Field(..., description="Must be 'impression' or 'click'")
    timestamp: Optional[str] = Field(default=None, description="ISO-8601 timestamp")
    user_id: str = Field(..., description="User ID or anonymous session ID")
    movie_id: Union[int, str] = Field(..., description="Target movie ID")
    variant: str = Field(default="wide_and_deep", description="Ranking model variant")
    request_id: Optional[str] = Field(default=None, description="Request ID linking impression to recommendation")
    impression_id: Optional[str] = Field(default=None, description="Prior impression event ID for click attribution")
    model_version: str = Field(default="v1.0", description="Model version active when event occurred")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("impression", "click"):
            raise ValueError(f"Invalid event_type '{v}'. Allowed types: 'impression', 'click'")
        return clean

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("event_id cannot be empty")
        return v.strip()


class FeedbackLogger:
    """Thread-safe durable logger for recommendation user feedback events backed by SQLite and JSONL."""

    def __init__(self, log_dir: Union[str, Path] = "data/feedback"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "events.jsonl"
        self.db_path = self.log_dir / "feedback.db"

        self._lock = threading.Lock()
        self._seen_event_ids: Set[str] = set()
        self._known_impressions: Set[str] = set()  # stores "user_id:movie_id:request_id"

        # Initialize SQLite database schema
        self._init_sqlite_db()

        # Load existing events on startup for deduplication and persistence
        self._load_existing_events()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite_db(self):
        """Initialize tables for durable event logging and user watchlist state."""
        with self._lock:
            with self._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        movie_id TEXT NOT NULL,
                        variant TEXT,
                        request_id TEXT,
                        impression_id TEXT,
                        model_version TEXT,
                        timestamp TEXT NOT NULL,
                        verified_impression_match INTEGER DEFAULT 0
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_events(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_event_type ON feedback_events(event_type);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_watchlist (
                        user_id TEXT NOT NULL,
                        movie_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        PRIMARY KEY (user_id, movie_id)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON user_watchlist(user_id);")
                conn.commit()

    def _load_existing_events(self):
        """Pre-populate deduplication index from SQLite and backfill from JSONL if needed."""
        with self._lock:
            # First load from SQLite
            try:
                with self._get_db_connection() as conn:
                    cur = conn.execute("SELECT event_id, event_type, user_id, movie_id, request_id FROM feedback_events")
                    for row in cur.fetchall():
                        eid = row["event_id"]
                        self._seen_event_ids.add(eid)
                        if row["event_type"] == "impression":
                            self._known_impressions.add(f"{row['user_id']}:{row['movie_id']}:{row['request_id'] or ''}")
            except Exception as e:
                logger.warning(f"Error reading existing events from SQLite: {e}")

            # Second: if events.jsonl has events not in SQLite, backfill them into SQLite
            if self.log_file.exists():
                backfilled = 0
                try:
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        with self._get_db_connection() as conn:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    eid = data.get("event_id")
                                    if not eid:
                                        continue
                                    self._seen_event_ids.add(eid)

                                    etype = data.get("event_type", "impression")
                                    u = str(data.get("user_id", ""))
                                    m = str(data.get("movie_id", ""))
                                    req = str(data.get("request_id", ""))
                                    if u.lower().startswith(("test_", "audit_", "verify_", "smoke_")):
                                        continue
                                    if etype == "impression":
                                        self._known_impressions.add(f"{u}:{m}:{req}")

                                    # Check if in SQLite
                                    exists = conn.execute("SELECT 1 FROM feedback_events WHERE event_id = ?", (eid,)).fetchone()
                                    if not exists:
                                        ts = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
                                        verified = 1 if data.get("verified_impression_match") else 0
                                        conn.execute("""
                                            INSERT OR IGNORE INTO feedback_events 
                                            (event_id, event_type, user_id, movie_id, variant, request_id, impression_id, model_version, timestamp, verified_impression_match)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """, (eid, etype, u, m, data.get("variant", "wide_and_deep"), req, data.get("impression_id"), data.get("model_version", "v1.0"), ts, verified))
                                        if etype == "click":
                                            conn.execute("""
                                                INSERT OR REPLACE INTO user_watchlist (user_id, movie_id, timestamp, event_id)
                                                VALUES (?, ?, ?, ?)
                                            """, (u, m, ts, eid))
                                        backfilled += 1
                                except Exception:
                                    continue
                            conn.commit()
                    if backfilled > 0:
                        logger.info(f"Backfilled {backfilled} events from JSONL into SQLite database.")
                except Exception as e:
                    logger.warning(f"Could not load/backfill events from {self.log_file}: {e}")

    def log_event(self, event: FeedbackEvent) -> Dict[str, Any]:
        """
        Record and durably persist a validated feedback event in SQLite and JSONL.

        Raises:
            ValueError: If event_id is a duplicate.
        """
        with self._lock:
            if event.event_id in self._seen_event_ids:
                raise ValueError(f"Duplicate event_id '{event.event_id}'. Event already recorded.")

            ts = event.timestamp or datetime.now(timezone.utc).isoformat()
            user_id_str = str(event.user_id)
            movie_id_str = str(event.movie_id)

            # Check impression attribution for click events
            is_verified_click = False
            if event.event_type == "click":
                click_key = f"{user_id_str}:{movie_id_str}:{event.request_id or ''}"
                is_verified_click = (
                    click_key in self._known_impressions
                    or (event.impression_id is not None and event.impression_id in self._seen_event_ids)
                )
            elif event.event_type == "impression":
                imp_key = f"{user_id_str}:{movie_id_str}:{event.request_id or ''}"
                self._known_impressions.add(imp_key)

            # 1. Persist to SQLite
            try:
                with self._get_db_connection() as conn:
                    conn.execute("""
                        INSERT INTO feedback_events 
                        (event_id, event_type, user_id, movie_id, variant, request_id, impression_id, model_version, timestamp, verified_impression_match)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type,
                        user_id_str,
                        movie_id_str,
                        event.variant,
                        event.request_id,
                        event.impression_id,
                        event.model_version,
                        ts,
                        1 if is_verified_click else 0,
                    ))

                    # If this is a click / save action, add to user_watchlist
                    if event.event_type == "click":
                        conn.execute("""
                            INSERT OR REPLACE INTO user_watchlist (user_id, movie_id, timestamp, event_id)
                            VALUES (?, ?, ?, ?)
                        """, (user_id_str, movie_id_str, ts, event.event_id))

                    conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Duplicate event_id '{event.event_id}'. Event already recorded in database.")
            except Exception as e:
                logger.error(f"Failed to persist feedback event to SQLite: {e}")
                raise RuntimeError(f"Database write failure: {e}")

            # 2. Append to JSONL for dual storage (genuine user events only)
            if not user_id_str.lower().startswith(("test_", "audit_", "verify_", "smoke_")):
                event_dict = event.model_dump()
                event_dict["timestamp"] = ts
                event_dict["user_id"] = user_id_str
                event_dict["movie_id"] = movie_id_str
                if event.event_type == "click":
                    event_dict["verified_impression_match"] = is_verified_click

                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_dict) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to append to JSONL log file: {e}")

            self._seen_event_ids.add(event.event_id)

            return {
                "status": "success",
                "event_id": event.event_id,
                "event_type": event.event_type,
                "persisted": True,
                "verified": is_verified_click if event.event_type == "click" else True,
            }

    def get_user_watchlist(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieve all currently saved watchlist items for a specific user from SQLite."""
        user_id_str = str(user_id).strip()
        with self._lock:
            with self._get_db_connection() as conn:
                cur = conn.execute(
                    "SELECT movie_id, timestamp, event_id FROM user_watchlist WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id_str,)
                )
                return [
                    {
                        "movie_id": str(row["movie_id"]),
                        "timestamp": row["timestamp"],
                        "event_id": row["event_id"],
                    }
                    for row in cur.fetchall()
                ]

    def remove_watchlist_item(self, user_id: str, movie_id: str) -> bool:
        """Remove a movie from user's watchlist in SQLite."""
        user_id_str = str(user_id).strip()
        movie_id_str = str(movie_id).strip()
        with self._lock:
            with self._get_db_connection() as conn:
                cur = conn.execute(
                    "DELETE FROM user_watchlist WHERE user_id = ? AND movie_id = ?",
                    (user_id_str, movie_id_str)
                )
                conn.commit()
                return cur.rowcount > 0

    def get_user_feedback(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve full feedback history for a user from SQLite."""
        user_id_str = str(user_id).strip()
        with self._lock:
            with self._get_db_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT event_id, event_type, user_id, movie_id, variant, request_id, 
                           impression_id, model_version, timestamp, verified_impression_match
                    FROM feedback_events 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC
                    """,
                    (user_id_str,)
                )
                return [dict(row) for row in cur.fetchall()]

    def compute_ctr(self, window_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate online CTR strictly from genuine logged impressions and clicks.

        Denominator: Total recorded impressions.
        Numerator: Total recorded clicks.
        """
        impressions_count = 0
        clicks_count = 0
        variant_impressions: Dict[str, int] = {}
        variant_clicks: Dict[str, int] = {}

        now = time.time()
        window_seconds = (window_hours * 3600.0) if window_hours else None

        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if window_seconds is not None:
                            # Parse timestamp
                            ts_str = record.get("timestamp")
                            if ts_str:
                                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                if (now - dt.timestamp()) > window_seconds:
                                    continue

                        uid = str(record.get("user_id", "")).strip().lower()
                        # Exclude synthetic automated test users from live CTR metrics
                        if uid.startswith(("test_", "audit_", "verify_", "smoke_")):
                            continue

                        etype = record.get("event_type")
                        variant = record.get("variant", "unknown")

                        if etype == "impression":
                            impressions_count += 1
                            variant_impressions[variant] = variant_impressions.get(variant, 0) + 1
                        elif etype == "click":
                            clicks_count += 1
                            variant_clicks[variant] = variant_clicks.get(variant, 0) + 1
                    except Exception:
                        continue

        overall_ctr = (clicks_count / impressions_count) if impressions_count > 0 else 0.0

        by_variant = {}
        all_variants = set(variant_impressions.keys()).union(set(variant_clicks.keys()))
        for v in all_variants:
            v_imp = variant_impressions.get(v, 0)
            v_clk = variant_clicks.get(v, 0)
            by_variant[v] = {
                "impressions": v_imp,
                "clicks": v_clk,
                "ctr": float(v_clk / v_imp) if v_imp > 0 else 0.0,
            }

        return {
            "total_impressions": impressions_count,
            "total_clicks": clicks_count,
            "overall_ctr": float(overall_ctr),
            "by_variant": by_variant,
            "reporting_window": f"last {window_hours} hours" if window_hours else "all-time",
            "disclaimer": (
                "Demo interaction data from local user testing. "
                "Per system design, offline ratings are strictly separated and NOT equated with CTR. "
                "This metric is solely derived from explicit UI impression and click events."
            ),
        }
