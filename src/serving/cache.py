"""
Redis Caching Layer for Recommendation Serving.

Provides:
  - Structured key generation: rec:{user_id}:{k}:{variant}:{model_version}
  - Graceful degradation: Continues serving without crashing if Redis is unavailable.
  - Metrics tracking: hits, misses, errors, writes.
  - Safe serialization & deserialization.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Manages Redis caching for recommendation responses with graceful fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600,
        enabled: bool = True,
        socket_timeout: float = 0.5,
        socket_connect_timeout: float = 0.5,
    ):
        self.default_ttl = default_ttl
        env_enabled = os.environ.get("REDIS_ENABLED", "true").lower() not in ("false", "0", "no")
        self.enabled = enabled and env_enabled
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout

        # Metrics counters
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.writes = 0

        self.client = None
        self._connected = False
        self._last_ping_time = 0.0

        if not self.enabled:
            logger.info("Redis caching is disabled.")
            return

        try:
            import redis

            url = redis_url or os.environ.get("REDIS_URL")
            if url:
                self.client = redis.Redis.from_url(
                    url,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    decode_responses=True,
                )
            else:
                h = os.environ.get("REDIS_HOST", host)
                if h == "localhost":
                    h = "127.0.0.1"  # Avoid IPv6 DNS resolution latency on Windows
                p = int(os.environ.get("REDIS_PORT", port))
                d = int(os.environ.get("REDIS_DB", db))
                self.client = redis.Redis(
                    host=h,
                    port=p,
                    db=d,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    decode_responses=True,
                )

            # Test connection
            self.client.ping()
            self._connected = True
            self._last_ping_time = time.time()
            logger.info(f"Successfully connected to Redis cache at {self.client.connection_pool.connection_kwargs.get('host', 'url')}")
        except Exception as e:
            logger.warning(
                f"Redis unavailable ({e}). Continuing in graceful degraded mode (no caching)."
            )
            self._connected = False
            self._last_ping_time = time.time()

    @property
    def is_connected(self) -> bool:
        """Check if Redis is currently connected with 10s backoff when offline."""
        if not self.enabled or self.client is None:
            return False
        now = time.time()
        if not self._connected and (now - self._last_ping_time < 10.0):
            return False
        self._last_ping_time = now
        try:
            self._connected = bool(self.client.ping())
            return self._connected
        except Exception:
            self._connected = False
            return False

    @staticmethod
    def make_key(user_id: Union[str, int], k: int, variant: str, model_version: str) -> str:
        """
        Construct structured cache key:
          rec:{user_id}:{k}:{variant}:{model_version}
        """
        return f"rec:{user_id}:{k}:{variant}:{model_version}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached JSON object by key.
        Returns None on miss, cache disabled, disconnected, or connection error.
        """
        if not self.enabled or not self._connected or self.client is None:
            self.misses += 1
            return None

        try:
            cached_data = self.client.get(key)
            if cached_data is not None:
                self.hits += 1
                return json.loads(cached_data)
            self.misses += 1
            return None
        except Exception as e:
            self.errors += 1
            self._connected = False
            logger.debug(f"Redis get error for key '{key}': {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache JSON-serializable dictionary with expiration TTL in seconds.
        Fails safely without raising exceptions.
        """
        if not self.enabled or not self._connected or self.client is None or value is None:
            return False

        effective_ttl = ttl if ttl is not None else self.default_ttl
        try:
            serialized = json.dumps(value)
            self.client.set(key, serialized, ex=effective_ttl)
            self.writes += 1
            return True
        except Exception as e:
            self.errors += 1
            logger.debug(f"Redis set error for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Remove a cached key."""
        if not self.enabled or self.client is None:
            return False
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and usage statistics."""
        total_queries = self.hits + self.misses
        hit_ratio = (self.hits / total_queries) if total_queries > 0 else 0.0
        return {
            "enabled": self.enabled,
            "connected": self.is_connected,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "total_queries": total_queries,
            "hit_ratio": float(hit_ratio),
        }
