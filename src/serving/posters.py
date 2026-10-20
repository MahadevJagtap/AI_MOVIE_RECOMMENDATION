"""
OMDb Movie Poster Integration Service for CineMatch.

Provides:
  - Title and year normalization for MovieLens catalog titles.
  - Fast IMDb ID lookup via links mapping (links.parquet / links.csv).
  - Robust OMDb API client using httpx with connection/read timeouts.
  - Multi-tier caching: in-memory dictionary + persistent SQLite storage.
  - Parallel batch fetching with bounded concurrency.
  - Comprehensive fallback handling (missing posters, rate limits, timeouts, invalid keys).
  - Non-blocking guarantees: recommendations never fail due to poster lookup errors.
"""

import concurrent.futures
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx

from configs.config import OMDB_API_KEY, OMDB_TIMEOUT, POSTER_CACHE_DIR

logger = logging.getLogger(__name__)

OMDB_BASE_URL = "https://www.omdbapi.com/"


def clean_movielens_title(raw_title: str) -> Tuple[str, str]:
    """
    Parse and normalize MovieLens catalog titles into a clean query title and release year.

    Examples:
      - "Shawshank Redemption, The (1994)" -> ("The Shawshank Redemption", "1994")
      - "Godfather: Part II, The (1974)" -> ("The Godfather: Part II", "1974")
      - "Matrix, The (1999)" -> ("The Matrix", "1999")
      - "City of God (Cidade de Deus) (2002)" -> ("City of God", "2002")
      - "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)" -> ("Amelie", "2001")
      - "Léon: The Professional (a.k.a. The Professional) (1994)" -> ("Léon: The Professional", "1994")
    """
    if not raw_title:
        return "", ""

    text = str(raw_title).strip()

    # 1. Extract 4-digit release year in parentheses
    year_match = re.search(r"\((\d{4})\)", text)
    year = year_match.group(1) if year_match else ""

    # Remove the year and anything following it
    text = re.sub(r"\s*\(\d{4}\).*", "", text).strip()

    # 2. Remove (a.k.a. ...) patterns
    text = re.sub(r"\s*\(a\.k\.a\..*?\)", "", text, flags=re.IGNORECASE).strip()

    # 3. Remove trailing parenthetical foreign/alternate title (e.g. "(Cidade de Deus)")
    text = re.sub(r"\s*\([^)]*\)$", "", text).strip()

    # 4. Handle trailing articles (e.g. "Shawshank Redemption, The" -> "The Shawshank Redemption")
    articles = ["The", "A", "An", "Le", "La", "Les", "Der", "Die", "Das", "Il", "L'"]
    for art in articles:
        suffix = f", {art}"
        if text.endswith(suffix):
            text = f"{art} {text[:-len(suffix)]}".strip()
            break
        elif text.endswith(f",{art}"):
            text = f"{art} {text[:-len(art)-1]}".strip()
            break

    # Clean up double spaces or trailing punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text, year


class MoviePosterService:
    """
    Thread-safe service for fetching, caching, and serving movie poster URLs via OMDb.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        links_data: Optional[Dict[str, str]] = None,
    ):
        self.api_key = (api_key if api_key is not None else OMDB_API_KEY).strip()
        self.timeout = float(timeout if timeout is not None else OMDB_TIMEOUT)

        # Cache directory & SQLite database path
        if cache_dir is None:
            cache_dir = POSTER_CACHE_DIR
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "posters.db"

        # Locks and in-memory caches
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, Optional[str]] = {}  # movie_id -> poster_url (or None)
        self._negative_cache_ttl = 86400 * 7  # 7 days for not_found

        # MovieLens movieId -> IMDb ID mapping (e.g. "318" -> "tt0111161")
        self.links: Dict[str, str] = links_data or {}
        if not self.links:
            self._load_links_mapping()

        # Initialize SQLite storage and preload existing cache
        self._init_sqlite()
        self._preload_cache()

    def _load_links_mapping(self):
        """Load links mapping from parquet or csv if available."""
        # 1. Try data/processed/links.parquet
        parquet_path = Path("data/processed/links.parquet")
        if parquet_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                for mid, iid in zip(df["movieId"].values, df["imdbId"].values):
                    iid_str = str(iid).strip()
                    if iid_str and iid_str != "nan":
                        self.links[str(mid)] = iid_str if iid_str.startswith("tt") else f"tt{int(iid):07d}"
                logger.info(f"Loaded {len(self.links)} IMDb links from {parquet_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load links from {parquet_path}: {e}")

        # 2. Try raw MovieLens links.csv
        movielens_dir = Path(os.environ.get("MOVIELENS_DIR", "d:/setup/Database/ml-25m"))
        csv_path = movielens_dir / "links.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, usecols=["movieId", "imdbId"])
                for mid, iid in zip(df["movieId"].values, df["imdbId"].values):
                    try:
                        ival = int(iid)
                        if ival > 0:
                            self.links[str(mid)] = f"tt{ival:07d}"
                    except (ValueError, TypeError):
                        continue
                logger.info(f"Loaded {len(self.links)} IMDb links from {csv_path}")
            except Exception as e:
                logger.warning(f"Could not load links from {csv_path}: {e}")

    def _get_db_conn(self) -> sqlite3.Connection:
        """Create a connection to SQLite database."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """Create poster cache schema in SQLite."""
        with self._lock:
            with self._get_db_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS poster_cache (
                        movie_id TEXT PRIMARY KEY,
                        imdb_id TEXT,
                        title TEXT,
                        year TEXT,
                        poster_url TEXT,
                        status TEXT,
                        updated_at REAL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_poster_imdb ON poster_cache(imdb_id);")
                conn.commit()

    def _preload_cache(self):
        """Preload cached poster entries from SQLite into in-memory dictionary for instant lookups."""
        try:
            with self._get_db_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT movie_id, poster_url, status FROM poster_cache")
                count = 0
                for row in cur.fetchall():
                    mid = str(row["movie_id"])
                    url = row["poster_url"]
                    status = row["status"]
                    if status == "success" and url:
                        self._memory_cache[mid] = url
                    elif status == "not_found":
                        self._memory_cache[mid] = None
                    count += 1
                logger.info(f"Preloaded {count} poster entries from SQLite into memory.")
        except Exception as e:
            logger.warning(f"Error preloading poster cache: {e}")

    def _save_cache(
        self,
        movie_id: str,
        poster_url: Optional[str],
        imdb_id: Optional[str] = None,
        title: Optional[str] = None,
        year: Optional[str] = None,
        status: str = "success",
    ):
        """Save poster result into both in-memory cache and persistent SQLite."""
        clean_mid = str(movie_id).strip()
        with self._lock:
            self._memory_cache[clean_mid] = poster_url
            try:
                with self._get_db_conn() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO poster_cache
                        (movie_id, imdb_id, title, year, poster_url, status, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            clean_mid,
                            imdb_id,
                            title,
                            year,
                            poster_url,
                            status,
                            time.time(),
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Error saving poster to SQLite for movie {clean_mid}: {e}")

    def get_poster(
        self,
        movie_id: Union[str, int],
        title: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> Optional[str]:
        """
        Retrieve poster URL for a movie with multi-tier caching and fallback strategies.
        Guaranteed not to raise an exception.
        """
        clean_mid = str(movie_id).strip()
        if not clean_mid:
            return None

        # 1. Check in-memory cache
        with self._lock:
            if clean_mid in self._memory_cache:
                return self._memory_cache[clean_mid]

        # 2. Check SQLite persistent cache
        try:
            with self._get_db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT poster_url, status FROM poster_cache WHERE movie_id = ?",
                    (clean_mid,),
                )
                row = cur.fetchone()
                if row:
                    url = row["poster_url"]
                    st = row["status"]
                    with self._lock:
                        self._memory_cache[clean_mid] = url if st == "success" else None
                    return self._memory_cache[clean_mid]
        except Exception as e:
            logger.debug(f"SQLite cache check error for {clean_mid}: {e}")

        # 3. If no API key configured, return None safely
        if not self.api_key:
            return None

        # 4. Perform OMDb API lookup
        return self._fetch_from_omdb(clean_mid, title=title, client=client)

    def _fetch_from_omdb(
        self,
        movie_id: str,
        title: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> Optional[str]:
        """Call OMDb API with IMDb ID or normalized title/year."""
        imdb_id = self.links.get(movie_id)
        clean_title, year = clean_movielens_title(title or "")

        def _do_request(params: Dict[str, str], http_client: httpx.Client) -> Tuple[Optional[str], str]:
            """Execute HTTP request to OMDb and return (poster_url, status)."""
            req_params = {"apikey": self.api_key, **params}
            try:
                resp = http_client.get(OMDB_BASE_URL, params=req_params, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.warning(f"OMDb returned HTTP {resp.status_code} for params {params}")
                    return None, "error"

                data = resp.json()
                if data.get("Response") == "True":
                    poster = data.get("Poster")
                    if poster and poster != "N/A" and str(poster).startswith("http"):
                        return str(poster), "success"
                    return None, "not_found"

                # Check error reason
                err = str(data.get("Error", ""))
                if "Movie not found" in err:
                    return None, "not_found"
                if "Invalid API key" in err or "Request limit reached" in err:
                    logger.warning(f"OMDb API key issue: {err}")
                    return None, "api_error"

                return None, "not_found"
            except httpx.TimeoutException:
                logger.warning(f"OMDb timeout ({self.timeout}s) for movie {movie_id}")
                return None, "timeout"
            except Exception as e:
                logger.warning(f"OMDb request error for movie {movie_id}: {e}")
                return None, "error"

        own_client = client is None
        http_client = client or httpx.Client(timeout=self.timeout)

        try:
            poster_url = None
            status = "not_found"

            # Strategy A: Use IMDb ID if mapped (most accurate)
            if imdb_id:
                poster_url, status = _do_request({"i": imdb_id}, http_client)

            # Strategy B: If no IMDb ID or Strategy A failed, search by clean title and year
            if not poster_url and status not in ("api_error", "timeout") and clean_title:
                params = {"t": clean_title}
                if year:
                    params["y"] = year
                poster_url, status = _do_request(params, http_client)

                # Strategy C: If year-specific search failed, try title only
                if not poster_url and year and status == "not_found":
                    poster_url, status = _do_request({"t": clean_title}, http_client)

            # Save to cache if result is success or not_found
            if status in ("success", "not_found"):
                self._save_cache(
                    movie_id=movie_id,
                    poster_url=poster_url,
                    imdb_id=imdb_id,
                    title=clean_title or title,
                    year=year,
                    status=status,
                )

            return poster_url
        finally:
            if own_client:
                http_client.close()

    def get_posters_batch(
        self,
        items: List[Dict[str, Any]],
        max_workers: int = 5,
    ) -> Dict[str, Optional[str]]:
        """
        Fetch posters for multiple movies concurrently with cache-first lookup.

        Args:
            items: List of dicts, each containing at least 'movie_id' and optional 'title'.
            max_workers: Bounded concurrency for outgoing OMDb requests.

        Returns:
            Dict mapping movie_id -> poster_url (or None).
        """
        results: Dict[str, Optional[str]] = {}
        missing_items: List[Dict[str, Any]] = []

        # 1. Fast cache check
        for item in items:
            mid = str(item.get("movie_id", "")).strip()
            if not mid:
                continue

            with self._lock:
                if mid in self._memory_cache:
                    results[mid] = self._memory_cache[mid]
                    continue

            missing_items.append(item)

        if not missing_items:
            return results

        # 2. Check SQLite for missing items in one pass
        uncached_items: List[Dict[str, Any]] = []
        try:
            with self._get_db_conn() as conn:
                cur = conn.cursor()
                for item in missing_items:
                    mid = str(item["movie_id"]).strip()
                    cur.execute("SELECT poster_url, status FROM poster_cache WHERE movie_id = ?", (mid,))
                    row = cur.fetchone()
                    if row:
                        url = row["poster_url"]
                        st = row["status"]
                        val = url if st == "success" else None
                        results[mid] = val
                        with self._lock:
                            self._memory_cache[mid] = val
                    else:
                        uncached_items.append(item)
        except Exception as e:
            logger.debug(f"SQLite batch check error: {e}")
            uncached_items = missing_items

        # If no uncached items or no API key, return immediately
        if not uncached_items or not self.api_key:
            for item in uncached_items:
                results[str(item["movie_id"])] = None
            return results

        # 3. Fetch uncached items concurrently with httpx.Client
        with httpx.Client(timeout=self.timeout) as client:
            def _fetch_worker(itm):
                m_id = str(itm["movie_id"]).strip()
                t_title = itm.get("title")
                return m_id, self.get_poster(m_id, title=t_title, client=client)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_mid = {executor.submit(_fetch_worker, itm): str(itm["movie_id"]) for itm in uncached_items}
                for future in concurrent.futures.as_completed(future_to_mid):
                    try:
                        m_id, url = future.result()
                        results[m_id] = url
                    except Exception as e:
                        m_id = future_to_mid[future]
                        logger.warning(f"Batch poster fetch error for {m_id}: {e}")
                        results[m_id] = None

        return results
