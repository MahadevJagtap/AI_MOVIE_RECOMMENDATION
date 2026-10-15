"""
Popularity-based baseline and cold-start fallback recommendation model.

Uses training-period interaction popularity:
score = rating_count * avg_rating
Provides filtered recommendations for unseen items.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """
    Popularity recommender based on training interactions.
    
    Serves as:
      1. Baseline candidate retrieval for offline experiments.
      2. Cold-start fallback for unknown users or empty retrieval results.
    """

    def __init__(
        self,
        popularity_path: Optional[Union[str, Path]] = None,
        movies_path: Optional[Union[str, Path]] = None,
    ):
        self.popularity_df: Optional[pd.DataFrame] = None
        self.movies_meta: Dict[int, Dict[str, str]] = {}
        self.movies_path = Path(movies_path) if movies_path else None
        self.popularity_path = Path(popularity_path) if popularity_path else None

        if self.movies_path and self.movies_path.exists():
            self._load_movies_meta(self.movies_path)

        if self.popularity_path and self.popularity_path.exists():
            self.load(self.popularity_path)

    def _load_movies_meta(self, movies_path: Path):
        """Load movie metadata from movies.csv for fast title/genre lookup."""
        try:
            df = pd.read_csv(movies_path)
            for _, row in df.iterrows():
                mid = int(row["movieId"])
                self.movies_meta[mid] = {
                    "title": str(row.get("title", "")),
                    "genres": str(row.get("genres", "")),
                }
            logger.info(f"Loaded metadata for {len(self.movies_meta)} catalog movies")
        except Exception as e:
            logger.warning(f"Could not load movie metadata from {movies_path}: {e}")

    def fit_from_df(self, train_ratings_df: pd.DataFrame, top_n: int = 500) -> "PopularityRecommender":
        """
        Fit popularity model from a pandas DataFrame of training ratings.
        
        Args:
            train_ratings_df: DataFrame with at least ['movieId', 'rating']
            top_n: Number of top items to keep
        """
        agg = (
            train_ratings_df.groupby("movieId")
            .agg(
                pop_count=("rating", "count"),
                pop_avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        agg["popularity_score"] = agg["pop_count"] * agg["pop_avg_rating"]
        agg = agg.sort_values(by="popularity_score", ascending=False).head(top_n).reset_index(drop=True)
        agg["popularity_rank"] = agg.index + 1

        self.popularity_df = agg
        logger.info(f"Fitted popularity model with top {len(self.popularity_df)} items")
        return self

    def load(self, popularity_path: Path):
        """Load pre-computed popularity rankings (Parquet or CSV)."""
        popularity_path = Path(popularity_path)
        if popularity_path.is_dir() or popularity_path.suffix == ".parquet":
            try:
                self.popularity_df = pd.read_parquet(popularity_path)
            except Exception:
                # If pyarrow/fastparquet not yet installed or spark dir with parquet parts
                import glob
                part_files = list(popularity_path.glob("*.parquet"))
                if part_files:
                    try:
                        self.popularity_df = pd.read_parquet(part_files[0])
                    except Exception as e:
                        logger.warning(f"Could not load parquet: {e}")
        elif popularity_path.suffix == ".csv":
            self.popularity_df = pd.read_csv(popularity_path)

        if self.popularity_df is not None:
            if "popularity_rank" not in self.popularity_df.columns:
                self.popularity_df = self.popularity_df.sort_values(
                    by="popularity_score", ascending=False
                ).reset_index(drop=True)
                self.popularity_df["popularity_rank"] = self.popularity_df.index + 1
            logger.info(f"Loaded popularity table with {len(self.popularity_df)} items")

    def recommend(
        self,
        user_id: Optional[int] = None,
        k: int = 10,
        filter_movie_ids: Optional[Set[int]] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k popular recommendations, optionally filtering consumed items.
        
        Returns:
            List of dictionaries containing:
                - movieId (int)
                - title (str)
                - genres (str)
                - popularity_score (float)
                - popularity_rank (int)
        """
        if self.popularity_df is None or len(self.popularity_df) == 0:
            logger.warning("Popularity model has no data to recommend from")
            return []

        filter_set = set(filter_movie_ids) if filter_movie_ids else set()
        results: List[Dict] = []

        for _, row in self.popularity_df.iterrows():
            mid = int(row["movieId"])
            if mid in filter_set:
                continue

            meta = self.movies_meta.get(mid, {})
            title = meta.get("title", f"Movie {mid}")
            genres = meta.get("genres", "")

            results.append({
                "movieId": mid,
                "title": title,
                "genres": genres,
                "popularity_score": float(row.get("popularity_score", 0.0)),
                "popularity_rank": int(row.get("popularity_rank", len(results) + 1)),
                "source": "popularity_baseline",
            })

            if len(results) >= k:
                break

        return results

    def get_candidate_ids(
        self,
        k: int = 200,
        filter_movie_ids: Optional[Set[int]] = None,
    ) -> List[int]:
        """Return list of candidate movie IDs only."""
        recs = self.recommend(k=k, filter_movie_ids=filter_movie_ids)
        return [r["movieId"] for r in recs]
