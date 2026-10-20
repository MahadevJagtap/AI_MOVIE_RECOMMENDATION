"""
Ranking Feature Engineering and Transformation Pipeline.

Transforms raw user interactions and metadata into leakage-safe feature dictionaries
for the Wide-only and Wide & Deep ranking models.

Components:
  - Categorical features: userId, movieId, genres
  - Crossed categorical features:
      - activity_pop_cross (5 user activity bins x 5 movie popularity bins = 25 buckets)
      - user_movie_rating_cross (4 user avg rating bins x 4 movie avg rating bins = 16 buckets)
  - Dense numerical features:
      - norm_user_rating_count (log1p transformed, z-score standardized from train stats)
      - norm_user_avg_rating (z-score standardized from train stats)
      - norm_movie_rating_count (log1p transformed, z-score standardized from train stats)
      - norm_movie_avg_rating (z-score standardized from train stats)
      - retrieval_score (cosine similarity / dot product between Two-Tower embeddings)
  - Target:
      - is_relevant: binary indicator (1.0 if rating >= 3.5, 0.0 otherwise)
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


class RankingFeaturePipeline:
    """Leakage-safe feature pipeline for ranking models."""

    def __init__(
        self,
        stats: Optional[Dict] = None,
        user_features_df: Optional[pd.DataFrame] = None,
        movie_features_df: Optional[pd.DataFrame] = None,
        user_embeddings: Optional[Dict[str, np.ndarray]] = None,
        movie_embeddings: Optional[Dict[str, np.ndarray]] = None,
        user_vocab: Optional[List[str]] = None,
        movie_vocab: Optional[List[str]] = None,
        genre_vocab: Optional[List[str]] = None,
    ):
        self.stats = stats or {}
        self.user_features_df = user_features_df
        self.movie_features_df = movie_features_df
        self.user_embeddings = user_embeddings or {}
        self.movie_embeddings = movie_embeddings or {}
        self.user_vocab = user_vocab or []
        self.movie_vocab = movie_vocab or []
        self.genre_vocab = genre_vocab or []

        # Fast lookup caches
        self._user_stats_cache: Dict[str, Tuple[float, float]] = {}
        self._movie_stats_cache: Dict[str, Tuple[float, float, str]] = {}
        self._populate_caches()
        # Free heavy DataFrames once lookup cache is populated
        self.user_features_df = None
        self.movie_features_df = None

    def _populate_caches(self):
        """Populate in-memory lookup dicts for high-throughput single-item/candidate inference."""
        if self.user_features_df is not None:
            for row in self.user_features_df.itertuples():
                uid = str(row.userId)
                cnt = float(getattr(row, "user_train_rating_count", 0.0))
                avg = float(getattr(row, "user_train_avg_rating", self.stats.get("global_rating_mean", 3.5)))
                self._user_stats_cache[uid] = (cnt, avg)

        if self.movie_features_df is not None:
            for row in self.movie_features_df.itertuples():
                mid = str(row.movieId)
                cnt = float(getattr(row, "movie_train_rating_count", 0.0))
                avg = float(getattr(row, "movie_train_avg_rating", self.stats.get("global_rating_mean", 3.5)))
                genres = str(getattr(row, "genres", "(no genres listed)"))
                self._movie_stats_cache[mid] = (cnt, avg, genres)

    @classmethod
    def fit_from_data(
        cls,
        processed_data_dir: Union[str, Path],
        retrieval_model_dir: Optional[Union[str, Path]] = None,
    ) -> "RankingFeaturePipeline":
        """
        Fit feature normalization statistics strictly on the training partition.
        No validation or test data is used to compute scaling parameters.
        """
        data_dir = Path(processed_data_dir)
        logger.info(f"Fitting RankingFeaturePipeline from {data_dir}...")

        # Load train partition
        train_path = data_dir / "train"
        train_df = pd.read_parquet(train_path)

        # Load user and movie features (precomputed strictly from train partition in Phase 1)
        user_feat_path = data_dir / "user_features"
        user_df = pd.read_parquet(user_feat_path)

        movie_feat_path = data_dir / "movie_features"
        movie_df = pd.read_parquet(movie_feat_path)

        # Load metadata
        meta_file = data_dir / "metadata.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)

        genre_vocab = meta.get("genre_vocabulary", [])

        # Compute train statistics
        global_rating_mean = float(train_df["rating"].mean())
        global_rating_std = float(train_df["rating"].std())

        # User stats
        user_log_counts = np.log1p(user_df["user_train_rating_count"].values.astype(np.float32))
        user_log_cnt_mean = float(np.mean(user_log_counts))
        user_log_cnt_std = float(np.std(user_log_counts)) or 1.0

        user_avg_ratings = user_df["user_train_avg_rating"].values.astype(np.float32)
        user_avg_mean = float(np.mean(user_avg_ratings))
        user_avg_std = float(np.std(user_avg_ratings)) or 1.0

        user_cnt_quantiles = [
            float(q) for q in np.quantile(user_df["user_train_rating_count"], [0.2, 0.4, 0.6, 0.8])
        ]

        # Movie stats (fit on rated movies to avoid zero-skew on long tail)
        rated_movies = movie_df[movie_df["movie_train_rating_count"] > 0]
        movie_log_counts = np.log1p(rated_movies["movie_train_rating_count"].values.astype(np.float32))
        movie_log_cnt_mean = float(np.mean(movie_log_counts)) if len(movie_log_counts) > 0 else 0.0
        movie_log_cnt_std = float(np.std(movie_log_counts)) if len(movie_log_counts) > 0 else 1.0

        movie_avg_ratings = rated_movies["movie_train_avg_rating"].values.astype(np.float32)
        movie_avg_mean = float(np.mean(movie_avg_ratings)) if len(movie_avg_ratings) > 0 else global_rating_mean
        movie_avg_std = float(np.std(movie_avg_ratings)) if len(movie_avg_ratings) > 0 else 1.0

        movie_cnt_quantiles = [
            float(q) for q in np.quantile(rated_movies["movie_train_rating_count"], [0.2, 0.4, 0.6, 0.8])
        ] if len(rated_movies) > 0 else [1.0, 3.0, 10.0, 40.0]

        stats = {
            "global_rating_mean": global_rating_mean,
            "global_rating_std": global_rating_std,
            "user_log_cnt_mean": user_log_cnt_mean,
            "user_log_cnt_std": user_log_cnt_std,
            "user_avg_mean": user_avg_mean,
            "user_avg_std": user_avg_std,
            "user_cnt_quantiles": user_cnt_quantiles,
            "movie_log_cnt_mean": movie_log_cnt_mean,
            "movie_log_cnt_std": movie_log_cnt_std,
            "movie_avg_mean": movie_avg_mean,
            "movie_avg_std": movie_avg_std,
            "movie_cnt_quantiles": movie_cnt_quantiles,
            "user_rating_quantiles": [3.0, 3.5, 4.0],
            "movie_rating_quantiles": [2.8, 3.3, 3.8],
        }

        # Load Two-Tower embeddings if directory provided
        user_embeddings = {}
        movie_embeddings = {}
        user_vocab = []
        movie_vocab = []

        if retrieval_model_dir is not None:
            retrieval_path = Path(retrieval_model_dir)
            cfg_file = retrieval_path / "model_config.json"
            if cfg_file.exists():
                with open(cfg_file, "r") as f:
                    cfg = json.load(f)
                user_vocab = cfg.get("user_vocab", [])
                movie_vocab = cfg.get("movie_vocab", [])

            # Load catalog movie embeddings
            movie_emb_file = retrieval_path / "movie_embeddings.npy"
            movie_ids_file = retrieval_path / "movie_ids.npy"
            if movie_emb_file.exists() and movie_ids_file.exists():
                m_embs = np.load(movie_emb_file)
                m_ids = np.load(movie_ids_file)
                for mid, emb in zip(m_ids, m_embs):
                    movie_embeddings[str(mid)] = emb

            # Extract user embeddings from trained UserTower
            try:
                from src.retrieval.two_tower import TwoTowerModel
                tt_model = TwoTowerModel.load_artifacts(retrieval_path)
                logger.info(f"Extracting user embeddings for {len(user_vocab)} users...")
                user_ids_tensor = tf.constant(user_vocab, dtype=tf.string)
                user_embs_array = tt_model.user_tower(user_ids_tensor).numpy()
                for uid, emb in zip(user_vocab, user_embs_array):
                    user_embeddings[str(uid)] = emb
                logger.info("Successfully cached user and movie embeddings.")
            except Exception as e:
                logger.warning(f"Could not extract user embeddings from TwoTowerModel: {e}")

        pipeline = cls(
            stats=stats,
            user_features_df=user_df,
            movie_features_df=movie_df,
            user_embeddings=user_embeddings,
            movie_embeddings=movie_embeddings,
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            genre_vocab=genre_vocab,
        )
        return pipeline

    def _bin_values(self, values: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """Assign values to discrete bins [0, len(thresholds)]."""
        return np.digitize(values, bins=thresholds).astype(np.int32)

    def compute_retrieval_scores(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """
        Compute vectorized cosine similarity / dot product scores between users and movies.
        Returns zeros for unlearned / cold-start IDs.
        """
        n = len(user_ids)
        scores = np.zeros(n, dtype=np.float32)

        if not self.user_embeddings or not self.movie_embeddings:
            return scores

        # Map to embeddings
        user_ids_str = [str(u) for u in user_ids]
        movie_ids_str = [str(m) for m in movie_ids]

        emb_dim = len(next(iter(self.user_embeddings.values()))) if self.user_embeddings else 64
        u_mat = np.zeros((n, emb_dim), dtype=np.float32)
        m_mat = np.zeros((n, emb_dim), dtype=np.float32)

        valid_mask = np.zeros(n, dtype=bool)

        for i, (u, m) in enumerate(zip(user_ids_str, movie_ids_str)):
            has_u = u in self.user_embeddings
            has_m = m in self.movie_embeddings
            if has_u:
                u_mat[i] = self.user_embeddings[u]
            if has_m:
                m_mat[i] = self.movie_embeddings[m]
            if has_u and has_m:
                valid_mask[i] = True

        dots = np.sum(u_mat * m_mat, axis=1)
        # Cosine similarity is in [-1, 1]
        scores[valid_mask] = np.clip(dots[valid_mask], -1.0, 1.0)
        return scores

    def transform_interactions(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Transform interaction DataFrame into model input dictionary and binary labels.

        Args:
            df: DataFrame containing ['userId', 'movieId'] and optionally ['rating']
            is_training: If True, returns binary labels (rating >= 3.5 -> 1.0)

        Returns:
            (features_dict, labels_array or None)
        """
        user_ids = df["userId"].astype(str).values
        movie_ids = df["movieId"].astype(str).values

        # User stats
        user_counts = np.zeros(len(df), dtype=np.float32)
        user_avgs = np.full(len(df), self.stats.get("user_avg_mean", 3.5), dtype=np.float32)

        for i, uid in enumerate(user_ids):
            if uid in self._user_stats_cache:
                cnt, avg = self._user_stats_cache[uid]
                user_counts[i] = cnt
                user_avgs[i] = avg

        # Movie stats & genres
        movie_counts = np.zeros(len(df), dtype=np.float32)
        movie_avgs = np.full(len(df), self.stats.get("movie_avg_mean", 3.5), dtype=np.float32)
        genres = np.full(len(df), "(no genres listed)", dtype=object)

        for i, mid in enumerate(movie_ids):
            if mid in self._movie_stats_cache:
                cnt, avg, g = self._movie_stats_cache[mid]
                movie_counts[i] = cnt
                movie_avgs[i] = avg
                genres[i] = g

        # Standardize continuous features using train stats
        u_log_cnt = np.log1p(user_counts)
        norm_user_count = (u_log_cnt - self.stats.get("user_log_cnt_mean", 0.0)) / self.stats.get("user_log_cnt_std", 1.0)
        norm_user_avg = (user_avgs - self.stats.get("user_avg_mean", 3.5)) / self.stats.get("user_avg_std", 1.0)

        m_log_cnt = np.log1p(movie_counts)
        norm_movie_count = (m_log_cnt - self.stats.get("movie_log_cnt_mean", 0.0)) / self.stats.get("movie_log_cnt_std", 1.0)
        norm_movie_avg = (movie_avgs - self.stats.get("movie_avg_mean", 3.5)) / self.stats.get("movie_avg_std", 1.0)

        # Retrieval similarity score
        retrieval_score = self.compute_retrieval_scores(user_ids, movie_ids)

        # Crossed categorical features
        # 1. Activity (5 bins) x Popularity (5 bins) -> 25 buckets
        u_act_bins = self._bin_values(user_counts, self.stats.get("user_cnt_quantiles", [31.0, 50.0, 89.0, 192.0]))
        m_pop_bins = self._bin_values(movie_counts, self.stats.get("movie_cnt_quantiles", [1.0, 3.0, 10.0, 41.0]))
        activity_pop_cross = (u_act_bins * 5 + m_pop_bins).astype(np.int32)

        # 2. User rating temperament (4 bins) x Movie rating level (4 bins) -> 16 buckets
        u_rat_bins = self._bin_values(user_avgs, self.stats.get("user_rating_quantiles", [3.0, 3.5, 4.0]))
        m_rat_bins = self._bin_values(movie_avgs, self.stats.get("movie_rating_quantiles", [2.8, 3.3, 3.8]))
        user_movie_rating_cross = (u_rat_bins * 4 + m_rat_bins).astype(np.int32)

        # Dense continuous feature matrix
        dense_features = np.column_stack([
            norm_user_count,
            norm_user_avg,
            norm_movie_count,
            norm_movie_avg,
            retrieval_score,
        ]).astype(np.float32)

        features = {
            "userId": user_ids.astype(str),
            "movieId": movie_ids.astype(str),
            "genres": genres.astype(str),
            "activity_pop_cross": activity_pop_cross,
            "user_movie_rating_cross": user_movie_rating_cross,
            "dense_features": dense_features,
        }

        labels = None
        if "rating" in df.columns:
            # Positive relevance label threshold: rating >= 3.5
            labels = (df["rating"].values >= 3.5).astype(np.float32)

        return features, labels

    def transform_candidates(
        self,
        user_id: Union[int, str],
        candidate_movie_ids: List[Union[int, str]],
    ) -> Dict[str, np.ndarray]:
        """
        High-throughput candidate list transformation for online serving and reranking.

        Args:
            user_id: Target user ID
            candidate_movie_ids: List of candidate movie IDs (e.g. top-100 from FAISS)

        Returns:
            Dictionary matching model input signatures
        """
        df = pd.DataFrame({
            "userId": [str(user_id)] * len(candidate_movie_ids),
            "movieId": [str(mid) for mid in candidate_movie_ids],
        })
        features, _ = self.transform_interactions(df, is_training=False)
        return features

    def save(self, save_dir: Union[str, Path]):
        """Persist feature pipeline configuration and vocabularies."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "stats": self.stats,
            "user_vocab": self.user_vocab,
            "movie_vocab": self.movie_vocab,
            "genre_vocab": self.genre_vocab,
            "n_user_embeddings": len(self.user_embeddings),
            "n_movie_embeddings": len(self.movie_embeddings),
        }
        with open(save_path / "ranking_pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved RankingFeaturePipeline configuration to {save_path}")

    @classmethod
    def load(
        cls,
        save_dir: Union[str, Path],
        processed_data_dir: Optional[Union[str, Path]] = None,
        retrieval_model_dir: Optional[Union[str, Path]] = None,
        two_tower_model: Optional[Any] = None,
    ) -> "RankingFeaturePipeline":
        """Load pipeline from persisted configuration with memory optimizations."""
        import gc

        save_path = Path(save_dir)
        config_file = save_path / "ranking_pipeline_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_file}")

        with open(config_file, "r") as f:
            config = json.load(f)

        stats = config["stats"]
        user_vocab = config.get("user_vocab", [])
        movie_vocab = config.get("movie_vocab", [])
        genre_vocab = config.get("genre_vocab", [])

        user_df = None
        movie_df = None
        if processed_data_dir is not None:
            p_path = Path(processed_data_dir)
            user_df = pd.read_parquet(
                p_path / "user_features",
                columns=["userId", "user_train_rating_count", "user_train_avg_rating"],
            )
            movie_df = pd.read_parquet(
                p_path / "movie_features",
                columns=["movieId", "movie_train_rating_count", "movie_train_avg_rating", "genres"],
            )

        user_embeddings = {}
        movie_embeddings = {}
        if retrieval_model_dir is not None:
            r_path = Path(retrieval_model_dir)
            movie_emb_file = r_path / "movie_embeddings.npy"
            movie_ids_file = r_path / "movie_ids.npy"
            if movie_emb_file.exists() and movie_ids_file.exists():
                m_embs = np.load(movie_emb_file)
                m_ids = np.load(movie_ids_file)
                for mid, emb in zip(m_ids, m_embs):
                    movie_embeddings[str(mid)] = emb

            try:
                if two_tower_model is not None:
                    tt_model = two_tower_model
                else:
                    from src.retrieval.two_tower import TwoTowerModel
                    tt_model = TwoTowerModel.load_artifacts(r_path)

                user_ids_tensor = tf.constant(user_vocab, dtype=tf.string)
                user_embs_array = tt_model.user_tower(user_ids_tensor).numpy()
                for uid, emb in zip(user_vocab, user_embs_array):
                    user_embeddings[str(uid)] = emb
                del user_ids_tensor, user_embs_array
                if two_tower_model is None:
                    del tt_model
            except Exception as e:
                logger.warning(f"Could not load TwoTower user embeddings: {e}")

        pipeline = cls(
            stats=stats,
            user_features_df=user_df,
            movie_features_df=movie_df,
            user_embeddings=user_embeddings,
            movie_embeddings=movie_embeddings,
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            genre_vocab=genre_vocab,
        )
        del user_df, movie_df
        gc.collect()
        return pipeline
