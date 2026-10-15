"""
FastAPI Serving Application for Large-Scale Personalized Recommendation Engine.

Endpoints:
  - GET  /health: System health, Redis connection status, model loading status, cache metrics.
  - GET  /recommend/{user_id}: Two-stage recommendation pipeline (Two-Tower FAISS -> Ranking).
  - GET  /candidates/{user_id}: Raw Stage-1 candidate generation inspection.
  - GET  /popular: Cold-start fallback popularity baseline.
  - POST /feedback: Genuine impression and click event logging.
  - GET  /metrics/ctr: Live demo CTR statistics and disclaimers.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np

FRONTEND_DIR = PROJECT_ROOT / "frontend"

from configs.config import (
    PROCESSED_DATA_DIR,
    RETRIEVAL_MODEL_DIR,
    FAISS_INDEX_DIR,
    RANKING_MODEL_DIR,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_TTL,
    TRITON_URL,
    TRITON_ENABLED,
)
from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndexManager
from src.retrieval.popularity import PopularityRecommender
from src.features.ranking_features import RankingFeaturePipeline
from src.ranking.wide_and_deep import WideOnlyModel, WideAndDeepModel
from src.serving.cache import RedisCacheManager
from src.serving.feedback import FeedbackEvent, FeedbackLogger
from src.serving.triton_client import TritonInferenceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serving_api")

MODEL_VERSION = "v1.0"


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class RecommendationItem(BaseModel):
    movie_id: str
    title: str
    genres: str
    score: float
    rank: int
    source: str


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    k: int
    variant: str
    model_version: str
    cached: bool
    is_cold_start: bool
    recommendations: List[RecommendationItem]
    latency_ms: float


class CandidateItem(BaseModel):
    movie_id: str
    title: str
    genres: str
    retrieval_score: float
    rank: int


class CandidateResponse(BaseModel):
    user_id: str
    k: int
    is_cold_start: bool
    candidates: List[CandidateItem]
    retrieval_latency_ms: float


class PopularMovieItem(BaseModel):
    movie_id: str
    title: str
    genres: str
    popularity_score: float
    rank: int


class PopularResponse(BaseModel):
    limit: int
    movies: List[PopularMovieItem]


class WatchlistItem(BaseModel):
    movie_id: str
    title: str
    genres: str
    timestamp: str
    event_id: str


class UserFeedbackResponse(BaseModel):
    user_id: str
    saved_movie_ids: List[str]
    total_saved: int
    watchlist: List[WatchlistItem]


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    models_loaded: Dict[str, bool]
    total_catalog_movies: int
    cache_stats: Dict[str, Any]
    version: str


# -----------------------------------------------------------------------------
# Application State Container & Lifespan
# -----------------------------------------------------------------------------
def safe_read_parquet(path, columns=None):
    """Read parquet with graceful engine fallback between pyarrow, fastparquet, and auto."""
    for engine in ["pyarrow", "fastparquet", "auto"]:
        try:
            return pd.read_parquet(path, columns=columns, engine=engine)
        except Exception:
            continue
    return pd.read_parquet(path, columns=columns)


class ModelManager:
    """Singleton container for thread-safe model and index instances."""

    def __init__(self):
        self.two_tower: Optional[TwoTowerModel] = None
        self.faiss_manager: Optional[FAISSIndexManager] = None
        self.feature_pipeline: Optional[RankingFeaturePipeline] = None
        self.wide_model: Optional[WideOnlyModel] = None
        self.wd_model: Optional[WideAndDeepModel] = None
        self.popularity: Optional[PopularityRecommender] = None
        self.cache: Optional[RedisCacheManager] = None
        self.feedback_logger: Optional[FeedbackLogger] = None
        self.triton_client: Optional[TritonInferenceClient] = None
        self.movie_meta: Dict[str, Tuple[str, str]] = {}  # mid -> (title, genres)
        self.train_consumed: Dict[str, Set[int]] = {}     # uid -> set of int(mid)
        self.user_vocab_set: Set[str] = set()
        self.is_ready = False

    def initialize(
        self,
        data_dir: Path = PROCESSED_DATA_DIR,
        retrieval_dir: Path = RETRIEVAL_MODEL_DIR,
        faiss_dir: Path = FAISS_INDEX_DIR,
        ranking_dir: Path = RANKING_MODEL_DIR,
    ):
        """Load all artifacts once during application startup."""
        logger.info("Initializing ModelManager and loading ML artifacts...")
        t0 = time.time()

        # 1. Movie Metadata
        meta_parquet = data_dir / "movie_features"
        if meta_parquet.exists():
            df_m = safe_read_parquet(meta_parquet, columns=["movieId", "title", "genres"])
            self.movie_meta = {
                str(r.movieId): (str(r.title), str(r.genres))
                for r in df_m.itertuples()
            }
            logger.info(f"Loaded {len(self.movie_meta)} movie metadata records.")

        # 2. Train History (Consumed Items for Filtering)
        train_parquet = data_dir / "train"
        if train_parquet.exists():
            df_t = safe_read_parquet(train_parquet, columns=["userId", "movieId"])
            for uid, mid in zip(df_t["userId"].values, df_t["movieId"].values):
                u_str = str(uid)
                if u_str not in self.train_consumed:
                    self.train_consumed[u_str] = set()
                self.train_consumed[u_str].add(int(mid))
            logger.info(f"Loaded consumed history for {len(self.train_consumed)} users.")

        # 3. Two-Tower Model
        if retrieval_dir.exists():
            self.two_tower = TwoTowerModel.load_artifacts(retrieval_dir)
            self.user_vocab_set = set(str(u) for u in self.two_tower.user_vocab)
            logger.info(f"Loaded TwoTowerModel with {len(self.user_vocab_set)} learned users.")

        # 4. FAISS Index
        if faiss_dir.exists():
            self.faiss_manager = FAISSIndexManager.load_artifacts(faiss_dir)
            logger.info(f"Loaded FAISSIndexManager with {self.faiss_manager.total_items} items.")

        # 5. Ranking Feature Pipeline
        pipeline_dir = ranking_dir / "pipeline"
        if pipeline_dir.exists():
            self.feature_pipeline = RankingFeaturePipeline.load(
                pipeline_dir,
                processed_data_dir=data_dir,
                retrieval_model_dir=retrieval_dir,
            )
            logger.info("Loaded RankingFeaturePipeline.")

        # 6. Wide-only Ranking Model (Variant A)
        wide_dir = ranking_dir / "wide_only"
        if wide_dir.exists():
            self.wide_model = WideOnlyModel.load_artifacts(wide_dir)
            logger.info("Loaded WideOnlyModel.")

        # 7. Wide & Deep Ranking Model (Variant B)
        wd_dir = ranking_dir / "wide_and_deep"
        if wd_dir.exists():
            self.wd_model = WideAndDeepModel.load_artifacts(wd_dir)
            logger.info("Loaded WideAndDeepModel.")

        # 8. Popularity Recommender (Cold-Start Fallback)
        pop_path = data_dir / "popularity"
        if pop_path.exists():
            self.popularity = PopularityRecommender(popularity_path=pop_path)
            # Pre-populate popularity model with catalog metadata to prevent generic labels
            for mid_str, (title, genres) in self.movie_meta.items():
                try:
                    self.popularity.movies_meta[int(mid_str)] = {"title": title, "genres": genres}
                except (ValueError, TypeError):
                    continue
            logger.info(f"Loaded PopularityRecommender with {len(self.popularity.movies_meta)} catalog titles.")

        # 9. Redis Cache Manager
        self.cache = RedisCacheManager(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            default_ttl=REDIS_TTL,
        )

        # 10. Feedback Logger
        self.feedback_logger = FeedbackLogger(PROJECT_ROOT / "data" / "feedback")

        # 11. Triton Inference Client
        self.triton_client = TritonInferenceClient(
            url=TRITON_URL,
            enabled=TRITON_ENABLED,
            connection_timeout=1.0,
            network_timeout=2.0,
        )
        if self.triton_client.is_available():
            logger.info(f"Triton Inference Server is ONLINE at {TRITON_URL}")
        else:
            logger.info(f"Triton Inference Server is OFFLINE / DISABLED (TRITON_ENABLED={TRITON_ENABLED}). Using local TensorFlow fallback.")

        self.is_ready = True
        logger.info(f"ModelManager successfully initialized in {time.time() - t0:.2f}s")


# Global model manager
manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: loads models once at startup, cleans up on shutdown."""
    manager.initialize()
    yield
    logger.info("Application shutting down.")


# -----------------------------------------------------------------------------
# FastAPI App Definition
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Large-Scale Personalized Recommendation Engine API",
    description="Production-grade two-stage recommendation service (Two-Tower Retrieval + Wide & Deep Ranking)",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Route Handlers
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def get_health():
    """System health status, Redis connectivity, and model availability."""
    models_status = {
        "two_tower": manager.two_tower is not None,
        "faiss_index": manager.faiss_manager is not None and manager.faiss_manager.is_built,
        "feature_pipeline": manager.feature_pipeline is not None,
        "ranking_wide_only": manager.wide_model is not None,
        "ranking_wide_and_deep": manager.wd_model is not None,
        "popularity": manager.popularity is not None,
        "triton_server": manager.triton_client.is_available() if manager.triton_client else False,
    }
    all_ready = all(
        v for k, v in models_status.items() if k != "triton_server"
    ) and manager.is_ready
    cache_stats = manager.cache.get_stats() if manager.cache else {}

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        redis_connected=manager.cache.is_connected if manager.cache else False,
        models_loaded=models_status,
        total_catalog_movies=len(manager.movie_meta),
        cache_stats=cache_stats,
        version=MODEL_VERSION,
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    k: int = Query(10, ge=1, le=100, description="Number of recommendations to return"),
    variant: str = Query("wide_and_deep", description="Ranking variant: 'wide_and_deep' or 'wide_only'"),
):
    """
    Two-Stage Personalized Recommendations:
      1. Stage 1: Candidate retrieval using Two-Tower embeddings and FAISS CPU vector index.
      2. Stage 2: Pointwise reranking using Wide-only or Wide & Deep neural network.
      3. Fallback: Popularity baseline for cold-start or unknown users.
    """
    t0 = time.perf_counter()
    clean_user = str(user_id).strip()
    if not clean_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id cannot be empty")

    clean_variant = variant.strip().lower()
    if clean_variant not in ("wide_and_deep", "wide_only"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid variant '{variant}'. Allowed: 'wide_and_deep', 'wide_only'",
        )

    # 1. Check Redis Cache
    cache_key = manager.cache.make_key(clean_user, k, clean_variant, MODEL_VERSION) if manager.cache else None
    if cache_key:
        cached_resp = manager.cache.get(cache_key)
        if cached_resp is not None:
            cached_resp["cached"] = True
            cached_resp["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return RecommendationResponse(**cached_resp)

    # 2. Check User Status: Warm vs Cold-Start
    is_cold_start = clean_user not in manager.user_vocab_set
    filter_consumed_set = manager.train_consumed.get(clean_user, set())

    recommendations: List[RecommendationItem] = []

    if not is_cold_start and manager.two_tower and manager.faiss_manager and manager.feature_pipeline:
        try:
            # Stage 1: Two-Tower Vector Candidate Retrieval (Triton or local fallback)
            if manager.triton_client and manager.triton_client.is_available():
                try:
                    user_emb = manager.triton_client.infer_user_embedding(clean_user)[0]
                except Exception as e:
                    logger.warning(f"Triton user_tower error ({e}); using local TwoTowerModel.")
                    user_emb = manager.two_tower.get_user_embedding(clean_user)
            else:
                user_emb = manager.two_tower.get_user_embedding(clean_user)

            # Retrieve pool of top-100 candidates to rank
            candidate_pool = manager.faiss_manager.search(
                query_embedding=user_emb,
                top_k=max(100, k * 2),
                filter_movie_ids=filter_consumed_set,
            )

            if candidate_pool:
                cand_ids = [str(mid) for mid, _ in candidate_pool]

                # Stage 2: Feature Transformation & Main Ranking
                cand_features = manager.feature_pipeline.transform_candidates(clean_user, cand_ids)

                if clean_variant == "wide_only" and manager.wide_model:
                    scores = manager.wide_model.predict_scores(cand_features)
                elif manager.triton_client and manager.triton_client.is_available():
                    try:
                        scores = manager.triton_client.infer_ranking_scores(cand_features)
                    except Exception as e:
                        logger.warning(f"Triton ranking_model error ({e}); using local WideAndDeepModel.")
                        scores = manager.wd_model.predict_scores(cand_features) if manager.wd_model else np.array([score for _, score in candidate_pool], dtype=np.float32)
                elif manager.wd_model:
                    scores = manager.wd_model.predict_scores(cand_features)
                else:
                    scores = np.array([score for _, score in candidate_pool], dtype=np.float32)

                # Sort candidates descending by predicted score
                sorted_indices = np.argsort(-scores)[:k]
                for rank_idx, idx in enumerate(sorted_indices, 1):
                    mid_str = cand_ids[idx]
                    title, genres = manager.movie_meta.get(mid_str, ("Unknown Title", "Unknown"))
                    recommendations.append(
                        RecommendationItem(
                            movie_id=mid_str,
                            title=title,
                            genres=genres,
                            score=float(round(scores[idx], 4)),
                            rank=rank_idx,
                            source="two_stage_ranking",
                        )
                    )
        except Exception as e:
            logger.warning(f"Error during two-stage ranking for user {clean_user}: {e}. Triggering fallback.")
            is_cold_start = True

    # 3. Fallback: Popularity Recommender (for cold-start or empty retrieval)
    if not recommendations and manager.popularity:
        pop_items = manager.popularity.recommend(k=k, filter_movie_ids=filter_consumed_set)
        for rank_idx, item in enumerate(pop_items[:k], 1):
            mid_str = str(item["movieId"])
            meta_title, meta_genres = manager.movie_meta.get(mid_str, (None, None))
            raw_title = str(item.get("title", "")).strip()
            # Ensure catalog title takes absolute priority over generic 'Movie {mid}' placeholders
            title = meta_title or (raw_title if not raw_title.startswith("Movie ") else None) or f"Movie {mid_str}"
            genres = meta_genres or item.get("genres") or "Unknown"
            recommendations.append(
                RecommendationItem(
                    movie_id=mid_str,
                    title=title,
                    genres=genres,
                    score=float(round(item.get("popularity_score", 0.0), 4)),
                    rank=rank_idx,
                    source="popularity_fallback",
                )
            )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    req_id = str(uuid.uuid4())

    response_payload = {
        "request_id": req_id,
        "user_id": clean_user,
        "k": len(recommendations),
        "variant": clean_variant,
        "model_version": MODEL_VERSION,
        "cached": False,
        "is_cold_start": is_cold_start,
        "recommendations": [rec.model_dump() for rec in recommendations],
        "latency_ms": latency_ms,
    }

    # 4. Cache in Redis (if enabled)
    if cache_key and manager.cache and not is_cold_start:
        manager.cache.set(cache_key, response_payload, ttl=REDIS_TTL)

    return RecommendationResponse(**response_payload)


@app.get("/candidates/{user_id}", response_model=CandidateResponse)
def get_candidates(
    user_id: str,
    k: int = Query(100, ge=1, le=200, description="Number of Stage-1 candidates"),
):
    """Inspect raw Stage-1 retrieval candidates and similarity scores before reranking."""
    t0 = time.perf_counter()
    clean_user = str(user_id).strip()
    if not clean_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id cannot be empty")

    is_cold_start = clean_user not in manager.user_vocab_set
    filter_consumed_set = manager.train_consumed.get(clean_user, set())

    candidates: List[CandidateItem] = []

    if not is_cold_start and manager.two_tower and manager.faiss_manager:
        try:
            user_emb = manager.two_tower.get_user_embedding(clean_user)
            pool = manager.faiss_manager.search(
                query_embedding=user_emb,
                top_k=k,
                filter_movie_ids=filter_consumed_set,
            )
            for rank_idx, (mid, score) in enumerate(pool[:k], 1):
                mid_str = str(mid)
                title, genres = manager.movie_meta.get(mid_str, ("Unknown Title", "Unknown"))
                candidates.append(
                    CandidateItem(
                        movie_id=mid_str,
                        title=title,
                        genres=genres,
                        retrieval_score=float(round(score, 4)),
                        rank=rank_idx,
                    )
                )
        except Exception as e:
            logger.warning(f"Error fetching candidates for {clean_user}: {e}")
            is_cold_start = True

    if not candidates and manager.popularity:
        pop_items = manager.popularity.recommend(k=k, filter_movie_ids=filter_consumed_set)
        for rank_idx, item in enumerate(pop_items[:k], 1):
            mid_str = str(item["movieId"])
            title, genres = manager.movie_meta.get(mid_str, (item.get("title", ""), item.get("genres", "")))
            candidates.append(
                CandidateItem(
                    movie_id=mid_str,
                    title=title,
                    genres=genres,
                    retrieval_score=float(round(item.get("popularity_score", 0.0), 4)),
                    rank=rank_idx,
                )
            )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return CandidateResponse(
        user_id=clean_user,
        k=len(candidates),
        is_cold_start=is_cold_start,
        candidates=candidates,
        retrieval_latency_ms=latency_ms,
    )


@app.get("/popular", response_model=PopularResponse)
def get_popular_movies(
    limit: int = Query(10, ge=1, le=100, description="Number of popular movies"),
):
    """Retrieve top global popular movies for fallback or cold users."""
    if not manager.popularity:
        raise HTTPException(status_code=503, detail="Popularity model unavailable")

    items = manager.popularity.recommend(k=limit)
    res = []
    for rank_idx, item in enumerate(items, 1):
        mid_str = str(item["movieId"])
        title, genres = manager.movie_meta.get(mid_str, (item.get("title", ""), item.get("genres", "")))
        res.append(
            PopularMovieItem(
                movie_id=mid_str,
                title=title,
                genres=genres,
                popularity_score=float(round(item.get("popularity_score", 0.0), 4)),
                rank=rank_idx,
            )
        )
    return PopularResponse(limit=len(res), movies=res)


@app.post("/feedback")
def record_feedback(event: FeedbackEvent):
    """
    Log genuine impression and click interaction events.
    Validates event structure, checks impression attribution for clicks, and persists to SQLite and JSONL.
    """
    if not manager.feedback_logger:
        raise HTTPException(status_code=503, detail="Feedback logging service unavailable")

    try:
        res = manager.feedback_logger.log_event(event)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to record feedback event: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal logging error")


@app.get("/feedback/{user_id}", response_model=UserFeedbackResponse)
def get_user_watchlist_endpoint(user_id: str):
    """Retrieve saved watchlist items and IDs for a specific user from durable SQLite storage."""
    if not manager.feedback_logger:
        raise HTTPException(status_code=503, detail="Feedback logger unavailable")

    clean_user = str(user_id).strip()
    if not clean_user:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")

    items = manager.feedback_logger.get_user_watchlist(clean_user)
    saved_ids = [item["movie_id"] for item in items]
    watchlist_detailed: List[WatchlistItem] = []

    for item in items:
        mid_str = str(item["movie_id"])
        meta_title, meta_genres = manager.movie_meta.get(mid_str, (f"Movie {mid_str}", "Unknown"))
        watchlist_detailed.append(
            WatchlistItem(
                movie_id=mid_str,
                title=meta_title,
                genres=meta_genres,
                timestamp=item["timestamp"],
                event_id=item["event_id"],
            )
        )

    return UserFeedbackResponse(
        user_id=clean_user,
        saved_movie_ids=saved_ids,
        total_saved=len(saved_ids),
        watchlist=watchlist_detailed,
    )


@app.delete("/feedback/{user_id}/{movie_id}")
def delete_watchlist_item_endpoint(user_id: str, movie_id: str):
    """Remove a movie from a user's watchlist in durable SQLite storage."""
    if not manager.feedback_logger:
        raise HTTPException(status_code=503, detail="Feedback logger unavailable")

    clean_user = str(user_id).strip()
    clean_movie = str(movie_id).strip()
    if not clean_user or not clean_movie:
        raise HTTPException(status_code=400, detail="user_id and movie_id cannot be empty")

    removed = manager.feedback_logger.remove_watchlist_item(clean_user, clean_movie)
    return {
        "status": "success" if removed else "not_found",
        "user_id": clean_user,
        "movie_id": clean_movie,
        "removed": removed,
    }


@app.get("/metrics/ctr")
def get_ctr_metrics(window_hours: Optional[float] = Query(None, ge=0.1, le=720.0)):
    """Return genuine CTR calculated exclusively from recorded UI impression and click events."""
    if not manager.feedback_logger:
        raise HTTPException(status_code=503, detail="Feedback logger unavailable")

    return manager.feedback_logger.compute_ctr(window_hours=window_hours)


@app.get("/metrics/offline")
def get_offline_metrics():
    """Return offline evaluation results from ranking_experiment.json."""
    import json
    exp_file = PROJECT_ROOT / "data" / "experiments" / "ranking_experiment.json"
    if exp_file.exists():
        with open(exp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Offline experiment metrics not found")


@app.get("/", include_in_schema=False)
def serve_frontend_root():
    """Serve CineMatch HTML frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "CineMatch API is running. Open /docs for Swagger documentation."}


# Mount static assets for frontend (css, js, assets)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting CineMatch server on {host}:{port}...")
    uvicorn.run("src.serving.api:app", host=host, port=port, reload=False)
