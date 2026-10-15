"""
Central configuration for the Recommendation Engine.

All paths, hyperparameters, and mode settings are configurable via
environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes")


# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(_env("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))

# ── Dataset paths ───────────────────────────────────────────────────
RAW_DATA_DIR = Path(_env("RAW_DATA_DIR", str(PROJECT_ROOT / "data" / "raw")))
PROCESSED_DATA_DIR = Path(_env("PROCESSED_DATA_DIR", str(PROJECT_ROOT / "data" / "processed")))

# The MovieLens 25M unzipped directory (contains ratings.csv, movies.csv, etc.)
MOVIELENS_DIR = Path(_env("MOVIELENS_DIR", str(Path("d:/setup/Database/ml-25m"))))

# ── Mode ────────────────────────────────────────────────────────────
DEV_MODE = _env_bool("DEV_MODE", True)
DEV_SAMPLE_SIZE = _env_int("DEV_SAMPLE_SIZE", 1_000_000)  # ~1M ratings for dev
DEV_RANDOM_SEED = _env_int("DEV_RANDOM_SEED", 42)

# ── Data split ratios (chronological) ──────────────────────────────
TRAIN_RATIO = _env_float("TRAIN_RATIO", 0.80)
VAL_RATIO = _env_float("VAL_RATIO", 0.10)
TEST_RATIO = _env_float("TEST_RATIO", 0.10)

# ── PySpark ─────────────────────────────────────────────────────────
SPARK_APP_NAME = _env("SPARK_APP_NAME", "RecEngineDataProcessing")
SPARK_DRIVER_MEMORY = _env("SPARK_DRIVER_MEMORY", "4g")
SPARK_LOCAL_DIR = _env("SPARK_LOCAL_DIR", str(PROJECT_ROOT / "data" / "spark_tmp"))

# ── Model paths ─────────────────────────────────────────────────────
MODELS_DIR = Path(_env("MODELS_DIR", str(PROJECT_ROOT / "models")))
RETRIEVAL_MODEL_DIR = MODELS_DIR / "retrieval"
RANKING_MODEL_DIR = MODELS_DIR / "ranking"
FAISS_INDEX_DIR = MODELS_DIR / "faiss"

# ── Retrieval model hyperparameters ─────────────────────────────────
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 64)
RETRIEVAL_EPOCHS = _env_int("RETRIEVAL_EPOCHS", 5)
RETRIEVAL_BATCH_SIZE = _env_int("RETRIEVAL_BATCH_SIZE", 4096)
RETRIEVAL_LEARNING_RATE = _env_float("RETRIEVAL_LEARNING_RATE", 0.1)

# ── Ranking model hyperparameters ───────────────────────────────────
RANKING_EPOCHS = _env_int("RANKING_EPOCHS", 5)
RANKING_BATCH_SIZE = _env_int("RANKING_BATCH_SIZE", 4096)
RANKING_LEARNING_RATE = _env_float("RANKING_LEARNING_RATE", 0.001)
RANKING_RELEVANCE_THRESHOLD = _env_float("RANKING_RELEVANCE_THRESHOLD", 3.5)

# ── FAISS ───────────────────────────────────────────────────────────
FAISS_NUM_CANDIDATES = _env_int("FAISS_NUM_CANDIDATES", 200)

# ── Redis ───────────────────────────────────────────────────────────
REDIS_HOST = _env("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _env_int("REDIS_PORT", 6379)
REDIS_DB = _env_int("REDIS_DB", 0)
REDIS_TTL = _env_int("REDIS_TTL", 3600)  # 1 hour

# ── FastAPI ─────────────────────────────────────────────────────────
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)

# ── Triton ──────────────────────────────────────────────────────────
TRITON_URL = _env("TRITON_URL", "localhost:8001")
TRITON_ENABLED = _env_bool("TRITON_ENABLED", False)
TRITON_MODEL_REPOSITORY = Path(_env("TRITON_MODEL_REPOSITORY", str(MODELS_DIR / "triton_model_repository")))

# ── Experiment ──────────────────────────────────────────────────────
EXPERIMENT_LOG_DIR = Path(_env("EXPERIMENT_LOG_DIR", str(PROJECT_ROOT / "data" / "experiments")))


@dataclass
class DatasetInfo:
    """Metadata about a processed dataset run."""
    mode: str  # "dev" or "full"
    total_ratings: int = 0
    sampled_ratings: int = 0
    n_users: int = 0
    n_movies: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_start_ts: Optional[str] = None
    train_end_ts: Optional[str] = None
    val_start_ts: Optional[str] = None
    val_end_ts: Optional[str] = None
    test_start_ts: Optional[str] = None
    test_end_ts: Optional[str] = None
    processing_duration_sec: float = 0.0
    sample_seed: Optional[int] = None
    sampling_method: Optional[str] = None
