"""
PySpark-based data processing for MovieLens 25M.

Handles:
  - Loading and sampling (dev mode vs full mode)
  - Chronological train/val/test splitting
  - Leakage-safe feature engineering
  - Saving processed datasets and metadata
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, LongType, StringType
)

logger = logging.getLogger(__name__)


# ── Explicit schema definitions ──────────────────────────────────────

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), False),
    StructField("movieId", IntegerType(), False),
    StructField("rating", FloatType(), False),
    StructField("timestamp", LongType(), False),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), False),
    StructField("title", StringType(), False),
    StructField("genres", StringType(), False),
])


def create_spark_session(
    app_name: str = "RecEngineDataProcessing",
    driver_memory: str = "4g",
    local_dir: Optional[str] = None,
) -> SparkSession:
    """Create a local-mode SparkSession."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # Windows Hadoop configuration
    hadoop_dir = os.environ.get("HADOOP_HOME")
    if not hadoop_dir:
        candidate_paths = [
            Path(__file__).resolve().parents[3] / "hadoop",
            Path(__file__).resolve().parents[2] / "hadoop",
            Path("d:/project_2/hadoop"),
            Path("d:/setup/hadoop"),
            Path("C:/hadoop"),
        ]
        for cp in candidate_paths:
            if (cp / "bin" / "winutils.exe").exists():
                hadoop_dir = str(cp.resolve())
                os.environ["HADOOP_HOME"] = hadoop_dir
                bin_path = str((cp / "bin").resolve())
                if bin_path not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
                break

    builder = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )
    if hadoop_dir:
        builder = (
            builder.config("spark.hadoop.home.dir", hadoop_dir)
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        )
    if local_dir:
        builder = builder.config("spark.local.dir", local_dir)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_ratings(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Load ratings.csv with explicit schema."""
    path = str(data_dir / "ratings.csv")
    logger.info(f"Loading ratings from {path}")
    df = spark.read.csv(path, header=True, schema=RATINGS_SCHEMA)
    return df


def load_movies(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Load movies.csv with explicit schema."""
    path = str(data_dir / "movies.csv")
    logger.info(f"Loading movies from {path}")
    df = spark.read.csv(path, header=True, schema=MOVIES_SCHEMA)
    return df


def sample_for_dev_mode(
    ratings_df: DataFrame,
    sample_size: int = 1_000_000,
    seed: int = 42,
) -> DataFrame:
    """
    Deterministic sampling for development mode.
    
    Strategy: Sample a fraction of USERS, then take all their ratings.
    This preserves per-user temporal sequences and yields meaningful
    collaborative signals. We adjust the user fraction to target
    approximately `sample_size` total ratings.
    
    Args:
        ratings_df: Full ratings DataFrame
        sample_size: Target number of ratings
        seed: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame preserving complete user histories
    """
    total_count = ratings_df.count()
    if total_count <= sample_size:
        logger.info(f"Dataset ({total_count:,}) <= sample_size ({sample_size:,}), no sampling needed")
        return ratings_df

    # Estimate fraction of users to sample
    target_fraction = sample_size / total_count
    # Over-sample slightly since we take complete user histories
    user_fraction = min(target_fraction * 1.1, 1.0)

    # Sample users deterministically
    all_users = ratings_df.select("userId").distinct()
    sampled_users = all_users.sample(fraction=user_fraction, seed=seed)
    
    # Join to get all ratings for sampled users
    sampled_ratings = ratings_df.join(sampled_users, on="userId", how="inner")
    
    actual_count = sampled_ratings.count()
    logger.info(
        f"Dev mode sampling: {total_count:,} -> {actual_count:,} ratings "
        f"(target: {sample_size:,}, user fraction: {user_fraction:.4f})"
    )
    
    return sampled_ratings


def chronological_split(
    ratings_df: DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict]:
    """
    Split ratings chronologically by global timestamp.
    
    Uses quantile-based boundaries on the timestamp column so that
    approximately train_ratio of rows fall into train, val_ratio into
    validation, and test_ratio into test.  Ties at boundaries are
    resolved deterministically: rows at the exact boundary timestamp
    go to the earlier partition.
    
    Returns:
        (train_df, val_df, test_df, split_metadata)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Compute quantile boundaries
    train_boundary = train_ratio
    val_boundary = train_ratio + val_ratio

    # Use approxQuantile for efficiency on large datasets
    boundaries = ratings_df.stat.approxQuantile(
        "timestamp",
        [train_boundary, val_boundary],
        relativeError=0.001,  # ~0.1% error is fine
    )
    train_end_ts = int(boundaries[0])
    val_end_ts = int(boundaries[1])

    # Split: ties at boundary go to earlier partition
    train_df = ratings_df.filter(F.col("timestamp") <= train_end_ts)
    val_df = ratings_df.filter(
        (F.col("timestamp") > train_end_ts) & (F.col("timestamp") <= val_end_ts)
    )
    test_df = ratings_df.filter(F.col("timestamp") > val_end_ts)

    # Compute counts
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    total = train_count + val_count + test_count

    # Timestamp boundaries for metadata
    ts_stats = ratings_df.agg(
        F.min("timestamp").alias("global_min_ts"),
        F.max("timestamp").alias("global_max_ts"),
    ).collect()[0]

    train_ts_range = train_df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
    ).collect()[0]

    val_ts_range = val_df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
    ).collect()[0]

    test_ts_range = test_df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
    ).collect()[0]

    def _ts_to_str(ts):
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    split_meta = {
        "train_ratio_requested": train_ratio,
        "val_ratio_requested": val_ratio,
        "test_ratio_requested": test_ratio,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_count": total,
        "train_actual_ratio": round(train_count / total, 4) if total > 0 else 0,
        "val_actual_ratio": round(val_count / total, 4) if total > 0 else 0,
        "test_actual_ratio": round(test_count / total, 4) if total > 0 else 0,
        "train_end_timestamp": train_end_ts,
        "val_end_timestamp": val_end_ts,
        "global_min_timestamp": int(ts_stats["global_min_ts"]),
        "global_max_timestamp": int(ts_stats["global_max_ts"]),
        "train_start_date": _ts_to_str(train_ts_range["min_ts"]),
        "train_end_date": _ts_to_str(train_ts_range["max_ts"]),
        "val_start_date": _ts_to_str(val_ts_range["min_ts"]),
        "val_end_date": _ts_to_str(val_ts_range["max_ts"]),
        "test_start_date": _ts_to_str(test_ts_range["min_ts"]),
        "test_end_date": _ts_to_str(test_ts_range["max_ts"]),
    }

    logger.info(
        f"Chronological split: train={train_count:,} ({split_meta['train_actual_ratio']:.1%}), "
        f"val={val_count:,} ({split_meta['val_actual_ratio']:.1%}), "
        f"test={test_count:,} ({split_meta['test_actual_ratio']:.1%})"
    )
    logger.info(
        f"Train period: {split_meta['train_start_date']} -> {split_meta['train_end_date']}"
    )
    logger.info(
        f"Val period:   {split_meta['val_start_date']} -> {split_meta['val_end_date']}"
    )
    logger.info(
        f"Test period:  {split_meta['test_start_date']} -> {split_meta['test_end_date']}"
    )

    return train_df, val_df, test_df, split_meta


def build_movie_features(
    train_df: DataFrame, movies_df: DataFrame
) -> DataFrame:
    """
    Build leakage-safe movie features using ONLY training-period data.
    
    Features:
      - genre list (from catalog, static)
      - movie_train_rating_count: number of training ratings
      - movie_train_avg_rating: average training rating
      - movie_train_rating_std: std dev of training ratings
    """
    # Movie popularity from training data only
    movie_train_stats = (
        train_df.groupBy("movieId")
        .agg(
            F.count("*").alias("movie_train_rating_count"),
            F.avg("rating").alias("movie_train_avg_rating"),
            F.stddev("rating").alias("movie_train_rating_std"),
        )
    )

    # Join with catalog metadata
    movie_features = (
        movies_df
        .join(movie_train_stats, on="movieId", how="left")
        .fillna({
            "movie_train_rating_count": 0,
            "movie_train_avg_rating": 0.0,
            "movie_train_rating_std": 0.0,
        })
    )

    return movie_features


def build_user_features(train_df: DataFrame) -> DataFrame:
    """
    Build leakage-safe user features from ONLY training-period data.
    
    Features:
      - user_train_rating_count: number of training ratings
      - user_train_avg_rating: average training rating
      - user_train_rating_std: std dev of training ratings
      - user_train_min_ts / user_train_max_ts: activity window
    """
    user_features = (
        train_df.groupBy("userId")
        .agg(
            F.count("*").alias("user_train_rating_count"),
            F.avg("rating").alias("user_train_avg_rating"),
            F.stddev("rating").alias("user_train_rating_std"),
            F.min("timestamp").alias("user_train_min_ts"),
            F.max("timestamp").alias("user_train_max_ts"),
        )
        .fillna({
            "user_train_rating_std": 0.0,
        })
    )

    return user_features


def build_popularity_baseline(train_df: DataFrame, top_n: int = 500) -> DataFrame:
    """
    Simple popularity baseline from training data.
    
    Ranks movies by (weighted) training popularity:
    score = rating_count * avg_rating (Bayesian-like weighting).
    """
    popularity = (
        train_df.groupBy("movieId")
        .agg(
            F.count("*").alias("pop_count"),
            F.avg("rating").alias("pop_avg_rating"),
        )
        .withColumn(
            "popularity_score",
            F.col("pop_count") * F.col("pop_avg_rating")
        )
        .orderBy(F.desc("popularity_score"))
        .limit(top_n)
        .withColumn("popularity_rank", F.row_number().over(
            Window.orderBy(F.desc("popularity_score"))
        ))
    )

    return popularity


def parse_genres(movies_df: DataFrame) -> DataFrame:
    """Explode genres pipe-separated string into array column."""
    return movies_df.withColumn(
        "genre_list", F.split(F.col("genres"), "\\|")
    )


def build_genre_vocabulary(movies_df: DataFrame) -> list:
    """Extract unique genres from the catalog."""
    genres_exploded = (
        movies_df
        .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
        .select("genre")
        .distinct()
        .orderBy("genre")
    )
    return [row["genre"] for row in genres_exploded.collect()]


def save_processed_data(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    movie_features: DataFrame,
    user_features: DataFrame,
    popularity_df: DataFrame,
    genre_vocab: list,
    split_meta: dict,
    output_dir: Path,
    dataset_mode: str,
    extra_meta: Optional[dict] = None,
):
    """
    Save all processed artifacts to the output directory.
    
    Saves as Parquet for DataFrames and JSON for metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_dir}")

    # Save splits as parquet
    train_df.write.parquet(str(output_dir / "train"), mode="overwrite")
    val_df.write.parquet(str(output_dir / "val"), mode="overwrite")
    test_df.write.parquet(str(output_dir / "test"), mode="overwrite")

    # Save features
    movie_features.write.parquet(str(output_dir / "movie_features"), mode="overwrite")
    user_features.write.parquet(str(output_dir / "user_features"), mode="overwrite")
    popularity_df.write.parquet(str(output_dir / "popularity"), mode="overwrite")

    # Save metadata as JSON
    meta = {
        "dataset_mode": dataset_mode,
        "split_metadata": split_meta,
        "genre_vocabulary": genre_vocab,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Metadata saved to {meta_path}")


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    dev_mode: bool = True,
    dev_sample_size: int = 1_000_000,
    dev_seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    spark_driver_memory: str = "4g",
    spark_app_name: str = "RecEngineDataProcessing",
    spark_local_dir: Optional[str] = None,
    spark: Optional[SparkSession] = None,
) -> Dict:
    """
    Full data processing pipeline.
    
    Args:
        data_dir: Path to MovieLens 25M directory
        output_dir: Where to save processed data
        dev_mode: If True, sample a subset
        dev_sample_size: Target sample size for dev mode
        dev_seed: Seed for dev mode sampling
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        spark_driver_memory: Spark driver memory
        spark_app_name: Spark application name
        spark_local_dir: Spark temp directory
        spark: Optional existing SparkSession
    
    Returns:
        Dictionary with processing results and metadata
    """
    start_time = time.time()

    # Create or reuse Spark session
    own_spark = spark is None
    if own_spark:
        spark = create_spark_session(
            app_name=spark_app_name,
            driver_memory=spark_driver_memory,
            local_dir=spark_local_dir,
        )

    try:
        # 1. Load data
        logger.info("Step 1: Loading raw data...")
        ratings_df = load_ratings(spark, data_dir)
        movies_df = load_movies(spark, data_dir)
        total_ratings = ratings_df.count()
        logger.info(f"Loaded {total_ratings:,} ratings")

        # 2. Dev mode sampling
        dataset_mode = "dev" if dev_mode else "full"
        if dev_mode:
            logger.info(f"Step 2: Sampling for dev mode (target: {dev_sample_size:,})...")
            ratings_df = sample_for_dev_mode(ratings_df, dev_sample_size, dev_seed)
            sampled_count = ratings_df.count()
        else:
            sampled_count = total_ratings
            logger.info("Step 2: Using full dataset (no sampling)")

        # 3. Chronological split
        logger.info("Step 3: Chronological splitting...")
        train_df, val_df, test_df, split_meta = chronological_split(
            ratings_df, train_ratio, val_ratio, test_ratio
        )

        # 4. Build features from training data only
        logger.info("Step 4: Building leakage-safe features...")
        movies_with_genres = parse_genres(movies_df)
        movie_features = build_movie_features(train_df, movies_with_genres)
        user_features = build_user_features(train_df)

        # 5. Build popularity baseline
        logger.info("Step 5: Building popularity baseline...")
        popularity = build_popularity_baseline(train_df)

        # 6. Build genre vocabulary
        genre_vocab = build_genre_vocabulary(movies_df)
        logger.info(f"Genre vocabulary: {genre_vocab}")

        # 7. Compute summary stats
        n_users_train = train_df.select("userId").distinct().count()
        n_movies_train = train_df.select("movieId").distinct().count()
        n_users_val = val_df.select("userId").distinct().count()
        n_movies_val = val_df.select("movieId").distinct().count()
        n_users_test = test_df.select("userId").distinct().count()
        n_movies_test = test_df.select("movieId").distinct().count()

        # Cold-start analysis
        val_only_users = val_df.select("userId").distinct().subtract(
            train_df.select("userId").distinct()
        ).count()
        test_only_users = test_df.select("userId").distinct().subtract(
            train_df.select("userId").distinct()
        ).count()
        val_only_movies = val_df.select("movieId").distinct().subtract(
            train_df.select("movieId").distinct()
        ).count()
        test_only_movies = test_df.select("movieId").distinct().subtract(
            train_df.select("movieId").distinct()
        ).count()

        extra_meta = {
            "total_ratings_in_source": total_ratings,
            "sampled_ratings": sampled_count,
            "sampling_method": "user-level sampling preserving complete histories" if dev_mode else "none",
            "sample_seed": dev_seed if dev_mode else None,
            "n_users_train": n_users_train,
            "n_movies_train": n_movies_train,
            "n_users_val": n_users_val,
            "n_movies_val": n_movies_val,
            "n_users_test": n_users_test,
            "n_movies_test": n_movies_test,
            "cold_start_analysis": {
                "val_users_not_in_train": val_only_users,
                "test_users_not_in_train": test_only_users,
                "val_movies_not_in_train": val_only_movies,
                "test_movies_not_in_train": test_only_movies,
                "handling": (
                    "Users/movies absent from training have no learned embeddings. "
                    "They receive popularity-based fallback recommendations."
                ),
            },
            "n_genres": len(genre_vocab),
            "n_popularity_movies": popularity.count(),
        }

        # 8. Save everything
        logger.info("Step 6: Saving processed data...")
        save_processed_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            movie_features=movie_features,
            user_features=user_features,
            popularity_df=popularity,
            genre_vocab=genre_vocab,
            split_meta=split_meta,
            output_dir=output_dir,
            dataset_mode=dataset_mode,
            extra_meta=extra_meta,
        )

        duration = time.time() - start_time
        extra_meta["processing_duration_sec"] = round(duration, 2)

        logger.info(f"Pipeline complete in {duration:.1f}s")

        return {
            "status": "success",
            "dataset_mode": dataset_mode,
            "total_ratings": total_ratings,
            "sampled_ratings": sampled_count,
            "split_metadata": split_meta,
            "extra_metadata": extra_meta,
            "output_dir": str(output_dir),
            "duration_sec": round(duration, 2),
        }

    finally:
        if own_spark:
            spark.stop()
