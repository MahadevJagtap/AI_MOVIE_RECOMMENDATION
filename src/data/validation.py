"""
Dataset validation for MovieLens 25M.

Validates file existence, schema, data types, nulls, duplicates,
and basic integrity before processing.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# Expected files and their required columns
EXPECTED_FILES = {
    "ratings.csv": {
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "dtypes": {
            "userId": "int",
            "movieId": "int",
            "rating": "float",
            "timestamp": "int",
        },
    },
    "movies.csv": {
        "columns": ["movieId", "title", "genres"],
        "dtypes": {
            "movieId": "int",
            "title": "string",
            "genres": "string",
        },
    },
    "tags.csv": {
        "columns": ["userId", "movieId", "tag", "timestamp"],
        "dtypes": {
            "userId": "int",
            "movieId": "int",
            "tag": "string",
            "timestamp": "int",
        },
    },
    "links.csv": {
        "columns": ["movieId", "imdbId", "tmdbId"],
        "dtypes": {
            "movieId": "int",
            "imdbId": "int",
            "tmdbId": "float",  # can have NaN
        },
    },
}


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.stats: Dict[str, any] = {}

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)
        logger.error(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def add_info(self, msg: str):
        self.info.append(msg)
        logger.info(msg)

    def summary(self) -> str:
        lines = ["=== Validation Summary ==="]
        lines.append(f"Status: {'PASSED' if self.passed else 'FAILED'}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.stats:
            lines.append("\nDataset Statistics:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def validate_file_existence(data_dir: Path) -> ValidationResult:
    """Check that all required MovieLens files exist."""
    result = ValidationResult()
    
    if not data_dir.exists():
        result.add_error(f"Data directory does not exist: {data_dir}")
        return result

    if not data_dir.is_dir():
        result.add_error(f"Data path is not a directory: {data_dir}")
        return result

    for filename in EXPECTED_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            result.add_error(f"Required file missing: {filename}")
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            result.add_info(f"{filename}: {size_mb:.1f} MB")
            result.stats[f"{filename}_size_mb"] = round(size_mb, 1)

    return result


def validate_with_spark(spark, data_dir: Path) -> ValidationResult:
    """
    Full validation using PySpark: schema, types, nulls, duplicates.
    
    Args:
        spark: Active SparkSession
        data_dir: Path to MovieLens directory
    
    Returns:
        ValidationResult with all findings
    """
    result = ValidationResult()

    # --- ratings.csv ---
    try:
        ratings_path = str(data_dir / "ratings.csv")
        ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
        
        # Check columns
        expected_cols = EXPECTED_FILES["ratings.csv"]["columns"]
        actual_cols = ratings_df.columns
        missing_cols = set(expected_cols) - set(actual_cols)
        if missing_cols:
            result.add_error(f"ratings.csv missing columns: {missing_cols}")
        else:
            result.add_info(f"ratings.csv columns OK: {actual_cols}")

        # Row count
        total_ratings = ratings_df.count()
        result.stats["total_ratings"] = total_ratings
        result.add_info(f"ratings.csv: {total_ratings:,} rows")

        # Check nulls
        from pyspark.sql import functions as F
        for col in expected_cols:
            if col in actual_cols:
                null_count = ratings_df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    result.add_error(f"ratings.csv has {null_count} nulls in '{col}'")

        # Check rating range
        rating_stats = ratings_df.agg(
            F.min("rating").alias("min_rating"),
            F.max("rating").alias("max_rating"),
            F.min("timestamp").alias("min_ts"),
            F.max("timestamp").alias("max_ts"),
        ).collect()[0]

        result.stats["min_rating"] = float(rating_stats["min_rating"])
        result.stats["max_rating"] = float(rating_stats["max_rating"])
        result.stats["min_timestamp"] = int(rating_stats["min_ts"])
        result.stats["max_timestamp"] = int(rating_stats["max_ts"])

        if rating_stats["min_rating"] < 0.5 or rating_stats["max_rating"] > 5.0:
            result.add_warning(
                f"Unexpected rating range: [{rating_stats['min_rating']}, {rating_stats['max_rating']}]"
            )

        # Check for duplicate (userId, movieId, timestamp)
        dup_count = (
            ratings_df.groupBy("userId", "movieId", "timestamp")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if dup_count > 0:
            result.add_warning(
                f"ratings.csv has {dup_count} duplicate (userId, movieId, timestamp) groups"
            )
        else:
            result.add_info("No exact duplicate interactions found")

        # Unique users and movies in ratings
        n_users = ratings_df.select("userId").distinct().count()
        n_movies = ratings_df.select("movieId").distinct().count()
        result.stats["n_users_ratings"] = n_users
        result.stats["n_movies_ratings"] = n_movies
        result.add_info(f"Unique users in ratings: {n_users:,}")
        result.add_info(f"Unique movies in ratings: {n_movies:,}")

        # Check for invalid ratings (not multiples of 0.5)
        invalid_ratings = ratings_df.filter(
            (F.col("rating") % 0.5 != 0) |
            (F.col("rating") < 0.5) |
            (F.col("rating") > 5.0)
        ).count()
        if invalid_ratings > 0:
            result.add_warning(f"{invalid_ratings} ratings outside [0.5, 5.0] or not multiples of 0.5")

    except Exception as e:
        result.add_error(f"Failed to validate ratings.csv: {e}")

    # --- movies.csv ---
    try:
        movies_path = str(data_dir / "movies.csv")
        movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
        
        expected_cols = EXPECTED_FILES["movies.csv"]["columns"]
        actual_cols = movies_df.columns
        missing_cols = set(expected_cols) - set(actual_cols)
        if missing_cols:
            result.add_error(f"movies.csv missing columns: {missing_cols}")

        n_movies_catalog = movies_df.count()
        result.stats["n_movies_catalog"] = n_movies_catalog
        result.add_info(f"movies.csv: {n_movies_catalog:,} movies in catalog")

        # Check for null movieId or title
        for col in ["movieId", "title"]:
            if col in actual_cols:
                null_count = movies_df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    result.add_error(f"movies.csv has {null_count} nulls in '{col}'")

        # Duplicate movieId check
        dup_movies = (
            movies_df.groupBy("movieId").count().filter(F.col("count") > 1).count()
        )
        if dup_movies > 0:
            result.add_error(f"movies.csv has {dup_movies} duplicate movieId entries")

    except Exception as e:
        result.add_error(f"Failed to validate movies.csv: {e}")

    # --- tags.csv ---
    try:
        tags_path = str(data_dir / "tags.csv")
        tags_df = spark.read.csv(tags_path, header=True, inferSchema=True)
        
        n_tags = tags_df.count()
        result.stats["n_tags"] = n_tags
        result.add_info(f"tags.csv: {n_tags:,} tags")
    except Exception as e:
        result.add_warning(f"Failed to validate tags.csv (non-critical): {e}")

    # --- links.csv ---
    try:
        links_path = str(data_dir / "links.csv")
        links_df = spark.read.csv(links_path, header=True, inferSchema=True)
        
        n_links = links_df.count()
        result.stats["n_links"] = n_links
        result.add_info(f"links.csv: {n_links:,} links")
    except Exception as e:
        result.add_warning(f"Failed to validate links.csv (non-critical): {e}")

    return result


def validate_dataset(data_dir: Path, spark=None) -> ValidationResult:
    """
    Run all validations.
    
    If a SparkSession is provided, runs full schema/content validation.
    Otherwise runs file-existence checks only.
    """
    result = validate_file_existence(data_dir)
    
    if not result.passed:
        return result

    if spark is not None:
        spark_result = validate_with_spark(spark, data_dir)
        # Merge results
        result.errors.extend(spark_result.errors)
        result.warnings.extend(spark_result.warnings)
        result.info.extend(spark_result.info)
        result.stats.update(spark_result.stats)

    return result
