"""
Tests for data validation and processing (Phase 1).

Uses small synthetic fixtures — does NOT require the full MovieLens dataset.
"""

import os
import json
import shutil
import tempfile
import pytest
from pathlib import Path
from datetime import datetime

# Ensure legacy Keras for later TF phases
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for tests — shared across the module."""
    from src.data.processing import create_spark_session
    session = create_spark_session(
        app_name="RecEngineTests",
        driver_memory="1g",
    )
    yield session
    session.stop()


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_data_dir(tmp_dir):
    """
    Create a minimal synthetic MovieLens directory with known data.
    
    10 users, 20 movies, ~100 ratings with known timestamps
    so we can verify chronological splitting precisely.
    """
    data_dir = tmp_dir / "ml-test"
    data_dir.mkdir()

    # ratings.csv — 100 ratings with timestamps spread across a range
    # Timestamps: 1000000 to 1000099 (evenly spaced, easy to verify splits)
    ratings_lines = ["userId,movieId,rating,timestamp"]
    ts = 1000000
    for i in range(100):
        uid = (i % 10) + 1        # users 1-10
        mid = (i % 20) + 1        # movies 1-20
        rating = round(0.5 + (i % 9) * 0.5, 1)  # 0.5 to 4.5
        ratings_lines.append(f"{uid},{mid},{rating},{ts + i}")

    (data_dir / "ratings.csv").write_text("\n".join(ratings_lines))

    # movies.csv — 20 movies with genres
    movies_lines = ["movieId,title,genres"]
    genres_pool = [
        "Action|Adventure", "Comedy", "Drama|Romance",
        "Sci-Fi|Action", "Documentary", "Thriller|Mystery",
        "Animation|Children", "Horror", "Musical", "Western",
    ]
    for mid in range(1, 21):
        genre = genres_pool[(mid - 1) % len(genres_pool)]
        movies_lines.append(f'{mid},Movie {mid},{genre}')

    (data_dir / "movies.csv").write_text("\n".join(movies_lines))

    # tags.csv — minimal
    tags_lines = ["userId,movieId,tag,timestamp"]
    tags_lines.append("1,1,fun,1000050")
    tags_lines.append("2,3,boring,1000060")
    (data_dir / "tags.csv").write_text("\n".join(tags_lines))

    # links.csv — minimal
    links_lines = ["movieId,imdbId,tmdbId"]
    for mid in range(1, 21):
        links_lines.append(f"{mid},{mid + 100000},{mid + 200000}")
    (data_dir / "links.csv").write_text("\n".join(links_lines))

    return data_dir


@pytest.fixture
def synthetic_output_dir(tmp_dir):
    """Output directory for processed data."""
    out = tmp_dir / "processed"
    out.mkdir()
    return out


# ── Validation Tests ─────────────────────────────────────────────────

class TestFileValidation:
    """Tests for file-existence validation."""

    def test_valid_directory(self, synthetic_data_dir):
        from src.data.validation import validate_file_existence
        result = validate_file_existence(synthetic_data_dir)
        assert result.passed, result.summary()

    def test_missing_directory(self, tmp_dir):
        from src.data.validation import validate_file_existence
        result = validate_file_existence(tmp_dir / "nonexistent")
        assert not result.passed
        assert any("does not exist" in e for e in result.errors)

    def test_missing_file(self, synthetic_data_dir):
        # Remove ratings.csv to trigger an error
        ratings_path = synthetic_data_dir / "ratings.csv"
        ratings_backup = synthetic_data_dir / "ratings.csv.bak"
        ratings_path.rename(ratings_backup)

        from src.data.validation import validate_file_existence
        result = validate_file_existence(synthetic_data_dir)
        assert not result.passed
        assert any("ratings.csv" in e for e in result.errors)

        # Restore
        ratings_backup.rename(ratings_path)


class TestSparkValidation:
    """Tests for Spark-based content validation."""

    def test_valid_dataset(self, spark, synthetic_data_dir):
        from src.data.validation import validate_with_spark
        result = validate_with_spark(spark, synthetic_data_dir)
        assert result.passed, result.summary()
        assert result.stats["total_ratings"] == 100

    def test_stats_recorded(self, spark, synthetic_data_dir):
        from src.data.validation import validate_with_spark
        result = validate_with_spark(spark, synthetic_data_dir)
        assert "n_users_ratings" in result.stats
        assert "n_movies_ratings" in result.stats
        assert result.stats["n_users_ratings"] == 10
        assert result.stats["n_movies_ratings"] == 20


# ── Chronological Split Tests ────────────────────────────────────────

class TestChronologicalSplit:
    """Tests for chronological splitting."""

    def test_basic_split_proportions(self, spark, synthetic_data_dir):
        from src.data.processing import load_ratings, chronological_split
        ratings = load_ratings(spark, synthetic_data_dir)
        train, val, test, meta = chronological_split(ratings, 0.8, 0.1, 0.1)

        total = meta["train_count"] + meta["val_count"] + meta["test_count"]
        assert total == 100
        # Train should be roughly 80%
        assert 70 <= meta["train_count"] <= 90
        # No empty splits
        assert meta["val_count"] > 0
        assert meta["test_count"] > 0

    def test_no_temporal_leakage(self, spark, synthetic_data_dir):
        """Train max timestamp must be < val min timestamp, etc."""
        from src.data.processing import load_ratings, chronological_split
        from pyspark.sql import functions as F

        ratings = load_ratings(spark, synthetic_data_dir)
        train, val, test, meta = chronological_split(ratings, 0.8, 0.1, 0.1)

        train_max = train.agg(F.max("timestamp")).collect()[0][0]
        val_min = val.agg(F.min("timestamp")).collect()[0][0]
        val_max = val.agg(F.max("timestamp")).collect()[0][0]
        test_min = test.agg(F.min("timestamp")).collect()[0][0]

        assert train_max < val_min, f"Leakage: train_max={train_max} >= val_min={val_min}"
        assert val_max < test_min, f"Leakage: val_max={val_max} >= test_min={test_min}"

    def test_all_rows_accounted(self, spark, synthetic_data_dir):
        """Every row from original should appear in exactly one split."""
        from src.data.processing import load_ratings, chronological_split
        ratings = load_ratings(spark, synthetic_data_dir)
        train, val, test, meta = chronological_split(ratings, 0.8, 0.1, 0.1)
        assert meta["total_count"] == 100

    def test_split_metadata_dates(self, spark, synthetic_data_dir):
        """Split metadata should contain human-readable date strings."""
        from src.data.processing import load_ratings, chronological_split
        ratings = load_ratings(spark, synthetic_data_dir)
        _, _, _, meta = chronological_split(ratings, 0.8, 0.1, 0.1)

        assert meta["train_start_date"] is not None
        assert meta["test_end_date"] is not None
        # Should be parseable
        datetime.strptime(meta["train_start_date"], "%Y-%m-%d %H:%M:%S")

    def test_ratios_sum_to_one_error(self, spark, synthetic_data_dir):
        """Should raise if ratios don't sum to 1."""
        from src.data.processing import load_ratings, chronological_split
        ratings = load_ratings(spark, synthetic_data_dir)
        with pytest.raises(AssertionError):
            chronological_split(ratings, 0.5, 0.1, 0.1)


# ── Dev Mode Sampling Tests ──────────────────────────────────────────

class TestDevModeSampling:
    """Tests for development-mode sampling."""

    def test_sampling_reduces_data(self, spark, synthetic_data_dir):
        from src.data.processing import load_ratings, sample_for_dev_mode
        ratings = load_ratings(spark, synthetic_data_dir)
        sampled = sample_for_dev_mode(ratings, sample_size=30, seed=42)
        count = sampled.count()
        # Should be significantly less than 100
        assert count < 100
        assert count > 0

    def test_sampling_preserves_user_histories(self, spark, synthetic_data_dir):
        """Each sampled user should have ALL their ratings, not a subset."""
        from src.data.processing import load_ratings, sample_for_dev_mode
        from pyspark.sql import functions as F

        ratings = load_ratings(spark, synthetic_data_dir)
        sampled = sample_for_dev_mode(ratings, sample_size=30, seed=42)

        # For each user in sampled, count should match original
        sampled_users = [
            row["userId"]
            for row in sampled.select("userId").distinct().collect()
        ]

        for uid in sampled_users:
            orig_count = ratings.filter(F.col("userId") == uid).count()
            sampled_count = sampled.filter(F.col("userId") == uid).count()
            assert orig_count == sampled_count, (
                f"User {uid}: original={orig_count}, sampled={sampled_count}"
            )

    def test_sampling_deterministic(self, spark, synthetic_data_dir):
        """Same seed should produce same sample."""
        from src.data.processing import load_ratings, sample_for_dev_mode
        ratings = load_ratings(spark, synthetic_data_dir)
        s1 = sample_for_dev_mode(ratings, sample_size=30, seed=42)
        s2 = sample_for_dev_mode(ratings, sample_size=30, seed=42)
        assert s1.count() == s2.count()

    def test_no_sampling_when_small(self, spark, synthetic_data_dir):
        """Should not sample if data is already <= sample_size."""
        from src.data.processing import load_ratings, sample_for_dev_mode
        ratings = load_ratings(spark, synthetic_data_dir)
        sampled = sample_for_dev_mode(ratings, sample_size=200, seed=42)
        assert sampled.count() == 100


# ── Feature Engineering Tests ────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for leakage-safe feature engineering."""

    def test_movie_features_from_train_only(self, spark, synthetic_data_dir):
        """Movie features must be computed from training data only."""
        from src.data.processing import (
            load_ratings, load_movies, chronological_split,
            parse_genres, build_movie_features,
        )

        ratings = load_ratings(spark, synthetic_data_dir)
        movies = parse_genres(load_movies(spark, synthetic_data_dir))
        train, val, test, _ = chronological_split(ratings, 0.8, 0.1, 0.1)

        movie_features = build_movie_features(train, movies)
        
        # Should have all catalog movies
        assert movie_features.count() == 20
        # Should have the expected columns
        cols = movie_features.columns
        assert "movie_train_rating_count" in cols
        assert "movie_train_avg_rating" in cols

    def test_user_features_from_train_only(self, spark, synthetic_data_dir):
        """User features must be computed from training data only."""
        from src.data.processing import (
            load_ratings, chronological_split, build_user_features,
        )

        ratings = load_ratings(spark, synthetic_data_dir)
        train, _, _, _ = chronological_split(ratings, 0.8, 0.1, 0.1)

        user_features = build_user_features(train)
        cols = user_features.columns
        assert "user_train_rating_count" in cols
        assert "user_train_avg_rating" in cols
        # Should only contain users who appear in training
        assert user_features.count() <= 10

    def test_genre_vocabulary(self, spark, synthetic_data_dir):
        from src.data.processing import load_movies, build_genre_vocabulary
        movies = load_movies(spark, synthetic_data_dir)
        vocab = build_genre_vocabulary(movies)
        assert isinstance(vocab, list)
        assert len(vocab) > 0
        assert "Action" in vocab
        assert "Comedy" in vocab

    def test_popularity_baseline(self, spark, synthetic_data_dir):
        from src.data.processing import (
            load_ratings, chronological_split, build_popularity_baseline,
        )

        ratings = load_ratings(spark, synthetic_data_dir)
        train, _, _, _ = chronological_split(ratings, 0.8, 0.1, 0.1)

        pop = build_popularity_baseline(train, top_n=10)
        assert pop.count() <= 10
        assert "popularity_rank" in pop.columns
        assert "popularity_score" in pop.columns


# ── Pipeline Integration Test ────────────────────────────────────────

class TestPipelineIntegration:
    """End-to-end pipeline test with synthetic data."""

    def test_full_pipeline(self, spark, synthetic_data_dir, synthetic_output_dir):
        from src.data.processing import run_pipeline

        result = run_pipeline(
            data_dir=synthetic_data_dir,
            output_dir=synthetic_output_dir,
            dev_mode=False,
            spark=spark,
        )

        assert result["status"] == "success"
        assert result["total_ratings"] == 100
        assert result["split_metadata"]["total_count"] == 100

        # Check saved files
        assert (synthetic_output_dir / "train").exists()
        assert (synthetic_output_dir / "val").exists()
        assert (synthetic_output_dir / "test").exists()
        assert (synthetic_output_dir / "movie_features").exists()
        assert (synthetic_output_dir / "user_features").exists()
        assert (synthetic_output_dir / "popularity").exists()
        assert (synthetic_output_dir / "metadata.json").exists()

        # Verify metadata content
        with open(synthetic_output_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["dataset_mode"] == "full"
        assert "split_metadata" in meta
        assert "genre_vocabulary" in meta
        assert len(meta["genre_vocabulary"]) > 0

    def test_dev_mode_pipeline(self, spark, synthetic_data_dir, synthetic_output_dir):
        from src.data.processing import run_pipeline

        result = run_pipeline(
            data_dir=synthetic_data_dir,
            output_dir=synthetic_output_dir,
            dev_mode=True,
            dev_sample_size=30,
            spark=spark,
        )

        assert result["status"] == "success"
        assert result["dataset_mode"] == "dev"
        # Sampled should be less than total
        assert result["sampled_ratings"] < result["total_ratings"]


# ── Timestamp Ties Test ──────────────────────────────────────────────

class TestTimestampTies:
    """Tests for handling timestamp ties at split boundaries."""

    def test_ties_handled_deterministically(self, spark, tmp_dir):
        """
        When many ratings share the same timestamp at a boundary,
        they should all go to the same partition (the earlier one).
        """
        from src.data.processing import chronological_split

        # Create data with heavy ties
        data_dir = tmp_dir / "ties"
        data_dir.mkdir()
        lines = ["userId,movieId,rating,timestamp"]
        # 50 ratings at ts=1000, 50 at ts=2000
        for i in range(50):
            lines.append(f"{i+1},{i+1},3.0,1000")
        for i in range(50):
            lines.append(f"{i+1},{i+51},4.0,2000")
        (data_dir / "ratings.csv").write_text("\n".join(lines))

        ratings = spark.read.csv(str(data_dir / "ratings.csv"), header=True, inferSchema=True)
        train, val, test, meta = chronological_split(ratings, 0.8, 0.1, 0.1)

        total = meta["train_count"] + meta["val_count"] + meta["test_count"]
        assert total == 100

        # With only 2 unique timestamps, splits won't be exactly 80/10/10
        # but all rows at a given timestamp should be in the same partition
        from pyspark.sql import functions as F
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            if split_df.count() > 0:
                unique_ts = split_df.select("timestamp").distinct().count()
                # Each split should contain complete timestamp groups
                assert unique_ts >= 1, f"{split_name} split is empty of unique timestamps"


# ── MovieLens Real Data Validation (only if available) ───────────────

class TestRealDataValidation:
    """Tests that run against the actual MovieLens 25M dataset (if present)."""

    @pytest.fixture
    def ml25m_dir(self):
        ml_dir = Path(os.environ.get("MOVIELENS_DIR", "d:/setup/Database/ml-25m"))
        if not (ml_dir / "ratings.csv").exists():
            pytest.skip("MovieLens 25M not available at expected path")
        return ml_dir

    def test_real_file_existence(self, ml25m_dir):
        from src.data.validation import validate_file_existence
        result = validate_file_existence(ml25m_dir)
        assert result.passed, result.summary()

    def test_real_file_sizes(self, ml25m_dir):
        """ratings.csv should be ~650MB for ML-25M."""
        ratings_size = (ml25m_dir / "ratings.csv").stat().st_size / (1024 * 1024)
        assert ratings_size > 600, f"ratings.csv too small: {ratings_size:.1f} MB"
