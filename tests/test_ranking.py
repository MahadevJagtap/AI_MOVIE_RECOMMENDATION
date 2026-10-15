"""
Unit and Integration Tests for Phase 3: Ranking Models and Evaluation.

Covers:
  - Feature engineering pipeline, leakage safety, and serialization
  - Wide-only baseline architecture, forward pass, and save/load
  - Wide & Deep primary model architecture, regularizers, and save/load
  - Ranking metrics: NDCG@K, Recall@K, Precision@K, MRR@K edge cases
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tf_keras as keras

from src.features.ranking_features import RankingFeaturePipeline
from src.ranking.wide_and_deep import WideOnlyModel, WideAndDeepModel
from src.evaluation.ranking_eval import (
    dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
)


# -------------------------------------------------------------------------
# Ranking Metrics Tests
# -------------------------------------------------------------------------
class TestRankingMetrics:
    """Rigorous mathematical tests for offline ranking metrics."""

    def test_ndcg_perfect_ranking(self):
        relevant = {"movie_1", "movie_2"}
        ranked = ["movie_1", "movie_2", "movie_3", "movie_4"]
        score = ndcg_at_k(relevant, ranked, k=2)
        assert pytest.approx(score, rel=1e-5) == 1.0

    def test_ndcg_suboptimal_ranking(self):
        relevant = {"movie_1", "movie_2"}
        # movie_2 is ranked 3rd instead of 2nd
        ranked = ["movie_1", "movie_3", "movie_2", "movie_4"]
        score = ndcg_at_k(relevant, ranked, k=3)
        assert 0.0 < score < 1.0
        # Expected:
        # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        expected = 1.5 / (1.0 + 1.0 / np.log2(3))
        assert pytest.approx(score, rel=1e-5) == expected

    def test_ndcg_no_relevant_items(self):
        relevant = set()
        ranked = ["movie_1", "movie_2"]
        assert ndcg_at_k(relevant, ranked, k=2) == 0.0

    def test_ndcg_zero_hits(self):
        relevant = {"movie_99"}
        ranked = ["movie_1", "movie_2", "movie_3"]
        assert ndcg_at_k(relevant, ranked, k=3) == 0.0

    def test_recall_at_k(self):
        relevant = {"1", "2", "3", "4"}
        ranked = ["1", "99", "2", "88", "77"]
        # Hits at k=3: '1', '2' -> 2 / 4 = 0.5
        assert pytest.approx(recall_at_k(relevant, ranked, k=3)) == 0.5
        # Hits at k=1: '1' -> 1 / 4 = 0.25
        assert pytest.approx(recall_at_k(relevant, ranked, k=1)) == 0.25
        # No relevant items
        assert recall_at_k(set(), ranked, k=3) == 0.0

    def test_precision_at_k(self):
        relevant = {"1", "2"}
        ranked = ["1", "99", "2", "88"]
        # Top-2 has 1 hit -> 1/2 = 0.5
        assert pytest.approx(precision_at_k(relevant, ranked, k=2)) == 0.5
        # Top-3 has 2 hits -> 2/3 = 0.66667
        assert pytest.approx(precision_at_k(relevant, ranked, k=3), rel=1e-3) == 0.6667
        # Top-4 has 2 hits -> 2/4 = 0.5
        assert pytest.approx(precision_at_k(relevant, ranked, k=4)) == 0.5

    def test_mrr_at_k(self):
        relevant = {"target"}
        assert mrr_at_k(relevant, ["target", "other"], k=2) == 1.0
        assert mrr_at_k(relevant, ["other", "target"], k=2) == 0.5
        assert mrr_at_k(relevant, ["a", "b", "target"], k=3) == pytest.approx(1.0 / 3)
        assert mrr_at_k(relevant, ["a", "b", "c"], k=3) == 0.0


# -------------------------------------------------------------------------
# Feature Pipeline Tests
# -------------------------------------------------------------------------
class TestRankingFeaturePipeline:
    """Tests for leakage safety and feature preparation."""

    @pytest.fixture
    def pipeline(self):
        stats = {
            "global_rating_mean": 3.5,
            "user_log_cnt_mean": 4.0,
            "user_log_cnt_std": 1.0,
            "user_avg_mean": 3.6,
            "user_avg_std": 0.5,
            "movie_log_cnt_mean": 2.0,
            "movie_log_cnt_std": 1.0,
            "movie_avg_mean": 3.2,
            "movie_avg_std": 0.6,
            "user_cnt_quantiles": [30.0, 50.0, 90.0, 190.0],
            "movie_cnt_quantiles": [1.0, 3.0, 10.0, 40.0],
            "user_rating_quantiles": [3.0, 3.5, 4.0],
            "movie_rating_quantiles": [2.8, 3.3, 3.8],
        }
        user_df = pd.DataFrame({
            "userId": [1, 2],
            "user_train_rating_count": [100, 20],
            "user_train_avg_rating": [3.8, 3.2],
        })
        movie_df = pd.DataFrame({
            "movieId": [10, 20],
            "movie_train_rating_count": [50, 5],
            "movie_train_avg_rating": [4.0, 2.5],
            "genres": ["Action|Adventure", "Drama"],
        })
        return RankingFeaturePipeline(
            stats=stats,
            user_features_df=user_df,
            movie_features_df=movie_df,
            user_embeddings={"1": np.ones(64, dtype=np.float32) / 8.0},
            movie_embeddings={"10": np.ones(64, dtype=np.float32) / 8.0},
            user_vocab=["1", "2"],
            movie_vocab=["10", "20"],
            genre_vocab=["Action", "Adventure", "Drama"],
        )

    def test_transform_interactions_known_entities(self, pipeline):
        df = pd.DataFrame({
            "userId": ["1"],
            "movieId": ["10"],
            "rating": [4.5],
        })
        feats, labels = pipeline.transform_interactions(df, is_training=True)
        assert feats["userId"][0] == "1"
        assert feats["movieId"][0] == "10"
        assert feats["genres"][0] == "Action|Adventure"
        assert feats["dense_features"].shape == (1, 5)
        # Dot product of normalized ones/8 is 64 * (1/64) = 1.0
        assert pytest.approx(feats["dense_features"][0, 4], rel=1e-4) == 1.0
        # Positive label since rating 4.5 >= 3.5
        assert labels[0] == 1.0

    def test_transform_interactions_cold_start_leakage_safety(self, pipeline):
        """Unseen users or movies in test must receive imputed train defaults and not crash."""
        df = pd.DataFrame({
            "userId": ["99999"],  # Unseen user
            "movieId": ["88888"], # Unseen movie
            "rating": [2.0],
        })
        feats, labels = pipeline.transform_interactions(df, is_training=True)
        assert feats["userId"][0] == "99999"
        assert feats["movieId"][0] == "88888"
        assert feats["genres"][0] == "(no genres listed)"
        # Cold start retrieval score defaults to 0.0
        assert feats["dense_features"][0, 4] == 0.0
        # Rating 2.0 < 3.5 -> negative label 0.0
        assert labels[0] == 0.0

    def test_transform_candidates(self, pipeline):
        cands = pipeline.transform_candidates("1", ["10", "20", "30"])
        assert len(cands["userId"]) == 3
        assert list(cands["userId"]) == ["1", "1", "1"]
        assert list(cands["movieId"]) == ["10", "20", "30"]
        assert cands["dense_features"].shape == (3, 5)

    def test_pipeline_serialization(self, pipeline, tmp_path):
        save_dir = tmp_path / "pipeline"
        pipeline.save(save_dir)
        assert (save_dir / "ranking_pipeline_config.json").exists()


# -------------------------------------------------------------------------
# Wide-only & Wide and Deep Model Tests
# -------------------------------------------------------------------------
class TestRankingModels:
    """Tests for model architectures, forward passes, outputs, and serialization."""

    @pytest.fixture
    def dummy_data(self):
        u_vocab = ["1", "2", "3"]
        m_vocab = ["10", "20", "30"]
        g_vocab = ["Action", "Comedy", "Drama"]
        batch = {
            "userId": tf.constant(["1", "2"], dtype=tf.string),
            "movieId": tf.constant(["10", "20"], dtype=tf.string),
            "genres": tf.constant(["Action|Comedy", "Drama"], dtype=tf.string),
            "activity_pop_cross": tf.constant([0, 1], dtype=tf.int32),
            "user_movie_rating_cross": tf.constant([2, 3], dtype=tf.int32),
            "dense_features": tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5], [-0.1, -0.2, -0.3, -0.4, -0.5]], dtype=tf.float32),
        }
        labels = tf.constant([[1.0], [0.0]], dtype=tf.float32)
        return u_vocab, m_vocab, g_vocab, batch, labels

    def test_wide_only_forward_and_loss(self, dummy_data):
        u_vocab, m_vocab, _, batch, labels = dummy_data
        model = WideOnlyModel(user_vocab=u_vocab, movie_vocab=m_vocab, n_dense_features=5)

        preds = model(batch, training=False)
        assert preds.shape == (2, 1)
        # Check sigmoid range [0, 1]
        assert np.all(preds.numpy() >= 0.0) and np.all(preds.numpy() <= 1.0)

        # Verify gradient calculation and training step
        loss_fn = keras.losses.BinaryCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss = loss_fn(labels, out)

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def test_wide_and_deep_forward_and_loss(self, dummy_data):
        u_vocab, m_vocab, g_vocab, batch, labels = dummy_data
        model = WideAndDeepModel(
            user_vocab=u_vocab,
            movie_vocab=m_vocab,
            genre_vocab=g_vocab,
            embedding_dim=16,
            hidden_units=(32, 16),
            n_dense_features=5,
        )

        preds = model(batch, training=False)
        assert preds.shape == (2, 1)
        assert np.all(preds.numpy() >= 0.0) and np.all(preds.numpy() <= 1.0)

        loss_fn = keras.losses.BinaryCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss = loss_fn(labels, out)

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def test_save_and_load_artifacts(self, dummy_data, tmp_path):
        u_vocab, m_vocab, g_vocab, batch, _ = dummy_data

        # Test WideOnly save/load
        wide = WideOnlyModel(user_vocab=u_vocab, movie_vocab=m_vocab, n_dense_features=5)
        _ = wide(batch)
        wide_dir = tmp_path / "wide"
        wide.save_artifacts(wide_dir)

        loaded_wide = WideOnlyModel.load_artifacts(wide_dir)
        pred_orig = wide.predict_scores({k: v.numpy() for k, v in batch.items()})
        pred_loaded = loaded_wide.predict_scores({k: v.numpy() for k, v in batch.items()})
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-5)

        # Test WideAndDeep save/load
        wd = WideAndDeepModel(
            user_vocab=u_vocab,
            movie_vocab=m_vocab,
            genre_vocab=g_vocab,
            embedding_dim=16,
            hidden_units=(32, 16),
            n_dense_features=5,
        )
        _ = wd(batch)
        wd_dir = tmp_path / "wd"
        wd.save_artifacts(wd_dir)

        loaded_wd = WideAndDeepModel.load_artifacts(wd_dir)
        pred_wd_orig = wd.predict_scores({k: v.numpy() for k, v in batch.items()})
        pred_wd_loaded = loaded_wd.predict_scores({k: v.numpy() for k, v in batch.items()})
        np.testing.assert_allclose(pred_wd_orig, pred_wd_loaded, rtol=1e-5)
