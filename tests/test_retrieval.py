"""
Unit and integration tests for Phase 2: Baseline and Retrieval.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.retrieval.popularity import PopularityRecommender


@pytest.fixture
def sample_ratings_df():
    """Create synthetic training interactions."""
    data = [
        {"userId": 1, "movieId": 101, "rating": 5.0},
        {"userId": 2, "movieId": 101, "rating": 4.0},
        {"userId": 3, "movieId": 101, "rating": 4.5},  # mid 101: count=3, avg=4.5 -> score=13.5
        {"userId": 1, "movieId": 102, "rating": 2.0},
        {"userId": 2, "movieId": 102, "rating": 3.0},  # mid 102: count=2, avg=2.5 -> score=5.0
        {"userId": 4, "movieId": 103, "rating": 5.0},  # mid 103: count=1, avg=5.0 -> score=5.0
        {"userId": 5, "movieId": 104, "rating": 1.0},  # mid 104: count=1, avg=1.0 -> score=1.0
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_movies_csv(tmp_path):
    """Create synthetic movies.csv file."""
    movies_file = tmp_path / "movies.csv"
    movies_data = [
        {"movieId": 101, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi"},
        {"movieId": 102, "title": "Toy Story (1995)", "genres": "Adventure|Animation"},
        {"movieId": 103, "title": "Inception (2010)", "genres": "Action|Sci-Fi|Thriller"},
        {"movieId": 104, "title": "Gigli (2003)", "genres": "Comedy|Romance"},
    ]
    pd.DataFrame(movies_data).to_csv(movies_file, index=False)
    return movies_file


class TestPopularityRecommender:
    """Tests for popularity baseline model."""

    def test_fit_and_recommend(self, sample_ratings_df, sample_movies_csv):
        model = PopularityRecommender(movies_path=sample_movies_csv)
        model.fit_from_df(sample_ratings_df, top_n=10)

        recs = model.recommend(user_id=999, k=3)
        assert len(recs) == 3
        # Top movie must be 101 (highest score 13.5)
        assert recs[0]["movieId"] == 101
        assert recs[0]["title"] == "The Matrix (1999)"
        assert recs[0]["popularity_rank"] == 1
        assert recs[0]["popularity_score"] > recs[1]["popularity_score"]

    def test_filter_consumed_items(self, sample_ratings_df, sample_movies_csv):
        model = PopularityRecommender(movies_path=sample_movies_csv)
        model.fit_from_df(sample_ratings_df, top_n=10)

        # Filter out movie 101 (user already watched it)
        recs = model.recommend(user_id=1, k=3, filter_movie_ids={101})
        rec_ids = [r["movieId"] for r in recs]
        assert 101 not in rec_ids
        assert len(recs) <= 3

    def test_get_candidate_ids(self, sample_ratings_df):
        model = PopularityRecommender()
        model.fit_from_df(sample_ratings_df, top_n=10)

        cand_ids = model.get_candidate_ids(k=2, filter_movie_ids={101})
        assert len(cand_ids) == 2
        assert 101 not in cand_ids

    def test_empty_model_graceful(self):
        model = PopularityRecommender()
        recs = model.recommend(user_id=1, k=10)
        assert recs == []


class TestFAISSIndexManager:
    """Tests for FAISS CPU candidate retrieval index."""

    def test_build_and_search(self):
        import numpy as np
        from src.retrieval.faiss_index import FAISSIndexManager

        manager = FAISSIndexManager(embedding_dim=4)
        # 3 movie embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        movie_ids = [10, 20, 30]

        manager.build(embeddings, movie_ids, model_version="test_v1")
        assert manager.is_built
        assert manager.total_items == 3

        # Query aligned with movie 20
        query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = manager.search(query, top_k=2)

        assert len(results) == 2
        # First result must be movie 20 with similarity ~ 1.0
        assert results[0][0] == 20
        assert pytest.approx(results[0][1], 0.01) == 1.0

    def test_search_with_filtering(self):
        import numpy as np
        from src.retrieval.faiss_index import FAISSIndexManager

        manager = FAISSIndexManager(embedding_dim=4)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        movie_ids = [101, 102, 103]

        manager.build(embeddings, movie_ids)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Exclude movie 101 (consumed)
        results = manager.search(query, top_k=2, filter_movie_ids={101})

        rec_ids = [mid for mid, _ in results]
        assert 101 not in rec_ids
        assert rec_ids[0] == 102

    def test_save_and_load(self, tmp_path):
        import numpy as np
        from src.retrieval.faiss_index import FAISSIndexManager

        manager = FAISSIndexManager(embedding_dim=4)
        embeddings = np.random.randn(10, 4).astype(np.float32)
        movie_ids = list(range(100, 110))
        manager.build(embeddings, movie_ids, model_version="v2.1")

        save_dir = tmp_path / "faiss_test"
        manager.save(save_dir)

        # Load into new manager
        loaded_manager = FAISSIndexManager()
        loaded_manager.load(save_dir)

        assert loaded_manager.is_built
        assert loaded_manager.total_items == 10
        assert loaded_manager.model_version == "v2.1"
        assert loaded_manager.idx_to_movie_id == movie_ids

        # Search should return identical movie IDs and approximately equal float scores
        q = np.random.randn(4).astype(np.float32)
        res1 = manager.search(q, top_k=3)
        res2 = loaded_manager.search(q, top_k=3)
        assert [mid for mid, _ in res1] == [mid for mid, _ in res2]
        for (_, s1), (_, s2) in zip(res1, res2):
            assert pytest.approx(s1, abs=1e-5) == s2

    def test_dimension_mismatch_error(self):
        import numpy as np
        from src.retrieval.faiss_index import FAISSIndexManager

        manager = FAISSIndexManager(embedding_dim=4)
        embeddings = np.random.randn(5, 4).astype(np.float32)
        manager.build(embeddings, list(range(5)))

        wrong_query = np.random.randn(8).astype(np.float32)
        with pytest.raises(ValueError, match="does not match index dim"):
            manager.search(wrong_query, top_k=2)


class TestTwoTowerModel:
    """Smoke tests for TwoTowerModel training and serialization."""

    def test_towers_and_training_smoke(self, tmp_path):
        import tensorflow as tf
        import tf_keras as keras
        from src.retrieval.two_tower import TwoTowerModel

        user_vocab = ["1", "2", "3", "4", "5"]
        movie_vocab = ["101", "102", "103", "104"]
        genre_vocab = ["Action", "Comedy", "Sci-Fi"]

        model = TwoTowerModel(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            genre_vocab=genre_vocab,
            embedding_dim=16,
        )

        # Test forward passes
        u_emb = model.get_user_embedding("1")
        assert u_emb.shape == (16,)
        # Check L2 normalization
        assert pytest.approx(float(np.linalg.norm(u_emb)), abs=1e-4) == 1.0

        m_emb = model.get_movie_embedding("101", "Action|Sci-Fi")
        assert m_emb.shape == (16,)
        assert pytest.approx(float(np.linalg.norm(m_emb)), abs=1e-4) == 1.0

        # Smoke training with tf.data
        synthetic_interactions = {
            "userId": tf.constant(["1", "2", "3", "4"], dtype=tf.string),
            "movieId": tf.constant(["101", "102", "103", "104"], dtype=tf.string),
            "genres": tf.constant(["Action", "Comedy", "Sci-Fi", "Action|Comedy"], dtype=tf.string),
        }
        train_ds = tf.data.Dataset.from_tensor_slices(synthetic_interactions).batch(2)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
        history = model.fit(train_ds, epochs=1, verbose=0)
        assert "loss" in history.history

        # Save and load artifacts
        post_train_u_emb = model.get_user_embedding("1")
        save_dir = tmp_path / "two_tower_smoke"
        model.save_artifacts(save_dir)

        loaded_model = TwoTowerModel.load_artifacts(save_dir)
        loaded_u_emb = loaded_model.get_user_embedding("1")
        assert np.allclose(post_train_u_emb, loaded_u_emb, atol=1e-4)


class TestRetrievalEvaluation:
    """Tests for retrieval evaluation metric calculations."""

    def test_recall_at_k(self):
        from src.evaluation.retrieval_eval import compute_recall_at_k

        ground_truth = {101, 102, 103}
        # Hits: 101, 102 (2 out of 3)
        predictions = [101, 999, 102, 888, 777]

        assert compute_recall_at_k(ground_truth, predictions, k=1) == pytest.approx(1 / 3)
        assert compute_recall_at_k(ground_truth, predictions, k=3) == pytest.approx(2 / 3)
        assert compute_recall_at_k(ground_truth, predictions, k=5) == pytest.approx(2 / 3)
        # Empty ground truth
        assert compute_recall_at_k(set(), predictions, k=5) == 0.0

    def test_evaluate_retrieval_function(self):
        from src.evaluation.retrieval_eval import evaluate_retrieval

        eval_df = pd.DataFrame([
            {"userId": 1, "movieId": 101, "rating": 5.0},
            {"userId": 1, "movieId": 102, "rating": 4.0},
            {"userId": 1, "movieId": 103, "rating": 2.0},  # < 3.5 -> not in positive ground truth
            {"userId": 2, "movieId": 201, "rating": 4.5},
        ])

        # Mock recommender that returns known candidates
        def mock_recommender(user_id, max_k, filter_items):
            if user_id == 1:
                return [101, 105, 106]
            return [201, 202, 203]

        metrics = evaluate_retrieval(
            eval_df=eval_df,
            train_history_df=None,
            recommender_fn=mock_recommender,
            relevance_threshold=3.5,
            k_list=(1, 3),
        )

        assert "recall@1" in metrics
        assert "recall@3" in metrics
        assert "avg_retrieval_latency_ms" in metrics
        assert metrics["evaluated_users_count"] == 2


