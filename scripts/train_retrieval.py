"""
CLI Script to train TFRS Two-Tower Retrieval Model, build FAISS CPU Index,
and evaluate Recall@K against the Popularity Baseline.

Usage:
    python scripts/train_retrieval.py [--epochs 5] [--batch-size 4096] [--embedding-dim 64]
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RETRIEVAL_MODEL_DIR, FAISS_INDEX_DIR,
    EMBEDDING_DIM, RETRIEVAL_EPOCHS, RETRIEVAL_BATCH_SIZE, RETRIEVAL_LEARNING_RATE,
    EXPERIMENT_LOG_DIR, MOVIELENS_DIR,
)
from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.embeddings import generate_movie_embeddings, save_embeddings
from src.retrieval.faiss_index import FAISSIndexManager
from src.retrieval.popularity import PopularityRecommender
from src.evaluation.retrieval_eval import evaluate_retrieval

logger = logging.getLogger("train_retrieval")


def load_dataset_and_metadata(processed_dir: Path):
    """Load train, val splits and metadata JSON."""
    meta_path = processed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Processed metadata not found at {meta_path}. Run data pipeline first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    logger.info("Loading training and validation Parquet files...")
    train_df = pd.read_parquet(processed_dir / "train")
    val_df = pd.read_parquet(processed_dir / "val")

    logger.info(f"Loaded {len(train_df):,} train ratings, {len(val_df):,} val ratings")
    return train_df, val_df, meta


def load_movie_metadata(movies_csv_path: Path) -> pd.DataFrame:
    """Load catalog movie metadata."""
    logger.info(f"Loading movie metadata from {movies_csv_path}")
    df = pd.read_csv(movies_csv_path)
    df["genres"] = df["genres"].fillna("")
    return df


def prepare_tf_dataset(
    ratings_df: pd.DataFrame,
    movies_meta_df: pd.DataFrame,
    batch_size: int = 4096,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Convert ratings and movie metadata to batched tf.data.Dataset."""
    # Merge genres onto interactions
    movie_genre_map = dict(zip(movies_meta_df["movieId"], movies_meta_df["genres"]))
    genres_series = ratings_df["movieId"].map(lambda mid: movie_genre_map.get(mid, ""))

    features = {
        "userId": tf.constant(ratings_df["userId"].astype(str).values, dtype=tf.string),
        "movieId": tf.constant(ratings_df["movieId"].astype(str).values, dtype=tf.string),
        "genres": tf.constant(genres_series.values, dtype=tf.string),
    }

    ds = tf.data.Dataset.from_tensor_slices(features)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(ratings_df), 100_000), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Retrieval Model and Build FAISS Index")
    parser.add_argument("--epochs", type=int, default=RETRIEVAL_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=RETRIEVAL_BATCH_SIZE, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=RETRIEVAL_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DATA_DIR), help="Path to processed data")
    parser.add_argument("--output-model-dir", type=str, default=str(RETRIEVAL_MODEL_DIR), help="Save path for retrieval model")
    parser.add_argument("--output-faiss-dir", type=str, default=str(FAISS_INDEX_DIR), help="Save path for FAISS index")
    parser.add_argument("--eval-users", type=int, default=500, help="Max users to evaluate")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if saved model and FAISS index already exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("=" * 70)
    logger.info("PHASE 2: TFRS TWO-TOWER RETRIEVAL TRAINING & FAISS INDEXING")
    logger.info("=" * 70)
    logger.info(f"Embedding Dim:   {args.embedding_dim}")
    logger.info(f"Epochs:          {args.epochs}")
    logger.info(f"Batch Size:      {args.batch_size}")
    logger.info(f"Learning Rate:   {args.lr}")
    logger.info("=" * 70)

    processed_dir = Path(args.processed_dir)
    train_df, val_df, meta = load_dataset_and_metadata(processed_dir)

    movies_csv_path = Path(MOVIELENS_DIR) / "movies.csv"
    movies_df = load_movie_metadata(movies_csv_path)

    model_dir = Path(args.output_model_dir)
    faiss_dir = Path(args.output_faiss_dir)

    unique_train_users = [str(u) for u in train_df["userId"].unique()]
    unique_train_movies = [str(m) for m in train_df["movieId"].unique()]
    genre_vocab = meta.get("genre_vocabulary", [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"
    ])

    train_duration = 0.0

    if args.skip_train and (model_dir / "model_config.json").exists() and (faiss_dir / "faiss_index.bin").exists():
        logger.info(f"Loading existing trained model from {model_dir}...")
        model = TwoTowerModel.load_artifacts(model_dir)
        logger.info(f"Loading existing FAISS index from {faiss_dir}...")
        faiss_manager = FAISSIndexManager()
        faiss_manager.load(faiss_dir)
    else:
        # 1. Build Vocabularies from training data only (Leakage prevention)
        logger.info(f"Vocabulary: {len(unique_train_users):,} train users, {len(unique_train_movies):,} train movies, {len(genre_vocab)} genres")

        # 2. Build TFRS Two-Tower Model
        logger.info("Constructing Two-Tower Retrieval Model...")
        model = TwoTowerModel(
            user_vocab=unique_train_users,
            movie_vocab=unique_train_movies,
            genre_vocab=genre_vocab,
            embedding_dim=args.embedding_dim,
        )
        optimizer = keras.optimizers.Adagrad(learning_rate=args.lr)
        model.compile(optimizer=optimizer)

        # 3. Prepare tf.data.Dataset
        logger.info("Preparing training tf.data.Dataset...")
        train_ds = prepare_tf_dataset(train_df, movies_df, batch_size=args.batch_size, shuffle=True)

        # 4. Train Model
        logger.info("Training Two-Tower Model on CPU...")
        start_train_time = time.time()
        history = model.fit(train_ds, epochs=args.epochs, verbose=1)
        train_duration = round(time.time() - start_train_time, 2)
        logger.info(f"Training completed in {train_duration}s")

        # 5. Save Model Artifacts
        model.save_artifacts(model_dir)

        # 6. Generate Movie Embeddings for all eligible catalog movies
        logger.info("Generating embeddings for all catalog movies...")
        movie_ids, embeddings = generate_movie_embeddings(model, movies_df, batch_size=2048)
        save_embeddings(movie_ids, embeddings, model_dir)

        # 7. Build FAISS CPU Index
        logger.info("Building FAISS CPU Candidate Index...")
        faiss_manager = FAISSIndexManager(embedding_dim=args.embedding_dim)
        faiss_manager.build(
            movie_embeddings=embeddings,
            movie_ids=movie_ids,
            model_version="tfrs_two_tower_v1",
            normalize=True,
        )
        faiss_manager.save(faiss_dir)

    # 8. Evaluation: Compare Two-Tower + FAISS vs Popularity Baseline on Validation Set
    logger.info("=" * 70)
    logger.info("EVALUATION: TWO-TOWER + FAISS VS POPULARITY BASELINE (VALIDATION SET)")
    logger.info("=" * 70)

    # Function for Two-Tower Retrieval
    def two_tower_recommender(user_id: int, max_k: int, filter_items: set) -> List[int]:
        uid_str = str(user_id)
        u_emb = model.get_user_embedding(uid_str)
        results = faiss_manager.search(u_emb, top_k=max_k, filter_movie_ids=filter_items)
        return [mid for mid, _ in results]

    logger.info("Evaluating Two-Tower + FAISS Retrieval...")
    two_tower_metrics = evaluate_retrieval(
        eval_df=val_df,
        train_history_df=train_df,
        recommender_fn=two_tower_recommender,
        relevance_threshold=3.5,
        k_list=(10, 20, 50),
        max_eval_users=args.eval_users,
    )

    # Popularity Baseline Recommender
    logger.info("Evaluating Popularity Baseline Recommender...")
    popularity_model = PopularityRecommender(
        popularity_path=processed_dir / "popularity",
        movies_path=movies_csv_path,
    )

    def popularity_recommender_fn(user_id: int, max_k: int, filter_items: set) -> List[int]:
        return popularity_model.get_candidate_ids(k=max_k, filter_movie_ids=filter_items)

    popularity_metrics = evaluate_retrieval(
        eval_df=val_df,
        train_history_df=train_df,
        recommender_fn=popularity_recommender_fn,
        relevance_threshold=3.5,
        k_list=(10, 20, 50),
        max_eval_users=args.eval_users,
    )

    # 9. Print Results Comparison Table
    print("\n" + "=" * 70)
    print("RETRIEVAL EXPERIMENT RESULTS COMPARISON (Held-Out Validation Set)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Popularity Baseline':<20} | {'Two-Tower (TFRS + FAISS)':<20}")
    print("-" * 70)
    for k in (10, 20, 50):
        pop_rec = popularity_metrics.get(f"recall@{k}", 0.0)
        tt_rec = two_tower_metrics.get(f"recall@{k}", 0.0)
        print(f"{f'Recall@{k}':<30} | {pop_rec:<20.4f} | {tt_rec:<20.4f}")
    print("-" * 70)
    print(f"{'Catalog Coverage':<30} | {popularity_metrics['candidate_catalog_coverage']:<20} | {two_tower_metrics['candidate_catalog_coverage']:<20}")
    print(f"{'Avg Latency (ms)':<30} | {popularity_metrics['avg_retrieval_latency_ms']:<20.2f} | {two_tower_metrics['avg_retrieval_latency_ms']:<20.2f}")
    print(f"{'P95 Latency (ms)':<30} | {popularity_metrics['p95_retrieval_latency_ms']:<20.2f} | {two_tower_metrics['p95_retrieval_latency_ms']:<20.2f}")
    print(f"{'Evaluated Users':<30} | {popularity_metrics['evaluated_users_count']:<20} | {two_tower_metrics['evaluated_users_count']:<20}")
    print("=" * 70 + "\n")

    # 10. Save Experiment Report
    experiment_dir = Path(EXPERIMENT_LOG_DIR)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_file = experiment_dir / "retrieval_experiment.json"

    experiment_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_duration_sec": train_duration,
            "n_users_train": len(unique_train_users),
            "n_movies_train": len(unique_train_movies),
            "n_movies_indexed": faiss_manager.total_items,
        },
        "metrics": {
            "two_tower": two_tower_metrics,
            "popularity_baseline": popularity_metrics,
        },
        "artifact_paths": {
            "retrieval_model_dir": str(model_dir),
            "faiss_index_dir": str(faiss_dir),
        },
    }
    with open(experiment_file, "w") as f:
        json.dump(experiment_report, f, indent=2)

    logger.info(f"Saved retrieval experiment report to {experiment_file}")


if __name__ == "__main__":
    main()
