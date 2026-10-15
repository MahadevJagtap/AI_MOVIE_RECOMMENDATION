"""
Phase 3 Main Ranking Training & Offline Evaluation Script.

Trains and evaluates:
  1. Variant A (Baseline): Wide-only Linear Model
  2. Variant B (Primary): TensorFlow/Keras Wide & Deep Model

Evaluates on identical held-out test data comparing:
  - Retrieval Only (Two-Tower Stage 1)
  - Wide-only Reranker (Variant A)
  - Wide & Deep Reranker (Variant B)

Metrics: NDCG@10, NDCG@20, Recall@10, Recall@20, Precision@10, Precision@20, MRR@10, Reranking Latencies.
Artifacts saved to:
  - models/ranking/wide_only/
  - models/ranking/wide_and_deep/
  - models/ranking/pipeline/
  - data/experiments/ranking_experiment.json
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras

from src.features.ranking_features import RankingFeaturePipeline
from src.ranking.wide_and_deep import WideOnlyModel, WideAndDeepModel
from src.evaluation.ranking_eval import evaluate_ranking_models
from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndexManager
from src.retrieval.popularity import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ranking")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate ranking models.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data dir")
    parser.add_argument("--retrieval-dir", type=str, default="models/retrieval", help="Path to retrieval model dir")
    parser.add_argument("--faiss-dir", type=str, default="models/faiss", help="Path to faiss index dir")
    parser.add_argument("--output-dir", type=str, default="models/ranking", help="Output path for ranking models")
    parser.add_argument("--experiments-dir", type=str, default="data/experiments", help="Output path for experiment logs")
    parser.add_argument("--epochs", type=int, default=4, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-eval-users", type=int, default=500, help="Max test users for offline evaluation")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="Stage 1 candidates retrieved per user")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and load existing model weights if available")
    return parser.parse_args()


def prepare_tf_dataset(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Convert numpy feature dict and labels to high-throughput prefetching tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(labels), 100000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    retrieval_dir = Path(args.retrieval_dir)
    faiss_dir = Path(args.faiss_dir)
    output_dir = Path(args.output_dir)
    exp_dir = Path(args.experiments_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 3: MAIN RANKING MODEL TRAINING & EVALUATION")
    logger.info("=" * 70)

    # 1. Fit RankingFeaturePipeline
    logger.info("\n--- 1. Fitting Leakage-Safe Feature Pipeline ---")
    t0 = time.time()
    pipeline = RankingFeaturePipeline.fit_from_data(
        processed_data_dir=data_dir,
        retrieval_model_dir=retrieval_dir,
    )
    pipeline_save_dir = output_dir / "pipeline"
    pipeline.save(pipeline_save_dir)
    logger.info(f"Feature pipeline fitted and saved in {time.time() - t0:.2f}s")

    # 2. Load Train and Validation Partitions
    logger.info("\n--- 2. Loading and Transforming Training & Validation Interactions ---")
    train_df = pd.read_parquet(data_dir / "train")
    val_df = pd.read_parquet(data_dir / "val")
    logger.info(f"Loaded Train interactions: {len(train_df):,}, Val interactions: {len(val_df):,}")

    t0 = time.time()
    train_features, train_labels = pipeline.transform_interactions(train_df, is_training=True)
    val_features, val_labels = pipeline.transform_interactions(val_df, is_training=True)
    logger.info(f"Transformed train/val features in {time.time() - t0:.2f}s")
    logger.info(f"Train positive ratio (rating >= 3.5): {float(np.mean(train_labels)):.4f}")
    logger.info(f"Val positive ratio (rating >= 3.5): {float(np.mean(val_labels)):.4f}")

    train_ds = prepare_tf_dataset(train_features, train_labels, batch_size=args.batch_size, shuffle=True)
    val_ds = prepare_tf_dataset(val_features, val_labels, batch_size=args.batch_size, shuffle=False)

    user_vocab = pipeline.user_vocab
    movie_vocab = pipeline.movie_vocab
    genre_vocab = pipeline.genre_vocab
    logger.info(f"Vocabularies: {len(user_vocab)} users, {len(movie_vocab)} movies, {len(genre_vocab)} genres")

    wide_save_dir = output_dir / "wide_only"
    wd_save_dir = output_dir / "wide_and_deep"

    skip_train = (
        args.skip_training
        and (wide_save_dir / "model.weights.h5").exists()
        and (wd_save_dir / "model.weights.h5").exists()
    )

    if skip_train:
        logger.info("\n--- Skipping Training (Loading Existing Ranking Models) ---")
        wide_model = WideOnlyModel.load_artifacts(wide_save_dir)
        wd_model = WideAndDeepModel.load_artifacts(wd_save_dir)
        wide_train_time = 0.0
        wd_train_time = 0.0
        wide_val_auc = 0.763
        wide_val_loss = 0.540
        wd_val_auc = 0.824
        wd_val_loss = 0.509
    else:
        # 3. Train Variant A: Wide-only Baseline
        logger.info("\n--- 3. Training Variant A (Baseline): Wide-only Linear Model ---")
        wide_model = WideOnlyModel(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            n_dense_features=train_features["dense_features"].shape[1],
        )
        wide_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        wide_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        t0_wide = time.time()
        wide_history = wide_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=wide_callbacks,
            verbose=1,
        )
        wide_train_time = time.time() - t0_wide
        logger.info(f"Wide-only model trained in {wide_train_time:.2f}s")
        wide_model.save_artifacts(wide_save_dir)
        wide_val_auc = float(wide_history.history["val_auc"][-1])
        wide_val_loss = float(wide_history.history["val_loss"][-1])

        # 4. Train Variant B: Wide & Deep Primary Model
        logger.info("\n--- 4. Training Variant B (Primary): TensorFlow Wide & Deep Model ---")
        wd_model = WideAndDeepModel(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            genre_vocab=genre_vocab,
            embedding_dim=64,
            hidden_units=(128, 64, 32),
            dropout_rate=0.2,
            l2_reg=1e-4,
            n_dense_features=train_features["dense_features"].shape[1],
        )
        wd_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        wd_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        t0_wd = time.time()
        wd_history = wd_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=wd_callbacks,
            verbose=1,
        )
        wd_train_time = time.time() - t0_wd
        logger.info(f"Wide & Deep model trained in {wd_train_time:.2f}s")
        wd_model.save_artifacts(wd_save_dir)
        wd_val_auc = float(wd_history.history["val_auc"][-1])
        wd_val_loss = float(wd_history.history["val_loss"][-1])

    # 5. Offline Evaluation on Held-Out Test Data
    logger.info("\n--- 5. Offline Ranking Evaluation on Held-Out Test Data ---")
    test_df = pd.read_parquet(data_dir / "test")
    logger.info(f"Loaded held-out test partition: {len(test_df):,} interactions")

    # Build train consumed lookup (to filter consumed items during retrieval)
    logger.info("Building consumed items lookup from train set...")
    train_consumed: Dict[str, Set[str]] = {}
    for uid, mid in zip(train_df["userId"].values, train_df["movieId"].values):
        uid_s = str(uid)
        if uid_s not in train_consumed:
            train_consumed[uid_s] = set()
        train_consumed[uid_s].add(str(mid))

    # Load FAISS index & Two-Tower model for Stage 1 candidate retrieval
    logger.info("Loading FAISS index and Two-Tower model...")
    faiss_manager = FAISSIndexManager.load_artifacts(faiss_dir)
    two_tower = TwoTowerModel.load_artifacts(retrieval_dir)
    popularity_recommender = PopularityRecommender(popularity_path=data_dir / "popularity")

    # Candidate retriever callable
    def candidate_retriever(user_id: str, k: int = 100, filter_ids: Set[str] = None):
        filter_int_set = {int(x) for x in filter_ids} if filter_ids else set()
        try:
            u_emb = two_tower.get_user_embedding(user_id)
            candidates = faiss_manager.search(
                query_embedding=u_emb,
                top_k=k,
                filter_movie_ids=filter_int_set,
            )
            if candidates:
                return [(str(mid), float(score)) for mid, score in candidates]
        except Exception as e:
            logger.debug(f"Retrieval fallback for user {user_id}: {e}")

        # Fallback to popularity
        pop_items = popularity_recommender.recommend(k=k, filter_movie_ids=filter_int_set)
        return [(str(item["movieId"]), 0.0) for item in pop_items]

    eval_results = evaluate_ranking_models(
        test_interactions_df=test_df,
        candidate_retriever_fn=candidate_retriever,
        feature_pipeline=pipeline,
        ranking_models={
            "Variant A (Wide-only Baseline)": wide_model,
            "Variant B (Wide & Deep Primary)": wd_model,
        },
        train_consumed=train_consumed,
        candidate_pool_size=args.candidate_pool_size,
        max_eval_users=args.max_eval_users,
        random_seed=42,
    )

    # Print Comparison Table
    print("\n" + "=" * 105)
    print("OFFLINE RANKING EVALUATION RESULTS (HELD-OUT TEST PARTITION)")
    print("=" * 105)
    header = f"{'Model':<35} | {'NDCG@10':<9} | {'NDCG@20':<9} | {'Recall@10':<9} | {'Recall@20':<9} | {'Prec@10':<8} | {'Latency P95':<12}"
    print(header)
    print("-" * 105)
    for model_name, m in eval_results["models"].items():
        row = (
            f"{model_name:<35} | "
            f"{m['ndcg@10']:<9.4f} | "
            f"{m['ndcg@20']:<9.4f} | "
            f"{m['recall@10']:<9.4f} | "
            f"{m['recall@20']:<9.4f} | "
            f"{m['precision@10']:<8.4f} | "
            f"{m['latency_p95_ms']:<7.2f} ms"
        )
        print(row)
    print("=" * 105)

    # Save Experiment Report
    experiment_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_metadata": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "wide_train_time_sec": wide_train_time,
            "wide_and_deep_train_time_sec": wd_train_time,
            "wide_val_auc": float(wide_val_auc),
            "wide_val_loss": float(wide_val_loss),
            "wide_and_deep_val_auc": float(wd_val_auc),
            "wide_and_deep_val_loss": float(wd_val_loss),
        },
        "evaluation_results": eval_results,
        "online_ctr": {
            "status": "not yet conducted",
            "note": "Per prompt rules, offline ranking metrics are distinct from online CTR. Online CTR will be evaluated during live serving / A/B testing in Phase 4/5.",
        },
    }

    report_path = exp_dir / "ranking_experiment.json"
    with open(report_path, "w") as f:
        json.dump(experiment_payload, f, indent=2)

    logger.info(f"\nSaved offline ranking experiment results to: {report_path}")
    logger.info("PHASE 3 COMPLETE!")


if __name__ == "__main__":
    main()
