"""
Ranking and Reranking Offline Evaluation Metrics.

Computes:
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Recall@K
  - Precision@K
  - MRR@K (Mean Reciprocal Rank)
  - Candidate Reranking Latency (Mean, P50, P95, P99 in ms)
  - Catalog Coverage across recommendations
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dcg_at_k(relevant_items: Set[Any], recommended_items: List[Any], k: int) -> float:
    """Discounted Cumulative Gain with binary relevance."""
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            # Binary relevance: rel_i = 1, so 2^rel_i - 1 = 1
            dcg += 1.0 / np.log2(i + 2.0)
    return float(dcg)


def ndcg_at_k(relevant_items: Set[Any], recommended_items: List[Any], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not relevant_items:
        return 0.0

    actual_dcg = dcg_at_k(relevant_items, recommended_items, k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal DCG: top min(|rel|, k) positions are all relevant
    n_ideal = min(len(relevant_items), k)
    ideal_dcg = sum(1.0 / np.log2(i + 2.0) for i in range(n_ideal))

    if ideal_dcg == 0.0:
        return 0.0

    return float(actual_dcg / ideal_dcg)


def recall_at_k(relevant_items: Set[Any], recommended_items: List[Any], k: int) -> float:
    """Proportion of relevant items captured within top-K recommendations."""
    if not relevant_items:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return float(hits / len(relevant_items))


def precision_at_k(relevant_items: Set[Any], recommended_items: List[Any], k: int) -> float:
    """Proportion of top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return float(hits / k)


def mrr_at_k(relevant_items: Set[Any], recommended_items: List[Any], k: int) -> float:
    """Mean Reciprocal Rank within top-K recommendations."""
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            return float(1.0 / (i + 1))
    return 0.0


def evaluate_ranking_models(
    test_interactions_df: pd.DataFrame,
    candidate_retriever_fn: Any,
    feature_pipeline: Any,
    ranking_models: Dict[str, Any],
    train_consumed: Optional[Dict[str, Set[str]]] = None,
    candidate_pool_size: int = 100,
    k_list: Tuple[int, ...] = (10, 20),
    max_eval_users: Optional[int] = 500,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Offline two-stage ranking evaluation comparing Retrieval-only vs Wide-only vs Wide & Deep.

    Args:
        test_interactions_df: DataFrame of test interactions with ['userId', 'movieId', 'rating']
        candidate_retriever_fn: Callable(user_id, k, filter_ids) -> List[(movie_id, retrieval_score)]
        feature_pipeline: RankingFeaturePipeline for candidate feature generation
        ranking_models: Dict mapping model_name -> model (e.g. {'Wide-only': m1, 'Wide & Deep': m2})
        train_consumed: Dict mapping userId -> set of consumed movieIds from train partition
        candidate_pool_size: Number of Stage-1 retrieval candidates per user (default 100)
        k_list: Cutoffs for metrics (e.g. (10, 20))
        max_eval_users: Max users to evaluate for fast reproducible reporting
        random_seed: Random seed for sampling users if max_eval_users is set

    Returns:
        Structured evaluation dictionary with per-model metrics, latency distributions, and catalog coverage.
    """
    logger.info("Starting offline ranking evaluation...")
    train_consumed = train_consumed or {}

    # Extract relevant test items per user (rating >= 3.5)
    relevant_by_user: Dict[str, Set[str]] = {}
    grouped = test_interactions_df[test_interactions_df["rating"] >= 3.5].groupby("userId")["movieId"]
    for uid, mids in grouped:
        relevant_by_user[str(uid)] = set(str(m) for m in mids)

    # Filter to users with at least 1 relevant test item
    eval_user_ids = [uid for uid, rels in relevant_by_user.items() if len(rels) > 0]
    logger.info(f"Total test users with relevant items: {len(eval_user_ids)}")

    if max_eval_users is not None and len(eval_user_ids) > max_eval_users:
        rng = np.random.RandomState(random_seed)
        eval_user_ids = list(rng.choice(eval_user_ids, size=max_eval_users, replace=False))
        logger.info(f"Subsampled to {len(eval_user_ids)} users for evaluation.")

    # Initialize metric containers
    # We include 'Retrieval Only' as baseline 0
    all_model_names = ["Retrieval Only (Stage 1)"] + list(ranking_models.keys())
    results: Dict[str, Dict[str, List[float]]] = {
        name: {
            "ndcg@10": [],
            "ndcg@20": [],
            "recall@10": [],
            "recall@20": [],
            "precision@10": [],
            "precision@20": [],
            "mrr@10": [],
            "latency_ms": [],
        }
        for name in all_model_names
    }
    recommended_items_per_model: Dict[str, Set[str]] = {name: set() for name in all_model_names}

    for user_idx, user_id in enumerate(eval_user_ids):
        actual_relevant = relevant_by_user[user_id]
        filter_ids = train_consumed.get(user_id, set())

        # Stage 1: Candidate Retrieval (retrieve top-N candidates)
        t0_retrieval = time.perf_counter()
        candidates_with_scores = candidate_retriever_fn(user_id, k=candidate_pool_size, filter_ids=filter_ids)
        retrieval_lat_ms = (time.perf_counter() - t0_retrieval) * 1000.0

        if not candidates_with_scores:
            continue

        candidate_ids = [str(item[0]) for item in candidates_with_scores]

        # 1. Baseline 0: Retrieval Only (candidates ranked by Stage-1 score)
        retrieval_ranked = candidate_ids
        results["Retrieval Only (Stage 1)"]["latency_ms"].append(retrieval_lat_ms)
        results["Retrieval Only (Stage 1)"]["ndcg@10"].append(ndcg_at_k(actual_relevant, retrieval_ranked, 10))
        results["Retrieval Only (Stage 1)"]["ndcg@20"].append(ndcg_at_k(actual_relevant, retrieval_ranked, 20))
        results["Retrieval Only (Stage 1)"]["recall@10"].append(recall_at_k(actual_relevant, retrieval_ranked, 10))
        results["Retrieval Only (Stage 1)"]["recall@20"].append(recall_at_k(actual_relevant, retrieval_ranked, 20))
        results["Retrieval Only (Stage 1)"]["precision@10"].append(precision_at_k(actual_relevant, retrieval_ranked, 10))
        results["Retrieval Only (Stage 1)"]["precision@20"].append(precision_at_k(actual_relevant, retrieval_ranked, 20))
        results["Retrieval Only (Stage 1)"]["mrr@10"].append(mrr_at_k(actual_relevant, retrieval_ranked, 10))
        recommended_items_per_model["Retrieval Only (Stage 1)"].update(retrieval_ranked[:20])

        # Feature preparation for rerankers
        cand_features = feature_pipeline.transform_candidates(user_id, candidate_ids)

        # 2. Score with each Main Ranking Model
        for model_name, model in ranking_models.items():
            t0_rank = time.perf_counter()
            pred_scores = model.predict_scores(cand_features)
            rank_lat_ms = (time.perf_counter() - t0_rank) * 1000.0

            # Sort descending by predicted relevance score
            sorted_indices = np.argsort(-pred_scores)
            reranked_ids = [candidate_ids[idx] for idx in sorted_indices]

            results[model_name]["latency_ms"].append(rank_lat_ms)
            results[model_name]["ndcg@10"].append(ndcg_at_k(actual_relevant, reranked_ids, 10))
            results[model_name]["ndcg@20"].append(ndcg_at_k(actual_relevant, reranked_ids, 20))
            results[model_name]["recall@10"].append(recall_at_k(actual_relevant, reranked_ids, 10))
            results[model_name]["recall@20"].append(recall_at_k(actual_relevant, reranked_ids, 20))
            results[model_name]["precision@10"].append(precision_at_k(actual_relevant, reranked_ids, 10))
            results[model_name]["precision@20"].append(precision_at_k(actual_relevant, reranked_ids, 20))
            results[model_name]["mrr@10"].append(mrr_at_k(actual_relevant, reranked_ids, 10))
            recommended_items_per_model[model_name].update(reranked_ids[:20])

    # Aggregate metric summaries
    summary: Dict[str, Dict[str, Any]] = {}
    for name in all_model_names:
        m = results[name]
        lats = np.array(m["latency_ms"]) if m["latency_ms"] else np.array([0.0])
        summary[name] = {
            "evaluated_users": len(m["ndcg@10"]),
            "ndcg@10": float(np.mean(m["ndcg@10"])) if m["ndcg@10"] else 0.0,
            "ndcg@20": float(np.mean(m["ndcg@20"])) if m["ndcg@20"] else 0.0,
            "recall@10": float(np.mean(m["recall@10"])) if m["recall@10"] else 0.0,
            "recall@20": float(np.mean(m["recall@20"])) if m["recall@20"] else 0.0,
            "precision@10": float(np.mean(m["precision@10"])) if m["precision@10"] else 0.0,
            "precision@20": float(np.mean(m["precision@20"])) if m["precision@20"] else 0.0,
            "mrr@10": float(np.mean(m["mrr@10"])) if m["mrr@10"] else 0.0,
            "latency_mean_ms": float(np.mean(lats)),
            "latency_p50_ms": float(np.percentile(lats, 50)),
            "latency_p95_ms": float(np.percentile(lats, 95)),
            "latency_p99_ms": float(np.percentile(lats, 99)),
            "unique_items_recommended@20": len(recommended_items_per_model[name]),
        }

    return {
        "candidate_pool_size": candidate_pool_size,
        "n_users_evaluated": len(eval_user_ids),
        "models": summary,
    }
