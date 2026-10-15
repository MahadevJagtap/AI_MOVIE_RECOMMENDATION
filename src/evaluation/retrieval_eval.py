"""
Candidate Retrieval Evaluation Module.

Calculates:
  - Recall@10, Recall@20, Recall@50
  - Candidate catalog coverage
  - Retrieval latency per request (ms)
"""

import time
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    ground_truth_items: Set[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Compute Recall@K for a single user.
    
    Recall@K = |recommended[:k] ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth_items:
        return 0.0

    top_k_recs = set(recommended_items[:k])
    hits = len(top_k_recs.intersection(ground_truth_items))
    return hits / float(len(ground_truth_items))


def evaluate_retrieval(
    eval_df: pd.DataFrame,
    train_history_df: Optional[pd.DataFrame],
    recommender_fn,
    relevance_threshold: float = 3.5,
    k_list: Tuple[int, ...] = (10, 20, 50),
    max_eval_users: Optional[int] = 500,
) -> Dict[str, float]:
    """
    Evaluate retrieval system Recall@K across evaluation users.
    
    Args:
        eval_df: Validation or test DataFrame with ['userId', 'movieId', 'rating']
        train_history_df: Training DataFrame to find consumed items to filter
        recommender_fn: Callable taking (user_id, max_k, filter_movie_ids) -> List[int] of candidate movie IDs
        relevance_threshold: Rating threshold defining relevant items (default 3.5)
        k_list: Tuple of K values for Recall@K (e.g. 10, 20, 50)
        max_eval_users: Optional limit on evaluation users for speed
        
    Returns:
        Dictionary of metrics including Recall@10, Recall@20, Recall@50, latency_ms, evaluated_users
    """
    # 1. Extract positive interactions in eval set
    pos_eval = eval_df[eval_df["rating"] >= relevance_threshold]
    user_ground_truth: Dict[int, Set[int]] = (
        pos_eval.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # 2. Extract consumed items in training set
    user_train_history: Dict[int, Set[int]] = {}
    if train_history_df is not None:
        user_train_history = (
            train_history_df.groupby("userId")["movieId"].apply(set).to_dict()
        )

    # Filter to users with at least 1 positive interaction
    eval_users = [uid for uid, items in user_ground_truth.items() if len(items) > 0]
    if max_eval_users and len(eval_users) > max_eval_users:
        # Deterministic sample for reproducibility
        np.random.seed(42)
        eval_users = list(np.random.choice(eval_users, size=max_eval_users, replace=False))

    if not eval_users:
        logger.warning("No evaluation users with positive interactions found")
        return {f"recall@{k}": 0.0 for k in k_list}

    max_k = max(k_list)
    recall_accumulators: Dict[int, List[float]] = {k: [] for k in k_list}
    unique_recommended_movies: Set[int] = set()
    latencies_ms: List[float] = []

    for uid in eval_users:
        gt_items = user_ground_truth[uid]
        filter_items = user_train_history.get(uid, set())

        t0 = time.perf_counter()
        candidates = recommender_fn(uid, max_k, filter_items)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)

        unique_recommended_movies.update(candidates)

        for k in k_list:
            rec_k = compute_recall_at_k(gt_items, candidates, k)
            recall_accumulators[k].append(rec_k)

    results = {
        f"recall@{k}": round(float(np.mean(recall_accumulators[k])), 4)
        for k in k_list
    }
    results["candidate_catalog_coverage"] = len(unique_recommended_movies)
    results["avg_retrieval_latency_ms"] = round(float(np.mean(latencies_ms)), 2)
    results["p95_retrieval_latency_ms"] = round(float(np.percentile(latencies_ms, 95)), 2)
    results["evaluated_users_count"] = len(eval_users)

    logger.info(f"Retrieval evaluation results: {results}")
    return results
