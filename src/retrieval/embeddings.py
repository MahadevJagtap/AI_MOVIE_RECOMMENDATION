"""
Movie embedding generator for candidate retrieval.

Uses trained MovieTower to generate and export normalized L2 embeddings
for all eligible catalog movies.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf

from src.retrieval.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


def generate_movie_embeddings(
    model: TwoTowerModel,
    movies_df: pd.DataFrame,
    batch_size: int = 1024,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate normalized embeddings for all movies in the given DataFrame.
    
    Args:
        model: Trained TwoTowerModel
        movies_df: DataFrame with 'movieId' and optionally 'genres'
        batch_size: Batch size for inference
        
    Returns:
        (movie_ids, embeddings) where embeddings is a (N, embedding_dim) float32 array
    """
    movie_ids = [int(mid) for mid in movies_df["movieId"].tolist()]
    movie_id_strs = [str(mid) for mid in movie_ids]
    genres = [str(g) if pd.notna(g) else "" for g in movies_df.get("genres", [""] * len(movie_ids))]

    total_movies = len(movie_ids)
    all_embeddings = []

    for start_idx in range(0, total_movies, batch_size):
        end_idx = min(start_idx + batch_size, total_movies)
        batch_ids = movie_id_strs[start_idx:end_idx]
        batch_genres = genres[start_idx:end_idx]

        inputs = {
            "movieId": tf.constant(batch_ids, dtype=tf.string),
            "genres": tf.constant(batch_genres, dtype=tf.string),
        }
        batch_emb = model.movie_tower(inputs)
        all_embeddings.append(batch_emb.numpy())

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    logger.info(
        f"Generated embeddings for {len(movie_ids)} movies (shape: {embeddings_matrix.shape})"
    )
    return movie_ids, embeddings_matrix


def save_embeddings(
    movie_ids: List[int],
    embeddings: np.ndarray,
    output_dir: Union[str, Path],
):
    """Save embeddings and ID mapping as numpy binary files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "movie_embeddings.npy", embeddings)
    np.save(out_path / "movie_ids.npy", np.array(movie_ids, dtype=np.int64))
    logger.info(f"Saved movie embeddings to {out_path}")


def load_embeddings(input_dir: Union[str, Path]) -> Tuple[List[int], np.ndarray]:
    """Load movie embeddings and IDs from disk."""
    in_path = Path(input_dir)
    embeddings = np.load(in_path / "movie_embeddings.npy")
    movie_ids = np.load(in_path / "movie_ids.npy").tolist()
    return movie_ids, embeddings
