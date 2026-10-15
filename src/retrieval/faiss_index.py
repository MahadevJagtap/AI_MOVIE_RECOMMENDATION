"""
FAISS CPU candidate retrieval index for two-tower movie embeddings.

Features:
  - Exact inner-product (cosine similarity on L2-normalized embeddings)
  - Persistent bidirectional mapping between FAISS positions and movie IDs
  - Save/load functionality with metadata and version tracking
  - Consumed item filtering during candidate search
  - Validation of returned candidate IDs
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manages creation, serialization, and querying of FAISS CPU candidate index."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.index = None
        self.idx_to_movie_id: List[int] = []
        self.movie_id_to_idx: Dict[int, int] = {}
        self.metadata: Dict = {}
        self.model_version: str = "v1"

    @property
    def is_built(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_items(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def build(
        self,
        movie_embeddings: np.ndarray,
        movie_ids: List[int],
        model_version: str = "v1",
        normalize: bool = True,
    ) -> "FAISSIndexManager":
        """
        Build a FAISS CPU index from movie embeddings and IDs.
        
        Args:
            movie_embeddings: Float32 array of shape (N, embedding_dim)
            movie_ids: List of N corresponding movie IDs
            model_version: Retrieval model identifier for version tracking
            normalize: If True, L2-normalizes embeddings so Inner Product = Cosine Similarity
        """
        import faiss

        movie_embeddings = np.ascontiguousarray(movie_embeddings, dtype=np.float32)
        if len(movie_embeddings) != len(movie_ids):
            raise ValueError(
                f"Count mismatch: {len(movie_embeddings)} embeddings vs {len(movie_ids)} movie IDs"
            )

        n_samples, dim = movie_embeddings.shape
        self.embedding_dim = dim
        self.model_version = model_version

        if normalize:
            faiss.normalize_L2(movie_embeddings)

        # IndexFlatIP calculates inner product
        index = faiss.IndexFlatIP(dim)
        index.add(movie_embeddings)

        self.index = index
        self.idx_to_movie_id = [int(mid) for mid in movie_ids]
        self.movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(movie_ids)}

        self.metadata = {
            "model_version": model_version,
            "embedding_dim": dim,
            "total_items": n_samples,
            "normalized": normalize,
            "metric": "INNER_PRODUCT (Cosine Similarity)",
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Built FAISS CPU index with {n_samples} items (dim={dim}, version={model_version})"
        )
        return self

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 200,
        filter_movie_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Search top-k most similar movies for a given user embedding vector.
        
        Args:
            query_embedding: Array of shape (dim,) or (1, dim)
            top_k: Number of candidates to retrieve
            filter_movie_ids: Set of movie IDs to exclude (already watched/rated)
            
        Returns:
            List of (movieId, similarity_score) tuples ordered by descending score.
        """
        import faiss

        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded")

        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim {query.shape[1]} does not match index dim {self.embedding_dim}"
            )

        faiss.normalize_L2(query)

        filter_set = set(filter_movie_ids) if filter_movie_ids else set()

        # Over-sample to guarantee top_k candidates after filtering
        fetch_k = min(top_k + len(filter_set), self.index.ntotal)
        if fetch_k <= 0:
            return []

        scores, indices = self.index.search(query, fetch_k)

        results: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.idx_to_movie_id):
                continue

            mid = self.idx_to_movie_id[idx]
            if mid in filter_set:
                continue

            results.append((mid, float(score)))
            if len(results) >= top_k:
                break

        return results

    def save(self, save_dir: Union[str, Path]):
        """Persist FAISS index binary and mapping metadata to disk."""
        import faiss

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        index_file = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))

        meta_file = save_path / "index_metadata.json"
        meta_payload = {
            **self.metadata,
            "idx_to_movie_id": self.idx_to_movie_id,
        }
        with open(meta_file, "w") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"Saved FAISS index and metadata to {save_path}")

    def load(self, save_dir: Union[str, Path]) -> "FAISSIndexManager":
        """Load persisted FAISS index and mapping metadata from disk."""
        import faiss

        load_path = Path(save_dir)
        index_file = load_path / "faiss_index.bin"
        meta_file = load_path / "index_metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"FAISS index metadata file not found: {meta_file}")

        self.index = faiss.read_index(str(index_file))

        with open(meta_file, "r") as f:
            meta_payload = json.load(f)

        self.idx_to_movie_id = [int(mid) for mid in meta_payload["idx_to_movie_id"]]
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.idx_to_movie_id)}
        self.embedding_dim = meta_payload.get("embedding_dim", self.index.d)
        self.model_version = meta_payload.get("model_version", "v1")
        self.metadata = meta_payload

        logger.info(
            f"Loaded FAISS index with {self.index.ntotal} items (dim={self.embedding_dim}, version={self.model_version})"
        )
        return self

    @classmethod
    def load_artifacts(cls, save_dir: Union[str, Path]) -> "FAISSIndexManager":
        """Classmethod constructor to load index from disk."""
        manager = cls()
        return manager.load(save_dir)
