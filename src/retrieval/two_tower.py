"""
TensorFlow Recommenders (TFRS) Two-Tower Retrieval Model.

Architecture:
  - User Tower: Trainable user ID embeddings + dense projection + L2 normalization
  - Movie Tower: Trainable movie ID embeddings + multi-hot genre representations + dense projection + L2 normalization
  - Objective: In-batch sampled retrieval loss with FactorizedTopK evaluation
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_recommenders as tfrs

logger = logging.getLogger(__name__)


class UserTower(keras.layers.Layer):
    """User tower projecting user ID to normalized embedding space."""

    def __init__(self, user_vocab: List[str], embedding_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.user_lookup = keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_embedding = keras.layers.Embedding(
            input_dim=self.user_lookup.vocabulary_size(),
            output_dim=embedding_dim,
            name="user_embedding",
        )
        self.dense = keras.layers.Dense(embedding_dim, activation=None, name="user_projection")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs: Tensor of user IDs (strings)
        Returns:
            L2-normalized user embedding tensor of shape (batch_size, embedding_dim)
        """
        x = self.user_lookup(inputs)
        x = self.user_embedding(x)
        x = self.dense(x)
        return tf.math.l2_normalize(x, axis=-1)


class MovieTower(keras.layers.Layer):
    """Movie tower combining movie ID and genre features into normalized embedding space."""

    def __init__(
        self,
        movie_vocab: List[str],
        genre_vocab: List[str],
        embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.movie_lookup = keras.layers.StringLookup(vocabulary=movie_vocab, mask_token=None)
        self.movie_embedding = keras.layers.Embedding(
            input_dim=self.movie_lookup.vocabulary_size(),
            output_dim=embedding_dim,
            name="movie_embedding",
        )

        genre_dim = max(16, embedding_dim // 4)
        self.genre_vectorizer = keras.layers.TextVectorization(
            vocabulary=genre_vocab,
            output_mode="multi_hot",
            standardize=lambda s: tf.strings.regex_replace(s, r"\|", " "),
            split="whitespace",
        )
        self.genre_dense = keras.layers.Dense(genre_dim, activation="relu", name="genre_dense")

        self.projection = keras.layers.Dense(embedding_dim, activation=None, name="movie_projection")

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Args:
            inputs: Dict with 'movieId' (string tensor) and optionally 'genres' (string tensor)
        Returns:
            L2-normalized movie embedding tensor of shape (batch_size, embedding_dim)
        """
        movie_id = inputs["movieId"]
        x_id = self.movie_lookup(movie_id)
        emb_id = self.movie_embedding(x_id)

        if "genres" in inputs:
            genres = inputs["genres"]
            x_genre = self.genre_vectorizer(genres)
            emb_genre = self.genre_dense(x_genre)
            combined = tf.concat([emb_id, emb_genre], axis=-1)
        else:
            combined = emb_id

        out = self.projection(combined)
        return tf.math.l2_normalize(out, axis=-1)


class TwoTowerModel(tfrs.Model):
    """
    Two-Tower Recommendation Model using TensorFlow Recommenders.
    """

    def __init__(
        self,
        user_vocab: List[str],
        movie_vocab: List[str],
        genre_vocab: List[str],
        embedding_dim: int = 64,
        candidate_dataset: Optional[tf.data.Dataset] = None,
    ):
        super().__init__()
        self.user_vocab = [str(u) for u in user_vocab]
        self.movie_vocab = [str(m) for m in movie_vocab]
        self.genre_vocab = [str(g) for g in genre_vocab]
        self.embedding_dim = embedding_dim

        self.user_tower = UserTower(self.user_vocab, embedding_dim=embedding_dim)
        self.movie_tower = MovieTower(
            self.movie_vocab, self.genre_vocab, embedding_dim=embedding_dim
        )

        metrics = None
        if candidate_dataset is not None:
            metrics = tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset.map(self.movie_tower)
            )

        self.task = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Compute retrieval loss for a batch of interactions."""
        user_embeddings = self.user_tower(features["userId"])
        movie_inputs = {"movieId": features["movieId"]}
        if "genres" in features:
            movie_inputs["genres"] = features["genres"]

        movie_embeddings = self.movie_tower(movie_inputs)
        return self.task(user_embeddings, movie_embeddings)

    def get_user_embedding(self, user_id: Union[int, str]) -> np.ndarray:
        """Get L2-normalized embedding vector for a single user ID."""
        inp = tf.constant([str(user_id)], dtype=tf.string)
        emb = self.user_tower(inp)
        return emb.numpy()[0]

    def get_movie_embedding(self, movie_id: Union[int, str], genres: str = "") -> np.ndarray:
        """Get L2-normalized embedding vector for a single movie."""
        inp = {
            "movieId": tf.constant([str(movie_id)], dtype=tf.string),
            "genres": tf.constant([str(genres)], dtype=tf.string),
        }
        emb = self.movie_tower(inp)
        return emb.numpy()[0]

    def save_artifacts(self, save_dir: Union[str, Path]):
        """Save model weights, configuration, and vocabularies."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_file = save_path / "model.weights.h5"
        self.save_weights(str(weights_file))

        # Save config and vocabularies
        config_payload = {
            "embedding_dim": self.embedding_dim,
            "user_vocab_size": len(self.user_vocab),
            "movie_vocab_size": len(self.movie_vocab),
            "genre_vocab_size": len(self.genre_vocab),
            "user_vocab": self.user_vocab,
            "movie_vocab": self.movie_vocab,
            "genre_vocab": self.genre_vocab,
        }
        with open(save_path / "model_config.json", "w") as f:
            json.dump(config_payload, f)

        logger.info(f"Saved Two-Tower model artifacts to {save_path}")

    @classmethod
    def load_artifacts(cls, save_dir: Union[str, Path]) -> "TwoTowerModel":
        """Load TwoTowerModel from saved weights and vocabulary config."""
        save_path = Path(save_dir)
        config_file = save_path / "model_config.json"
        weights_file = save_path / "model.weights.h5"

        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")
        if not weights_file.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_file}")

        with open(config_file, "r") as f:
            cfg = json.load(f)

        model = cls(
            user_vocab=cfg["user_vocab"],
            movie_vocab=cfg["movie_vocab"],
            genre_vocab=cfg["genre_vocab"],
            embedding_dim=cfg["embedding_dim"],
        )

        # Initialize layers with dummy forward pass before loading weights
        dummy_user = tf.constant(["0"], dtype=tf.string)
        dummy_movie = {"movieId": tf.constant(["0"], dtype=tf.string), "genres": tf.constant(["Action"], dtype=tf.string)}
        _ = model.user_tower(dummy_user)
        _ = model.movie_tower(dummy_movie)

        try:
            model.load_weights(str(weights_file))
        except Exception as e:
            logger.warning(f"Standard load_weights failed ({e}), loading directly from HDF5 datasets...")
            import h5py
            with h5py.File(weights_file, "r") as h5f:
                for k in h5f.keys():
                    norm_k = k.replace("\\", "/")
                    if "user_embedding" in norm_k and "vars" in h5f[k]:
                        model.user_tower.user_embedding.set_weights([h5f[k]["vars"]["0"][:]])
                    elif "user_tower" in norm_k and "dense" in norm_k and "vars" in h5f[k]:
                        model.user_tower.dense.set_weights([h5f[k]["vars"]["0"][:], h5f[k]["vars"]["1"][:]])
                    elif "movie_embedding" in norm_k and "vars" in h5f[k]:
                        model.movie_tower.movie_embedding.set_weights([h5f[k]["vars"]["0"][:]])
                    elif "genre_dense" in norm_k and "vars" in h5f[k]:
                        model.movie_tower.genre_dense.set_weights([h5f[k]["vars"]["0"][:], h5f[k]["vars"]["1"][:]])
                    elif ("projection" in norm_k or "movie_projection" in norm_k) and "vars" in h5f[k]:
                        model.movie_tower.projection.set_weights([h5f[k]["vars"]["0"][:], h5f[k]["vars"]["1"][:]])

        logger.info(f"Loaded TwoTowerModel from {save_path}")
        return model
