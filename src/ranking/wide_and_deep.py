"""
TensorFlow / Keras Wide & Deep Ranking Models.

Implements two variants:
  1. WideOnlyModel (Variant A - Linear memorization baseline)
  2. WideAndDeepModel (Variant B - Primary deep ranking model combining memorization and generalization)

Both models operate on identical feature signatures from RankingFeaturePipeline:
  - userId: string tensor
  - movieId: string tensor
  - genres: string tensor (pipe-delimited)
  - activity_pop_cross: int32 tensor (0..24)
  - user_movie_rating_cross: int32 tensor (0..15)
  - dense_features: float32 tensor (5 continuous standardized features)
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

logger = logging.getLogger(__name__)


class WideComponent(keras.layers.Layer):
    """
    Generalized linear model layer capturing sparse ID and crossed-feature effects.
    """

    def __init__(
        self,
        user_vocab: List[str],
        movie_vocab: List[str],
        n_activity_pop_buckets: int = 25,
        n_user_movie_rating_buckets: int = 16,
        n_dense_features: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_lookup = keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_linear = keras.layers.Embedding(
            input_dim=self.user_lookup.vocabulary_size(),
            output_dim=1,
            embeddings_initializer="zeros",
            name="wide_user_emb",
        )

        self.movie_lookup = keras.layers.StringLookup(vocabulary=movie_vocab, mask_token=None)
        self.movie_linear = keras.layers.Embedding(
            input_dim=self.movie_lookup.vocabulary_size(),
            output_dim=1,
            embeddings_initializer="zeros",
            name="wide_movie_emb",
        )

        self.act_pop_linear = keras.layers.Embedding(
            input_dim=n_activity_pop_buckets + 1,
            output_dim=1,
            embeddings_initializer="zeros",
            name="wide_act_pop_emb",
        )

        self.rating_cross_linear = keras.layers.Embedding(
            input_dim=n_user_movie_rating_buckets + 1,
            output_dim=1,
            embeddings_initializer="zeros",
            name="wide_rating_cross_emb",
        )

        self.dense_linear = keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer="zeros",
            name="wide_dense_linear",
        )

        self.bias = self.add_weight(
            name="wide_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute scalar linear logit:
          logit = w_user + w_movie + w_act_pop + w_rating_cross + w_dense^T x + b
        """
        # User 1D weight
        u_idx = self.user_lookup(inputs["userId"])
        w_u = tf.squeeze(self.user_linear(u_idx), axis=-1)

        # Movie 1D weight
        m_idx = self.movie_lookup(inputs["movieId"])
        w_m = tf.squeeze(self.movie_linear(m_idx), axis=-1)

        # Crossed feature 1D weights
        w_act_pop = tf.squeeze(self.act_pop_linear(inputs["activity_pop_cross"]), axis=-1)
        w_rat_cross = tf.squeeze(self.rating_cross_linear(inputs["user_movie_rating_cross"]), axis=-1)

        # Continuous feature linear dot product
        w_dense = tf.squeeze(self.dense_linear(inputs["dense_features"]), axis=-1)

        # Sum linear terms + bias
        logits = w_u + w_m + w_act_pop + w_rat_cross + w_dense + self.bias
        return tf.reshape(logits, (-1, 1))


class DeepComponent(keras.layers.Layer):
    """
    Deep neural network component learning non-linear feature interactions
    via dense embeddings and multi-layer feedforward network with regularizers.
    """

    def __init__(
        self,
        user_vocab: List[str],
        movie_vocab: List[str],
        genre_vocab: List[str],
        embedding_dim: int = 64,
        hidden_units: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

        # User Embedding
        self.user_lookup = keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_emb = keras.layers.Embedding(
            input_dim=self.user_lookup.vocabulary_size(),
            output_dim=embedding_dim,
            name="deep_user_emb",
        )

        # Movie Embedding
        self.movie_lookup = keras.layers.StringLookup(vocabulary=movie_vocab, mask_token=None)
        self.movie_emb = keras.layers.Embedding(
            input_dim=self.movie_lookup.vocabulary_size(),
            output_dim=embedding_dim,
            name="deep_movie_emb",
        )

        # Genre Multi-Hot Projection
        genre_dim = max(16, embedding_dim // 2)
        self.genre_vectorizer = keras.layers.TextVectorization(
            vocabulary=genre_vocab,
            output_mode="multi_hot",
            standardize=lambda s: tf.strings.regex_replace(s, r"\|", " "),
            split="whitespace",
        )
        self.genre_dense = keras.layers.Dense(
            genre_dim,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name="deep_genre_dense",
        )

        # Feedforward MLP layers
        self.mlp_layers = []
        for i, units in enumerate(hidden_units):
            self.mlp_layers.append(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"deep_dense_{i}",
                )
            )
            self.mlp_layers.append(keras.layers.BatchNormalization(name=f"deep_bn_{i}"))
            self.mlp_layers.append(keras.layers.Dropout(dropout_rate, name=f"deep_dropout_{i}"))

        self.projection = keras.layers.Dense(1, activation=None, name="deep_projection")

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Deep forward pass:
          1. Lookup user and movie embeddings
          2. Multi-hot project genre string
          3. Concat [user_emb, movie_emb, genre_proj, dense_features]
          4. Pass through MLP with Batch Normalization & Dropout
          5. Final linear logit projection
        """
        u_idx = self.user_lookup(inputs["userId"])
        e_u = self.user_emb(u_idx)

        m_idx = self.movie_lookup(inputs["movieId"])
        e_m = self.movie_emb(m_idx)

        g_vec = self.genre_vectorizer(inputs["genres"])
        e_g = self.genre_dense(g_vec)

        dense_feat = inputs["dense_features"]

        # Concatenate dense representations
        x = tf.concat([e_u, e_m, e_g, dense_feat], axis=-1)

        # Feed through MLP
        for layer in self.mlp_layers:
            if isinstance(layer, (keras.layers.Dropout, keras.layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)

        deep_logits = self.projection(x)
        return deep_logits


class WideOnlyModel(keras.Model):
    """
    Variant A (Baseline): Pure linear memorization model.
    """

    def __init__(
        self,
        user_vocab: List[str],
        movie_vocab: List[str],
        n_dense_features: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
        self.n_dense_features = n_dense_features

        self.wide = WideComponent(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            n_dense_features=n_dense_features,
            name="wide_component",
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        logits = self.wide(inputs)
        return tf.sigmoid(logits)

    def predict_scores(self, features: Dict[str, np.ndarray], batch_size: int = 4096) -> np.ndarray:
        """High-throughput inference returning 1D probability array."""
        n_samples = len(features["userId"])
        probs = np.zeros(n_samples, dtype=np.float32)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = {
                "userId": tf.constant(features["userId"][i:end_idx], dtype=tf.string),
                "movieId": tf.constant(features["movieId"][i:end_idx], dtype=tf.string),
                "activity_pop_cross": tf.constant(features["activity_pop_cross"][i:end_idx], dtype=tf.int32),
                "user_movie_rating_cross": tf.constant(features["user_movie_rating_cross"][i:end_idx], dtype=tf.int32),
                "dense_features": tf.constant(features["dense_features"][i:end_idx], dtype=tf.float32),
            }
            batch_probs = self.call(batch, training=False).numpy().flatten()
            probs[i:end_idx] = batch_probs

        return probs

    def save_artifacts(self, save_dir: Union[str, Path]):
        """Save model weights and architecture config."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.save_weights(str(save_path / "model.weights.h5"))
        config = {
            "model_type": "WideOnlyModel",
            "user_vocab_size": len(self.user_vocab),
            "movie_vocab_size": len(self.movie_vocab),
            "n_dense_features": self.n_dense_features,
            "user_vocab": self.user_vocab,
            "movie_vocab": self.movie_vocab,
        }
        with open(save_path / "model_config.json", "w") as f:
            json.dump(config, f)
        logger.info(f"Saved WideOnlyModel artifacts to {save_path}")

    @classmethod
    def load_artifacts(cls, save_dir: Union[str, Path]) -> "WideOnlyModel":
        """Load WideOnlyModel from weights and config."""
        save_path = Path(save_dir)
        with open(save_path / "model_config.json", "r") as f:
            cfg = json.load(f)

        model = cls(
            user_vocab=cfg["user_vocab"],
            movie_vocab=cfg["movie_vocab"],
            n_dense_features=cfg.get("n_dense_features", 5),
        )

        # Build graph with dummy input before loading weights
        dummy = {
            "userId": tf.constant(["0"], dtype=tf.string),
            "movieId": tf.constant(["0"], dtype=tf.string),
            "activity_pop_cross": tf.constant([0], dtype=tf.int32),
            "user_movie_rating_cross": tf.constant([0], dtype=tf.int32),
            "dense_features": tf.constant([[0.0] * cfg.get("n_dense_features", 5)], dtype=tf.float32),
        }
        _ = model(dummy)

        try:
            model.load_weights(str(save_path / "model.weights.h5"))
        except Exception as e:
            logger.warning(f"Standard load_weights failed for WideOnlyModel ({e}), loading directly from HDF5...")
            import h5py
            with h5py.File(save_path / "model.weights.h5", "r") as h5f:
                for k in h5f.keys():
                    norm_k = k.replace("\\", "/")
                    if "user_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.user_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "movie_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.movie_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "act_pop_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.act_pop_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "rating_cross_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.rating_cross_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "dense_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.dense_linear.kernel.assign(h5f[k]["vars"]["0"][:])
                    elif norm_k == "wide" and "vars" in h5f[k] and "0" in h5f[k]["vars"]:
                        model.wide.bias.assign(h5f[k]["vars"]["0"][:])

        logger.info(f"Loaded WideOnlyModel from {save_path}")
        return model


class WideAndDeepModel(keras.Model):
    """
    Variant B (Primary): Google Wide & Deep Ranking Model combining
    linear memorization with deep non-linear feature interaction learning.
    """

    def __init__(
        self,
        user_vocab: List[str],
        movie_vocab: List[str],
        genre_vocab: List[str],
        embedding_dim: int = 64,
        hidden_units: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4,
        n_dense_features: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
        self.genre_vocab = genre_vocab
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.n_dense_features = n_dense_features

        # Wide component
        self.wide = WideComponent(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            n_dense_features=n_dense_features,
            name="wide_component",
        )

        # Deep component
        self.deep = DeepComponent(
            user_vocab=user_vocab,
            movie_vocab=movie_vocab,
            genre_vocab=genre_vocab,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name="deep_component",
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        wide_logits = self.wide(inputs)
        deep_logits = self.deep(inputs, training=training)
        total_logits = wide_logits + deep_logits
        return tf.sigmoid(total_logits)

    def predict_scores(self, features: Dict[str, np.ndarray], batch_size: int = 4096) -> np.ndarray:
        """High-throughput candidate inference returning 1D probability array."""
        n_samples = len(features["userId"])
        probs = np.zeros(n_samples, dtype=np.float32)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = {
                "userId": tf.constant(features["userId"][i:end_idx], dtype=tf.string),
                "movieId": tf.constant(features["movieId"][i:end_idx], dtype=tf.string),
                "genres": tf.constant(features["genres"][i:end_idx], dtype=tf.string),
                "activity_pop_cross": tf.constant(features["activity_pop_cross"][i:end_idx], dtype=tf.int32),
                "user_movie_rating_cross": tf.constant(features["user_movie_rating_cross"][i:end_idx], dtype=tf.int32),
                "dense_features": tf.constant(features["dense_features"][i:end_idx], dtype=tf.float32),
            }
            batch_probs = self.call(batch, training=False).numpy().flatten()
            probs[i:end_idx] = batch_probs

        return probs

    def save_artifacts(self, save_dir: Union[str, Path]):
        """Save model weights and architecture config."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.save_weights(str(save_path / "model.weights.h5"))
        config = {
            "model_type": "WideAndDeepModel",
            "user_vocab_size": len(self.user_vocab),
            "movie_vocab_size": len(self.movie_vocab),
            "genre_vocab_size": len(self.genre_vocab),
            "embedding_dim": self.embedding_dim,
            "hidden_units": list(self.hidden_units),
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "n_dense_features": self.n_dense_features,
            "user_vocab": self.user_vocab,
            "movie_vocab": self.movie_vocab,
            "genre_vocab": self.genre_vocab,
        }
        with open(save_path / "model_config.json", "w") as f:
            json.dump(config, f)
        logger.info(f"Saved WideAndDeepModel artifacts to {save_path}")

    @classmethod
    def load_artifacts(cls, save_dir: Union[str, Path]) -> "WideAndDeepModel":
        """Load WideAndDeepModel from weights and config."""
        save_path = Path(save_dir)
        with open(save_path / "model_config.json", "r") as f:
            cfg = json.load(f)

        model = cls(
            user_vocab=cfg["user_vocab"],
            movie_vocab=cfg["movie_vocab"],
            genre_vocab=cfg["genre_vocab"],
            embedding_dim=cfg.get("embedding_dim", 64),
            hidden_units=tuple(cfg.get("hidden_units", [128, 64, 32])),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            l2_reg=cfg.get("l2_reg", 1e-4),
            n_dense_features=cfg.get("n_dense_features", 5),
        )

        dummy = {
            "userId": tf.constant(["0"], dtype=tf.string),
            "movieId": tf.constant(["0"], dtype=tf.string),
            "genres": tf.constant(["Action"], dtype=tf.string),
            "activity_pop_cross": tf.constant([0], dtype=tf.int32),
            "user_movie_rating_cross": tf.constant([0], dtype=tf.int32),
            "dense_features": tf.constant([[0.0] * cfg.get("n_dense_features", 5)], dtype=tf.float32),
        }
        _ = model(dummy)

        try:
            model.load_weights(str(save_path / "model.weights.h5"))
        except Exception as e:
            logger.warning(f"Standard load_weights failed for WideAndDeepModel ({e}), loading directly from HDF5...")
            import h5py
            with h5py.File(save_path / "model.weights.h5", "r") as h5f:
                for k in h5f.keys():
                    norm_k = k.replace("\\", "/")
                    # Wide component weights
                    if "user_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.user_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "movie_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.movie_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "act_pop_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.act_pop_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "rating_cross_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.rating_cross_linear.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "dense_linear" in norm_k and "vars" in h5f[k]:
                        model.wide.dense_linear.kernel.assign(h5f[k]["vars"]["0"][:])
                    elif norm_k == "wide" and "vars" in h5f[k] and "0" in h5f[k]["vars"]:
                        model.wide.bias.assign(h5f[k]["vars"]["0"][:])
                    # Deep component weights
                    elif "user_emb" in norm_k and "vars" in h5f[k]:
                        model.deep.user_emb.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "movie_emb" in norm_k and "vars" in h5f[k]:
                        model.deep.movie_emb.embeddings.assign(h5f[k]["vars"]["0"][:])
                    elif "genre_dense" in norm_k and "vars" in h5f[k]:
                        model.deep.genre_dense.kernel.assign(h5f[k]["vars"]["0"][:])
                        model.deep.genre_dense.bias.assign(h5f[k]["vars"]["1"][:])
                    elif "mlp_layers" in norm_k and "vars" in h5f[k]:
                        if "dense_2" in norm_k:
                            model.deep.mlp_layers[6].kernel.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[6].bias.assign(h5f[k]["vars"]["1"][:])
                        elif "dense_1" in norm_k:
                            model.deep.mlp_layers[3].kernel.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[3].bias.assign(h5f[k]["vars"]["1"][:])
                        elif norm_k.endswith("dense"):
                            model.deep.mlp_layers[0].kernel.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[0].bias.assign(h5f[k]["vars"]["1"][:])
                        elif "batch_normalization_2" in norm_k:
                            model.deep.mlp_layers[7].gamma.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[7].beta.assign(h5f[k]["vars"]["1"][:])
                            model.deep.mlp_layers[7].moving_mean.assign(h5f[k]["vars"]["2"][:])
                            model.deep.mlp_layers[7].moving_variance.assign(h5f[k]["vars"]["3"][:])
                        elif "batch_normalization_1" in norm_k:
                            model.deep.mlp_layers[4].gamma.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[4].beta.assign(h5f[k]["vars"]["1"][:])
                            model.deep.mlp_layers[4].moving_mean.assign(h5f[k]["vars"]["2"][:])
                            model.deep.mlp_layers[4].moving_variance.assign(h5f[k]["vars"]["3"][:])
                        elif norm_k.endswith("batch_normalization"):
                            model.deep.mlp_layers[1].gamma.assign(h5f[k]["vars"]["0"][:])
                            model.deep.mlp_layers[1].beta.assign(h5f[k]["vars"]["1"][:])
                            model.deep.mlp_layers[1].moving_mean.assign(h5f[k]["vars"]["2"][:])
                            model.deep.mlp_layers[1].moving_variance.assign(h5f[k]["vars"]["3"][:])
                    elif "projection" in norm_k and "vars" in h5f[k]:
                        model.deep.projection.kernel.assign(h5f[k]["vars"]["0"][:])
                        model.deep.projection.bias.assign(h5f[k]["vars"]["1"][:])

        logger.info(f"Loaded WideAndDeepModel from {save_path}")
        return model
