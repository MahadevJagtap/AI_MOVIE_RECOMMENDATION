"""
Triton Inference Server client for recommendation engine.

Supports:
- User tower embedding inference ('user_tower' model).
- Candidate ranking probability scoring ('ranking_model' model).
- Health check and model readiness inspection.
- Resilient local TensorFlow inference fallback when Triton is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    httpclient = None  # type: ignore
    InferenceServerException = Exception  # type: ignore

logger = logging.getLogger("triton_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [triton_client]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TritonInferenceClient:
    """
    Client for querying NVIDIA Triton Inference Server (or KServe v2 HTTP compatible server).
    Provides automatic fallback to local TensorFlow models on connection loss or errors.
    """

    def __init__(
        self,
        url: str = "127.0.0.1:8001",
        enabled: bool = True,
        connection_timeout: float = 1.0,
        network_timeout: float = 3.0,
        user_tower_model_name: str = "user_tower",
        ranking_model_name: str = "ranking_model",
    ):
        self.url = url
        self.enabled = enabled and (httpclient is not None)
        self.connection_timeout = connection_timeout
        self.network_timeout = network_timeout
        self.user_tower_model_name = user_tower_model_name
        self.ranking_model_name = ranking_model_name

        self._client: Optional[httpclient.InferenceServerClient] = None
        self._is_connected = False
        self._last_check_time = 0.0

        if self.enabled:
            self._init_client()

    def _init_client(self):
        """Initialize HTTP client instance."""
        try:
            self._client = httpclient.InferenceServerClient(
                url=self.url,
                connection_timeout=self.connection_timeout,
                network_timeout=self.network_timeout,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Triton client for {self.url}: {e}")
            self._client = None
            self._is_connected = False

    def is_available(self, force_check: bool = False) -> bool:
        """
        Check if Triton server is reachable and required models are ready.
        Caches health status for 5 seconds unless force_check is True.
        """
        if not self.enabled or self._client is None:
            return False

        now = time.time()
        if not force_check and (now - self._last_check_time < 5.0):
            return self._is_connected

        self._last_check_time = now
        try:
            if not self._client.is_server_live():
                self._is_connected = False
                return False
            if not self._client.is_server_ready():
                self._is_connected = False
                return False

            # Check model readiness
            ut_ready = self._client.is_model_ready(self.user_tower_model_name)
            rk_ready = self._client.is_model_ready(self.ranking_model_name)
            self._is_connected = bool(ut_ready and rk_ready)
            return self._is_connected
        except Exception as e:
            logger.debug(f"Triton availability check failed: {e}")
            self._is_connected = False
            return False

    def infer_user_embedding(
        self,
        user_ids: Union[str, List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Query Triton for user tower embedding vectors.

        Args:
            user_ids: Single user ID or list/array of user IDs.

        Returns:
            np.ndarray: float32 array of shape (N, 64).
        """
        if not self.is_available():
            raise ConnectionError(f"Triton server at {self.url} is not available")

        # Standardize to 1D string array
        if isinstance(user_ids, (str, int)):
            uids = [str(user_ids)]
        else:
            uids = [str(u) for u in user_ids]

        n_samples = len(uids)
        uid_arr = np.array(uids, dtype=object)

        # Prepare Triton inputs/outputs
        inputs = [httpclient.InferInput("user_id", [n_samples], "BYTES")]
        inputs[0].set_data_from_numpy(uid_arr, binary_data=False)

        outputs = [httpclient.InferRequestedOutput("user_embedding", binary_data=False)]

        try:
            res = self._client.infer(
                model_name=self.user_tower_model_name,
                inputs=inputs,
                outputs=outputs,
            )
            embedding: np.ndarray = res.as_numpy("user_embedding")
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Triton user_tower inference failed: {e}")
            raise

    def infer_ranking_scores(
        self,
        features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Query Triton for candidate ranking probabilities.

        Args:
            features: Dictionary containing:
                - userId: array of user ID strings
                - movieId: array of movie ID strings
                - genres: array of genre strings
                - activity_pop_cross: array of int32 bucket indices
                - user_movie_rating_cross: array of int32 bucket indices
                - dense_features: float32 array of shape (N, 5)

        Returns:
            np.ndarray: 1D float32 array of probability scores.
        """
        if not self.is_available():
            raise ConnectionError(f"Triton server at {self.url} is not available")

        n_samples = len(features["userId"])
        if n_samples == 0:
            return np.empty(0, dtype=np.float32)

        # Prepare numpy inputs
        u_arr = np.array([str(x) for x in features["userId"]], dtype=object)
        m_arr = np.array([str(x) for x in features["movieId"]], dtype=object)
        g_arr = np.array([str(x) for x in features["genres"]], dtype=object)
        act_pop_arr = np.array(features["activity_pop_cross"], dtype=np.int32)
        rat_cross_arr = np.array(features["user_movie_rating_cross"], dtype=np.int32)
        dense_arr = np.array(features["dense_features"], dtype=np.float32)

        # Ensure dense_features is (N, 5)
        if len(dense_arr.shape) == 1:
            dense_arr = dense_arr.reshape((n_samples, -1))

        # Build Triton inputs
        inputs = [
            httpclient.InferInput("userId", [n_samples], "BYTES"),
            httpclient.InferInput("movieId", [n_samples], "BYTES"),
            httpclient.InferInput("genres", [n_samples], "BYTES"),
            httpclient.InferInput("activity_pop_cross", [n_samples], "INT32"),
            httpclient.InferInput("user_movie_rating_cross", [n_samples], "INT32"),
            httpclient.InferInput("dense_features", list(dense_arr.shape), "FP32"),
        ]
        inputs[0].set_data_from_numpy(u_arr, binary_data=False)
        inputs[1].set_data_from_numpy(m_arr, binary_data=False)
        inputs[2].set_data_from_numpy(g_arr, binary_data=False)
        inputs[3].set_data_from_numpy(act_pop_arr, binary_data=False)
        inputs[4].set_data_from_numpy(rat_cross_arr, binary_data=False)
        inputs[5].set_data_from_numpy(dense_arr, binary_data=False)

        outputs = [httpclient.InferRequestedOutput("probabilities", binary_data=False)]

        try:
            res = self._client.infer(
                model_name=self.ranking_model_name,
                inputs=inputs,
                outputs=outputs,
            )
            probs: np.ndarray = res.as_numpy("probabilities")
            return probs.flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Triton ranking_model inference failed: {e}")
            raise

    def infer_user_embedding_with_fallback(
        self,
        user_ids: Union[str, List[str], np.ndarray],
        local_user_tower: Any = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Query Triton for user embedding with automatic fallback to local model.

        Returns:
            Tuple[np.ndarray, str]: (embedding_array, source_backend)
            where source_backend is 'triton' or 'local_tensorflow'.
        """
        if self.is_available():
            try:
                emb = self.infer_user_embedding(user_ids)
                return emb, "triton"
            except Exception as e:
                logger.warning(f"Triton user embedding failed ({e}); falling back to local TensorFlow.")

        if local_user_tower is not None:
            import tensorflow as tf
            if isinstance(user_ids, (str, int)):
                uids = [str(user_ids)]
            else:
                uids = [str(u) for u in user_ids]
            inp_tensor = tf.constant(uids, dtype=tf.string)
            local_emb = local_user_tower(inp_tensor).numpy()
            return local_emb.astype(np.float32), "local_tensorflow"

        raise RuntimeError("Triton unavailable and no local_user_tower provided for fallback")

    def infer_ranking_scores_with_fallback(
        self,
        features: Dict[str, Any],
        local_ranking_model: Any = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Query Triton for ranking probabilities with automatic fallback to local model.

        Returns:
            Tuple[np.ndarray, str]: (probabilities_array, source_backend)
            where source_backend is 'triton' or 'local_tensorflow'.
        """
        if self.is_available():
            try:
                scores = self.infer_ranking_scores(features)
                return scores, "triton"
            except Exception as e:
                logger.warning(f"Triton ranking scoring failed ({e}); falling back to local TensorFlow.")

        if local_ranking_model is not None:
            scores = local_ranking_model.predict_scores(features)
            return scores.astype(np.float32), "local_tensorflow"

        raise RuntimeError("Triton unavailable and no local_ranking_model provided for fallback")
