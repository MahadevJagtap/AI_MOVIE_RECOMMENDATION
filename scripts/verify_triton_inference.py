"""
Verify Triton Inference Server vs Local TensorFlow Inference.

Steps:
1. Spawns Triton HTTP v2 server on 127.0.0.1:8001.
2. Waits for server readiness and model availability.
3. Issues real inference requests via tritonclient.http for:
   - user_tower: shape, dtype, and numerical parity.
   - ranking_model: shape, dtype, and numerical parity.
4. Compares against local TensorFlow inference with np.allclose(atol=1e-5).
5. Shuts down server and outputs verification report.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import time
import subprocess
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import RETRIEVAL_MODEL_DIR, RANKING_MODEL_DIR, TRITON_MODEL_REPOSITORY
from src.retrieval.two_tower import TwoTowerModel
from src.ranking.wide_and_deep import WideAndDeepModel
from src.serving.triton_client import TritonInferenceClient

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("verify_triton")

PYTHON_EXE = sys.executable
TRITON_PORT = 8001
TRITON_URL = f"127.0.0.1:{TRITON_PORT}"


def main():
    logger.info("=== STEP 1: Launching Triton HTTP v2 Server on 127.0.0.1:8001 ===")
    cmd = [
        PYTHON_EXE,
        "-m", "src.serving.triton_server_runner",
        "--host", "127.0.0.1",
        "--port", str(TRITON_PORT),
    ]
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    server_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)

    try:
        # Wait for server readiness
        client = TritonInferenceClient(url=TRITON_URL, enabled=True, connection_timeout=1.0)
        ready = False
        for attempt in range(20):
            time.sleep(1.0)
            if client.is_available(force_check=True):
                ready = True
                logger.info(f"Triton server connected and ready on attempt {attempt + 1}!")
                break
            logger.info(f"Waiting for Triton server... (attempt {attempt + 1}/20)")

        if not ready:
            raise TimeoutError("Triton server failed to become ready within 20 seconds.")

        # === STEP 2: Load Local Models for Ground Truth Parity Check ===
        logger.info("=== STEP 2: Loading Local TensorFlow Models for Parity Comparison ===")
        retrieval_dir = Path(RETRIEVAL_MODEL_DIR)
        two_tower = TwoTowerModel.load_artifacts(retrieval_dir)
        local_user_tower = two_tower.user_tower

        wd_dir = Path(RANKING_MODEL_DIR) / "wide_and_deep"
        local_ranking_model = WideAndDeepModel.load_artifacts(wd_dir)

        # === STEP 3: Verify User Tower ===
        logger.info("=== STEP 3: Testing User Tower Inference ===")
        test_uids = ["198", "662", "805", "924"]

        # Local inference
        t0 = time.perf_counter()
        local_tensor = tf.constant(test_uids, dtype=tf.string)
        local_embs = local_user_tower(local_tensor).numpy()
        local_ut_latency = (time.perf_counter() - t0) * 1000

        # Triton inference
        t0 = time.perf_counter()
        triton_embs = client.infer_user_embedding(test_uids)
        triton_ut_latency = (time.perf_counter() - t0) * 1000

        logger.info(f"Local user_tower output: shape={local_embs.shape}, dtype={local_embs.dtype}, latency={local_ut_latency:.2f}ms")
        logger.info(f"Triton user_tower output: shape={triton_embs.shape}, dtype={triton_embs.dtype}, latency={triton_ut_latency:.2f}ms")

        assert triton_embs.shape == (4, 64), f"Unexpected Triton shape: {triton_embs.shape}"
        assert triton_embs.dtype == np.float32, f"Unexpected Triton dtype: {triton_embs.dtype}"
        ut_max_diff = np.max(np.abs(triton_embs - local_embs))
        logger.info(f"User Tower Max Numerical Difference: {ut_max_diff:.8e}")
        assert np.allclose(triton_embs, local_embs, atol=1e-5), f"Numerical mismatch in user_tower: max diff {ut_max_diff}"

        # === STEP 4: Verify Ranking Model ===
        logger.info("=== STEP 4: Testing Ranking Model Inference ===")
        features = {
            "userId": np.array(["198", "198", "662", "662"], dtype=object),
            "movieId": np.array(["1", "356", "296", "593"], dtype=object),
            "genres": np.array([
                "Adventure|Animation|Children",
                "Comedy|Drama|Romance",
                "Crime|Drama",
                "Crime|Horror|Thriller",
            ], dtype=object),
            "activity_pop_cross": np.array([0, 1, 2, 3], dtype=np.int32),
            "user_movie_rating_cross": np.array([1, 2, 0, 3], dtype=np.int32),
            "dense_features": np.array([
                [0.1, -0.2, 0.5, 1.2, 0.8],
                [0.4, 0.1, -0.4, 0.2, -0.1],
                [-0.3, 0.8, 0.1, -0.5, 0.3],
                [0.2, -0.1, 0.7, 0.3, 0.0],
            ], dtype=np.float32),
        }

        # Local inference
        t0 = time.perf_counter()
        local_probs = local_ranking_model.predict_scores(features)
        local_rk_latency = (time.perf_counter() - t0) * 1000

        # Triton inference
        t0 = time.perf_counter()
        triton_probs = client.infer_ranking_scores(features)
        triton_rk_latency = (time.perf_counter() - t0) * 1000

        logger.info(f"Local ranking output: shape={local_probs.shape}, dtype={local_probs.dtype}, latency={local_rk_latency:.2f}ms")
        logger.info(f"Triton ranking output: shape={triton_probs.shape}, dtype={triton_probs.dtype}, latency={triton_rk_latency:.2f}ms")
        logger.info(f"Local scores: {local_probs}")
        logger.info(f"Triton scores: {triton_probs}")

        assert triton_probs.shape == (4,), f"Unexpected Triton shape: {triton_probs.shape}"
        assert triton_probs.dtype == np.float32, f"Unexpected Triton dtype: {triton_probs.dtype}"
        rk_max_diff = np.max(np.abs(triton_probs - local_probs))
        logger.info(f"Ranking Model Max Numerical Difference: {rk_max_diff:.8e}")
        assert np.allclose(triton_probs, local_probs, atol=1e-5), f"Numerical mismatch in ranking_model: max diff {rk_max_diff}"

        # === STEP 5: Verify Fallback Mechanism ===
        logger.info("=== STEP 5: Testing Local Fallback ===")
        disabled_client = TritonInferenceClient(url="127.0.0.1:9999", enabled=False)
        fallback_embs, backend_ut = disabled_client.infer_user_embedding_with_fallback(
            test_uids, local_user_tower=local_user_tower
        )
        assert backend_ut == "local_tensorflow"
        assert np.allclose(fallback_embs, local_embs, atol=1e-5)
        logger.info("Fallback test passed: successfully fell back to local_tensorflow.")

        # Print Summary Table
        print("\n" + "=" * 80)
        print("TRITON INFERENCE SERVER VERIFICATION REPORT")
        print("=" * 80)
        print(f"{'Test Component':<25} | {'Local TF Shape':<14} | {'Triton Shape':<14} | {'Max Diff':<12} | {'Parity':<8}")
        print("-" * 80)
        print(f"{'user_tower (4 users)':<25} | {str(local_embs.shape):<14} | {str(triton_embs.shape):<14} | {ut_max_diff:<12.2e} | {'MATCH [OK]':<10}")
        print(f"{'ranking_model (4 items)':<25} | {str(local_probs.shape):<14} | {str(triton_probs.shape):<14} | {rk_max_diff:<12.2e} | {'MATCH [OK]':<10}")
        print(f"{'local fallback':<25} | {'(4, 64)':<14} | {'(4, 64)':<14} | {'0.00e+00':<12} | {'MATCH [OK]':<10}")
        print("=" * 80 + "\n")

    finally:
        logger.info("Shutting down Triton server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        logger.info("Triton server process terminated cleanly.")


if __name__ == "__main__":
    main()
