"""
Export trained TensorFlow models to NVIDIA Triton Model Repository format.

Exports:
1. user_tower:
   Input: 'user_id' [None] (string)
   Output: 'user_embedding' [None, 64] (float32)
   Path: models/triton_model_repository/user_tower/1/model.savedmodel
   Config: models/triton_model_repository/user_tower/config.pbtxt

2. ranking_model:
   Inputs:
     - 'userId' [None] (string)
     - 'movieId' [None] (string)
     - 'genres' [None] (string)
     - 'activity_pop_cross' [None] (int32)
     - 'user_movie_rating_cross' [None] (int32)
     - 'dense_features' [None, 5] (float32)
   Output: 'probabilities' [None, 1] (float32)
   Path: models/triton_model_repository/ranking_model/1/model.savedmodel
   Config: models/triton_model_repository/ranking_model/config.pbtxt
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import shutil
import logging
from pathlib import Path
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import RETRIEVAL_MODEL_DIR, RANKING_MODEL_DIR, TRITON_MODEL_REPOSITORY
from src.retrieval.two_tower import TwoTowerModel
from src.ranking.wide_and_deep import WideAndDeepModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("export_triton")


class UserTowerServingModule(tf.Module):
    """Wrapper module providing a clean concrete function for user_tower serving."""

    def __init__(self, user_tower):
        super().__init__()
        self.user_tower = user_tower

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="user_id"),
    ])
    def serve(self, user_id):
        embedding = self.user_tower(user_id)
        return {"user_embedding": embedding}


class RankingServingModule(tf.Module):
    """Wrapper module providing a clean concrete function for ranking serving."""

    def __init__(self, ranking_model):
        super().__init__()
        self.ranking_model = ranking_model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="userId"),
        tf.TensorSpec(shape=[None], dtype=tf.string, name="movieId"),
        tf.TensorSpec(shape=[None], dtype=tf.string, name="genres"),
        tf.TensorSpec(shape=[None], dtype=tf.int32, name="activity_pop_cross"),
        tf.TensorSpec(shape=[None], dtype=tf.int32, name="user_movie_rating_cross"),
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32, name="dense_features"),
    ])
    def serve(self, userId, movieId, genres, activity_pop_cross, user_movie_rating_cross, dense_features):
        inputs = {
            "userId": userId,
            "movieId": movieId,
            "genres": genres,
            "activity_pop_cross": activity_pop_cross,
            "user_movie_rating_cross": user_movie_rating_cross,
            "dense_features": dense_features,
        }
        probs = self.ranking_model(inputs, training=False)
        return {"probabilities": probs}


def export_user_tower(repo_dir: Path) -> Path:
    """Export TwoTowerModel.user_tower to Triton SavedModel format."""
    logger.info("Exporting user_tower model for Triton...")
    retrieval_dir = Path(RETRIEVAL_MODEL_DIR)
    two_tower = TwoTowerModel.load_artifacts(retrieval_dir)
    user_tower = two_tower.user_tower

    serving_module = UserTowerServingModule(user_tower)

    # Destination directory: models/triton_model_repository/user_tower/1/model.savedmodel
    target_dir = repo_dir / "user_tower" / "1" / "model.savedmodel"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    tf.saved_model.save(
        serving_module,
        str(target_dir),
        signatures={"serving_default": serving_module.serve},
    )
    logger.info(f"SavedModel user_tower exported to {target_dir}")

    # Generate config.pbtxt
    pbtxt_content = """name: "user_tower"
platform: "tensorflow_savedmodel"
max_batch_size: 1024

input [
  {
    name: "user_id"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

output [
  {
    name: "user_embedding"
    data_type: TYPE_FP32
    dims: [ 64 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
}
"""
    pbtxt_path = repo_dir / "user_tower" / "config.pbtxt"
    with open(pbtxt_path, "w", encoding="utf-8") as f:
        f.write(pbtxt_content)
    logger.info(f"Generated Triton config at {pbtxt_path}")
    return target_dir


def export_ranking_model(repo_dir: Path) -> Path:
    """Export WideAndDeepModel to Triton SavedModel format."""
    logger.info("Exporting ranking_model (Wide & Deep) for Triton...")
    wd_dir = Path(RANKING_MODEL_DIR) / "wide_and_deep"
    ranking_model = WideAndDeepModel.load_artifacts(save_dir=wd_dir)

    serving_module = RankingServingModule(ranking_model)

    # Destination directory: models/triton_model_repository/ranking_model/1/model.savedmodel
    target_dir = repo_dir / "ranking_model" / "1" / "model.savedmodel"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    tf.saved_model.save(
        serving_module,
        str(target_dir),
        signatures={"serving_default": serving_module.serve},
    )
    logger.info(f"SavedModel ranking_model exported to {target_dir}")

    # Generate config.pbtxt
    pbtxt_content = """name: "ranking_model"
platform: "tensorflow_savedmodel"
max_batch_size: 1024

input [
  {
    name: "userId"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "movieId"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "genres"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "activity_pop_cross"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "user_movie_rating_cross"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "dense_features"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]

output [
  {
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
}
"""
    pbtxt_path = repo_dir / "ranking_model" / "config.pbtxt"
    with open(pbtxt_path, "w", encoding="utf-8") as f:
        f.write(pbtxt_content)
    logger.info(f"Generated Triton config at {pbtxt_path}")
    return target_dir


def main():
    repo_dir = Path(TRITON_MODEL_REPOSITORY)
    repo_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting models to Triton repository: {repo_dir}")

    export_user_tower(repo_dir)
    export_ranking_model(repo_dir)

    # Clean up scratch if exists
    scratch_dir = PROJECT_ROOT / "models" / "triton_temp"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    logger.info("Successfully exported all Triton models and configurations.")


if __name__ == "__main__":
    main()
