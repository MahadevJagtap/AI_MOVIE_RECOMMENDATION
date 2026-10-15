"""
Triton Inference Server HTTP v2 Protocol Service.

Implements the official NVIDIA Triton / KServe v2 HTTP inference protocol
for environments without Linux container runtimes (such as native Windows).
Serves TensorFlow SavedModels directly from the standard Triton Model Repository.

Protocol Endpoints:
- GET  /v2/health/live
- GET  /v2/health/ready
- GET  /v2/models/{model_name}/ready
- GET  /v2/models/{model_name}
- GET  /v2/models/{model_name}/config
- POST /v2/models/{model_name}/infer
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import io
import json
import struct
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("triton_server")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [triton_server]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Resolve default repository path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_REPO = PROJECT_ROOT / "models" / "triton_model_repository"


class TritonServerState:
    """Holds loaded model signatures and metadata from Triton Model Repository."""

    def __init__(self, repo_dir: Path = DEFAULT_MODEL_REPO):
        self.repo_dir = Path(repo_dir)
        self.models: Dict[str, Any] = {}
        self.signatures: Dict[str, Any] = {}
        self.is_live = True
        self.is_ready = False

    def load_models(self):
        """Discover and load SavedModels from repository."""
        logger.info(f"Loading Triton Model Repository from {self.repo_dir}")
        for model_dir in self.repo_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            savedmodel_dir = model_dir / "1" / "model.savedmodel"
            if savedmodel_dir.exists():
                logger.info(f"Loading SavedModel for Triton model '{model_name}' from {savedmodel_dir}")
                loaded = tf.saved_model.load(str(savedmodel_dir))
                sig = loaded.signatures.get("serving_default")
                if sig is not None:
                    self.models[model_name] = loaded
                    self.signatures[model_name] = sig
                    logger.info(f"Model '{model_name}' ready with serving_default signature.")
                else:
                    logger.warning(f"Model '{model_name}' has no serving_default signature.")

        self.is_ready = len(self.signatures) > 0
        logger.info(f"Triton server ready: {len(self.signatures)} models active ({list(self.signatures.keys())})")


def create_triton_app(repo_dir: Path = DEFAULT_MODEL_REPO) -> FastAPI:
    """Create FastAPI app implementing Triton HTTP v2 protocol."""
    app = FastAPI(title="Triton HTTP v2 Protocol Inference Server", version="2.0")
    state = TritonServerState(repo_dir=repo_dir)

    @app.on_event("startup")
    async def startup():
        state.load_models()

    @app.get("/v2/health/live")
    async def health_live():
        if state.is_live:
            return Response(status_code=200)
        return Response(status_code=503)

    @app.get("/v2/health/ready")
    async def health_ready():
        if state.is_ready:
            return Response(status_code=200)
        return Response(status_code=503)

    @app.get("/v2/models/{model_name}/ready")
    async def model_ready(model_name: str):
        if model_name in state.signatures:
            return Response(status_code=200)
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found or not ready")

    @app.get("/v2/models/{model_name}")
    async def model_metadata(model_name: str):
        if model_name not in state.signatures:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return {
            "name": model_name,
            "versions": ["1"],
            "platform": "tensorflow_savedmodel",
            "inputs": [],
            "outputs": [],
        }

    @app.get("/v2/models/{model_name}/config")
    async def model_config(model_name: str):
        if model_name not in state.signatures:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        pbtxt_path = state.repo_dir / model_name / "config.pbtxt"
        config_text = pbtxt_path.read_text(encoding="utf-8") if pbtxt_path.exists() else ""
        return {"name": model_name, "raw_config": config_text}

    @app.post("/v2/models/{model_name}/infer")
    async def model_infer(model_name: str, request: Request):
        if model_name not in state.signatures:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        sig = state.signatures[model_name]
        raw_body = await request.body()

        # Handle either binary data payload or standard JSON
        json_len_hdr = request.headers.get("Inference-Header-Content-Length")
        if json_len_hdr:
            json_len = int(json_len_hdr)
            json_bytes = raw_body[:json_len]
            body = json.loads(json_bytes.decode("utf-8"))
            binary_buf = raw_body[json_len:]
        else:
            body = json.loads(raw_body.decode("utf-8"))
            binary_buf = b""

        request_id = body.get("id", "0")
        inputs_list = body.get("inputs", [])
        outputs_requested = [o["name"] for o in body.get("outputs", [])]

        # Parse inputs into TensorFlow tensors
        feed_dict = {}
        binary_offset = 0

        for inp in inputs_list:
            inp_name = inp["name"]
            datatype = inp.get("datatype", "FP32")
            shape = inp.get("shape", [])
            data = inp.get("data")

            if data is not None:
                # Direct JSON array format
                if datatype == "BYTES":
                    arr = np.array(data, dtype=object)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.string)
                elif datatype == "INT32":
                    arr = np.array(data, dtype=np.int32).reshape(shape)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.int32)
                elif datatype == "INT64":
                    arr = np.array(data, dtype=np.int64).reshape(shape)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.int64)
                elif datatype in ("FP32", "FLOAT"):
                    arr = np.array(data, dtype=np.float32).reshape(shape)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.float32)
                else:
                    arr = np.array(data)
                    feed_dict[inp_name] = tf.constant(arr)
            else:
                # Raw binary tensor data parsing
                params = inp.get("parameters", {})
                b_size = params.get("binary_data_size", 0)
                tensor_bytes = binary_buf[binary_offset:binary_offset + b_size]
                binary_offset += b_size

                if datatype == "BYTES":
                    # Triton byte tensor format: 4-byte length prefix per string
                    str_list = []
                    offset = 0
                    while offset < len(tensor_bytes):
                        s_len = struct.unpack_from("<I", tensor_bytes, offset)[0]
                        offset += 4
                        s_val = tensor_bytes[offset:offset + s_len].decode("utf-8", errors="replace")
                        offset += s_len
                        str_list.append(s_val)
                    arr = np.array(str_list, dtype=object)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.string)
                elif datatype == "INT32":
                    arr = np.frombuffer(tensor_bytes, dtype=np.int32).reshape(shape)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.int32)
                elif datatype in ("FP32", "FLOAT"):
                    arr = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
                    feed_dict[inp_name] = tf.constant(arr, dtype=tf.float32)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported datatype: {datatype}")

        # Run inference through concrete function
        try:
            results = sig(**feed_dict)
        except Exception as e:
            logger.error(f"Inference error in model '{model_name}': {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

        # Format Triton v2 response
        response_outputs = []
        for out_name, tensor_val in results.items():
            if outputs_requested and out_name not in outputs_requested:
                continue

            arr = tensor_val.numpy()
            dtype_str = "FP32" if arr.dtype in (np.float32, np.float64) else "INT32"
            response_outputs.append({
                "name": out_name,
                "datatype": dtype_str,
                "shape": list(arr.shape),
                "data": arr.flatten().tolist(),
            })

        return {
            "model_name": model_name,
            "model_version": "1",
            "id": request_id,
            "outputs": response_outputs,
        }

    return app


def run_triton_server(host: str = "127.0.0.1", port: int = 8001):
    """Start the Triton HTTP v2 protocol server."""
    app = create_triton_app()
    logger.info(f"Starting Triton HTTP v2 Protocol Server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Triton HTTP v2 Protocol Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    args = parser.parse_args()

    run_triton_server(host=args.host, port=args.port)
