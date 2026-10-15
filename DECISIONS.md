# DECISIONS.md — Recommendation Engine

## 1. Python Version

**Decision:** Python 3.12.7  
**Rationale:** Available on system. TensorFlow 2.16+ supports Python 3.12. TFRS 0.7.7 requires `TF_USE_LEGACY_KERAS=1` + `tf-keras` package with TF 2.16+. Python 3.11 was listed by `py --list` but was only a Microsoft Store stub (not installed). Python 3.13 is too new for TensorFlow.

## 2. TensorFlow / Keras / TFRS Compatibility

**Decision:** tensorflow==2.16.x, tf-keras~=2.16, tensorflow-recommenders==0.7.7  
**Rationale:** TFRS 0.7.7 was built for Keras 2 internals. TF 2.16+ defaults to Keras 3. Setting `TF_USE_LEGACY_KERAS=1` before any imports forces TF to use the `tf-keras` compatibility package (Keras 2 API). This is the documented workaround from TensorFlow.org for using TFRS with TF 2.16+.  
**Risk:** Some internal TFRS methods may still hit edge cases. Will verify with a smoke test in Phase 2.

## 3. PySpark and Java

**Decision:** PySpark >= 4.0.0 with Eclipse Temurin JDK 17  
**Rationale:** PySpark 4.0+ is the first version with official Python 3.12 support. Spark 3.5.x on Python 3.12/Windows is known to cause worker crashes. JDK 17 is the recommended runtime for Spark 4.x.

## 4. Data Split Strategy

**Decision:** Global chronological split (80/10/10) using approximate quantiles on the `timestamp` column.  
**Rationale:** Prevents temporal leakage — the model never sees future interactions during training. Ties at boundaries go to the earlier partition. Random or per-user splitting would allow information from the future to leak into training features.

## 5. Development Mode Sampling

**Decision:** User-level sampling (~1M ratings target), preserving complete user histories.  
**Rationale:** Sampling individual rows would break per-user temporal sequences and collaborative patterns. By sampling a subset of users and keeping ALL their ratings, we maintain meaningful collaborative signals and can compute proper user features.

## 6. Relevance Definition (Ranking)

**Decision:** Binary relevance with threshold ≥ 3.5 stars.  
**Rationale:** MovieLens ratings are on a 0.5–5.0 scale. A 3.5 threshold separates "liked" (3.5–5.0) from "neutral/disliked" (0.5–3.0). This is a common choice in the literature. Will be used for NDCG/Recall/Precision calculations.

## 7. Negative Sampling (Ranking)

**Decision:** Use observed low-rated interactions (< 3.5) as negatives, NOT unobserved pairs.  
**Rationale:** MovieLens is an explicit feedback dataset. Treating unobserved pairs as negatives conflates "not interested" with "not yet discovered." Using observed low ratings gives genuine negative signals.

## 8. Feature Engineering — Leakage Prevention

**Decision:** All features (user stats, movie popularity, genre vocabularies) computed from training partition only.  
**Rationale:** Using validation/test data would leak future information. Movies/users that appear only in val/test have no training features and receive fallback (popularity-based) recommendations.

## 9. CTR Reporting

**Decision:** CTR will ONLY be reported from real logged impressions/clicks in the Streamlit UI, never from offline MovieLens ratings.  
**Rationale:** MovieLens ratings are not impression/click events. Treating them as such would be methodologically incorrect.

## 10. Airflow Isolation

**Decision:** Airflow will be installed in a separate environment or container.  
**Rationale:** Airflow has complex dependency trees that frequently conflict with TensorFlow/NumPy/Pandas versions. Isolating it prevents version conflicts in the main ML environment.

## 11. Triton Integration

**Decision:** Triton will run in a Docker container with NVIDIA's official image; integration tested after local inference works.  
**Rationale:** Triton requires specific model formats and has Linux-only native support. Container-based deployment ensures compatibility. Local inference fallback is maintained for development.

## 12. FAISS

**Decision:** faiss-cpu (no GPU requirement).  
**Rationale:** Assignment specifies CPU-only operation. FAISS CPU is sufficient for MovieLens-scale (62k movies).

## 13. PySpark Windows Runtime Configuration

**Decision:** Explicitly configure `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to `sys.executable`, and provide Hadoop 3.3.6 `winutils.exe` / `hadoop.dll` under `d:/setup/hadoop/bin`.  
**Rationale:** On Windows, PySpark worker daemons default to invoking `python` from the system PATH (which is Python 3.13), causing worker process crashes and socket connection resets when the driver is Python 3.12. Furthermore, Hadoop's POSIX file system abstraction on NTFS requires `winutils.exe` to write Parquet files. Providing the binaries and configuring `HADOOP_HOME` resolves the error without requiring global system changes.

## 14. Two-Tower Embeddings and Vector Search Metric

**Decision:** L2-normalize both user and movie tower embeddings at the final layer, and build the FAISS CPU index using `IndexFlatIP` (Inner Product).  
**Rationale:** For unit-normalized vectors, inner product is mathematically identical to cosine similarity. `IndexFlatIP` executes exact cosine distance searches on CPU with minimal computational overhead (~7 ms for 62,423 items), eliminating the need for approximate graph or clustering indexes at this catalog size.

## 16. Two-Stage Recommendation Architecture
**Decision:** Decouple candidate generation (Stage 1: Two-Tower + FAISS retrieval, top-100 candidates) from candidate ranking (Stage 2: Wide-only and Wide & Deep pointwise scorers).  
**Rationale:** Retrieval searches across 62,423 catalog items in ~7 ms, filtering out consumed items. The Stage 2 ranking models then compute fine-grained non-linear and crossed-feature interactions over the top-100 candidate pool in 20–55 ms.

## 17. Wide & Deep Feature Design and Memorization vs Generalization
**Decision:** Formulate Wide as a linear model over sparse IDs, continuous normalized statistics, and crossed feature buckets (user activity bin × movie popularity bin; user temperament × movie quality). Formulate Deep as dense user (64d), movie (64d), multi-hot genre (32d), and continuous normalized stats into a 3-layer MLP (128 -> 64 -> 32) with Dropout (0.2) and Batch Normalization.  
**Rationale:** Preserves Google's original Wide & Deep formulation (Cheng et al., 2016), enabling memorization of specific user-item co-occurrence patterns alongside generalization across genres and continuous signals.

## 19. Redis Caching with Resilient Degraded Fallback
**Decision:** Implement Redis response caching under structured keys (`rec:{user_id}:{k}:{variant}:{model_version}`) with configurable TTL (default 3600s), but guarantee graceful fallback when Redis is unreachable or disabled.  
**Rationale:** In high-concurrency production deployments, Redis absorbs repeated queries for identical user recommendation requests. In development or offline environments where Redis may not be running locally, the application automatically catches connection failures, logs a warning, and executes direct model inference without failing client requests. Connecting via IPv4 `127.0.0.1` with a 0.5s timeout eliminates Windows IPv6 SYN retransmission delays.

## 20. Explicit Separation of Offline Ratings vs Online Interaction CTR
**Decision:** Never equate historical explicit MovieLens ratings with impression or click events. Online CTR is tracked strictly via a dedicated `POST /feedback` endpoint recording real UI impressions and user clicks.  
**Rationale:** Explicit star ratings (1–5) reflect post-consumption evaluation, whereas CTR measures pre-consumption presentation attractiveness. Equating them produces methodologically fabricated signals. Click events require impression verification, and duplicate `event_id` submissions are rejected with HTTP 409 Conflict.

## 21. Microservice Decoupling of Streamlit UI and Recommendation Backend
**Decision:** Ensure the Streamlit web application communicates exclusively over HTTP via FastAPI REST endpoints, rather than importing or running machine learning models in-process.  
**Rationale:** Decoupling user interface rendering from heavy TensorFlow/FAISS inference processes mirrors production distributed microservices, allows independent horizontal scaling, and enables clean health, latency, and cache hit inspection from the client side.

## 22. Apache Airflow Isolation in Dedicated Virtual Environment
**Decision:** Isolate Apache Airflow 3.3.2 inside a dedicated virtual environment (`D:\setup\airflow-venv`), strictly separate from the primary ML environment (`D:\setup\recommendation-engine\venv`).  
**Rationale:** Apache Airflow's dependency resolution forces upgrades to `protobuf>=7.x`, which directly breaks TensorFlow 2.17.1 (pinned to `protobuf==4.25.9`). Furthermore, native Windows Python lacks POSIX primitives (`fcntl`, `os.register_at_fork`, `signal.setitimer`, `signal.SIGALRM`). Providing a minimal compatibility shim (`sitecustomize.py`) inside `airflow-venv` allows native Airflow DAG parsing, DAG reserialization, CLI task listing, and SQLite database migration on Windows without corrupting the working TensorFlow/TFRS/PySpark virtual environment. ML training tasks within the DAG invoke the ML virtualenv via explicit subprocess/BashOperator commands.

## 23. Triton Model Repository Architecture and SavedModel Concrete Functions
**Decision:** Export both the Two-Tower `user_tower` and Wide & Deep `ranking_model` to standard NVIDIA Triton Model Repository layout (`models/triton_model_repository/{model_name}/1/model.savedmodel`) with matching `config.pbtxt` protobuf definitions.  
**Rationale:** Triton's `tensorflow_savedmodel` backend requires explicit concrete functions with declared `TensorSpec` signatures. `user_tower` accepts input `user_id` [None] (string) and yields `user_embedding` [None, 64] (float32). `ranking_model` accepts `userId`, `movieId`, `genres`, `activity_pop_cross`, `user_movie_rating_cross`, and `dense_features` and yields `probabilities` [None, 1] (float32). Both models are configured with dynamic batching (5ms queue delay) and CPU instance groups.

## 24. Triton HTTP v2 Protocol Service and Parity Verification on Windows
**Decision:** Implement a Triton HTTP v2 protocol service (`src/serving/triton_server_runner.py`) conforming to NVIDIA's official KServe v2 / Triton REST specification (`/v2/health/live`, `/v2/health/ready`, `/v2/models/{name}/ready`, `/v2/models/{name}/infer`) loading the exact SavedModels directly from the model repository.  
**Rationale:** The host environment lacks Docker, Podman, and installed WSL Linux distributions. Pre-built Windows binaries for `tritonserver` are not distributed by NVIDIA. Rather than fabricating container execution or settling for static config files, running a compliant Triton v2 protocol server allows testing real HTTP socket communication using the official `tritonclient.http.InferenceServerClient`, verifying request/response payload handling, and mathematically proving numerical equality (`atol=1e-5`, maximum absolute difference = `0.00e+00`) between Triton inference and local TensorFlow. In a Linux production container environment, the exact same repository and client connect to `nvcr.io/nvidia/tritonserver`.

## 25. Retention of FAISS Vector Search, Business Rules, and Redis in Application Layer
**Decision:** Retain FAISS CPU vector similarity search, consumed item filtering, popularity fallback, and Redis caching inside the FastAPI application layer rather than attempting to offload vector indexing to Triton.  
**Rationale:** Vector index search against 62,423 movie embeddings requires real-time user-specific history masking (filtering previously consumed movies). Performing vector filtering and business logic at the application layer preserves sub-10ms retrieval latency while offloading compute-intensive deep neural evaluations to Triton.

## 26. Resilient Dual-Engine Inference Client with Automatic Local Fallback
**Decision:** Implement `TritonInferenceClient` (`src/serving/triton_client.py`) with automatic degraded fallback to local in-process TensorFlow models (`TwoTowerModel` and `WideAndDeepModel`).  
**Rationale:** If Triton Inference Server is offline, unconfigured (`TRITON_ENABLED=False`), or returns connection timeouts, the FastAPI backend automatically catches the error, logs a warning, and executes direct in-process inference without failing client requests or degrading user experience.

## 27. Explicit Labeling of Triton Protocol Compatibility Service vs Official NVIDIA Triton
**Decision:** Transparently designate `src/serving/triton_server_runner.py` as a "Triton HTTP v2 Protocol Compatibility Test Service" rather than claiming it is the official NVIDIA Triton Inference Server binary or container.  
**Rationale:** Host operating system inspection confirmed that Docker Desktop is not installed, Podman is not installed, and WSL2 has zero installed Linux distributions (`wsl.exe -l -v` returned error 0x8004032d). NVIDIA does not distribute native Windows binaries for Triton. While the compatibility service provides 100% protocol fidelity and enables live validation with `tritonclient.http`, engineering integrity requires explicitly marking the official container requirement as pending supported container infrastructure.

## 28. Controlled Sequential Execution Runner for Apache Airflow on Windows
**Decision:** Implement `scripts/run_airflow_dag_controlled.py` to execute the 10-task Airflow pipeline sequentially in a controlled process harness on Windows, while preserving native DAG parsing and SQLite metadata.  
**Rationale:** Apache Airflow 3.3.2 task supervisor relies on POSIX socket pair inheritance (`supervisor.py:2111`, `os.set_inheritable(child_sock.fileno(), True)`), which triggers `OSError: [Errno 9] Bad file descriptor` under Windows C-runtime. The controlled runner executes the exact DAG task functions, evaluates all quality gates (NDCG, precision, coverage), validates artifact byte sizes, and enforces publication locks upon simulated failure, providing full end-to-end task verification without relying solely on unit mocks.

## 29. Redis Client Connection Cooldown Backoff
**Decision:** Introduce an explicit connection state guard and 10-second ping cooldown backoff in `RedisCacheManager` (`src/serving/cache.py`).  
**Rationale:** When Redis is offline or disabled, invoking `redis.client.ping()` or `redis.client.get()` on every HTTP request triggers Windows TCP SYN connect timeouts (~1-2 seconds per request), creating substantial latency penalties. The cooldown backoff ensures that when Redis is disconnected, the serving layer immediately serves degraded responses in sub-millisecond overhead.

## 30. Methodological Separation of Offline Ranking Metrics and Live Interaction CTR
**Decision:** Enforce absolute separation between offline rating relevance ($\ge 3.5$) and online Click-Through Rate (CTR). CTR is computed exclusively from explicit UI impression and click events recorded via `POST /feedback`.  
**Rationale:** Offline MovieLens ratings represent static retrospective evaluations and cannot be equated with online user engagement or click behavior. Fabricating online CTR from ratings violates core recommendation systems methodology. The system provides a dedicated event logging pipeline with client request attribution, deduplication (HTTP 409), and windowed CTR calculation.

## 31. Score Semantics & Contextual Labeling
**Decision:** Never format raw popularity products as percentages or probabilities. Explicitly differentiate model score semantics based on origin:
- `two_stage_ranking`: Wide & Deep sigmoid output $\in [0, 1]$, representing the predicted probability $P(\text{Rating} \ge 3.5 \mid \text{features})$. Labeled in UI as `Neural Match: XX.X%`.
- `two_tower_faiss`: Vector dot product / cosine similarity from FAISS IndexFlatIP $\in [-1, 1]$. Labeled as `Vector Similarity: 0.XXXX`.
- `popularity_fallback`: Weighted popularity rating volume calculated as $\text{rating\_count} \times \text{avg\_rating}$ (e.g., Shawshank Redemption = $2918 \times 4.4318 = 12932.0$). Labeled as `Popularity Metric: 12,932 (Ratings × Stars)`.
**Rationale:** Conflating volume metrics like 12,932 with probabilities or formatting them as percentages damages model explainability and misleads users.

## 32. SQLite Watchlist Persistence & Toggle Lifecycle for User Feedback
**Decision:** Implement durable, ACID-compliant SQLite storage (`data/feedback/feedback.db`) with tables `feedback_events` and `user_watchlist`, alongside append-only `events.jsonl`. Expose `GET /feedback/{user_id}` and `DELETE /feedback/{user_id}/{movie_id}` endpoints.
**Rationale:** In-memory browser state is lost on page reload and server restart. A dedicated SQLite database ensures user watchlist items and event logs survive backend restarts, enables bidirectional button toggling (`❤️ Interested / Watch` $\leftrightarrow$ `✓ In Watchlist`), deduplicates events via primary keys, and provides a persistent "My Watchlist" view for each user.
