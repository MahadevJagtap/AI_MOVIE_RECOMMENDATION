# CineMatch: Prioritized Next Steps & Operational Checklist

**Project Root:** `D:\setup\recommendation-engine`  
**Current System Status:** **100/100 Tests Passing** &bull; **32/32 Targeted Regression Tests Passing** &bull; Submission Ready  
**Generated Date:** October 15, 2026  

---

## 🟢 Immediate / Submission-Ready Tasks (Completed & Verified ✅)

- [x] **Verify 100/100 Full Regression Suite:** All 8 test modules pass via `$env:TF_USE_LEGACY_KERAS="1"; pytest tests/ -v`.
- [x] **Verify 32/32 Targeted Regression Suite:** Frontend, feedback persistence, and serving suites pass with zero errors (`tests/test_frontend.py`, `tests/test_feedback_persistence.py`, `tests/test_serving.py`).
- [x] **Empirically Reconcile Viewer Profiles:** Correct profile labels in `frontend/index.html` to reflect actual MovieLens training histories without fabricating preferences:
  - Alex (198): Comedy & Action (mean rating 4.29; 21 Comedy, 13 Action).
  - Jordan (662): Drama & Romance (mean rating 4.31; 12 Drama, 12 Romance).
  - Taylor (805): Action & Sci-Fi (mean rating 4.29; 174 Comedy, 166 Action, 112 Sci-Fi; corrected from Thriller & Crime).
  - Morgan (924): Drama & Action (mean rating 3.20; 98 Drama, 83 Action, 76 Romance; corrected from Adventure & Fantasy).
- [x] **Delineate Official NVIDIA Triton (`NOT VERIFIED`):** Formally distinguish SavedModel exports and client unit tests (`tests/test_triton_client.py`) from the official server container. Because the official server was not launched with real inference requests on Windows, it is explicitly documented as **`NOT VERIFIED`**.
- [x] **Document Native Airflow Limitations (`BLOCKED ON WINDOWS`):** Distinguish DAG validation (25 tests passing) and controlled sequential execution (10/10 tasks passing) from native Airflow supervisor execution, which fails on Windows due to POSIX `SIGALRM` (`ValueError`) and child socket inheritance (`os.set_inheritable`).
- [x] **Clarify Score Semantics & Uncalibrated Likelihood:** Label predicted neural scores as `🎯 Match Score: XX.X% (Likelihood Rating >= 3.5)` and popularity products as `🌟 Popularity Score: 12,932 (Ratings × Stars)`. Explicitly note that the neural output is an uncalibrated point estimate of $P(\text{rating} \ge 3.5)$, not a frequentist calibrated probability.
- [x] **Insulate Live CTR & Watchlist from Test Pollution:** Prevent test users (`test_*`, `audit_*`, `verify_*`) from polluting user-facing CTR or the `events.jsonl` log. Purge synthetic records while preserving 100% of legitimate user watchlists (Users 1, 8, 12, 89, 123, 198, 662, 924, 5656).
- [x] **Verify Full Restart Persistence Lifecycle:** Verify end-to-end save, retrieve, server termination, restart, reload, and delete using dedicated test user `test_restart_verifier` and test movie `318`.

---

## 🟡 Optional Enhancements (Post-Submission / Production Scale)

### 1. Model Calibration Evaluation (Production Monetization)
- **Objective:** Evaluate Expected Calibration Error (ECE) and produce reliability diagrams.
- **Current State:** Model outputs raw sigmoid logits trained with binary cross-entropy.
- **Action:** If calibrated probabilities are required for downstream business bids or monetization, fit Platt scaling (logistic regression on logits) or Isotonic Regression on the validation set.

### 2. Containerized NVIDIA Triton Serving (Linux / Cloud)
- **Objective:** Serve inference requests via the official NVIDIA Triton container.
- **Current State:** Model repository (`models/triton_model_repository/`), SavedModels, `config.pbtxt`, and client protocol verification are complete (`0.00e+00` numerical difference). Official container is marked `NOT VERIFIED` on Windows.
- **Action:** When deploying to Linux/WSL2/Cloud, mount `models/triton_model_repository/` into `nvcr.io/nvidia/tritonserver:24.08-py3` and set `TRITON_ENABLED=True` in `.env`.

### 3. Production Airflow Orchestration (Linux / Cloud)
- **Objective:** Run native Airflow scheduler and task supervisor.
- **Current State:** 10-task DAG is verified and all tasks run via `scripts/run_airflow_dag_controlled.py`. Native execution is blocked on Windows by POSIX signal and socket requirements.
- **Action:** Run Airflow using the official `apache/airflow:2.10.2` Docker image or a managed cloud service (AWS MWAA, Google Cloud Composer).

### 4. Redis Cache Service
- **Objective:** Enable sub-5ms repeat request caching.
- **Current State:** Handled gracefully via a 10-second cooldown backoff with zero latency penalty.
- **Action:** If Redis is desired, launch `redis-server.exe` on port 6379 or deploy a Redis container.

---

## 🚀 Daily Operational Commands

### Start CineMatch Application
```powershell
Set-Location -Path "D:\setup\recommendation-engine"
$env:TF_USE_LEGACY_KERAS="1"
.\venv\Scripts\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```
Open in browser: `http://127.0.0.1:8000`

### Run Targeted Regression Tests (32 Tests)
```powershell
Set-Location -Path "D:\setup\recommendation-engine"
$env:TF_USE_LEGACY_KERAS="1"
.\venv\Scripts\python.exe -m pytest tests/test_frontend.py tests/test_feedback_persistence.py tests/test_serving.py -v
```

### Run Full Regression Suite (100 Tests)
```powershell
Set-Location -Path "D:\setup\recommendation-engine"
$env:TF_USE_LEGACY_KERAS="1"
.\venv\Scripts\python.exe -m pytest tests/ -v
```

### Verify Persistence Lifecycle
```powershell
Set-Location -Path "D:\setup\recommendation-engine"
.\venv\Scripts\python.exe scripts/verify_feedback_persistence_live.py phase1
# Terminate & restart FastAPI backend
.\venv\Scripts\python.exe scripts/verify_feedback_persistence_live.py phase2
```
