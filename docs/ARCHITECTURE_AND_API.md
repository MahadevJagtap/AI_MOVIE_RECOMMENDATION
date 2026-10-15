# CineMatch — Architecture & API Documentation

**Project Name:** CineMatch — Large-Scale Personalized Recommendation Engine  
**GitHub Repository:** [https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION](https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION)  
**Live Demo:** Native Web Service on Render (`https://cinematch.onrender.com`)  
**Dataset:** MovieLens 25M (25,000,095 ratings across 62,423 movies)

---

## 1. System Overview

CineMatch is a production-grade, two-stage personalized movie recommendation engine built on industry standards (inspired by Netflix and YouTube recommender architectures). 

Recommending personalized items over a 62,000+ item catalog cannot be done via a single brute-force neural network due to strict latency budgets (<100 ms). CineMatch decomposes recommendation into a high-throughput **two-stage funnel**:
1. **Stage 1 — Candidate Retrieval:** Narrows down the catalog from 62,423 titles to top-K (~50-100) high-affinity candidates in under 5 milliseconds using a **TensorFlow Recommenders (TFRS) Two-Tower neural model** and a **FAISS CPU** inner-product vector index.
2. **Stage 2 — Pointwise Re-ranking:** Scores and re-orders the candidate pool using a **Wide & Deep neural network** that combines memorization (wide feature crosses) with generalization (deep continuous embeddings), outputting predicted engagement probabilities.
3. **Serving & Caching:** Exposed via a **FastAPI** REST backend with **Redis** caching, automated cold-start popularity fallback, and SQLite-backed user interaction and watchlist logging.
4. **Interactive UI:** A responsive **HTML5/CSS3/JavaScript** frontend with dark-mode glassmorphism, instant profile switching, and real-time CTR analytics.

---

## 2. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph DataPipeline [Offline ETL & Feature Store]
        ML[MovieLens 25M Dataset] --> Spark[PySpark Chronological Processing]
        Spark --> TrainData[Train Set: 893k Interactions]
        Spark --> MovieFeatures[Movie Features Parquet: 62k Items]
        Spark --> UserFeatures[User Activity Parquet: 6.1k Users]
        Spark --> PopularityTable[Top 500 Popularity Parquet]
    end

    subgraph ModelTraining [Model Training & Indexing]
        TrainData --> TwoTowerTrain[TFRS Two-Tower Model]
        TwoTowerTrain --> MovieEmbeddings[64-dim Catalog Embeddings]
        TwoTowerTrain --> UserTowerWeights[User Tower Saved Weights]
        MovieEmbeddings --> FAISSBuild[FAISS IndexFlatIP Index]
        FAISSBuild --> FAISSBin[faiss_index.bin]
        
        TrainData --> FeaturePipelines[Feature Pipeline & Crosses]
        FeaturePipelines --> WideDeepTrain[Wide & Deep Model Training]
        WideDeepTrain --> RankingWeights[Wide & Deep Weights & Config]
    end

    subgraph ServingLayer [FastAPI Production Serving]
        ClientReq[Incoming User Request: /recommend/{user_id}] --> CacheCheck{Redis Cache Hit?}
        CacheCheck -- Yes --> CacheReturn[Return Cached JSON < 5ms]
        CacheCheck -- No / Degraded --> ModelManager[ModelManager Orchestrator]

        ModelManager --> UserCheck{User in Train Set?}
        UserCheck -- Cold Start --> PopFallback[Popularity Fallback Recommender]
        UserCheck -- Known User --> Stage1[Stage 1: Two-Tower User Embedding]
        
        Stage1 --> FAISSSearch[FAISS Vector Search over 62k Items]
        FAISSSearch --> TopKCandidates[Top 100 Candidate IDs]
        TopKCandidates --> FilterHistory[Deduplicate Previously Watched Movies]
        
        FilterHistory --> Stage2[Stage 2: Wide & Deep Neural Re-ranking]
        Stage2 --> TopRanked[Top-N Ranked Recommendations]
        TopRanked --> WriteCache[Write to Redis with TTL]
        TopRanked --> JSONResp[Return HTTP 200 Response]
        PopFallback --> JSONResp
    end

    subgraph FrontendUI [Frontend & Interaction Loop]
        JSONResp --> WebUI[CineMatch Modern Web UI]
        WebUI --> UserClicks[User Watchlist / Impression Events]
        UserClicks --> FeedbackAPI[POST /feedback]
        FeedbackAPI --> FeedbackDB[(SQLite feedback.db)]
        FeedbackDB --> CTRMetrics[GET /metrics/ctr]
    end
```

---

## 3. Detailed Component Architecture

### 3.1 Stage 1: Candidate Retrieval (Two-Tower + FAISS)

- **Objective:** Rapidly retrieve candidate items relevant to user preferences from 62,423 catalog items in < 5ms.
- **Model Architecture:**
  - **User Tower:** Maps `user_id` string into a learned 64-dimensional dense representation via an embedding layer followed by dense projection.
  - **Item Tower:** Maps `movie_id` string into a 64-dimensional dense representation.
  - **Objective Function:** Sampled softmax cross-entropy loss with in-batch negatives.
- **Vector Search Index:**
  - Item embeddings for all 62,423 movies are precomputed, normalized to unit length, and indexed in a **FAISS `IndexFlatIP`** (Inner Product / Cosine Similarity).
  - During live serving, the User Tower generates a 64-dim query vector for the target user. FAISS performs an exhaustive inner product search returning the top 50–100 candidate IDs in **~1.7 milliseconds**.

### 3.2 Stage 2: Pointwise Re-ranking (Wide & Deep)

- **Objective:** Accurately score candidate items based on complex feature interactions to maximize engagement and NDCG@10.
- **Model Architecture:**
  - **Wide Component (Memorization):** Generalized linear model over cross-product feature transformations:
    - User ID $\times$ Movie ID
    - User Activity Level $\times$ Movie Release Year
    - High-order genre indicator interactions
  - **Deep Component (Generalization):** Feed-forward multi-layer perceptron (MLP) with dense layers (128 -> 64 -> 32) and ReLU activations:
    - Continuous features: User rating count, user mean rating, movie average rating, rating standard deviation, movie release year.
    - Low-dimensional entity embeddings for categorical features.
  - **Output Layer:** Single output neuron with sigmoid activation computing $P(\text{rating} \ge 3.5)$.
- **Feature Pipeline:**
  - Standardized continuous features using precomputed mean and variance statistics stored in `ranking_pipeline_config.json`.
  - Missing or unseen features default to population averages.

### 3.3 Serving & Resilience Architecture

- **FastAPI Core:** Asynchronous ASGI application with lifespan-managed model initialization (`ModelManager`).
- **Redis Caching:** Recommendation results are cached with a configurable TTL (default: 300s). Key schema: `rec:{user_id}:{k}:{variant}`.
- **Graceful Redis Degradation:** If Redis is down, unreachable, or absent (such as on standard Render Free tier), `src/serving/cache.py` catches `redis.exceptions.ConnectionError`, logs a warning, and executes direct model inference. No HTTP 500 errors are returned to users.
- **Cold-Start Fallback:** When an unobserved user requests recommendations, the system automatically falls back to `PopularityRecommender`, serving top-rated movies with at least 50 historical ratings.

---

## 4. Complete REST API Reference

The serving application exposes interactive Swagger UI docs at `GET /docs` and OpenAPI JSON at `GET /openapi.json`.

### 4.1 `GET /health`
Validates readiness of the application, model weights, catalog data, and vector index.

- **Request:** `GET /health`
- **Response (HTTP 200):**
```json
{
  "status": "healthy",
  "catalog_movies": 62423,
  "faiss_index": true,
  "two_tower": true,
  "ranking_wide_only": true,
  "ranking_wide_and_deep": true,
  "redis_cache": false,
  "triton_inference": false
}
```

---

### 4.2 `GET /recommend/{user_id}`
Returns top-$k$ personalized movie recommendations for a user.

- **Request:** `GET /recommend/198?k=5&variant=wide_and_deep`
- **Query Parameters:**
  - `k` *(int, default: 10, min: 1, max: 100)*: Number of recommendations to return.
  - `variant` *(str, default: "wide_and_deep")*: Ranking model variant (`wide_and_deep` or `wide_only`).
- **Response (HTTP 200):**
```json
{
  "user_id": "198",
  "recommendations": [
    {
      "movie_id": 2571,
      "title": "Matrix, The (1999)",
      "genres": "Action|Sci-Fi|Thriller",
      "score": 0.9412,
      "source": "wide_and_deep"
    },
    {
      "movie_id": 589,
      "title": "Terminator 2: Judgment Day (1991)",
      "genres": "Action|Sci-Fi",
      "score": 0.9285,
      "source": "wide_and_deep"
    }
  ],
  "latency_ms": 28.4,
  "cached": false,
  "model_variant": "wide_and_deep"
}
```

---

### 4.3 `GET /candidates/{user_id}`
Inspects Stage 1 Candidate Retrieval outputs prior to Stage 2 re-ranking.

- **Request:** `GET /candidates/198?k=10`
- **Response (HTTP 200):**
```json
{
  "user_id": "198",
  "candidates": [
    {
      "movie_id": 2571,
      "title": "Matrix, The (1999)",
      "genres": "Action|Sci-Fi|Thriller",
      "similarity_score": 0.8124
    }
  ],
  "total_candidates": 10,
  "retrieval_method": "two_tower_faiss"
}
```

---

### 4.4 `GET /popular`
Returns globally popular movies from the offline-calculated popularity table.

- **Request:** `GET /popular?limit=5`
- **Response (HTTP 200):**
```json
{
  "movies": [
    {
      "movie_id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "score": 81482.0
    }
  ],
  "count": 5
}
```

---

### 4.5 `POST /feedback`
Logs user interactions (impression, click, watchlist addition).

- **Request:** `POST /feedback`
- **Payload:**
```json
{
  "user_id": "198",
  "movie_id": 2571,
  "event_type": "click",
  "dwell_time_sec": 12.5,
  "rating": 5.0
}
```
- **Response (HTTP 200):**
```json
{
  "status": "success",
  "event_id": 1402,
  "recorded_at": "2026-10-15T21:20:00Z"
}
```

---

### 4.6 `GET /feedback/{user_id}` & `DELETE /feedback/{user_id}/{movie_id}`
Retrieves or removes saved watchlist movies for a user.

- **Retrieve Watchlist:** `GET /feedback/198`
- **Remove from Watchlist:** `DELETE /feedback/198/2571`
- **Response:**
```json
{
  "user_id": "198",
  "watchlist": [
    {
      "movie_id": 2571,
      "title": "Matrix, The (1999)",
      "saved_at": "2026-10-15T21:18:42"
    }
  ],
  "count": 1
}
```

---

### 4.7 `GET /metrics/ctr`
Computes live demo Click-Through Rate (CTR) and interaction analytics.

- **Request:** `GET /metrics/ctr`
- **Response (HTTP 200):**
```json
{
  "impressions": 482,
  "clicks": 94,
  "ctr": 0.1950,
  "unique_users": 18
}
```

---

### 4.8 `GET /metrics/offline`
Exposes precomputed offline model benchmark comparisons from `data/experiments/ranking_experiment.json`.

- **Request:** `GET /metrics/offline`
- **Response (HTTP 200):**
```json
{
  "models": {
    "wide_and_deep": {
      "ndcg_10": 0.835,
      "mrr": 0.435,
      "auc": 0.824,
      "test_loss": 0.509
    },
    "wide_only": {
      "ndcg_10": 0.792,
      "mrr": 0.388,
      "auc": 0.763,
      "test_loss": 0.540
    }
  },
  "dataset": "MovieLens 25M"
}
```

---

## 5. Data Schemas & Processing Details

| File Location | Format | Rows | Size | Description |
| :--- | :---: | :---: | :---: | :--- |
| `data/processed/movie_features/` | Snappy Parquet | 62,423 | 1.82 MB | Complete movie catalog with cleaned titles, release years, and multi-hot genre vectors. |
| `data/processed/user_features/` | Snappy Parquet | 6,141 | 0.21 MB | Precomputed user activity statistics (rating counts, mean ratings, variance). |
| `data/processed/popularity/` | Snappy Parquet | 500 | 0.01 MB | Top 500 highest-rated movies with $\ge 50$ ratings for cold-start fallbacks. |
| `data/processed/train/` | Snappy Parquet | 893,000 | 6.00 MB | Chronologically partitioned training interactions used for deduplicating watched titles. |
| `models/faiss/faiss_index.bin` | Binary | 62,423 | 15.99 MB | FAISS `IndexFlatIP` storing normalized 64-dim movie embeddings. |
| `models/retrieval/model.weights.h5` | HDF5 | - | 11.23 MB | TFRS Two-Tower learned weights. |
| `models/ranking/wide_and_deep/model.weights.h5` | HDF5 | - | 17.66 MB | Wide & Deep neural ranking model weights. |
