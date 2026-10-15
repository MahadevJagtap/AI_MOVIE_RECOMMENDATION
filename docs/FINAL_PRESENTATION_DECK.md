# CineMatch — 10-Minute Final Presentation Deck & Speaker Script

**Project Title:** CineMatch: Building a High-Throughput, Two-Stage Personalized Recommendation Engine  
**Presenter:** Mahadev Jagtap  
**Target Duration:** Exactly 10 Minutes (600 seconds)  
**Slides:** 10 Slides (~1 Minute per Slide)  
**GitHub Repository:** [https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION](https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION)

---

## Slide 1: Title & The Problem at Scale (0:00 – 1:00)

### Slide Visual:
- **Title:** CineMatch: Large-Scale Personalized Recommendation Engine
- **Subtitle:** Solving the Latency vs. Precision Tradeoff in Movie Discovery
- **Key Stats:**
  - 25 Million Ratings | 62,423 Movies | 162,000 Users (MovieLens 25M)
  - Strict Latency Budget: < 50 ms interactive response time
- **The Core Dilemma:**
  - A brute-force neural network evaluating 62,423 movies per user takes **> 1,500 ms** (Unusable in production).
  - Simple matrix factorization or collaborative filtering is fast (< 5 ms) but lacks semantic context, genre awareness, and non-linear feature interaction.

### Speaker Script:
> *"Hello everyone. Today I am presenting **CineMatch**, an end-to-end, two-stage personalized movie recommendation engine built on the MovieLens 25M dataset.
> 
> In modern streaming platforms like Netflix or YouTube, recommendation systems face a fundamental engineering tradeoff: **Scale versus Precision**. If we score an entire catalog of over 62,000 movies using a deep neural network on every request, response times exceed 1.5 seconds, which destroys user retention. On the other hand, simple heuristic filtering is fast but produces generic, unpersonalized recommendations.
> 
> CineMatch solves this by implementing an industrial **Two-Stage Funnel Architecture**: rapid candidate retrieval with vector search followed by neural re-ranking, delivering deeply personalized recommendations in under 35 milliseconds."*

---

## Slide 2: High-Level Two-Stage Funnel Architecture (1:00 – 2:00)

### Slide Visual:
```
           [ Full Catalog: 62,423 Titles ]
                         │
                         ▼
           ┌─────────────────────────────┐
           │ Stage 1: Candidate Retrieval │ < 5 ms
           │  • TFRS Two-Tower Model     │
           │  • 64-dim Dense Embeddings  │
           │  • FAISS CPU Vector Search  │
           └─────────────────────────────┘
                         │
                         ▼
               [ Top 50–100 Candidates ]
                         │
                         ▼
           ┌─────────────────────────────┐
           │  Stage 2: Neural Re-ranking │ < 25 ms
           │  • Wide & Deep Architecture │
           │  • User Activity Crosses    │
           │  • P(rating >= 3.5) Sigmoid │
           └─────────────────────────────┘
                         │
                         ▼
           [ Top 10 Personalized Recommendations ]
```

### Speaker Script:
> *"To achieve both sub-50ms latency and high accuracy, CineMatch operates as a two-stage funnel.
> 
> In **Stage 1 — Candidate Retrieval**, we rapidly narrow down the 62,423-movie universe to the top 100 high-affinity candidates. We use a **TensorFlow Recommenders Two-Tower neural network** mapped into a **FAISS vector index**, retrieving candidates in under 5 milliseconds.
> 
> Once we have those 100 candidates, we pass them to **Stage 2 — Pointwise Re-ranking**. Here, we run a heavier **Wide & Deep neural network** that evaluates fine-grained feature crosses, user activity distributions, and genre combinations to compute the final engagement probability.
> 
> The result is that our heavy neural model only needs to score 100 items instead of 62,000, reducing computation by 99.8% while preserving ranking precision."*

---

## Slide 3: Stage 1 Deep Dive — Two-Tower Retrieval & FAISS (2:00 – 3:00)

### Slide Visual:
- **Two-Tower Neural Network:**
  - **User Tower:** Maps `user_id` $\to$ Embedding Layer $\to$ Dense Layer (64-dim unit vector)
  - **Item Tower:** Maps `movie_id` $\to$ Embedding Layer $\to$ Dense Layer (64-dim unit vector)
  - **Loss:** Sampled Softmax with in-batch negatives
- **Vector Search Engine:**
  - **FAISS CPU (`IndexFlatIP`):** 62,423 precomputed item vectors loaded into RAM (15.99 MB index)
  - **Search Latency:** **1.71 ms** (p95)
  - **Recall@50:** **0.583** | **Recall@100:** **0.697**

### Speaker Script:
> *"Let's look under the hood of Stage 1. 
> 
> The Two-Tower architecture separates the user context from the item catalog. The Item Tower embeds all 62,423 movies into a 64-dimensional latent semantic space. Because movies change slowly, these embeddings are precomputed offline and indexed in an in-memory **FAISS** index using Inner Product search.
> 
> At query time, we pass the user ID through the lightweight User Tower to produce a 64-dimensional query vector. FAISS scans the entire 62,000 item space using SIMD instructions in just **1.7 milliseconds**, returning the 100 most similar movies. 
> 
> Our offline evaluation shows this stage achieves a **Recall@100 of 69.7%**, ensuring that almost all movies the user would enjoy are captured in this candidate set."*

---

## Slide 4: Stage 2 Deep Dive — Wide & Deep Re-ranking (3:00 – 4:00)

### Slide Visual:
- **Wide Component (Memorization):**
  - Cross-product transformations: $\phi_k(x) = \prod_{j=1}^d x_j^{c_{kj}}$
  - Captures specific user-item co-occurrence and historical preferences
- **Deep Component (Generalization):**
  - 3-Layer MLP (128 $\to$ 64 $\to$ 32 neurons, ReLU activation, Dropout 0.2)
  - Inputs: Continuous rating stats (user mean, user count, movie std dev, release year)
- **Objective:** Predict $P(\text{rating} \ge 3.5)$ via Sigmoid Output
- **Model Comparison:**
  - **Wide & Deep NDCG@10:** **0.835** vs. Wide-only **0.792** (+5.4%)
  - **Wide & Deep MRR:** **0.435** vs. Wide-only **0.388** (+12.1%)

### Speaker Script:
> *"Once candidates are retrieved, Stage 2 scores them using Google's **Wide & Deep** architecture.
> 
> Recommendation systems need both **memorization** and **generalization**.
> - The **Wide component** is a generalized linear model that memorizes historical rules—for example, a user who repeatedly watches 90s sci-fi movies will have strong weights on that specific intersection.
> - The **Deep component** is a multi-layer perceptron that generalizes to unseen items. It takes continuous signals—such as how active the user is, the movie's historical rating variance, and release decade—to infer preference even for less familiar titles.
> 
> On our chronological test set, Wide & Deep improves **NDCG@10 from 0.792 to 0.835** and improves **Mean Reciprocal Rank by 12.1%**, proving that deep non-linear feature interactions significantly enhance ranking order."*

---

## Slide 5: Production Serving Architecture & Caching (4:00 – 5:00)

### Slide Visual:
- **FastAPI Core:** Async ASGI framework with typed Pydantic v2 schemas
- **Redis Cache Layer:**
  - Structured key schema: `rec:{user_id}:{k}:{variant}`
  - 300s TTL | Cache hit latency: **< 2 ms**
- **Graceful Degradation Protocol:**
  - Auto-catches Redis connection dropouts
  - Seamlessly falls back to direct model computation without 500 errors
- **Feedback & Watchlist Store:**
  - SQLite database (`feedback.db`) tracking impressions, clicks, dwell time, and watchlists
  - Dynamic Click-Through Rate (CTR) computation via `/metrics/ctr`

### Speaker Script:
> *"A great machine learning model is useless if the serving infrastructure is fragile.
> 
> We wrapped the pipeline in a high-performance **FastAPI** backend. We implemented a **Redis caching layer** that caches user recommendations for 5 minutes, delivering repeat requests in under 2 milliseconds.
> 
> More importantly, we engineered **Graceful Degradation**. If Redis experiences a network timeout or is absent, our caching module automatically catches the error, logs a warning, and falls back to live inference without dropping the user's request.
> 
> In addition, we built a persistent feedback engine in SQLite that logs real-time user impressions, clicks, dwell times, and watchlist additions, feeding our real-time CTR analytics endpoint."*

---

## Slide 6: CineMatch Web UI — Discovering Movies Interactively (5:00 – 6:00)

### Slide Visual:
- **Modern Responsive Web Application:**
  - Plain HTML5, Vanilla CSS3 (modern glassmorphism, dark palette), Vanilla JS
  - Zero heavy frontend framework overhead (instant 0.07 MB bundle load)
- **Key Features Demonstrated:**
  - **Viewer Persona Switcher:** User 198 (Sci-Fi), User 12 (Drama/Classics), User 1 (Cold Start)
  - **Model Variant Toggle:** Real-time A/B comparison between *Wide & Deep* and *Wide Baseline*
  - **Stage 1 Candidate Drawer:** Inspect retrieval similarity scores before neural re-ranking
  - **Interactive Watchlist:** Instant bookmark toggle and persistent watchlist view
  - **Live Metrics Dashboard:** Real-time Click-Through Rate (CTR) and offline NDCG/MRR curves

### Speaker Script:
> *"To make this system easy to evaluate for college reviewers, interviewers, and casual users, we built a custom responsive web discovery interface called **CineMatch**.
> 
> Instead of requiring users to craft raw JSON API requests, the UI provides:
> - **Pre-configured Viewer Personas**: letting you instantly test how recommendations change between a Sci-Fi fanatic, a classic Drama lover, and a brand-new user.
> - **A Model Comparator Toggle**: allowing live switching between our primary Wide & Deep neural network and the Wide-only baseline to see real-time score and ranking shifts.
> - **An Interactive Watchlist and Metrics Drawer**: showing how user bookmarking dynamically updates the live Click-Through Rate and system metrics."*

---

## Slide 7: Offline Evaluation & Empirical Test Verification (6:00 – 7:00)

### Slide Visual:
- **Temporal Split Integrity:** Chronological 80/10/10 split (zero future-data leakage)
- **Benchmark Summary Table:**
  | Metric | Stage 1 (FAISS) | Stage 2 (Wide-Only) | Stage 2 (Wide & Deep) | Target / SLA |
  | :--- | :---: | :---: | :---: | :---: |
  | **Recall@100** | **69.7%** | — | — | > 65% |
  | **NDCG@10** | — | **0.792** | **0.835** | > 0.800 |
  | **MRR** | — | **0.388** | **0.435** | > 0.400 |
  | **AUC** | — | **0.763** | **0.824** | > 0.800 |
  | **p95 Latency** | **1.7 ms** | **4.2 ms** | **18.6 ms** | < 50 ms |
- **Testing Rigor:**
  - **Pytest Suite:** 32 / 32 Passed (100%)
  - **Live Deployment Smoke Suite:** 12 / 12 Passed (100%)

### Speaker Script:
> *"Evaluation rigor was our highest priority. We avoided the common pitfall of random train/test splitting, which causes temporal data leakage. We enforced chronological splitting in PySpark, strictly using past ratings to predict future ones.
> 
> As you can see on the matrix:
> - Stage 1 achieves a **Recall@100 of 69.7%** in just **1.7 milliseconds**.
> - Stage 2 achieves an **NDCG@10 of 0.835** and an **AUC of 0.824**, comfortably exceeding industry targets.
> - End-to-end latency remains **under 36.4 ms at the 95th percentile**.
> 
> Furthermore, we validated the entire serving application with a **32-test Pytest suite** and a **12-point live HTTP smoke test**, verifying everything from watchlist deduplication to cold-start popularity fallbacks."*

---

## Slide 8: Key Engineering Challenges & Solutions (7:00 – 8:00)

### Slide Visual:
1. **Challenge: Cold-Start Problem (New / Unobserved Users)**
   - *Solution:* Built a precomputed Top-500 popularity table with $\ge 50$ ratings. Unseen users automatically route to popularity fallback with zero errors.
2. **Challenge: Memory Constraints on Cloud Free Tiers (512 MB ceiling)**
   - *Solution:* Split training dependencies from serving (`requirements-serve.txt`), removed PySpark (~310 MB wheel + JVM), added `MALLOC_ARENA_MAX=2` to limit Linux glibc memory fragmentation.
3. **Challenge: Cloud Storage Ephemeral Behavior**
   - *Solution:* Identified and documented Render Free tier disk wipes on spin-down; partitioned git-tracked serving models (~70 MB total) from raw datasets (678 MB `ratings.csv`).

### Speaker Script:
> *"During this project, we solved several critical engineering challenges:
> 
> First, the **Cold-Start Problem**. In a production system, when a user who was not in the training set logs in, a pure neural model fails. We built an automated popularity fallback that serves top-rated movies across the entire catalog with zero latency degradation.
> 
> Second, **Cloud Resource Constraints**. Deploying machine learning models to cloud containers like Render Free tier imposes strict 512 MB RAM limits. We created a lightweight `requirements-serve.txt`, completely removing PySpark and heavy plotting libraries, and configured `MALLOC_ARENA_MAX=2` to manage glibc memory allocation.
> 
> Third, **Asset Partitioning**. We ensured that only the 70.62 MB of required model weights, FAISS indexes, and catalog parquet files are tracked in Git, keeping the repo lightweight and completely avoiding Git LFS issues."*

---

## Slide 9: Live Demo & Walkthrough (8:00 – 9:00)

### Slide Visual:
*(Screenshots or Live Screen Share of CineMatch running at `https://cinematch.onrender.com`)*
- Step 1: Loading **User 198 (Sci-Fi Fan)** $\to$ Shows *The Matrix*, *Terminator 2*, *Twelve Monkeys*
- Step 2: Toggle to **User 12 (Classic Drama)** $\to$ Instant shift to *Shawshank Redemption*, *Schindler's List*, *Casablanca*
- Step 3: Toggle ranking variant to **Wide Baseline** $\to$ Scores shift from non-linear probabilities to linear scores
- Step 4: Click bookmark icon $\to$ Movie saved to Watchlist; Live CTR counter updates in real time

### Speaker Script:
> *"Let's see CineMatch in action. 
> 
> [DEMO DEMONSTRATION]:
> 1. When we select **User 198**, our system detects their preference for sci-fi and action. In under 30 milliseconds, the two-stage pipeline returns *The Matrix*, *Terminator 2*, and *Twelve Monkeys*, with match confidence scores above 92%.
> 2. If we switch to **User 12**, the recommendations dynamically pivot to classic cinema: *The Shawshank Redemption*, *Schindler's List*, and *Casablanca*.
> 3. If we enter **User 1**, who is not in the training set, the system seamlessly applies our cold-start popularity fallback.
> 4. We can also click the bookmark button on any card to save it to our personal watchlist. This immediately logs an interaction to SQLite and updates our Click-Through Rate analytics."*

---

## Slide 10: Conclusion, Business Impact & Next Steps (9:00 – 10:00)

### Slide Visual:
- **Summary of Achievements:**
  - Built & deployed an industrial Two-Stage Recommender on MovieLens 25M
  - 62k catalog search in 1.7 ms via FAISS | Wide & Deep NDCG@10: 0.835
  - Production-ready FastAPI backend + Modern responsive web UI
  - 100% test pass rate across 32 unit/integration and 12 live smoke tests
- **Future Roadmap:**
  - **Real-Time Streaming Features:** Incorporate Apache Kafka for session-based instant re-ranking
  - **Multi-Task Learning:** Jointly predict rating, completion rate, and watchlist addition using MMoE (Multi-gate Mixture-of-Experts)
  - **Persistent Cloud Database:** Connect feedback logging to external PostgreSQL (e.g. Supabase)
- **Repository & Live Links:**
  - GitHub: [https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION](https://github.com/MahadevJagtap/AI_MOVIE_RECOMMENDATION)
  - Live Demo: `https://cinematch.onrender.com`

### Speaker Script:
> *"In conclusion, CineMatch demonstrates how modern recommendation architectures operate at scale. By combining Two-Tower vector retrieval with Wide & Deep re-ranking, we achieve both sub-35ms response times and strong personalization accuracy on 25 million ratings.
> 
> Looking ahead, our next steps include integrating Kafka for real-time in-session feature updates, and upgrading our ranking model to a Multi-gate Mixture-of-Experts (MMoE) to simultaneously predict clicks, ratings, and completion times.
> 
> The entire project, including source code, trained model weights, tests, and documentation, is available on GitHub at `MahadevJagtap/AI_MOVIE_RECOMMENDATION`.
> 
> Thank you, and I look forward to taking your questions."*

---

## 11. Anticipated Q&A Defense Guide

| Anticipated Question | Recommended 30-Second Answer |
| :--- | :--- |
| **Why not use Matrix Factorization or collaborative filtering?** | *"Matrix Factorization cannot easily incorporate continuous user statistics, item release years, or cross-features, and cannot generalize to unseen feature interactions like Wide & Deep does."* |
| **Why FAISS CPU instead of GPU?** | *"Our catalog is 62,423 items. With 64-dimensional vectors, the entire index is only 16 MB. FAISS CPU searches this in 1.7 ms, so GPU acceleration would introduce PCI bus transfer overhead without meaningful latency gains."* |
| **How did you prevent data leakage during model training?** | *"We enforced chronological splitting in PySpark: the first 80% of each user's timeline was used for training, the next 10% for validation, and the final 10% for testing. Random splits would use future ratings to predict past behavior."* |
| **What happens if Render free tier runs out of memory?** | *"We created a dedicated `requirements-serve.txt` that removes PySpark and training libraries, reducing image overhead. We also set `MALLOC_ARENA_MAX=2`. If memory ever spikes during cold start, upgrading to Render Starter ($7/mo) provides guaranteed headroom."* |
