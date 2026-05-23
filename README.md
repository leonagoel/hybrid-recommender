```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    H Y B R I D R E C                                             ║
║    ─────────────────────────────────────────────────────────     ║
║    Hybrid Recommender System · Leona Goel      
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-VADER_NLP-154f3c?style=flat-square)](https://nltk.org)
[![MIT License](https://img.shields.io/badge/License-MIT-ff6b35?style=flat-square)](LICENSE)

</div>

---

> [!IMPORTANT]
> **🟢 This is the active GSSoC project repo — open all issues and PRs here only.**

---

> A production-ready recommender fusing **Content-Based Filtering (TF-IDF)**, **Collaborative Filtering (SVD)**, and **NLP Sentiment Analysis (VADER)** with a tunable weighted scoring engine — backed by Supabase PostgreSQL, served via FastAPI, and built to be **dataset-agnostic by design**.

```
25,000+ products  ·  Sub-50ms search  ·  3 ML models fused  ·  ~60% faster integration
```

---

## 01 — Architecture

The core insight: blend three independent signals, each capturing something the others miss.

```
User Reviews (text)           ──→  NLP Engine (VADER Sentiment)    ──┐
Item Metadata (title/desc)    ──→  Content Vectorization (TF-IDF)  ──┼──→  Weighted Hybrid  ──→  Ranked Results
User Purchases (clicks/buys)  ──→  Matrix Factorization (SVD)      ──┘         Engine

     Hybrid Score  =  α · content_score        [TF-IDF cosine similarity]
                    + β · collab_score          [Truncated SVD latent space]
                    + γ · sentiment_score       [VADER compound polarity]

     // α, β, γ are live-tunable via API or UI sliders
```

<details>
<summary><b>α — Content Model &nbsp;·&nbsp; TF-IDF + Cosine Similarity</b></summary>
<br/>

Item metadata (`title` + `description` + `category`) vectorized with TF-IDF (unigrams + bigrams, max 5,000 features). On-the-fly cosine similarity yields `content_score ∈ [0, 1]`. Fast, interpretable, and requires **zero user history** — ideal for cold-start.

</details>

<details>
<summary><b>β — Collaborative Model &nbsp;·&nbsp; Truncated SVD</b></summary>
<br/>

User-item interaction matrix built from purchases + implicit feedback (views, clicks). SVD reduces to 50 latent factors; cosine similarity in latent space yields `collab_score`. **Adaptive rank** automatically reduces SVD components for sparse matrices.

</details>

<details>
<summary><b>γ — Sentiment Model &nbsp;·&nbsp; NLTK VADER</b></summary>
<br/>

Review text analyzed for compound polarity ∈ [-1, 1]. Per-item aggregation → Min-Max normalization → `sentiment_score ∈ [0, 1]`. Surfaces genuinely loved products, not just popular ones.

</details>

<details>
<summary><b>❄ Cold-Start Handling</b></summary>
<br/>

- **Bayesian average rating** — prevents 1-review, 5-star bias
- **Popularity-based fallback** — ranks new items by review count and category similarity
- **Mock user seeding** — synthetic purchase history to bootstrap collaborative filtering

</details>

---

## 02 — Features

| Feature | Detail |
|---|---|
| `PostgreSQL FTS` | GIN-indexed full-text search — sub-50ms on 250k+ rows |
| `Supabase Auth` | Guest (anonymous) and email/password, Row-Level Security on all tables |
| `Tunable Weights` | Live α/β/γ sliders to adjust recommendation blend in real time |
| `Dataset-Agnostic` | Fuzzy column detection (`product_name` → `title`) cuts integration time by ~60% |
| `Cold-Start Resilient` | Bayesian avg rating + popularity fallback for new users and items |
| `Type-to-Search` | Global keyboard capture — start typing anywhere to search instantly |
| `Responsive UI` | Amazon-inspired dark header, 4→3→2→1 column card grid across breakpoints |
| `Secure by Default` | Pydantic validation, parameterized queries, CORS-restricted, no stack-trace leakage |

---

## 03 — Tech Stack

```
┌─────────────────┬────────────────────────────────────────────────┐
│ Layer           │ Technology                                      │
├─────────────────┼────────────────────────────────────────────────┤
│ Backend         │ Python 3.10+, FastAPI, Uvicorn                 │
│ Database        │ Supabase (PostgreSQL), Row-Level Security       │
│ Search          │ PostgreSQL FTS (GIN indexes, ts_rank)          │
│ Auth            │ Supabase Auth (anonymous + email/password)      │
│ ML — Content    │ scikit-learn: TF-IDF Vectorizer, Cosine Sim    │
│ ML — Collab     │ scikit-learn: TruncatedSVD, SciPy sparse       │
│ NLP             │ NLTK VADER SentimentIntensityAnalyzer           │
│ Data            │ Pandas, NumPy                                   │
│ Frontend        │ HTML5, CSS3, Vanilla JS, Supabase JS v2        │
└─────────────────┴────────────────────────────────────────────────┘
```

---

## 04 — Project Structure

```
hybrid-recommender/
│
├── backend/
│   └── main.py                  # FastAPI server — search, upload, build, recommend
│
├── frontend/
│   ├── index.html               # Single-page UI (Amazon-like layout)
│   ├── styles.css               # Design system (dark header, cards, animations)
│   └── app.js                   # Frontend logic (auth, search, rendering)
│
├── scripts/
│   ├── generate_sample_data.py  # Synthetic test dataset generator
│   ├── import_to_supabase.py    # Batch import CSV/JSON → PostgreSQL
│   └── seed_mock_data.py        # Mock users + purchases for cold-start bootstrap
│
├── data_adapter.py              # ⭐ Auto column detection + schema normalization
├── content_model.py             # TF-IDF content-based recommender
├── collaborative_model.py       # SVD collaborative recommender + implicit feedback
├── hybrid_model.py              # Weighted hybrid engine (Bayesian avg, popularity)
├── nlp_engine.py                # VADER sentiment analysis pipeline
├── evaluation.py                # Precision@K, Recall@K, NDCG@K benchmarks
├── db.py                        # Supabase client singleton (anon + admin)
├── requirements.txt
├── .env.example
└── SETUP.md
```

---

## 05 — Quick Start

**Prerequisites:** Python 3.10+ · Supabase account *(free tier works)*

```bash
# 1 — Clone & install
git clone https://github.com/Bhumi1701/hybrid-recommender.git
cd hybrid-recommender
pip install -r requirements.txt
```

```bash
# 2 — Configure Supabase
cp .env.example .env
# Fill in from: Supabase Dashboard → Settings → API
```

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key   # Required for bulk import
```

```bash
# 3 — Run SQL migrations
# See SETUP.md for full schema → paste into Supabase SQL Editor

# 4 — Start the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, upload any CSV/JSON from `datasets/`, click **Build Models**, then start typing to search.

## 06 — API Reference

```
GET    /api/config                   →  Supabase public config
GET    /api/status                   →  System status + product count
GET    /api/search?q=...&limit=20    →  Full-text search (PostgreSQL FTS)
POST   /api/upload                   →  Upload CSV/JSON dataset
POST   /api/build                    →  Train TF-IDF, SVD, VADER models
GET    /api/recommend/{title}        →  Hybrid recommendations for an item
GET    /api/items?page=1&per_page=50 →  Paginated product listing
GET    /api/categories               →  All available categories
GET    /api/weights                  →  Current α, β, γ blend weights
PUT    /api/weights                  →  Update blend weights live
GET    /api/purchases/{user_id}      →  User purchase history
POST   /api/purchases                →  Record a purchase event
```

---

## 07 — Evaluation

```bash
python evaluation.py
```

Benchmarks **Content-Only**, **Collab-Only**, **Sentiment-Only**, and **Hybrid** across:

```
Precision@K  —  fraction of relevant items in top-K
Recall@K     —  fraction of all relevant items retrieved
NDCG@K       —  ranking quality (discounted cumulative gain)
```

---

## 08 — Security

```
✓  No hardcoded credentials — config served via /api/config
✓  .env excluded from git via .gitignore
✓  CORS restricted to configured origins
✓  Row-Level Security (RLS) on all Supabase tables
✓  Input validation via Pydantic models
✓  Generic error messages — no stack trace leakage
✓  SQL injection safe (Supabase SDK parameterized queries)
```

---
## 09 — FAQ

<details>
<summary><strong>How do I set up the project locally?</strong></summary>

Clone the repository and install the required dependencies using the package manager mentioned in the project documentation. After that, configure the environment variables if needed and start both the frontend and backend servers. Make sure your database or dataset files are also available before running the app.

</details>

<details>
<summary><strong>What datasets does this project use?</strong></summary>

This project uses datasets related to user interactions, ratings, and item metadata to generate recommendations. The exact dataset files are usually stored inside the data or datasets directory. You can check the project documentation for download links and formatting details.

</details>

<details>
<summary><strong>How do the alpha/beta/gamma weights affect recommendations?</strong></summary>

The alpha, beta, and gamma weights control how much influence different recommendation factors have in the final score. Changing these values can prioritize popularity, similarity, or personalized behavior differently. Experimenting with the weights helps fine-tune recommendation quality for your use case.

</details>

<details>
<summary><strong>What is Bayesian rating and why is it used?</strong></summary>

Bayesian rating is a method used to balance average ratings with the number of votes an item has received. It prevents items with very few ratings from unfairly appearing at the top of recommendations. This makes the ranking system more stable and reliable.

</details>

<details>
<summary><strong>How do I run the tests?</strong></summary>

Run the test command provided in the project, usually through a package manager like npm or a testing framework command. Make sure all dependencies are installed before running tests. The test results will help verify that the application works correctly after changes.

</details>

<details>
<summary><strong>The backend shows "Backend offline" — what do I do?</strong></summary>

First, check whether the backend server is running on the correct port. Verify that your environment variables and database connections are configured properly. If the issue continues, restart the backend server and review the console logs for errors.

</details>

<details>
<summary><strong>Can I use my own dataset with this project?</strong></summary>

Yes, you can use your own dataset as long as it follows the expected format used by the project. You may need to update file paths or preprocessing steps depending on your data structure. Testing with smaller datasets first is recommended to ensure compatibility.

</details>
---

## License

MIT — see [`LICENSE`](LICENSE)

---

<div align="center">

```
Built by Leona Goel
B.Tech CSE · Vellore Institute of Technology
National Finalist · Smart India Hackathon 2025 · Top 8% of 950+ Teams
```

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leona-goel)
[![GitHub](https://img.shields.io/badge/Follow-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/leonagoel)
[![Email](https://img.shields.io/badge/Email-leona.goel23%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:leona.goel23@gmail.com)

</div>
