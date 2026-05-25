```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    H Y B R I D R E C                                             ║
║    ─────────────────────────────────────────────────────────     ║
║    Hybrid Recommender System · Leona Goel                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

<div align="center">

[![CI](https://github.com/leonagoel/hybrid-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/leonagoel/hybrid-recommender/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/leonagoel/hybrid-recommender)](https://github.com/leonagoel/hybrid-recommender/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Contributors](https://img.shields.io/github/contributors/leonagoel/hybrid-recommender.svg?style=flat-square)](https://github.com/leonagoel/hybrid-recommender/graphs/contributors)
[![PRs Welcome](https://img.shields.io/badge/PRs_welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![GSSoC 2026](https://img.shields.io/badge/GSSoC_2026-orange.svg?style=flat-square)](https://gssoc.girlscript.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-VADER_NLP-154f3c?style=flat-square)](https://nltk.org)

</div>

---

> [!IMPORTANT]
> **🟢 This is the active GSSoC project repo — open all issues and PRs here only.**

---

> A production-ready recommender fusing **Content-Based Filtering (TF-IDF)**, **Collaborative Filtering (SVD)**, and **NLP Sentiment Analysis (VADER)** with a tunable weighted scoring engine — backed by Supabase PostgreSQL, served via FastAPI, and built to be **dataset-agnostic by design**.

```text
25,000+ products  ·  Sub-50ms search  ·  3 ML models fused  ·  ~60% faster integration
```

---

## 01 — Architecture

The core insight: blend three independent signals, each capturing something the others miss.

```text
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
| `Streamlit UI` | Local CSV upload → build models → recommendations, no Supabase or server required |

---

## 03 — Tech Stack

```text
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

```text
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
├── app.py                       # Streamlit UI — upload CSV, build models, get recommendations
├── requirements.txt
├── .env.example
└── SETUP.md
```

---

## 05 — Quick Start

**Prerequisites:** Python 3.10+ · Supabase account *(free tier works)*

```bash
# 1 — Clone & install
git clone https://github.com/leonagoel/hybrid-recommender.git
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
SUPABASE_SERVICE_KEY=your-service-role-key
```

```bash
# 3 — Run SQL migrations
# See SETUP.md for full schema → paste into Supabase SQL Editor
```

```bash
# 4 — Start the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, upload any CSV/JSON from `datasets/`, click **Build Models**, then start typing to search.

### Async Recommendations — Celery Worker Setup

Async recommendation tasks require Redis and a running Celery worker.

**1 — Start Redis** (Docker recommended):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**2 — Add to `.env`**:
```env
REDIS_URL=redis://localhost:6379/0
```

**3 — Start the Celery worker** (separate terminal, from project root):
```bash
celery -A celery_app worker --loglevel=info
```

**4 — Use async recommendations**:
```bash
# Dispatch — returns task_id instantly (202 Accepted)
curl -X POST "http://localhost:8000/api/recommend?item_title=YourItem&top_n=10"

# Poll for results using the returned task_id
curl "http://localhost:8000/api/task/<task_id>"
```

**Response flow:**
```
POST /api/recommend  →  { "task_id": "abc123", "status": "PENDING" }
GET  /api/task/abc123  →  { "status": "SUCCESS", "result": { ... } }
```

### Alternative — Streamlit UI *(no Supabase required)*

```bash
streamlit run app.py
```

Upload any CSV file, click **Build Models**, then enter an item name or User ID to get recommendations directly in your browser — no database or server setup needed.

---

## 06 — API Reference

**Retrieve frontend configuration (Supabase URL + anon key):**
```http
GET /api/config
```

**Check if the API server is running:**
```http
GET /api/status
```

**Full-text search across items (PostgreSQL FTS):**
```http
GET /api/search?q=...&limit=20
```

**Upload a CSV or JSON dataset:**
```http
POST /api/upload
```

**Build / rebuild the ML models from uploaded data:**
```http
POST /api/build
```

**Get hybrid recommendations for a given item title:**
```http
GET /api/recommend/{title}
```

**Paginated list of all items:**
```http
GET /api/items?page=1&per_page=50
```

**List all distinct product categories:**
```http
GET /api/categories
```

**Read the current α / β / γ blending weights:**
```http
GET /api/weights
```

**Update the α / β / γ blending weights:**
```http
PUT /api/weights
```

**Get purchase history for a specific user:**
```http
GET /api/purchases/{user_id}
```

**Record a new purchase event:**
```http
POST /api/purchases
```

---

## 07 — Evaluation

```python
# Run evaluation benchmarks
python evaluation.py
```

Benchmarks **Content-Only**, **Collab-Only**, **Sentiment-Only**, and **Hybrid** across:

```text
Precision@K  —  fraction of relevant items in top-K
Recall@K     —  fraction of all relevant items retrieved
NDCG@K       —  ranking quality (discounted cumulative gain)
```

---

## 08 — Security

```text
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

## 09 — Screenshots

### Home Page
![Home Page](assets/homepage.png)

### Recommendation Results
![Recommendations](assets/recommendations.png)

### API Documentation
![Swagger Docs](assets/swagger.png)

---

## 10 — Troubleshooting

### ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Port Already In Use

```bash
python -m uvicorn backend.main:app --port 8001
```

### NLTK VADER Download Error

```python
import nltk
nltk.download('vader_lexicon')
```

### Supabase Connection Error

Check your `.env` file — no extra spaces, no quotes, correct project credentials:

```env
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
SUPABASE_SERVICE_KEY=your_service_key
```

---

## 11 — Setup Verification

```bash
# Backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Visit: http://localhost:8000/api/status → { "status": "ok" }

# Streamlit
streamlit run app.py
# Browser opens automatically with CSV upload interface
```

---

## 12 — Beginner Contributor Tips

### Sync Your Fork Before Starting

```bash
git remote add upstream https://github.com/leonagoel/hybrid-recommender.git
git fetch upstream
git merge upstream/main
```

### Resolve Merge Conflicts

1. Open conflicted files
2. Remove conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
3. Keep correct code, save, then commit

### Pull Request Checklist

- [ ] Project runs successfully
- [ ] README formatting checked
- [ ] No unnecessary files added
- [ ] Branch name follows guidelines
- [ ] Commit message follows convention
- [ ] PR linked to issue

---

## License

MIT — see [`LICENSE`](LICENSE)

---

## Documentation

- [CHANGELOG](CHANGELOG.md)

<div align="center">

```text
Built by Leona Goel
B.Tech CSE · Vellore Institute of Technology
National Finalist · Smart India Hackathon 2025 · Top 8% of 950+ Teams
```

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leona-goel)
[![GitHub](https://img.shields.io/badge/Follow-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/leonagoel)
[![Email](https://img.shields.io/badge/Email-leona.goel123%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:leona.goel123@gmail.com)

</div>

---

## 👥 Contributors

Thanks to all the amazing people who contribute to this project ❤️

[![Good First Issues](https://img.shields.io/github/issues/leonagoel/hybrid-recommender/good%20first%20issue?color=brightgreen&label=good+first+issues&style=flat-square)](https://github.com/leonagoel/hybrid-recommender/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
[![Open Issues](https://img.shields.io/github/issues/leonagoel/hybrid-recommender?style=flat-square)](https://github.com/leonagoel/hybrid-recommender/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

### Contributor Grid

<a href="https://github.com/leonagoel/hybrid-recommender/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=leonagoel/hybrid-recommender" alt="Contributors" />
</a>

### Want to contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) to get started — all skill levels welcome!

<div align="center">

| Step | Action |
|------|--------|
| 1️⃣ | [Fork the repo](https://github.com/leonagoel/hybrid-recommender/fork) |
| 2️⃣ | Pick a [good first issue](https://github.com/leonagoel/hybrid-recommender/issues?q=is%3Aopen+label%3A%22good+first+issue%22) |
| 3️⃣ | Submit a Pull Request |

</div>

---
