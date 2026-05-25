# SETUP.md — HybridRec Setup Guide

## Prerequisites
- Python 3.10+
- pip

## Quick Start

### 1. Clone & Install
```bash
cd hybrid-recommender
pip install -r requirements.txt
```

### 2. Configure Environment
The `.env` file should already be configured with your Supabase credentials.
If not, copy the template:
```bash
cp .env.example .env
```

Fill in:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
```

> **Where to find the Service Key:**
> Go to [Supabase Dashboard](https://supabase.com/dashboard) → Your Project → Settings → API → `service_role` key.

### 3. Import Data
Place your CSV or JSON dataset files in the `datasets/` folder, then run:
```bash
python scripts/import_to_supabase.py
```

Options:
```bash
python scripts/import_to_supabase.py --file datasets/Books.csv --batch-size 2000
python scripts/import_to_supabase.py --sentiment  # Enable sentiment analysis (slow)
```

### 4. Seed Mock Data (Optional)
Creates fake users and purchase history for testing collaborative filtering:
```bash
python scripts/seed_mock_data.py
python scripts/seed_mock_data.py --users 50 --purchases 2000  # Custom amounts
```

### 5. Start the Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
## Windows Local Setup (Recommended)

If you encounter dependency issues on Windows, it is recommended to create a clean conda environment.

### Create Environment

```bash
conda create -n hybridrec python=3.10 -y
conda activate hybridrec

pip install -r requirements.txt
pip install sentence-transformers

## Run your streamlit app easy way to run the app 
$env:PYTHONPATH="."
streamlit run src/api/app.py

### 6. Run with Docker
Build the container image from the repository root:
```bash
docker build -t hybrid-recommender .
```

Run the FastAPI app on port 8000:
```bash
docker run --env-file .env -p 8000:8000 hybrid-recommender
```

The API will be available at [http://localhost:8000](http://localhost:8000).

### 7. Open the App
Navigate to: [http://localhost:8000](http://localhost:8000)

## Architecture
```
hybrid-recommender/
├── backend/
│   └── main.py          # FastAPI server
├── frontend/
│   ├── index.html       # UI
│   ├── styles.css       # Design system
│   └── app.js           # Client logic + Supabase auth
├── scripts/
│   ├── import_to_supabase.py  # One-time data import
│   └── seed_mock_data.py      # Generate fake users + purchases
├── content_model.py     # TF-IDF content-based recommender
├── collaborative_model.py # SVD collaborative filtering
├── hybrid_model.py      # Weighted hybrid scoring
├── nlp_engine.py        # VADER sentiment analysis
├── data_adapter.py      # CSV/JSON schema adapter
├── dataset_manager.py   # Multi-dataset manager
├── db.py                # Supabase client
├── app.py               # Streamlit UI (no Supabase needed)
├── Dockerfile           # FastAPI container image
├── .dockerignore        # Docker build exclusions
├── .env                 # Credentials (git-ignored)
└── requirements.txt
```

## Features
- **Search:** PostgreSQL full-text search (<50ms on 250k+ products)
- **Auth:** Guest (anonymous) + Email/Password
- **Recommendations:** Hybrid (Content + Collaborative + Sentiment)
- **Upload:** CSV and JSON format support
- **UI:** Amazon-like modern design with animations
- **Streamlit UI:** Upload a CSV locally, build models, and get recommendations — no Supabase or server setup needed (`streamlit run src/api/app.py`)
