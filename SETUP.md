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
if (-not $env:HOST) { $env:HOST = "0.0.0.0" }
if (-not $env:PORT) { $env:PORT = "8000" }

python -m uvicorn backend.main:app --host $env:HOST --port $env:PORT --reload
```

### 6. Open the App
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
├── .env                 # Credentials (git-ignored)
└── requirements.txt
```

## Features
- **Search:** PostgreSQL full-text search (<50ms on 250k+ products)
- **Auth:** Guest (anonymous) + Email/Password
- **Recommendations:** Hybrid (Content + Collaborative + Sentiment)
- **Upload:** CSV and JSON format support
- **UI:** Amazon-like modern design with animations
