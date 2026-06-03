# 💡 idea.md — HybridRec Project Plan

## Vision
Build a production-grade hybrid recommendation system that combines three AI approaches:
1. **Content-Based Filtering** — TF-IDF vectorization on product metadata
2. **Collaborative Filtering** — SVD matrix factorization on user interactions
3. **Sentiment Analysis** — VADER NLP for review-based scoring

## Architecture

```
┌──────────────────────────────────────────────┐
│  Frontend (Vanilla JS + Supabase Auth)       │
│  ├─ Type-to-search with FTS dropdown         │
│  ├─ Product grid with staggered animations   │
│  ├─ Recommendation strip (horizontal scroll) │
│  └─ Auth modal (Guest / Email+Password)      │
└──────────────────┬───────────────────────────┘
                   │ REST API
┌──────────────────┴───────────────────────────┐
│  FastAPI Backend                              │
│  ├─ /api/search    → PostgreSQL FTS (instant) │
│  ├─ /api/build     → Build ML models          │
│  ├─ /api/recommend → Hybrid scoring            │
│  ├─ /api/upload    → CSV/JSON import           │
│  └─ /api/purchases → User history              │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────┴───────────────────────────┐
│  Supabase (PostgreSQL)                        │
│  ├─ products  (250k+ rows, GIN-indexed FTS)   │
│  ├─ profiles  (extends auth.users)             │
│  ├─ purchases (user-product interactions)      │
│  ├─ reviews   (sentiment-analyzed reviews)     │
│  └─ auth.users (guest + email/password)        │
└──────────────────────────────────────────────┘
```

## Key Decisions

### Why Supabase over raw PostgreSQL?
- Free tier with 500MB storage (enough for 250k products)
- Built-in Auth with anonymous sign-in
- REST API + realtime subscriptions
- Row Level Security for data isolation
- Zero-config setup

### Why PostgreSQL FTS over in-memory TF-IDF for search?
- GIN index pre-computes token positions → <50ms on 250k rows
- TF-IDF on 250k items took 5+ minutes to build in memory
- `websearch_to_tsquery` supports natural language queries
- Stemming and ranking built-in
- TF-IDF is still used for content-based item similarity (model internals)

### Data Sparsity Solutions
| Strategy | Impact |
|----------|--------|
| Implicit feedback | Views/purchases as soft signals → 10-50x denser matrix |
| Popularity fallback | Category-avg for items with <5 interactions |
| Bayesian average | `(v/(v+m))×R + (m/(v+m))×C` prevents rating bias |
| Adaptive SVD rank | Lower n_factors when density <0.1% |
| Category warm-start | New users get popular items from browsed categories |

## Implementation Phases
1. ✅ Database schema + RLS + FTS function
2. ✅ Authentication (Guest + Email/Password)
3. ⬜ Data import (CSV/JSON → PostgreSQL)
4. ✅ Backend (Supabase integration, improved models)
5. ✅ Frontend (Amazon-like UI, type-to-search)
6. ✅ Documentation

## Tech Stack
- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **ML:** scikit-learn (TF-IDF, SVD), NLTK (VADER)
- **Database:** Supabase (PostgreSQL 15)
- **Frontend:** Vanilla JS, Supabase-JS, CSS3
- **Auth:** Supabase Auth (Anonymous + Email/Password)

## Future Work
- pgvector semantic search (sentence-transformers embeddings)
- Real-time recommendations via Supabase Realtime
- A/B testing framework for model weights
- User preference learning from click patterns
- Mobile-responsive PWA conversion
