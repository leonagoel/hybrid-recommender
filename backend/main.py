"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import sys
import io
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from db import get_supabase, get_supabase_admin
from data_adapter import adapt_data, read_file
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="3.0")

# CORS — restrict in production; allow localhost for development
allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── State ────────────────────────────────────────────────────────────
models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
}


class WeightsUpdate(BaseModel):
    alpha: float = 0.4
    beta: float = 0.35
    gamma: float = 0.25


class PurchaseCreate(BaseModel):
    user_id: str
    product_id: int
    rating: float = 0.0
    review_text: str = ""


# ── Config (for frontend — serves only public keys) ─────────────────

@app.get("/api/config")
def get_config():
    """Serve Supabase public config to the frontend. Only exposes the anon key (safe for public use)."""
    return {
        "supabase_url": os.environ.get("SUPABASE_URL", ""),
        "supabase_anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
    }


# ── Status ───────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    sb = get_supabase()
    count_result = sb.table('products').select('id', count='exact').limit(0).execute()
    product_count = count_result.count or 0
    return {
        "status": "ready" if models["ready"] else ("has_data" if product_count > 0 else "no_data"),
        "product_count": product_count,
        "model_ready": models["ready"],
        "build_time": models["build_time"],
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": models["ready"]
    }

# ── Search (PostgreSQL FTS) ─────────────────────────────────────────

@app.get("/api/search")
def search_items(
    q: str = "",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Search products using PostgreSQL full-text search.
    Falls back to top-rated products when query is empty.
    """
    sb = get_supabase()

    if q.strip():
        try:
            result = sb.rpc('search_products', {
                'query_text': q.strip(),
                'match_count': limit,
                'offset_val': offset,
            }).execute()
            products = result.data or []
        except Exception:
            # Fallback: do a LIKE search if FTS parsing fails
            result = sb.table('products') \
                .select('id, title, description, category, rating, avg_sentiment, review_count') \
                .ilike('title', f'%{q.strip()}%') \
                .order('rating', desc=True) \
                .limit(limit) \
                .execute()
            products = result.data or []
            for p in products:
                p['rank'] = 0.0
    else:
        result = sb.table('products') \
            .select('id, title, description, category, rating, avg_sentiment, review_count') \
            .order('rating', desc=True) \
            .order('review_count', desc=True) \
            .limit(limit) \
            .offset(offset) \
            .execute()
        products = result.data or []

    # Format response
    results = []
    for p in products:
        results.append({
            'id': p.get('id'),
            'title': p.get('title', ''),
            'description': str(p.get('description', ''))[:200],
            'category': p.get('category', ''),
            'rating': p.get('rating', 0.0),
            'avg_sentiment': p.get('avg_sentiment', 0.0),
            'review_count': p.get('review_count', 0),
            'rank': p.get('rank', 0.0),
        })

    return {
        "results": results,
        "total": len(results),
        "query": q,
        "is_fallback": not q.strip(),
    }


# ── Upload + Import ─────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or JSON dataset and import into Supabase."""
    import math
    filename = file.filename or "data.csv"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ('.csv', '.json'):
        raise HTTPException(400, "Only CSV and JSON files are supported.")

    try:
        contents = await file.read()
        buf = io.BytesIO(contents)
        raw_df = read_file(buf, file_format=ext.replace('.', ''))
        adapted_df, meta = adapt_data(raw_df)
        adapted_df = adapted_df.drop_duplicates(subset='title', keep='first')

        # Use admin client if available, otherwise fall back to anon
        try:
            sb = get_supabase_admin()
        except RuntimeError:
            sb = get_supabase()

        batch_size = 500
        total = len(adapted_df)
        imported = 0
        errors = []

        for start in range(0, total, batch_size):
            chunk = adapted_df.iloc[start:start + batch_size]
            rows = []
            for _, row in chunk.iterrows():
                # Safely convert rating — handle NaN, inf, None
                raw_rating = row.get('rating', 0)
                try:
                    rating_val = float(raw_rating)
                    if math.isnan(rating_val) or math.isinf(rating_val):
                        rating_val = 0.0
                except (ValueError, TypeError):
                    rating_val = 0.0

                title = str(row.get('title', 'Unknown')).strip()
                if not title or title == 'nan' or title == 'Unknown':
                    continue

                rows.append({
                    'title': title[:500],
                    'description': str(row.get('description', ''))[:2000],
                    'category': str(row.get('category', ''))[:200],
                    'rating': round(rating_val, 2),
                    'metadata': {},
                })

            if not rows:
                continue

            try:
                sb.table('products').upsert(
                    rows, on_conflict='title', ignore_duplicates=True
                ).execute()
                imported += len(rows)
            except Exception as e:
                errors.append(f"Batch {start}-{start+len(rows)}: {str(e)[:100]}")

        models["ready"] = False  # Force rebuild

        result = {
            "message": f"Imported {imported:,} products from {filename}",
            "imported": imported,
            "total_rows": total,
            "meta": {
                "has_user_data": meta['has_user_data'],
                "has_reviews": meta['has_reviews'],
            },
        }
        if errors:
            result["warnings"] = errors[:5]  # Return first 5 errors
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Don't leak internal details — log server-side, return generic message
        raise HTTPException(400, "Upload failed. Check file format and try again.")


# ── Build Models ────────────────────────────────────────────────────

@app.post("/api/build")
def build_models():
    """Build recommendation models from Supabase data."""
    sb = get_supabase()

    # Fetch products
    all_products = []
    page_size = 1000
    offset = 0
    while True:
        result = sb.table('products') \
            .select('id, title, description, category, rating, avg_sentiment, review_count') \
            .range(offset, offset + page_size - 1) \
            .execute()
        batch = result.data or []
        all_products.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not all_products:
        raise HTTPException(400, "No products in database. Upload data first.")

    import pandas as pd
    item_df = pd.DataFrame(all_products)
    item_df['combined'] = (
        item_df['title'].astype(str) + ' ' +
        item_df['description'].fillna('').astype(str) + ' ' +
        item_df['category'].fillna('').astype(str)
    )
    item_df['review_count'] = item_df['review_count'].fillna(0).astype(int)

    start_time = time.time()

    # Content model
    content_model = ContentRecommender(item_df)

    # Collaborative model (from purchases)
    collab_model = None
    try:
        purchases_result = sb.table('purchases') \
            .select('user_id, product_id, rating') \
            .limit(50000) \
            .execute()
        purchases = purchases_result.data or []

        if len(purchases) > 10:
            # Map product_id → title
            product_title_map = {p['id']: p['title'] for p in all_products}
            interaction_rows = []
            for p in purchases:
                title = product_title_map.get(p['product_id'])
                if title:
                    interaction_rows.append({
                        'user_id': p['user_id'],
                        'title': title,
                        'rating': p.get('rating', 3.0),
                    })

            if len(interaction_rows) > 10:
                interaction_df = pd.DataFrame(interaction_rows)
                if interaction_df['user_id'].nunique() > 1:
                    collab_model = CollaborativeRecommender(interaction_df)
    except Exception:
        pass

    # Hybrid model
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

    build_time = round(time.time() - start_time, 2)

    models["content"] = content_model
    models["collab"] = collab_model
    models["hybrid"] = hybrid_model
    models["item_df"] = item_df
    models["ready"] = True
    models["build_time"] = build_time

    return {
        "message": "Models built successfully!",
        "items": len(item_df),
        "has_collaborative": collab_model is not None,
        "build_time_seconds": build_time,
    }


# ── Recommendations ────────────────────────────────────────────────

@app.get("/api/recommend/{item_title}")
def get_recommendations(item_title: str, top_n: int = 10):
    """Get hybrid recommendations for an item."""
    if not models["ready"]:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    recs = models["hybrid"].recommend(item_title, top_n=top_n)
    if not recs:
        raise HTTPException(404, "Item not found or no recommendations.")
    return {
        "query_item": item_title,
        "recommendations": recs,
        "weights": models["hybrid"].get_weights(),
    }


# ── Weights ─────────────────────────────────────────────────────────

@app.get("/api/weights")
def get_weights():
    if not models["ready"]:
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}
    return models["hybrid"].get_weights()


@app.put("/api/weights")
def update_weights(w: WeightsUpdate):
    if not models["ready"]:
        raise HTTPException(400, "Models not built.")
    models["hybrid"].set_weights(w.alpha, w.beta, w.gamma)
    return {"message": "Weights updated", "weights": models["hybrid"].get_weights()}


# ── Items ───────────────────────────────────────────────────────────

@app.get("/api/items")
def list_items(page: int = 1, per_page: int = 50):
    """List products from Supabase with pagination."""
    sb = get_supabase()
    offset = (page - 1) * per_page
    result = sb.table('products') \
        .select('id, title, description, category, rating, avg_sentiment, review_count') \
        .order('rating', desc=True) \
        .range(offset, offset + per_page - 1) \
        .execute()

    count_result = sb.table('products').select('id', count='exact').limit(0).execute()

    items = []
    for p in (result.data or []):
        items.append({
            'id': p.get('id'),
            'title': p.get('title', ''),
            'category': p.get('category', ''),
            'rating': round(float(p.get('rating', 0)), 2),
            'avg_sentiment': round(float(p.get('avg_sentiment', 0)), 4),
            'description': str(p.get('description', ''))[:200],
        })

    return {
        "items": items,
        "total": count_result.count or 0,
        "page": page,
        "per_page": per_page,
    }


# ── Categories ──────────────────────────────────────────────────────

@app.get("/api/categories")
def get_categories():
    """Get all unique categories."""
    sb = get_supabase()
    result = sb.rpc('get_categories', {}).execute()
    if result.data:
        return {"categories": result.data}

    # Fallback: distinct query
    result = sb.table('products').select('category').limit(5000).execute()
    cats = list(set(p['category'] for p in (result.data or []) if p.get('category')))
    cats.sort()
    return {"categories": cats}


# ── Purchases ───────────────────────────────────────────────────────

@app.get("/api/purchases/{user_id}")
def get_user_purchases(user_id: str, limit: int = 50):
    """Get purchase history for a user (via anon client — RLS enforced)."""
    sb = get_supabase()
    result = sb.table('purchases') \
        .select('id, product_id, rating, review_text, purchased_at, products(title, category, rating)') \
        .eq('user_id', user_id) \
        .order('purchased_at', desc=True) \
        .limit(limit) \
        .execute()
    return {"purchases": result.data or []}


@app.post("/api/purchases")
def create_purchase(data: PurchaseCreate):
    """Record a purchase (validated input)."""
    sb = get_supabase()
    result = sb.table('purchases').insert({
        'user_id': data.user_id,
        'product_id': data.product_id,
        'rating': max(0, min(5, data.rating)),
        'review_text': data.review_text[:1000],
    }).execute()
    return {"purchase": result.data}


# ── Frontend Serving ────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
