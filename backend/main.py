"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import sys
import io
import time
import logging
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

from db import get_supabase, get_supabase_admin
from data_adapter import adapt_data, read_file
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender, bayesian_rating
from llm_explainer import get_explainer

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="3.0")
logger = logging.getLogger("hybrid_recommender.api")
RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0


def _get_slow_response_threshold_ms() -> float:
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS

# CORS — restrict in production; allow localhost for development
allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def response_time_middleware(request: Request, call_next):
    """Attach response duration headers and log every API request."""
    started_at = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        status_code = response.status_code if response is not None else 500

        if response is not None:
            response.headers[RESPONSE_TIME_HEADER] = f"{duration_ms:.2f}"

        log_fn = logger.warning if duration_ms >= _get_slow_response_threshold_ms() else logger.info
        log_fn(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

# ── State ────────────────────────────────────────────────────────────
models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
    "last_trained_at": None,
}

trending_cache = {}
TRENDING_CACHE_TTL = 60 * 60  # 1 hour


class WeightsUpdate(BaseModel):
    alpha: float = 0.4
    beta: float = 0.35
    gamma: float = 0.25


class PurchaseCreate(BaseModel):
    user_id: str
    product_id: int
    rating: float = 0.0
    review_text: str = ""

class FeedbackCreate(BaseModel):
    user_id: str
    item: str
    feedback: str


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

    return {
        "status": "healthy",
        "products": 120,
        "message": "Mock status running locally"
    }


# ── Dashboard (admin metrics — issue #71) ───────────────────────────

@app.get("/api/dashboard")
def dashboard():
    """Aggregate metrics for the admin dashboard."""
    sb = get_supabase()

    try:
        product_count = sb.table('products').select('id', count='exact').limit(0).execute().count or 0
    except Exception as e:
        logger.warning("Dashboard: product count failed: %s", e)
        product_count = 0

    try:
        interaction_count = sb.table('purchases').select('id', count='exact').limit(0).execute().count or 0
    except Exception as e:
        logger.warning("Dashboard: interaction count failed: %s", e)
        interaction_count = 0

    # Distinct users from purchases (capped scan)
    total_users = 0
    purchase_counts: Counter = Counter()
    try:
        purchase_rows = sb.table('purchases') \
            .select('user_id, product_id') \
            .limit(50000).execute().data or []
        total_users = len({r['user_id'] for r in purchase_rows if r.get('user_id')})
        purchase_counts = Counter(
            r['product_id'] for r in purchase_rows if r.get('product_id') is not None
        )
    except Exception as e:
        logger.warning("Dashboard: purchases scan failed: %s", e)

    # Averages over products
    avg_recommendation_score = 0.0
    avg_sentiment_score = 0.0
    try:
        prod_stats = sb.table('products') \
            .select('rating, avg_sentiment') \
            .limit(50000).execute().data or []
        ratings = [
            float(p['rating']) for p in prod_stats
            if p.get('rating') not in (None, 0)
        ]
        sentiments = [
            float(p['avg_sentiment']) for p in prod_stats
            if p.get('avg_sentiment') is not None
        ]
        if ratings:
            avg_recommendation_score = round(sum(ratings) / len(ratings), 4)
        if sentiments:
            avg_sentiment_score = round(sum(sentiments) / len(sentiments), 4)
    except Exception as e:
        logger.warning("Dashboard: averages query failed: %s", e)

    # Top 5 by purchase count; fallback to top-rated when no purchases
    top_products = []
    try:
        if purchase_counts:
            top_ids = [pid for pid, _ in purchase_counts.most_common(5)]
            prod_result = sb.table('products') \
                .select('id, title, category, rating') \
                .in_('id', top_ids).execute().data or []
            prod_map = {p['id']: p for p in prod_result}
            for pid in top_ids:
                p = prod_map.get(pid)
                if p:
                    top_products.append({
                        'id': p['id'],
                        'title': p.get('title', ''),
                        'category': p.get('category', ''),
                        'rating': round(float(p.get('rating', 0) or 0), 2),
                        'interactions': purchase_counts[pid],
                    })
        if not top_products:
            fallback = sb.table('products') \
                .select('id, title, category, rating') \
                .order('rating', desc=True) \
                .order('review_count', desc=True) \
                .limit(5).execute().data or []
            for p in fallback:
                top_products.append({
                    'id': p['id'],
                    'title': p.get('title', ''),
                    'category': p.get('category', ''),
                    'rating': round(float(p.get('rating', 0) or 0), 2),
                    'interactions': 0,
                })
    except Exception as e:
        logger.warning("Dashboard: top products query failed: %s", e)

    return {
        "total_products": product_count,
        "total_users": total_users,
        "total_interactions": interaction_count,
        "avg_recommendation_score": avg_recommendation_score,
        "avg_sentiment_score": avg_sentiment_score,
        "top_5_recommended_products": top_products,
        "model_last_trained": models.get("last_trained_at"),
    }


# ── Search (PostgreSQL FTS) ─────────────────────────────────────────
@app.get("/api/search")
def search_items(q: str = "", limit: int = 8):

    mock_items = [
        {"title": "iPhone 15", "rating": 4.8},
        {"title": "Samsung Galaxy S24", "rating": 4.7},
        {"title": "MacBook Air M3", "rating": 4.9},
        {"title": "Sony WH-1000XM5", "rating": 4.6},
        {"title": "Apple Watch Ultra", "rating": 4.7},
    ]

    return mock_items

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
        except Exception as e:
            logger.warning("Full-text search failed for query '%s': %s", q.strip(), e)
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

    filtered = [
        item for item in mock_items
        if q.lower() in item["title"].lower()
    ]

    return {
        "items": filtered[:limit]
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
            logger.warning("Imported dataset with %d batch warnings", len(errors))

        logger.info("Imported %d products from %s", imported, filename)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed for %s: %s", filename, e, exc_info=True)
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
        logger.warning("Model build requested with no products in database")
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
    except Exception as e:
        logger.warning("Collaborative model data load failed: %s", e)

    # Hybrid model
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

    build_time = round(time.time() - start_time, 2)

    models["content"] = content_model
    models["collab"] = collab_model
    models["hybrid"] = hybrid_model
    models["item_df"] = item_df
    models["ready"] = True
    models["build_time"] = build_time
    models["last_trained_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Built recommendation models for %d items in %.2f seconds",
        len(item_df),
        build_time,
    )

    return {
        "message": "Models built successfully!",
        "items": len(item_df),
        "has_collaborative": collab_model is not None,
        "build_time_seconds": build_time,
    }


# ── Recommendations ────────────────────────────────────────────────

@app.get("/api/recommend/{item_title}")
def get_recommendations(item_title: str, top_n: int = 10, explain: bool = Query(False), llm_explain: bool = Query(False)):
    """Get hybrid recommendations for an item with optional LLM explanations."""
    if not models["ready"]:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    recs = models["hybrid"].recommend(item_title, top_n=top_n, explain=explain)
    if not recs:
        raise HTTPException(404, "Item not found or no recommendations.")
    
    # Add LLM explanations if requested
    if llm_explain:
        try:
            explainer = get_explainer()
            recs = explainer.explain_multiple(recs, item_title)
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}. Returning recommendations without LLM explanations.")
    
    return {
        "query_item": item_title,
        "recommendations": recs,
        "weights": models["hybrid"].get_weights(),
        "explain": explain,
        "llm_explain": llm_explain,
    }


@app.get("/api/explain")
def explain_recommendation(item: str, user: str):
    """Explain WHY an item was recommended to a specific user."""
    if not models["ready"]:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
        
    hybrid = models["hybrid"]
    
    # Check if item exists in our models
    if item not in hybrid._rating_map:
        raise HTTPException(404, "Item not found in recommendations database.")
        
    # Extract item scores
    sentiment_score = hybrid._sentiment_map.get(item, 0.0)
    bayesian_score = hybrid._rating_map.get(item, 0.0)
    norm_sentiment = (sentiment_score + 1) / 2
    
    collab_score = 0.0
    content_score = 0.0
    
    collab_model = models.get("collab")
    if collab_model:
        # Predict rating for the user and item
        pred = collab_model.predict_rating(user, item)
        if pred is not None:
            collab_score = max(0.0, min(1.0, pred / 5.0))
            
        # For content score, compare against the user's top-rated item
        user_history = collab_model.df[collab_model.df['user_id'] == user]
        if not user_history.empty:
            top_item = user_history.loc[user_history['rating'].idxmax()]['title']
            content_model = models.get("content")
            if content_model:
                try:
                    recs = content_model.recommend(top_item, top_n=100)
                    for r in recs:
                        if r['title'] == item:
                            content_score = r['content_score']
                            break
                except Exception:
                    pass
    
    # Build reasons
    reasons = []
    if collab_score > 0.7:
        reasons.append("Similar to your top rated items")
    elif collab_score > 0.5:
        reasons.append("Matches your user profile")
        
    if norm_sentiment > 0.65:
        reasons.append("High sentiment score")
        
    if bayesian_score > 4.0:
        reasons.append("Popular in your category")
        
    reasons = reasons[:3]
    if not reasons:
        reasons.append("Recommended based on general popularity")

    return {
        "item": item,
        "reasons": reasons,
        "scores": {
            "content": round(content_score, 4),
            "collab": round(collab_score, 4),
            "sentiment": round(norm_sentiment, 4),
            "bayesian": round(bayesian_score, 4)
        }
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
def list_items(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """List products from Supabase with cursor-style pagination.

    Supports ``?page=1&limit=20`` for infinite-scroll on the frontend.
    Returns a ``has_more`` flag so the client knows when to stop fetching.
    """
    sb = get_supabase()
    offset = (page - 1) * limit
    result = sb.table('products') \
        .select('id, title, description, category, rating, avg_sentiment, review_count') \
        .order('rating', desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()

    count_result = sb.table('products').select('id', count='exact').limit(0).execute()
    total = count_result.count or 0

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
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (offset + len(items)) < total,
    }


# ── Similarity Matrix ──────────────────────────────────────────────

@app.get("/api/similarity-matrix")
def similarity_matrix(items: str = Query(..., description="Comma-separated product titles")):
    """Compute an NxN cosine similarity matrix for the given product titles.

    Uses the content model's TF-IDF vectors to calculate pairwise cosine
    similarity scores.  Accepts up to 20 items to keep response size
    manageable.

    Example::

        GET /api/similarity-matrix?items=ProductA,ProductB,ProductC
    """
    if not models["ready"] or models["content"] is None:
        raise HTTPException(400, "Models not built. Build first via /api/build.")

    titles = [t.strip() for t in items.split(",") if t.strip()]
    if len(titles) < 2:
        raise HTTPException(400, "Provide at least 2 comma-separated item titles.")
    if len(titles) > 20:
        raise HTTPException(400, "Maximum 20 items allowed per request.")

    content_model = models["content"]
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Resolve indices and filter out unknown titles
    indices = []
    valid_titles = []
    not_found = []
    for title in titles:
        idx = content_model._title_to_idx.get(title.lower())
        if idx is not None:
            indices.append(idx)
            valid_titles.append(content_model.df.iloc[idx]['title'])  # canonical case
        else:
            not_found.append(title)

    if len(valid_titles) < 2:
        raise HTTPException(
            404,
            f"Need at least 2 valid items. Not found: {not_found}",
        )

    # Compute NxN similarity from the TF-IDF matrix rows
    sub_matrix = content_model.matrix[indices]
    sim = cos_sim(sub_matrix, sub_matrix)

    # Build JSON-serializable matrix (rounded to 4 decimals)
    matrix = [[round(float(sim[i][j]), 4) for j in range(len(valid_titles))]
              for i in range(len(valid_titles))]

    result = {
        "labels": valid_titles,
        "matrix": matrix,
        "size": len(valid_titles),
    }
    if not_found:
        result["not_found"] = not_found

    return result


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
# ── Dashboard ───────────────────────────────────────────────────────

# ── Feedback ────────────────────────────────────────────────────────

@app.post("/api/feedback")
def submit_feedback(data: FeedbackCreate):

    return {
        "message": "Feedback submitted successfully",
        "feedback": {
            "user_id": data.user_id,
            "item": data.item,
            "feedback": data.feedback
        }
    }
# ── Frontend Serving ────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/dashboard.html")
    def serve_dashboard():
        return FileResponse(os.path.join(frontend_dir, "dashboard.html"))
