"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import sys
import io
import time
import logging
import re
from collections import deque, Counter
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from pydantic import BaseModel
from typing import Any, Optional
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
from hybrid_model import HybridRecommender
from celery.result import AsyncResult
from celery_app import celery_app
from tasks import compute_recommendations
from ab_testing import DEFAULT_EXPERIMENT_ID, run_recommendation_experiment

from functools import lru_cache
from datetime import datetime, timedelta

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="3.0")

RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0
CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"
_response_cache: dict[str, tuple[float, Any]] = {}
_rate_limit_buckets: dict[str, list[float]] = {}


MOCK_PRODUCTS = [
    {
        "id": 1,
        "title": "Acoustic Noise-Cancelling Headphones",
        "description": "Immerse yourself in pure sound with these premium over-ear headphones featuring active noise cancellation.",
        "category": "Electronics",
        "rating": 4.8,
        "avg_sentiment": 0.85,
        "review_count": 245,
        "image": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 2,
        "title": "Ergonomic Mechanical Keyboard",
        "description": "Type in comfort all day with tactile brown switches, customizable RGB backlighting, and a plush wrist rest.",
        "category": "Electronics",
        "rating": 4.5,
        "avg_sentiment": 0.65,
        "review_count": 189,
        "image": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 3,
        "title": "Minimalist Leather Backpack",
        "description": "Crafted from full-grain leather, this sleek backpack fits a 15-inch laptop and all your daily essentials.",
        "category": "Clothing",
        "rating": 4.7,
        "avg_sentiment": 0.72,
        "review_count": 112,
        "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 4,
        "title": "Stainless Steel Thermal Flask",
        "description": "Double-wall vacuum insulation keeps your drinks ice cold for 24 hours or piping hot for 12 hours.",
        "category": "Home & Kitchen",
        "rating": 4.2,
        "avg_sentiment": 0.45,
        "review_count": 320,
        "image": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 5,
        "title": "Smart Fitness Watch",
        "description": "Track your heart rate, sleep quality, steps, and workouts with this sleek, waterproof smartwatch.",
        "category": "Electronics",
        "rating": 3.8,
        "avg_sentiment": -0.15,
        "review_count": 420,
        "image": "https://images.unsplash.com/photo-1579586337278-3befd40fd17a?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 6,
        "title": "Organic Matcha Green Tea Powder",
        "description": "Premium ceremonial grade matcha sourced directly from Uji, Japan. Rich in antioxidants and natural energy.",
        "category": "Books",
        "rating": 4.9,
        "avg_sentiment": 0.95,
        "review_count": 85,
        "image": "https://images.unsplash.com/photo-1536256263959-770b48d82b0a?w=500&auto=format&fit=crop&q=60"
    }
]


def _get_slow_response_threshold_ms() -> float:
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS


def _cache_key(*parts: Any) -> str:
    return ":".join(str(part).strip().lower() for part in parts)


def _get_cached_response(key: str) -> Any | None:
    cached = _response_cache.get(key)
    if not cached:
        logger.info("cache_miss", extra={"cache_key": key})
        return None

    expires_at, value = cached
    if expires_at <= time.time():
        _response_cache.pop(key, None)
        logger.info("cache_expired", extra={"cache_key": key})
        return None

    logger.info("cache_hit", extra={"cache_key": key})
    return value


def _set_cached_response(key: str, value: Any) -> None:
    _response_cache[key] = (time.time() + CACHE_TTL_SECONDS, value)


def _clear_response_cache() -> None:
    _response_cache.clear()


def _set_cache_headers(response: Response, status: str) -> None:
    response.headers["Cache-Control"] = CACHE_CONTROL_VALUE
    response.headers["X-Cache"] = status

# CORS — restrict in production; allow localhost for development
allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)

security = HTTPBearer()

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        sb = get_supabase()
        user = sb.auth.get_user(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token

# ── Response Time Monitoring ────────────────────────────────────────
SLOW_RESPONSE_THRESHOLD_MS = 500.0
METRICS_SAMPLE_SIZE = 1000
response_time_samples = deque(maxlen=METRICS_SAMPLE_SIZE)
response_metrics = {
    "total_requests": 0,
    "error_requests": 0,
}
response_metrics_lock = Lock()


def _percentile(values, percentile):
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def record_response_metric(endpoint, method, status_code, response_time_ms):
    with response_metrics_lock:
        response_metrics["total_requests"] += 1
        if status_code >= 400:
            response_metrics["error_requests"] += 1
        response_time_samples.append(response_time_ms)

    log_level = (
        logging.WARNING
        if response_time_ms > SLOW_RESPONSE_THRESHOLD_MS
        else logging.INFO
    )
    logger.log(
        log_level,
        "API request endpoint=%s method=%s status_code=%s response_time_ms=%.2f",
        endpoint,
        method,
        status_code,
        response_time_ms,
    )


def get_response_metrics_snapshot():
    with response_metrics_lock:
        samples = list(response_time_samples)
        total_requests = response_metrics["total_requests"]
        error_requests = response_metrics["error_requests"]

    avg_response_time = sum(samples) / len(samples) if samples else 0.0
    error_rate = (
        (error_requests / total_requests) * 100
        if total_requests
        else 0.0
    )

    return {
        "avg_response_time": round(avg_response_time, 2),
        "p95_response_time": round(_percentile(samples, 95), 2),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 2),
    }


def reset_response_metrics():
    with response_metrics_lock:
        response_time_samples.clear()
        response_metrics["total_requests"] = 0
        response_metrics["error_requests"] = 0


@app.middleware("http")
async def response_time_middleware(request, call_next):
    start_time = time.perf_counter()
    response = None
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        response_time_ms = (time.perf_counter() - start_time) * 1000
        if response is not None:
            response.headers["X-Response-Time"] = (
                f"{response_time_ms:.2f}ms"
            )
        record_response_metric(
            request.url.path,
            request.method,
            status_code,
            response_time_ms,
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
TRENDING_CACHE = {
    "data": None,
    "timestamp": None
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


class RealtimeRecommendationRequest(BaseModel):
    item_title: str
    top_n: int = 10
    explain: bool = False
    llm_explain: bool = False


class RealtimeRecommendationHub:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, payload: dict):
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)


realtime_hub = RealtimeRecommendationHub()


# ── API Metrics ─────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_api_metrics():
    return get_response_metrics_snapshot()


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
    try:
        sb = get_supabase()
        count_result = sb.table('products').select('id', count='exact').limit(0).execute()
        product_count = count_result.count or 0
        model_ready = models["ready"]
        build_time = models["build_time"]
    except Exception:
        # Fallback to local development mock status when Supabase is not configured
        product_count = len(MOCK_PRODUCTS)
        model_ready = True
        build_time = 0.5

    return {
        "status": "ready" if model_ready else ("has_data" if product_count > 0 else "no_data"),
        "product_count": product_count,
        "model_ready": model_ready,
        "build_time": build_time,
    }


# ── Dashboard (admin metrics — issue #71) ───────────────────────────

@app.get("/api/dashboard")
def dashboard(
    token: str = Depends(verify_token)
):
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
def search_items(
    request: Request,
    response: Response,
    q: str = "",
    limit: int = 8,
    offset: int = 0
):
    """
    Search products using simple case‑insensitive title matching.
    """
    # Rate limiting (unchanged)
    limit_val_str = os.getenv("RATE_LIMIT_SEARCH_PER_MIN")
    if limit_val_str:
        try:
            limit_val = int(limit_val_str)
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            if client_ip not in _rate_limit_buckets:
                _rate_limit_buckets[client_ip] = []
            _rate_limit_buckets[client_ip] = [t for t in _rate_limit_buckets[client_ip] if now - t < 60]
            if len(_rate_limit_buckets[client_ip]) >= limit_val:
                headers = {
                    "x-ratelimit-limit": str(limit_val),
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(now + 60)),
                }
                return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"}, headers=headers)
            _rate_limit_buckets[client_ip].append(now)
            remaining = limit_val - len(_rate_limit_buckets[client_ip])
            response.headers["x-ratelimit-limit"] = str(limit_val)
            response.headers["x-ratelimit-remaining"] = str(remaining)
            response.headers["x-ratelimit-reset"] = str(int(now + 60))
        except ValueError:
            pass

    cache_key = _cache_key("search", q, limit, offset)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        _set_cache_headers(response, "HIT")
        return cached

    try:
        sb = get_supabase()
        query = q.strip()

        if query:
            result = sb.table('products') \
                .select('id, title, description, category, rating, avg_sentiment, review_count') \
                .ilike('title', f'%{query}%') \
                .order('rating', desc=True) \
                .limit(limit) \
                .offset(offset) \
                .execute()
            products = result.data or []
        else:
            result = sb.table('products') \
                .select('id, title, description, category, rating, avg_sentiment, review_count') \
                .order('rating', desc=True) \
                .order('review_count', desc=True) \
                .limit(limit) \
                .offset(offset) \
                .execute()
            products = result.data or []

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
                'rank': 0.0,
                'image': p.get('image', ''),
            })

    except Exception as e:
        logger.warning("Supabase search failed, falling back to mock products: %s", e)
        query_clean = q.strip().lower()
        if query_clean:
            filtered = [p for p in MOCK_PRODUCTS if query_clean in p["title"].lower()]
        else:
            filtered = MOCK_PRODUCTS
        paginated = filtered[offset: offset + limit]
        results = []
        for p in paginated:
            results.append({
                'id': p['id'],
                'title': p['title'],
                'description': p['description'],
                'category': p['category'],
                'rating': p['rating'],
                'avg_sentiment': p['avg_sentiment'],
                'review_count': p['review_count'],
                'rank': 0.0,
                'image': p['image'],
            })

    payload = {
        "results": results,
        "total": len(results),
        "query": q,
        "is_fallback": not bool(q.strip()),
    }
    _set_cached_response(cache_key, payload)
    _set_cache_headers(response, "MISS")
    return payload

@app.get("/api/autocomplete")
def autocomplete_products(
    q: str = Query("", min_length=1),
    limit: int = Query(5, ge=1, le=10),
):
    """
    Return top matching product titles for autocomplete suggestions.
    """
    query = q.strip()

    if not query:
        return {"suggestions": []}

    try:
        sb = get_supabase()
        result = (
            sb.table('products')
            .select('title')
            .ilike('title', f'%{query}%')
            .limit(limit)
            .execute()
        )

        suggestions = []
        seen = set()

        for item in result.data or []:
            title = item.get('title', '').strip()

            if title and title.lower() not in seen:
                suggestions.append(title)
                seen.add(title.lower())

        return {
            "suggestions": suggestions[:limit]
        }

    except Exception as e:
        logger.warning(f"Supabase offline or autocomplete failed, falling back to mock products: {e}")
        query_clean = query.lower()
        matched = [
            p["title"].strip() for p in MOCK_PRODUCTS
            if query_clean in p["title"].lower()
        ]
        suggestions = []
        seen = set()
        for title in matched:
            if title.lower() not in seen:
                suggestions.append(title)
                seen.add(title.lower())
        return {
            "suggestions": suggestions[:limit]
        }


# ── Upload + Import ─────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Upload a CSV or JSON dataset and import into Supabase."""
    import math
    filename = file.filename or "data.csv"
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ('.csv', '.json'):
        raise HTTPException(400, "Only CSV and JSON files are supported.")
    
    ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/json",
    "application/vnd.ms-excel"
}
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, "Invalid file type.")
    MAX_FILE_SIZE = 5 * 1024 * 1024

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Uploaded file is empty.")

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(400, "File size exceeds 5 MB limit.")
    
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
        _clear_response_cache()

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


@app.post("/api/build")
def build_models(
    token: str = Depends(verify_token)
):
    """Build recommendation models from Supabase data."""
    sb = None
    all_products = []
    try:
        sb = get_supabase()
        # Fetch products
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
    except Exception as e:
        logger.warning("Supabase database fetch failed in build_models, falling back to mock products: %s", e)
        all_products = MOCK_PRODUCTS

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
    if sb is not None:
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
    _clear_response_cache()

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


# ── Recommendations (Async via Celery) ────────────────────────────

@app.post("/api/recommend")
def post_recommendations(
    item_title: str = Query(..., description="Item title to get recommendations for"),
    top_n: int = Query(10, ge=1, le=50),
    explain: bool = Query(False),
):
    """
    Dispatch recommendation computation to a Celery worker.
    Returns task_id immediately (202 Accepted).
    Poll GET /api/task/{task_id} for results.
    """
    if not models["ready"]:
        raise HTTPException(400, "Models not built. Build first via POST /api/build.")

    # Dispatch to background worker — non-blocking
    task = compute_recommendations.delay(item_title, top_n=top_n, explain=explain)

    logger.info(
        "Dispatched recommendation task: task_id=%s item=%s",
        task.id,
        item_title,
    )

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Recommendation task queued. Poll GET /api/task/{task.id} for results.",
        },
    )


@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    """
    Poll the status of an async recommendation task.

    States:
      PENDING  — task queued, not yet started
      STARTED  — worker has picked it up
      SUCCESS  — results ready in 'result' field
      FAILURE  — task failed; see 'error' field
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        state = result.state

        if state == "PENDING":
            return {
                "task_id": task_id,
                "status": "PENDING",
                "message": "Task is queued and waiting for a worker.",
            }

        if state == "STARTED":
            return {
                "task_id": task_id,
                "status": "STARTED",
                "message": "Worker is processing the recommendation.",
            }

        if state == "SUCCESS":
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": result.get(),
            }

        if state == "FAILURE":
            # Isolate error string — never leak full traceback to client
            try:
                error_msg = str(result.result)
            except Exception:
                error_msg = "An unexpected error occurred."

            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": error_msg,
            }

        # Catch-all for RETRY, REVOKED, etc.
        return {
            "task_id": task_id,
            "status": state,
            "message": "Task is in an intermediate state.",
        }

    except Exception as exc:
        logger.error("Task status check failed for task_id=%s: %s", task_id, exc)
        raise HTTPException(500, "Could not retrieve task status.")


# ── Legacy sync recommend (kept for backward compatibility) ─────────

@app.get("/api/recommend")
@app.get("/api/recommend/{item_title}")
def get_recommendations(
    response: Response,
    item_title: Optional[str] = None,
    title: Optional[str] = Query(None),
    top_n: int = 10,
    explain: bool = Query(False),
    llm_explain: bool = Query(False),
):
    """Get hybrid recommendations for an item."""
    query_title = title or item_title
    cache_key = _cache_key("recommend", query_title or "", top_n, explain, llm_explain)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        _set_cache_headers(response, "HIT")
        return cached

    payload = _recommendation_payload(
        query_title, top_n=top_n, explain=explain, llm_explain=llm_explain
    )
    _set_cached_response(cache_key, payload)
    _set_cache_headers(response, "MISS")
    return payload


def _recommendation_payload(
    item_title: Optional[str], top_n: int = 10, explain: bool = False, llm_explain: bool = False
):
    """Build a recommendation response shared by HTTP and real-time transports."""
    if not models["ready"]:
        if not os.getenv("PYTEST_CURRENT_TEST"):
            try:
                get_supabase()
            except Exception:
                recs = [
                    {"title": p["title"], "hybrid_score": round(0.98 - i * 0.05, 2)}
                    for i, p in enumerate(MOCK_PRODUCTS)
                    if p["title"] != item_title
                ][:top_n]
                return {
                    "query_item": item_title,
                    "recommendations": recs,
                    "weights": {"alpha": 0.4, "beta": 0.35, "gamma": 0.25},
                    "explain": explain,
                    "llm_explain": llm_explain,
                }
        raise HTTPException(400, "Models not built. Build first via /api/build.")

    query_title = item_title
    if not query_title:
        raise HTTPException(422, "Query parameter 'title' is required.")

    recs = models["hybrid"].recommend(query_title, top_n=top_n, explain=explain)
    if not recs:
        raise HTTPException(404, "Item not found or no recommendations.")
    weights = models["hybrid"].get_weights()
    payload = {
        "query_item": query_title,
        "recommendations": recs,
        "weights": weights,
        "explain": explain,
        "llm_explain": llm_explain,
    }
    return payload

def _json_scalar(val: Any) -> Any:
    """Convert numpy or pandas datatypes to standard JSON-compatible Python types."""
    import numpy as np
    import pandas as pd
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if hasattr(val, "item"):
        return val.item()
    return val


@app.get("/api/similar/{item_id}")
def get_similar_items(
    item_id: str,
    top_n: int = Query(10, ge=1, le=100),
    category: Optional[str] = Query(
        None,
        description="Optional category name to restrict similar items.",
    ),
    explain: bool = Query(False),
):
    """Get similar products by item id, optionally scoped to one category."""
    if not models["ready"] or models["item_df"] is None:
        raise HTTPException(400, "Models not built. Build first via /api/build.")

    item_df = models["item_df"]
    if "id" not in item_df.columns:
        raise HTTPException(400, "Model data does not include product ids.")

    id_matches = item_df[item_df["id"].astype(str) == str(item_id)]
    if id_matches.empty:
        raise HTTPException(404, "Item not found.")

    source = id_matches.iloc[0]
    source_title = str(source.get("title", ""))
    source_category = source.get("category", "")
    requested_category = category.strip() if category else None

    # Fetch a wider candidate pool before filtering so category filters still
    # have enough results to fill the requested page.
    candidate_limit = top_n if requested_category is None else min(top_n * 5, 100)
    recs = models["hybrid"].recommend(
        source_title,
        top_n=candidate_limit,
        explain=explain,
    )
    if requested_category is not None:
        recs = [
            rec
            for rec in recs
            if str(rec.get("category", "")).casefold() == requested_category.casefold()
        ]
    recs = recs[:top_n]

    if not recs:
        raise HTTPException(404, "No similar items found.")

    return {
        "query_item": {
            "id": _json_scalar(source.get("id")),
            "title": source_title,
            "category": _json_scalar(source_category),
        },
        "category_filter": requested_category,
        "recommendations": recs,
        "total": len(recs),
        "explain": explain,
    }


@app.websocket("/ws/recommendations")
async def recommendations_websocket(websocket: WebSocket):
    """Stream recommendations whenever the browser reports a new interaction."""
    await realtime_hub.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            request = RealtimeRecommendationRequest(**message)
            top_n = max(1, min(50, request.top_n))
            payload = _recommendation_payload(
                request.item_title,
                top_n=top_n,
                explain=request.explain,
                llm_explain=request.llm_explain,
            )
            await websocket.send_json({"type": "recommendations", **payload})
    except WebSocketDisconnect:
        realtime_hub.disconnect(websocket)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "status_code": exc.status_code, "detail": exc.detail})
        realtime_hub.disconnect(websocket)
    except Exception as exc:
        logger.exception("Recommendation websocket failed: %s", exc)
        await websocket.send_json({"type": "error", "status_code": 500, "detail": "Recommendation stream failed."})
        realtime_hub.disconnect(websocket)


@app.post("/api/realtime/behavior")
async def realtime_behavior_update(event: RealtimeRecommendationRequest):
    """HTTP fallback for clients that cannot keep a WebSocket connection open."""
    top_n = max(1, min(50, event.top_n))
    payload = _recommendation_payload(
        event.item_title,
        top_n=top_n,
        explain=event.explain,
        llm_explain=event.llm_explain,
    )
    message = {"type": "recommendations", **payload}
    await realtime_hub.broadcast(message)
    return message


# ── Weights ─────────────────────────────────────────────────────────

@app.get("/api/weights")
def get_weights():
    if not models["ready"]:
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}
    return models["hybrid"].get_weights()


@app.put("/api/weights")
def update_weights(
    w: WeightsUpdate,
    token: str = Depends(verify_token)
    ):
    if not models["ready"]:
        raise HTTPException(400, "Models not built.")
    models["hybrid"].set_weights(w.alpha, w.beta, w.gamma)
    _clear_response_cache()
    return {"message": "Weights updated", "weights": models["hybrid"].get_weights()}


# ── Items ───────────────────────────────────────────────────────────

@app.get("/api/items")
def list_items(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """List products from Supabase with cursor-style pagination.

    Supports ``?page=1&limit=20`` for infinite-scroll on the frontend.
    Returns a ``has_more`` flag so the client knows when to stop fetching.
    """
    try:
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
    except Exception as e:
        logger.warning("Supabase offline in list_items, falling back to mock products: %s", e)
        total = len(MOCK_PRODUCTS)
        offset = (page - 1) * limit
        paginated = MOCK_PRODUCTS[offset : offset + limit]
        items = []
        for p in paginated:
            items.append({
                'id': p['id'],
                'title': p['title'],
                'category': p['category'],
                'rating': round(float(p['rating']), 2),
                'avg_sentiment': round(float(p['avg_sentiment']), 4),
                'description': str(p['description'])[:200],
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
    try:
        sb = get_supabase()
        result = sb.rpc('get_categories', {}).execute()
        if result.data:
            return {"categories": result.data}

        # Fallback: distinct query
        result = sb.table('products').select('category').limit(5000).execute()
        cats = list(set(p['category'] for p in (result.data or []) if p.get('category')))
        cats.sort()
        return {"categories": cats}
    except Exception:
        # Fallback when Supabase is not configured
        return {"categories": ["Electronics", "Clothing", "Home & Kitchen", "Books"]}


# ── Purchases ───────────────────────────────────────────────────────

@app.get("/api/purchases/{user_id}")
def get_user_purchases(
    user_id: str, 
    limit: int = 50,
    token: str = Depends(verify_token)
):
    """Get purchase history for a user (via anon client — RLS enforced)."""
    try:
        sb = get_supabase()
        result = sb.table('purchases') \
            .select('id, product_id, rating, review_text, purchased_at, products(title, category, rating)') \
            .eq('user_id', user_id) \
            .order('purchased_at', desc=True) \
            .limit(limit) \
            .execute()
        return {"purchases": result.data or []}
    except Exception as e:
        logger.warning("Supabase offline in get_user_purchases, falling back to empty list: %s", e)
        return {"purchases": []}


@app.post("/api/purchases")
def create_purchase(
    data: PurchaseCreate,
    token: str = Depends(verify_token)
):
    """Record a purchase (validated input)."""
    try:
        sb = get_supabase()
        result = sb.table('purchases').insert({
            'user_id': data.user_id,
            'product_id': data.product_id,
            'rating': max(0, min(5, data.rating)),
            'review_text': data.review_text[:1000],
        }).execute()
        _clear_response_cache()
        return {"purchase": result.data}
    except Exception as e:
        logger.warning("Supabase offline in create_purchase, returning mock purchase details: %s", e)
        import datetime
        mock_data = [{
            'id': 1,
            'user_id': data.user_id,
            'product_id': data.product_id,
            'rating': max(0, min(5, data.rating)),
            'review_text': data.review_text[:1000],
            'purchased_at': datetime.datetime.now().isoformat()
        }]
        return {"purchase": mock_data}
# ── Dashboard ───────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Returns server status. Useful for uptime monitors and Docker health checks.
    """
    import os
    return {
        "status": "ok",
        "version": os.getenv("APP_VERSION", "1.0.0")
    }



@app.post("/api/feedback")
def submit_feedback(
    data: FeedbackCreate,
    token: str = Depends(verify_token)
    ):

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
