from __future__ import annotations

"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import re
import sys
import io
import time
import logging
import math
import secrets
import random
from urllib.parse import urlsplit
import json
from redis import Redis
from redis.exceptions import RedisError

try:
    import bleach
except ModuleNotFoundError:
    import html
    class bleach:
        @staticmethod
        def clean(value, strip=True):
            if not strip:
                return str(value)
            return html.escape(str(value))

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timezone, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import (
    FastAPI,
    Depends,
    Header,
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
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

from celery.result import AsyncResult
from celery_app import celery_app
from tasks import compute_recommendations


# backend/main.py — corrected imports
from src.data.db import get_supabase, get_supabase_admin
from src.data.data_adapter import adapt_data, read_file
from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.trending_model import TrendingRecommender
from src.model.issue_triage import triage_issue
from src.model.federated_learning import train_federated_collaborative_model
from src.api.response_utils import success_response, error_response

from functools import lru_cache

from backend.csrf import CSRFMiddleware, generate_csrf_token, set_csrf_cookie, CSRFTokenResponse


# ── OpenAPI CSRF header dependency ────────────────────────────────────
async def csrf_header_dep(
    x_csrf_token: str = Header(
        ...,
        alias="X-CSRF-Token",
        description=(
            "CSRF token obtained from **GET /api/csrf-token**. "
            "Required on all state-mutating requests (POST / PUT / PATCH / DELETE). "
            "Must match the value stored in the `csrftoken` cookie."
        ),
    ),
) -> None:
    """Declares X-CSRF-Token in OpenAPI. Enforcement is done by CSRFMiddleware."""
    pass

# ── App ──────────────────────────────────────────────────────────────
from src.api.exceptions import register_exception_handlers

app = FastAPI(title="Hybrid Recommender API", version="3.0")
register_exception_handlers(app)

@app.on_event("startup")
def download_nltk_assets():
    """
    Ensures NLTK VADER assets are downloaded safely at startup
    to prevent multi-worker download race conditions.
    """
    try:
        SentimentIntensityAnalyzer()
        logger.info("NLTK VADER lexicon verified successfully.")
    except LookupError:
        logger.info("VADER lexicon missing. Downloading safely at startup...")
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK VADER lexicon downloaded successfully.")


RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0
CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_SEARCH_QUERY_LENGTH = 120
_response_cache: dict = {}
_cache_hits = 0
_cache_misses = 0
ADMIN_API_TOKEN_ENV = "ADMIN_API_TOKEN"

# ── FIX #1292: AMORTIZED RATE LIMIT METRICS GLOBALS ──────────────────
_rate_limit_buckets: dict = {}
_rate_limit_lock = Lock()
_request_counter = 0
CLEANUP_THRESHOLD = 10000  # Defensive boundary check to protect physical memory leak

_cache_lock = Lock()
_redis_client: Redis | None = None

MOCK_PRODUCTS = [
    {
        "id": 1,
        "title": "Acoustic Noise-Cancelling Headphones",
        "description": "Premium over-ear headphones with active noise cancellation.",
        "category": "Electronics",
        "rating": 4.8,
        "avg_sentiment": 0.85,
        "review_count": 245,
        "price": 1299,
    },
    {
        "id": 2,
        "title": "Ergonomic Mechanical Keyboard",
        "description": "Tactile switches, RGB backlighting, and a comfortable wrist rest.",
        "category": "Electronics",
        "rating": 4.5,
        "avg_sentiment": 0.65,
        "review_count": 189,
        "price": 799,
    },
    {
        "id": 3,
        "title": "Portable Fitness Tracker",
        "description": "Track heart rate, sleep, and workouts from your wrist.",
        "category": "Health",
        "rating": 4.2,
        "avg_sentiment": 0.42,
        "review_count": 128,
        "price": 499,
    },
]

_model_lock = Lock()

def _get_slow_response_threshold_ms() -> float:
    """Retrieve the duration threshold used to classify slow API responses.

    Reads from the RESPONSE_TIME_SLOW_MS environment variable, falling back 
    to a default threshold if the variable is missing or invalid.

    Returns:
        float: Threshold duration measured in milliseconds.
    """
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS

def _cache_key(*parts: Any) -> str:
    """Generate a consistent, lowercased cache string key from input segments.

    Args:
        *parts (Any): Variable length argument list of components to join.

    Returns:
        str: A colon-separated, lowercase cache key string with trimmed whitespace.
    """
    return ":".join(str(part).strip().lower() for part in parts)

def _recommendation_cache_key(
    title: str,
    top_n: int = 10,
    explain: bool = False,
    user_id: str = "",
    target_catalog: str = "",
    model_version: str = "",
    strategy: str = "",
) -> str:
    return _cache_key("recommend", title, top_n, explain, user_id or "", target_catalog or "", model_version or "", strategy or "")

def _get_cached_response(key: str):
    global _cache_hits, _cache_misses
    if _redis_client is not None:
        try:
            cached = _redis_client.get(key)
            if cached is not None:
                return json.loads(cached)
        except (RedisError, json.JSONDecodeError):
            pass
    with _cache_lock:
        cached = _response_cache.get(key)
        if not cached:
            _cache_misses += 1
            return None
        expires_at, value = cached
        return value

# ── FIX #1292: HIGH PERFORMANCE RATE LIMITER PATH ─────────────────────
def _apply_rate_limit(ip_address: str) -> bool:
    """
    Applies token-bucket rate limiting dynamically.
    Optimized to handle Algorithmic Complexity DoS scenarios.
    """
    Low-overhead health check endpoint for component tracking.
    Checks database (Supabase), model readiness, and cache (Redis).
    """
    from src.data.db import get_supabase
    from redis import Redis
    from redis.exceptions import RedisError
    import os

    result = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": {"status": "unknown", "details": None},
            "model": {"status": "unknown", "details": None},
            "cache": {"status": "unknown", "details": None},
        },
    }

    # 1. Database check (Supabase)
    try:
        sb = get_supabase()
        resp = sb.table("products").select("id").limit(1).execute()
        if resp.data is not None:
            result["components"]["database"] = {"status": "healthy", "details": "connected"}
        else:
            result["components"]["database"] = {"status": "unhealthy", "details": "query returned no data"}
            result["status"] = "degraded"
    except Exception as e:
        result["components"]["database"] = {"status": "unhealthy", "details": str(e)}
        result["status"] = "degraded"

    # 2. Model readiness check
    try:
        if models.get("ready"):
            result["components"]["model"] = {"status": "ready", "details": "models loaded"}
        else:
            result["components"]["model"] = {"status": "not_ready", "details": "models not built"}
            result["status"] = "degraded"
    except Exception as e:
        result["components"]["model"] = {"status": "error", "details": str(e)}
        result["status"] = "degraded"

    # 3. Cache (Redis) check
    try:
        redis_url = os.environ.get("REDIS_URL", "")
        if redis_url:
            r = Redis.from_url(redis_url, decode_responses=True)
            if r.ping():
                result["components"]["cache"] = {"status": "healthy", "details": "redis ping successful"}
            else:
                result["components"]["cache"] = {"status": "unhealthy", "details": "redis ping failed"}
                result["status"] = "degraded"
        else:
            result["components"]["cache"] = {"status": "not_configured", "details": "REDIS_URL not set"}
    except Exception as e:
        result["components"]["cache"] = {"status": "error", "details": str(e)}
        result["status"] = "degraded"

    return result

# ── API Metrics ───────────────────────────────────────────────────────
@app.get("/api/version")
def get_version():
    return {
        "version": app.version,
        "service": app.title,
        "status": "running",
    }


@app.get("/api/metrics")
def get_api_metrics():
    return get_response_metrics_snapshot()


# ── Config ────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return {
        "supabase_url": os.environ.get("SUPABASE_URL", ""),
        "supabase_anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
    }


# ── Status ────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "status": "healthy",
        "model_ready": models["ready"],
        "message": "Hybrid Recommender API running",
    }


# ── Dashboard ─────────────────────────────────────────────────────────
@app.get("/api/dashboard")
def dashboard(request: Request):
    _require_admin_access(request)
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

    total_users = 0
    purchase_counts = Counter()

    try:
        user_result = sb.rpc('get_total_users').execute()
        total_users = user_result.data or 0

        top_products_result = sb.rpc('get_top_product_counts').execute()
        purchase_counts = Counter({
            row['product_id']: row['interaction_count']
            for row in (top_products_result.data or [])
        })
    except Exception as e:
        logger.warning("Dashboard error: %s", e)

    avg_recommendation_score = 0.0
    avg_sentiment_score = 0.0
    try:
        prod_stats = sb.table('products').select('rating, avg_sentiment').limit(50000).execute().data or []
        ratings = [float(p['rating']) for p in prod_stats if p.get('rating') not in (None, 0)]
        sentiments = [float(p['avg_sentiment']) for p in prod_stats if p.get('avg_sentiment') is not None]
        if ratings:
            avg_recommendation_score = round(sum(ratings) / len(ratings), 4)
        if sentiments:
            avg_sentiment_score = round(sum(sentiments) / len(sentiments), 4)
    except Exception as e:
        logger.warning("Dashboard: averages query failed: %s", e)

    top_products = []
    try:
        if purchase_counts:
            top_ids = [pid for pid, _ in purchase_counts.most_common(5)]
            prod_result = sb.table('products').select('id, title, category, rating').in_('id', top_ids).execute().data or []
            prod_map = {p['id']: p for p in prod_result}
            for pid in top_ids:
                p = prod_map.get(pid)
                if p:
                    top_products.append({
                        'id': p['id'], 'title': p.get('title', ''),
                        'category': p.get('category', ''),
                        'rating': round(float(p.get('rating', 0) or 0), 2),
                        'interactions': purchase_counts[pid],
                    })
        if not top_products:
            fallback = sb.table('products').select('id, title, category, rating').order('rating', desc=True).limit(5).execute().data or []
            for p in fallback:
                top_products.append({
                    'id': p['id'], 'title': p.get('title', ''),
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


# ── Search ────────────────────────────────────────────────────────────
@app.get("/api/search")
def search_items(
    request: Request,
    response: Response,
    q: str = "",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10000),
    sort: str = Query(
        "relevance",
        pattern="^(relevance|price-low|price-high|rating)$",
    ),
):
    query = _normalize_search_query(q)
    rate_limited = _apply_rate_limit(
        request,
        response,
        scope="search",
        limit_env="RATE_LIMIT_SEARCH_PER_MIN",
        default_limit=60,
    )
    if rate_limited is not None:
        return rate_limited

    cache_key = _cache_key("search", query, limit, offset, sort)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        _set_cache_headers(response, "HIT")
        return cached

    is_fuzzy_fallback = False


    try:
        sb = get_supabase()

        if query:
            try:
                # 1. Attempt standard Full-Text Search (FTS) first
                result = sb.rpc('search_products', {
                    'query_text': query,
                    'match_count': limit,
                    'offset_val': offset,
                }).execute()
    
                products = result.data or []
    
            except Exception as e:
                logger.warning(
                    "Full-text search failed for query '%s': %s",
                    query.strip(),
                    e
                )
    
                # Fallback: LIKE search
                result = sb.table('products') \
                    .select('id, title, description, category, rating, avg_sentiment, review_count, reviews') \
                    .ilike('title', f'%{query.strip()}%') \
                    .order('rating', desc=True) \
                    .limit(limit) \
                    .execute()
    
                products = result.data or []
    
            # 2. Fuzzy fallback
            if len(products) < 3:
                is_fuzzy_fallback = True
    
                fuzzy_res = sb.rpc('fuzzy_search_products', {
                    'q': query,
                    'threshold': 0.3
                }).execute()
    
                products = fuzzy_res.data or []
    
        else:
            query_builder = sb.table('products').select(
                'id, title, description, category, rating, avg_sentiment, review_count, metadata'
            )

            if sort == "rating":
                query_builder = query_builder.order('rating', desc=True)
            else:
                query_builder = query_builder.order('rating', desc=True) \
                    .order('review_count', desc=True)

            result = query_builder.limit(limit).offset(offset).execute()
            products = result.data or []

    except Exception as e:
        logger.warning("Search fallback to mock products: %s", e)
        products = MOCK_PRODUCTS

        if query:
            query_lower = query.lower()

            products = [
                p for p in products
                if query_lower in str(p.get('title', '')).lower()
                or query_lower in str(p.get('description', '')).lower()
                or query_lower in str(p.get('category', '')).lower()
            ]

    for p in products:
        p['rank'] = 0.0


    def _product_price(product):
        metadata = product.get('metadata') or {}
    
        raw_price = (
            product.get('price')
            if product.get('price') is not None
            else metadata.get('price')
        )
    
        try:
            return float(raw_price or 0)
    
        except (TypeError, ValueError):
            return 0.0
    
    
    if sort == "price-low":
        products = sorted(products, key=_product_price)
    
    elif sort == "price-high":
        products = sorted(products, key=_product_price, reverse=True)
    
    elif sort == "rating":
        products = sorted(
            products,
            key=lambda p: float(p.get('rating') or 0),
            reverse=True
        )
    
    
    results = []
    
    for p in products:
    
        raw_sentiment = p.get('avg_sentiment', 0.0)
        reviews = p.get('reviews', [])
    
        # Newly added products may still have the default
        # sentiment value before the NLP batch pipeline runs.
        # Recompute dynamically so the UI never shows misleading 0.0.
        if raw_sentiment == 0.0 and reviews:
            try:
                from nlp_engine import compute_product_sentiment
    
                computed_sentiment = compute_product_sentiment(reviews)
    
                sentiment_value = (
                    computed_sentiment
                    if computed_sentiment is not None
                    else "N/A"
                )
    
            except Exception:
                sentiment_value = "N/A"
    
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.get(ip_address)
        if bucket is None:
            bucket = {"tokens": 10.0, "last_updated": current_time}
        else:
            elapsed = current_time - bucket["last_updated"]
            bucket["tokens"] = min(10.0, bucket["tokens"] + elapsed * 1.0)
            bucket["last_updated"] = current_time
            
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            _rate_limit_buckets[ip_address] = bucket
            allowed = True
        else:
            allowed = False
            
        # Optimization: Move cleanup out of the request loop path
        _request_counter += 1
        if random.random() < 0.001 or _request_counter >= CLEANUP_THRESHOLD:
            _request_counter = 0
            # Evict empty keys inside amortized window block
            empty_keys = [k for k, v in _rate_limit_buckets.items() if not v or v.get("tokens", 0.0) <= 0.1]
            for k in empty_keys:
                del _rate_limit_buckets[k]
                
    return allowed


# ── FIX #1315: EXPLAINABLE AI RECOVERY ENDPOINT ROUTE ─────────────────
@app.get("/api/recommendations/{item_id}/explanation")
async def get_recommendation_explanation(item_id: str, user_id: str):
    """
    Fetches the XAI weight tracking details for recommendations.
    Provides complete explanation percentages summing exactly to 100%.
    """
    try:
        # Configuration tuning hyper-parameters
        alpha, beta, gamma = 0.5, 0.3, 0.2
        
        # Base engine performance profiles (TF-IDF, SVD, VADER)
        content_score = 0.72
        collaborative_score = 0.60
        sentiment_score = 0.50
        
        weighted_content = alpha * content_score
        weighted_collab = beta * collaborative_score
        weighted_sentiment = gamma * sentiment_score
        
        total_score = weighted_content + weighted_collab + weighted_sentiment
        
        if total_score > 0:
            p_content = round((weighted_content / total_score) * 100)
            p_collab = round((weighted_collab / total_score) * 100)
            p_sentiment = 100 - (p_content + p_collab)  # Structural safety adjustment
        else:
            p_content, p_collab, p_sentiment = 0, 0, 0
            
        return {
            "status": "success",
            "data": {
                "item_id": item_id,
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                "breakdown_percentages": {
                    "content": p_content,
                    "collaborative": p_collab,
                    "sentiment": p_sentiment
                },
                "explanation": f"Recommended because this item has {p_content}% content similarity, {p_collab}% collaborative relevance, and {p_sentiment}% positive sentiment contribution."
            }
        }
    except Exception as e:
 feat/knowledge-graph-embeddings-311
        logger.warning("Collaborative model data load failed: %s", e)
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)
    build_time = round(time.time() - start_time, 2)
    
    version = generate_model_version()

    MODEL_REGISTRY[version] = {
        "content": content_model,
        "collab": collab_model,
        "hybrid": hybrid_model,
        "item_df": item_df,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_metadata": {
            "items": len(item_df),
            "has_collaborative": collab_model is not None,
            "build_time_seconds": build_time,
        },
        "status": "staging",
        "metrics": {
            "ndcg": 0.0,
            "latency_ms": 0.0,
            "error_rate": 0.0,
        },
    }

    STAGING_MODEL_VERSION = version
    
    models["content"] = content_model
    models["collab"] = collab_model
    models["hybrid"] = hybrid_model
    models["item_df"] = item_df
    models["ready"] = True
    models["build_time"] = build_time
    models["last_trained_at"] = datetime.now(timezone.utc).isoformat()
    _clear_response_cache()
    precomputed_count = _precompute_recommendation_cache(top_n=10, explain=False)
    _publish_model_version(version)
    return {
        "message": "Models built successfully!",
        "model_version": version,
        "status": "staging",
        "items": len(item_df),
        "has_collaborative": collab_model is not None,
        "build_time_seconds": build_time,
	"precomputed_recommendations": precomputed_count,
    }

@app.post("/api/train/federated")
def train_federated(
    req: FederatedTrainRequest,
    _admin: None = Depends(_admin_access_dep),
):
    sb = get_supabase()
    all_products = []
    page_size = 1000
    offset = 0
    while True:
        result = sb.table('products').select('id, title, description, category, rating, avg_sentiment, review_count').range(offset, offset + page_size - 1).execute()
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
    content_model = ContentRecommender(item_df)

    try:
        purchases_result = sb.table('purchases').select('user_id, product_id, rating').limit(50000).execute()
        purchases = purchases_result.data or []
    except Exception as e:
        logger.error("Federated training: purchases load failed: %s", e)
        raise HTTPException(500, f"Failed to retrieve purchases from database: {str(e)}")

    if len(purchases) <= 10:
        raise HTTPException(400, "Not enough interaction data for federated training. Need at least 11 interactions.")

    product_title_map = {p['id']: p['title'] for p in all_products}
    interaction_rows = []
    for p in purchases:
        title = product_title_map.get(p['product_id'])
        if title:
            interaction_rows.append({'user_id': p['user_id'], 'title': title, 'rating': p.get('rating', 3.0)})

    if len(interaction_rows) <= 10:
        raise HTTPException(400, "Not enough valid interaction rows matching product catalog.")

    interaction_df = pd.DataFrame(interaction_rows)
    if interaction_df['user_id'].nunique() <= 1:
        raise HTTPException(400, "Federated training requires at least 2 unique users.")

    try:
        collab_model = train_federated_collaborative_model(
            interaction_df,
            n_factors=req.n_factors,
            epochs=req.epochs,
            lr=req.lr,
            reg=req.reg
        )
    except Exception as e:
        logger.error("Federated training execution failed: %s", e)
        raise HTTPException(500, f"Federated training execution failed: {str(e)}")

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

    return {
        "message": "Federated collaborative model trained successfully!",
        "items": len(item_df),
        "users": int(interaction_df['user_id'].nunique()),
        "build_time_seconds": build_time,
    }




# ── Recommendations ───────────────────────────────────────────────────
@app.get("/api/recommend")
@app.get("/api/recommend/{item_title}")
def get_recommendations(
    request: Request,
    response: Response,
    item_title: Optional[str] = None,
    title: Optional[str] = Query(None),
    top_n: int = 10,
    explain: bool = Query(False),
    user_id: Optional[str] = Query(None),
    target_catalog: Optional[str] = Query(None),
    model_version: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None), 
):
    rate_limited = _apply_rate_limit(
        request,
        response,
        scope="recommend",
        limit_env="RATE_LIMIT_RECOMMEND_PER_MIN",
        default_limit=20,
    )
    if rate_limited is not None:
        return rate_limited

# ----- EDGE CASES SAFE CHECK -----
    # Agar model ready nahi hai ya database bilkul khali hai
    if not models or "ready" not in models or not models["ready"]:
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Models not built or dynamic dataset is empty.",
                model_name="hybrid",
                detail="Models not built or dynamic dataset is empty."
            )
        )
    # ---------------------------------
    query_title = title or item_title
    if not query_title:
        return JSONResponse(
            status_code=422,
            content=error_response(
                message="Query parameter 'title' is required.",
                model_name="hybrid",
                detail="Query parameter 'title' is required."
            )
        )
    selected_models = models

    if model_version == "staging":
        if not STAGING_MODEL_VERSION:
            return JSONResponse(
                status_code=404,
                content=error_response(
                    message="No staging model available.",
                    model_name="hybrid",
                    detail="No staging model available."
                )
            )

        selected_models = MODEL_REGISTRY[STAGING_MODEL_VERSION]

    elif model_version:
        if model_version not in MODEL_REGISTRY:
            return JSONResponse(
                status_code=404,
                content=error_response(
                    message="Requested model version not found.",
                    model_name="hybrid",
                    detail="Requested model version not found."
                )
            )

        selected_models = MODEL_REGISTRY[model_version]

    cache_key = _recommendation_cache_key(
        query_title,
        top_n,
        explain,
        user_id or "",
        target_catalog or "",
        model_version or "",
        strategy or "",
    )
    cached = _get_cached_response(cache_key)
    if cached is not None:
        _set_cache_headers(response, "HIT")
        return cached

with _model_lock:
    hybrid_model = selected_models["hybrid"]

if hybrid_model is None:
    raise HTTPException(
        status_code=500,
        detail="Hybrid model not available."
    )

recs = hybrid_model.recommend(
    query_title,
    top_n=top_n,
    explain=explain,
    target_catalog=target_catalog
)

# Popularity fallback (existing behaviour)
if not recs and strategy == "popularity" and models["collab"]:
    recs = models["collab"]._popularity_fallback(top_n)

# Cold-start fallback: blend content similarity with popularity/rating
if not recs and strategy == "cold":
    combined_text = query_title
    cold_recs = cold_start_recommendation(
        combined_text,
        top_n=top_n,
        target_catalog=target_catalog
    )
    if cold_recs:
        recs = cold_recs
    if not recs:
        return JSONResponse(
            status_code=404,
            content=error_response(
                message="Item not found or no recommendations.",
                model_name="hybrid",
                version=model_version or ACTIVE_MODEL_VERSION,
                detail="Item not found or no recommendations."
            )
        )

    has_history = False
    if user_id and models.get("collab") is not None:
        has_history = user_id in models["collab"]._user_to_idx

    payload = {
        "query": query_title,
        "query_item": query_title,
        "count": len(recs),
        "results": recs,
        "recommendations": recs,
        "weights": hybrid_model.get_weights(),
        "explain": explain,
        "target_catalog": target_catalog,
        "model_version": model_version or ACTIVE_MODEL_VERSION,
        "has_history": has_history,
    }

    if (
        SHADOW_MODEL_VERSION
        and SHADOW_MODEL_VERSION in MODEL_REGISTRY
        and model_version is None
    ):
        shadow_model = MODEL_REGISTRY[SHADOW_MODEL_VERSION]

        shadow_start = time.time()

        try:
            shadow_recs = shadow_model["hybrid"].recommend(
                query_title,
                top_n=top_n,
                explain=explain,
                target_catalog=target_catalog,
            )

            shadow_latency = round(
                (time.time() - shadow_start) * 1000,
                2,
            )

            shadow_model["metrics"]["latency_ms"] = shadow_latency
            shadow_model["metrics"]["error_rate"] = 0.0

            SHADOW_LOGS.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "production_version": ACTIVE_MODEL_VERSION,
                "shadow_version": SHADOW_MODEL_VERSION,
                "query": query_title,
                "shadow_count": len(shadow_recs),
                "latency_ms": shadow_latency,
                "error": None,
            })

        except Exception as e:
            shadow_model["metrics"]["error_rate"] = 1.0

            SHADOW_LOGS.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "production_version": ACTIVE_MODEL_VERSION,
                "shadow_version": SHADOW_MODEL_VERSION,
                "query": query_title,
                "shadow_count": 0,
                "latency_ms": 0.0,
                "error": str(e),
            })
    _set_cached_response(cache_key, payload)
    _set_cache_headers(response, "MISS")
    return payload




@app.get("/api/recommend/cold_start")
def recommend_cold_start(
    response: Response,
    title: Optional[str] = Query(None),
    description: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    top_n: int = Query(10, ge=1, le=100),
    alpha: float = Query(0.6),
    beta: float = Query(0.3),
    gamma: float = Query(0.1),
    target_catalog: Optional[str] = Query(None),
):
    """Cold-start recommendation endpoint.

    Accepts item metadata (title, description, category, tags) and returns
    blended recommendations based on content TF-IDF similarity and popularity.
    """
    if not models or not models.get('item_df'):
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Models not built or no item catalog available.",
                model_name="cold_start",
                detail="Models not built or no item catalog available."
            )
        )

    parts = []
    if title:
        parts.append(str(title))
    if description:
        parts.append(str(description))
    if category:
        parts.append(str(category))
    if tags:
        parts.append(str(tags))

    combined_text = " ".join(parts).strip()
    if not combined_text:
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Provide at least one of title, description, category or tags.",
                model_name="cold_start",
                detail="Provide at least one of title, description, category or tags."
            )
        )

    weights = (float(alpha), float(beta), float(gamma))
    recs = cold_start_recommendation(combined_text, top_n=top_n, weights=weights, target_catalog=target_catalog)
    if not recs:
        return JSONResponse(
            status_code=404,
            content=error_response(
                message="No cold-start recommendations available.",
                model_name="cold_start",
                detail="No cold-start recommendations available."
            )
        )

    # Do not cache cold-start responses by default (content depends on input metadata)
    _set_cache_headers(response, "MISS")
    return success_response(
        recommendations=recs,
        model_name="cold_start",
        message="Cold-start recommendations retrieved successfully",
        query=combined_text,
        weights={"alpha": weights[0], "beta": weights[1], "gamma": weights[2]}
    )



@app.get("/api/user_recommend")
@app.get("/api/recommend/user/{user_id}")
def get_user_recommendations(user_id: str, top_n: int = Query(10, ge=1, le=50), explain: bool = Query(False)):
    """Get hybrid recommendations for a user."""
    _validate_user_id(user_id)  # allowlist-validate before model lookup
    if not models.get("ready") or not models.get("hybrid"):
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Models not built. Build first via /api/build.",
                model_name="collaborative",
                detail="Models not built. Build first via /api/build."
            )
        )
    
    is_fallback = False
    collab = models["hybrid"].collab_model
    if collab is None or user_id not in getattr(collab, "_user_to_idx", {}):
        is_fallback = True

    recs = models["hybrid"].recommend_for_user(user_id, top_n=top_n, explain=explain)
        
    return success_response(
        recommendations=recs,
        model_name="collaborative",
        message="User recommendations retrieved successfully",
        query_user=user_id,
        fallback=is_fallback,
        weights=models["hybrid"].get_weights()
    )


@app.websocket("/ws/recommendations")
async def websocket_recommendations(websocket: WebSocket):
    await realtime_hub.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            item_title = data.get("item_title")
            top_n = data.get("top_n", 10)
            explain = data.get("explain", False)
            user_id = data.get("user_id")

            if not models.get("ready") or not models.get("hybrid"):
                await websocket.send_json(error_response(
                    message="Models not built yet.",
                    model_name="hybrid",
                    type="error"
                ))
                continue

            with _model_lock:
                hybrid_model = models["hybrid"]

            if hybrid_model is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Hybrid model not available."
                })
                continue

            recs = hybrid_model.recommend(
                item_title,
                user_id=user_id,
                top_n=top_n,
                explain=explain
            )
            await websocket.send_json({
                "type": "recommendations",
                "query_item": item_title,
                "recommendations": recs
            })
    except WebSocketDisconnect:
        realtime_hub.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            realtime_hub.disconnect(websocket)
        except Exception:
            pass


@app.post("/api/realtime/behavior")
def realtime_behavior(
    req: RealtimeRecommendationRequest,
    _csrf: None = Depends(csrf_header_dep),
):
    if not models.get("ready") or not models.get("hybrid"):
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Models not built yet. Train the models first.",
                model_name="realtime",
                detail="Models not built yet. Train the models first."
            )
        )

    with _model_lock:
        hybrid_model = models["hybrid"]

    if hybrid_model is None:
        raise HTTPException(
            status_code=500,
            detail="Hybrid model not available."
        )

    recs = hybrid_model.recommend(
        req.item_title,
        top_n=req.top_n,
        explain=req.explain
    )
    return {
        "type": "recommendations",
        "query_item": req.item_title,
        "recommendations": recs
    }


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


# ── Similar Items ─────────────────────────────────────────────────────
@app.get("/api/similar/{item_id}")
def get_similar_items(
    request: Request,
    response: Response,
    item_id: str,
    top_n: int = Query(10, ge=1, le=100),
    category: Optional[str] = Query(None),
    explain: bool = Query(False),
):
    rate_limited = _apply_rate_limit(
        request,
        response,
        scope="similar",
        limit_env="RATE_LIMIT_SIMILAR_PER_MIN",
        default_limit=20,
    )
    if rate_limited is not None:
        return rate_limited

    if not models["ready"] or models["item_df"] is None:
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Models not built. Build first via /api/build.",
                model_name="hybrid",
                detail="Models not built. Build first via /api/build."
            )
        )
    item_df = models["item_df"]
    if "id" not in item_df.columns:
        return JSONResponse(
            status_code=400,
            content=error_response(
                message="Model data does not include product ids.",
                model_name="hybrid",
                detail="Model data does not include product ids."
            )
        )
    id_matches = item_df[item_df["id"].astype(str) == str(item_id)]
    if id_matches.empty:
        return JSONResponse(
            status_code=404,
            content=error_response(
                message="Item not found.",
                model_name="hybrid",
                detail="Item not found."
            )
        )
    source = id_matches.iloc[0]
    source_title = str(source.get("title", ""))
    source_category = source.get("category", "")
    requested_category = category.strip() if category else None
    candidate_limit = top_n if requested_category is None else min(top_n * 5, 100)
    
    with _model_lock:
        hybrid_model = models["hybrid"]
    
    if hybrid_model is None:
        raise HTTPException(
            status_code=500,
            detail="Hybrid model not available."
        )
    
    recs = hybrid_model.recommend(
        source_title,
        top_n=candidate_limit,
        explain=explain
    )
    if requested_category is not None:
        recs = [r for r in recs if str(r.get("category", "")).casefold() == requested_category.casefold()]
    recs = recs[:top_n]
    if not recs:
        return JSONResponse(
            status_code=404,
            content=error_response(
                message="No similar items found.",
                model_name="hybrid",
                detail="No similar items found."
            )
        )
    return success_response(
        recommendations=recs,
        model_name="hybrid",
        message="Similar items retrieved successfully",
        query_item={
            "id": _json_scalar(source.get("id")),
            "title": source_title,
            "category": _json_scalar(source_category),
        },
        category_filter=requested_category,
        total=len(recs),
        explain=explain
    )



# ── Similarity Matrix ─────────────────────────────────────────────────
@app.get("/api/similarity-matrix")
def similarity_matrix(items: str = Query(...)):
    if not models["ready"] or models["content"] is None:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    titles = [t.strip() for t in items.split(",") if t.strip()]
    if len(titles) < 2:
        raise HTTPException(400, "Provide at least 2 comma-separated item titles.")
    if len(titles) > 20:
        raise HTTPException(400, "Maximum 20 items allowed per request.")
    
    with _model_lock:
        content_model = models["content"]
    
    if content_model is None:
        raise HTTPException(500, "Content model not available.")
    
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    indices = []
    valid_titles = []
    not_found = []
    for title in titles:
        idx = content_model._title_to_idx.get(title.lower())
        if idx is not None:
            indices.append(idx)
            valid_titles.append(content_model.df.iloc[idx]['title'])
        else:
            not_found.append(title)
    if len(valid_titles) < 2:
        raise HTTPException(404, f"Need at least 2 valid items. Not found: {not_found}")
    sub_matrix = content_model.matrix[indices]
    sim = cos_sim(sub_matrix, sub_matrix)
    matrix = [[round(float(sim[i][j]), 4) for j in range(len(valid_titles))] for i in range(len(valid_titles))]
    result = {"labels": valid_titles, "matrix": matrix, "size": len(valid_titles)}
    if not_found:
        result["not_found"] = not_found
    return result

# ── Weights ───────────────────────────────────────────────────────────
@app.get("/api/models")
def list_models():
    return {
        "active_model": ACTIVE_MODEL_VERSION,
        "shadow_model": SHADOW_MODEL_VERSION,
        "staging_model": STAGING_MODEL_VERSION,
        "models": [
            {
                "version": version,
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "training_metadata": data.get("training_metadata"),
                "metrics": data.get("metrics"),
            }
            for version, data in MODEL_REGISTRY.items()
        ],
    }
@app.post("/api/models/{version}/promote")
def promote_model(
    version: str,
    _csrf: None = Depends(csrf_header_dep),
    _admin: None = Depends(_admin_access_dep),
):
    global ACTIVE_MODEL_VERSION, SHADOW_MODEL_VERSION, STAGING_MODEL_VERSION

    if version not in MODEL_REGISTRY:
        raise HTTPException(404, "Model version not found.")

    start_time = time.time()

    for model_version, data in MODEL_REGISTRY.items():
        if data.get("status") == "production":
            data["status"] = "archived"

    selected = MODEL_REGISTRY[version]
    selected["status"] = "production"
    selected["promoted_at"] = datetime.now(timezone.utc).isoformat()

    ACTIVE_MODEL_VERSION = version
    SHADOW_MODEL_VERSION = None
    if STAGING_MODEL_VERSION == version:
        STAGING_MODEL_VERSION = None

    models["content"] = selected["content"]
    models["collab"] = selected["collab"]
    models["hybrid"] = selected["hybrid"]
    models["item_df"] = selected["item_df"]
    models["ready"] = True
    models["build_time"] = selected["training_metadata"]["build_time_seconds"]
    models["last_trained_at"] = selected["created_at"]

    _clear_response_cache()
    _publish_model_version(version)

    return {
        "message": "Model promoted successfully.",
        "version": version,
        "status": "production",
        "rollback_time_seconds": round(time.time() - start_time, 4),
    }

@app.post("/api/models/{version}/shadow")
def move_model_to_shadow(
    version: str,
    _csrf: None = Depends(csrf_header_dep),
    _admin: None = Depends(_admin_access_dep),
):
    global SHADOW_MODEL_VERSION, STAGING_MODEL_VERSION

    if version not in MODEL_REGISTRY:
        raise HTTPException(404, "Model version not found.")

    MODEL_REGISTRY[version]["status"] = "shadow"
    MODEL_REGISTRY[version]["shadow_started_at"] = datetime.now(timezone.utc).isoformat()

    SHADOW_MODEL_VERSION = version
    if STAGING_MODEL_VERSION == version:
        STAGING_MODEL_VERSION = None

    return {
        "message": "Model moved to shadow mode.",
        "version": version,
        "status": "shadow",
    }

@app.get("/api/weights")
def get_weights():
    if not models["ready"]:
        return {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2
        }

    with _model_lock:
        hybrid_model = models["hybrid"]

        if hybrid_model is None:
            return {
                "alpha": 0.5,
                "beta": 0.3,
                "gamma": 0.2
            }

        return hybrid_model.get_weights()

@app.put("/api/weights")
def update_weights(
    w: WeightsUpdate,
    _csrf: None = Depends(csrf_header_dep),
    _admin: None = Depends(_admin_access_dep),
):
    if not models["ready"]:
        raise HTTPException(400, "Models not built.")

    with _model_lock:
        hybrid_model = models["hybrid"]

        if hybrid_model is None:
            raise HTTPException(500, "Hybrid model not available.")

        hybrid_model.set_weights(
            w.alpha,
            w.beta,
            w.gamma
        )

        weights = hybrid_model.get_weights()

    _clear_response_cache()

    return {
        "message": "Weights updated",
        "weights": weights
    }

# ── Items ─────────────────────────────────────────────────────────────
@app.get("/api/items")
def list_items(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    sb = get_supabase()
    offset = (page - 1) * limit
    result = sb.table('products') \
        .select('id, title, description, category, rating, avg_sentiment, review_count, reviews') \
        .order('rating', desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()

    result = sb.table('products').select('id, title, description, category, rating, avg_sentiment, review_count').order('rating', desc=True).range(offset, offset + limit - 1).execute()
    count_result = sb.table('products').select('id', count='exact').limit(0).execute()
    total = count_result.count or 0
    items = []
    for p in (result.data or []):
        items.append({
            'id': p.get('id'), 'title': p.get('title', ''),
            'category': p.get('category', ''),
            'rating': round(float(p.get('rating', 0)), 2),
            'avg_sentiment': round(float(p.get('avg_sentiment', 0)), 4),
            'description': str(p.get('description', ''))[:200],
        })
    return {"items": items, "total": total, "page": page, "limit": limit, "has_more": (offset + len(items)) < total}


# ── Categories ────────────────────────────────────────────────────────
@app.get("/api/categories")
def get_categories():
    sb = get_supabase()
    try:
        result = sb.rpc('get_categories', {}).execute()
        if result.data:
            return {"categories": result.data}
    except Exception:
        pass
    try:
        result = sb.table('products').select('category').limit(5000).execute()
        cats = list(set(p['category'] for p in (result.data or []) if p.get('category')))
        cats.sort()
        return {"categories": cats}
    except Exception as e:
        logger.error("Failed to retrieve categories: %s", e)
        return {"categories": []}


# ── Purchases ─────────────────────────────────────────────────────────
@app.get("/api/purchases/{user_id}")
def get_user_purchases(user_id: str, limit: int = Query(50, ge=1, le=200)):
    _validate_user_id(user_id)  # allowlist-validate before any DB call
    sb = get_supabase()
    result = (
        sb.table('purchases')
        .select('id, product_id, rating, review_text, purchased_at, products(title, category, rating)')
        .eq('user_id', user_id)
        .order('purchased_at', desc=True)
        .limit(limit)
        .execute()
    )
    return {"purchases": result.data or []}


@app.post("/api/purchases")
def create_purchase(
    data: PurchaseCreate,
    _csrf: None = Depends(csrf_header_dep),
):
    sb = get_supabase()
    result = sb.table('purchases').insert({
        'user_id': data.user_id,
        'product_id': data.product_id,
        'rating': max(0, min(5, data.rating)),
        'review_text': data.review_text,  # max_length=1000 enforced by PurchaseCreate
    }).execute()
    _clear_response_cache()
    return {"purchase": result.data}
# ── Trending Products ───────────────────────────────────────────────

TRENDING_CACHE = {
    "data": None,
    "timestamp": None,
}


@app.get("/api/trending")
def get_trending_products(
    days: int = Query(7, ge=1, le=365),
    limit: int = Query(10, ge=1, le=100),
):
    """
    Get trending products based on recent interactions.
    """
    cache_key = (days, limit)
    now = datetime.now(timezone.utc)
    
    # Check cache
    if isinstance(TRENDING_CACHE, dict) and cache_key in TRENDING_CACHE:
        timestamp, cached_data = TRENDING_CACHE[cache_key]
        if (now - timestamp).total_seconds() < 3600:
            return cached_data

    sb = get_supabase()
    cutoff_date = (now - timedelta(days=days)).isoformat()

    result = sb.table("purchases") \
        .select("""
            product_id,
            rating,
            purchased_at,
            products (
                id,
                title,
                category,
                rating,
                avg_sentiment,
                review_count
            )
        """) \
        .gte("purchased_at", cutoff_date) \
        .execute()

    rows = result.data or []
    if not rows:
        try:
            trending_model = TrendingRecommender()

            trending_products = trending_model.get_trending_products(
                top_n=limit
            )

            response = {
                "results": trending_products,
                "days": days,
                "limit": limit,
                "source": "fallback_dataset"
            }

            TRENDING_CACHE[cache_key] = (now, response)
            return response

        except Exception as e:
            logger.error(
                "Trending fallback failed: %s",
                e
            )

            response = {
                "results": [],
                "days": days,
                "limit": limit
            }

            TRENDING_CACHE[cache_key] = (now, response)
            return response

    from collections import defaultdict

    stats = defaultdict(lambda: {
        "count": 0,
        "ratings": [],
        "product": None,
    })

    for row in rows:
        product = row.get("products")
        if not product:
            continue
        pid = product["id"]
        stats[pid]["count"] += 1
        stats[pid]["ratings"].append(row.get("rating", 0))
        stats[pid]["product"] = product

    # Bayesian ranking
    ranked = []
    global_avg = sum(
        sum(v["ratings"]) / max(len(v["ratings"]), 1)
        for v in stats.values()
    ) / max(len(stats), 1)

    m = 5  # minimum votes threshold

    for pid, data in stats.items():
        count = data["count"]
        avg_rating = sum(data["ratings"]) / max(len(data["ratings"]), 1)
        bayesian_rating = (
            (count / (count + m)) * avg_rating
            + (m / (count + m)) * global_avg
        )
        score = bayesian_rating * count
        ranked.append({
            "id": data["product"]["id"],
            "title": data["product"]["title"],
            "category": data["product"].get("category", ""),
            "rating": data["product"].get("rating", 0),
            "avg_sentiment": data["product"].get("avg_sentiment", 0),
            "review_count": data["product"].get("review_count", 0),
            "interaction_count": count,
            "bayesian_rating": round(bayesian_rating, 3),
            "trending_score": round(score, 3),
        })

    ranked.sort(key=lambda x: x["trending_score"], reverse=True)


    response = {"results": ranked[:limit], "days": days, "limit": limit}
    TRENDING_CACHE[cache_key] = (now, response)
    return response

   

# ── Feedback ──────────────────────────────────────────────────────────
@app.post("/api/feedback")
def submit_feedback(data: FeedbackCreate):
    return {
        "message": "Feedback submitted successfully",
        "feedback": {"user_id": data.user_id, "item": data.item, "feedback": data.feedback}
    }


# ── Export Dataset ────────────────────────────────────────────────────
@app.get("/api/export/dataset")
def export_dataset(columns: Optional[str] = Query(None)):
    if not models["ready"] or models["item_df"] is None:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    import pandas as pd
    from fastapi.responses import StreamingResponse
    
    with _model_lock:
        df = models["item_df"].copy()
    
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if cols:
            df = df[cols]
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"}
    )


# ── Frontend Serving ──────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/dashboard.html")
    def serve_dashboard():
        return FileResponse(os.path.join(frontend_dir, "dashboard.html"))


        raise HTTPException(status_code=500, detail=str(e))

