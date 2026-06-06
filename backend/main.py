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
from pydantic import BaseModel
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
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
    Low‑overhead health check endpoint for component tracking.
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
        raise HTTPException(status_code=500, detail=str(e))
