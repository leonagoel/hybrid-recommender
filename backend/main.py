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

# ── FIX #1292: O(1) LRU RATE LIMIT METRICS GLOBALS ──────────────────
from collections import OrderedDict
_rate_limit_buckets = OrderedDict()
_rate_limit_lock = Lock()
MAX_RATE_LIMIT_IPS = 10000

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
    import time
    import random
    current_time = time.time()
    
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.get(ip_address)
        if bucket is None:
            bucket = {"tokens": 10.0, "last_updated": current_time}
            _rate_limit_buckets[ip_address] = bucket
        else:
            _rate_limit_buckets.move_to_end(ip_address)
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
            
        # Optimization: O(1) Eviction to prevent memory leak and Algorithmic Complexity DoS
        if len(_rate_limit_buckets) > MAX_RATE_LIMIT_IPS:
            _rate_limit_buckets.popitem(last=False)
        # Optimization: Move cleanup out of the request loop path
        global _request_counter
        _request_counter += 1
        if random.random() < 0.001 or _request_counter >= CLEANUP_THRESHOLD:
            _request_counter = 0
            # Evict empty keys inside amortized window block
            empty_keys = [k for k, v in _rate_limit_buckets.items() if not v or v.get("tokens", 0.0) <= 0.1]
            for k in empty_keys:
                del _rate_limit_buckets[k]
                
    return allowed
def _set_cached_response(key: str, value: Any) -> None:
    try:
        with _cache_lock:
            _response_cache[key] = (time.time() + CACHE_TTL_SECONDS, value)
    except (RedisError, TypeError):
        pass

def _clear_response_cache() -> None:
    with _cache_lock:
        _response_cache.clear()
        global _cache_hits, _cache_misses
        _cache_hits = 0
        _cache_misses = 0


@app.get("/api/cache_metrics")
def get_cache_metrics():
    """Expose simple cache hit/miss metrics and configured TTL."""
    return {
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "hits": int(_cache_hits),
        "misses": int(_cache_misses),
        "current_items": len(_response_cache),
    }


from backend.services.ml_service import _build_tfidf_for_items, cold_start_recommendation, _precompute_recommendation_cache


def _normalize_search_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    if len(normalized) > MAX_SEARCH_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Search query must be {MAX_SEARCH_QUERY_LENGTH} characters or fewer.",
        )
    return normalized


def _escape_like_pattern(value: str) -> str:
    """Escape special LIKE metacharacters to prevent pattern injection."""
    return (
        value
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.@]{1,128}$")


def _validate_user_id(user_id: str) -> str:
    """Allowlist-validate user_id to block injection via path parameters."""
    if not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format.")
    return user_id


def _set_cache_headers(response: Response, status: str) -> None:
    response.headers["Cache-Control"] = CACHE_CONTROL_VALUE
    response.headers["X-Cache"] = status


def _get_rate_limit(limit_env: str, default_limit: int) -> int:
    try:
        limit = int(os.environ.get(limit_env, str(default_limit)))
    except ValueError:
        return default_limit
    return max(1, limit)


def _rate_limit_exceeded_response(rate_limit: int, reset_time: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
        },
        headers={
            "x-ratelimit-limit": str(rate_limit),
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset_time),
        },
    )


def _apply_rate_limit(
    request: Request,
    response: Response,
    scope: str,
    limit_env: str,
    default_limit: int,
) -> JSONResponse | None:
    rate_limit = _get_rate_limit(limit_env, default_limit)
    client_ip = request.client.host if request.client else "127.0.0.1"
    bucket_key = (scope, client_ip)
    now = time.time()

    with _rate_limit_lock:
        timestamps = _rate_limit_buckets.setdefault(bucket_key, [])
        timestamps[:] = [timestamp for timestamp in timestamps if now - timestamp < 60]

        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

        if len(timestamps) >= rate_limit:
            return _rate_limit_exceeded_response(rate_limit, reset_time)

        timestamps.append(now)
        remaining = rate_limit - len(timestamps)
        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

        # Garbage Collection: Remove empty buckets to prevent memory leak
        empty_keys = [k for k, v in _rate_limit_buckets.items() if not v]
        for k in empty_keys:
            del _rate_limit_buckets[k]

    response.headers["x-ratelimit-limit"] = str(rate_limit)
    response.headers["x-ratelimit-remaining"] = str(remaining)
    response.headers["x-ratelimit-reset"] = str(reset_time)
    return None



def _extract_bearer_token(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _require_admin_access(request: Request) -> None:
    expected_token = os.environ.get(ADMIN_API_TOKEN_ENV, "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin token not configured.",
        )

    provided_token = (
        request.headers.get("x-admin-token", "").strip()
        or _extract_bearer_token(request.headers.get("authorization"))
    )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=401, detail="Admin token required.")
    def _admin_access_dep(request: Request):
        _require_admin_access(request)

_admin_access_dep = _require_admin_access



def _admin_access_dep(request: Request) -> None:
    """FastAPI dependency wrapper around _require_admin_access."""
    _require_admin_access(request)


def _admin_access_dep(request: Request) -> None:
    """FastAPI dependency wrapper around _require_admin_access."""
    _require_admin_access(request)


CORS_ORIGINS_ENV = "CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000")
ALLOWED_CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
ALLOWED_CORS_HEADERS = ["Accept", "Authorization", "Content-Type", "X-Admin-Token", "X-CSRF-Token"]


def _normalize_cors_origin(origin: str) -> str:
    normalized = origin.strip().rstrip("/")
    if not normalized:
        raise RuntimeError("CORS_ORIGINS cannot contain empty entries.")
    if normalized == "*":
        raise RuntimeError("CORS_ORIGINS must not include wildcard origin '*'.")

    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"Invalid CORS origin: {origin}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeError(f"Invalid CORS origin: {origin}")

    return f"{parsed.scheme}://{parsed.netloc}"


def _parse_cors_origins(raw_value: str | None = None) -> list[str]:
    configured_value = os.environ.get(CORS_ORIGINS_ENV, "") if raw_value is None else raw_value
    if not configured_value.strip():
        return list(DEFAULT_CORS_ORIGINS)

    origins = []
    seen = set()
    for raw_origin in configured_value.split(","):
        normalized_origin = _normalize_cors_origin(raw_origin)
        if normalized_origin not in seen:
            origins.append(normalized_origin)
            seen.add(normalized_origin)

    return origins


@app.on_event("startup")
def validate_cors_configuration() -> None:
    _parse_cors_origins()


# CORS
allowed_origins = _parse_cors_origins()

allow_creds = True
if "*" in allowed_origins:
    allow_creds = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*", "X-CSRF-Token"],
)

app.add_middleware(CSRFMiddleware)

# ── Response Time Monitoring ─────────────────────────────────────────
SLOW_RESPONSE_THRESHOLD_MS = 500.0
METRICS_SAMPLE_SIZE = 1000
response_time_samples = deque(maxlen=METRICS_SAMPLE_SIZE)
METRICS_WINDOW_SECONDS = 600
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
        response_time_samples.append(
          (time.time(), response_time_ms)
        )

        current_time = time.time()

        while (
          response_time_samples
          and current_time - response_time_samples[0][0] > METRICS_WINDOW_SECONDS
        ):
          response_time_samples.popleft()
    log_level = logging.WARNING if response_time_ms > SLOW_RESPONSE_THRESHOLD_MS else logging.INFO
    if log_level == logging.WARNING:
        logger.warning("API request slow endpoint=%s method=%s status=%s time=%.2fms response_time_ms=%.2f endpoint=%s",
                       endpoint, method, status_code, response_time_ms, response_time_ms, endpoint)
    else:
        logger.info("API request endpoint=%s method=%s status=%s time=%.2fms",
                    endpoint, method, status_code, response_time_ms)


def reset_response_metrics():
    with response_metrics_lock:
        response_metrics["total_requests"] = 0
        response_metrics["error_requests"] = 0
        response_time_samples.clear()


def get_response_metrics_snapshot():
    with response_metrics_lock:
        samples = [value for _, value in response_time_samples]
        total_requests = response_metrics["total_requests"]
        error_requests = response_metrics["error_requests"]
    avg_response_time = sum(samples) / len(samples) if samples else 0.0
    error_rate = (error_requests / total_requests) * 100 if total_requests else 0.0
    return {
        "avg_response_time": round(avg_response_time, 2),
        "p95_response_time": round(_percentile(samples, 95), 2),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 2),
    }


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
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        record_response_metric(request.url.path, request.method, status_code, response_time_ms)


# ── State ─────────────────────────────────────────────────────────────
models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
    "last_trained_at": None,
}

MODEL_REGISTRY = {}
ACTIVE_MODEL_VERSION = None
SHADOW_MODEL_VERSION = None
STAGING_MODEL_VERSION = None

SHADOW_LOGS = []

def generate_model_version():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"1.0.0-{timestamp}"


from backend.core.websockets import realtime_hub


class WeightsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2


class PurchaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    product_id: int = Field(..., gt=0)
    rating: float = Field(0.0, ge=0.0, le=5.0)
    review_text: str = Field("", max_length=1000)


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    item: str = Field(..., min_length=1, max_length=500)
    feedback: str = Field(..., min_length=1, max_length=2000)
    thumbs: str = Field(..., pattern=r"^(up|down)$")

class RealtimeRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_title: str
    top_n: int = 10
    explain: bool = False
    target_catalog: Optional[str] = None


# ── CSRF Token ───────────────────────────────────────────────────────
@app.get(
    "/api/csrf-token",
    response_model=CSRFTokenResponse,
    summary="Issue a CSRF token",
    tags=["Security"],
)
def get_csrf_token(response: Response):
    token = generate_csrf_token()
    set_csrf_cookie(response, token)
    return CSRFTokenResponse(csrfToken=token)


class FederatedTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_factors: int = 20
    epochs: int = 5
    lr: float = 0.05
    reg: float = 0.05


# ── Health ────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/api/health")
def health_check():
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
    
    
# ── FIX #1315: EXPLAINABLE AI RECOVERY ENDPOINT ROUTE ─────────────────
@app.get("/api/recommendations/{item_id}/explanation")
async def get_recommendation_explanation(item_id: str, user_id: str):
    """
    Fetches the XAI weight tracking details for recommendations.
    Provides complete explanation percentages summing exactly to 100%.
    """
    try:
        from backend.core.state import models, _model_lock
        
        if not models.get("ready") or not models.get("hybrid"):
            raise HTTPException(status_code=400, detail="Models not built yet.")
            
        with _model_lock:
            hybrid_model = models["hybrid"]
            weights = hybrid_model.get_weights()
            alpha, beta, gamma = weights['alpha'], weights['beta'], weights['gamma']
            
            item_df = models["item_df"]
            title = str(item_id)
            if item_df is not None and "id" in item_df.columns:
                matches = item_df[item_df["id"].astype(str) == str(item_id)]
                if not matches.empty:
                    title = matches.iloc[0]["title"]
                    
            recs = hybrid_model.recommend_for_user(user_id, top_n=50, explain=True)
            
            content_score = 0.0
            collaborative_score = 0.0
            sentiment_score = hybrid_model._sentiment_map.get(title, 0.5)
            
            for rec in recs:
                if rec['title'] == title:
                    content_score = rec.get('content_score', 0.0)
                    collaborative_score = rec.get('collab_score', 0.0)
                    sentiment_score = rec.get('sentiment_score', sentiment_score)
                    break
            
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
