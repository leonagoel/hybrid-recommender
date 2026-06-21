from __future__ import annotations
from fastapi import FastAPI # type: ignore
from backend.routers import recommend

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

import json
from urllib.parse import urlsplit
from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

try:
    import bleach
except ModuleNotFoundError:
    class bleach:
        @staticmethod
        def clean(value, strip=True):
            if not strip:
                return str(value)
            return re.sub(r"<[^>]*>", "", str(value))

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timezone, timedelta

from collections import defaultdict

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src", "data"))
sys.path.insert(0, os.path.join(_project_root, "src", "model"))

from fastapi import ( # type: ignore
    FastAPI,
    APIRouter,
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
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import FileResponse, JSONResponse # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Dict, List, Optional # type: ignore
from pydantic import BaseModel, ConfigDict, Field # type: ignore
from typing import Any, Optional
from dotenv import load_dotenv # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from db import get_supabase, get_supabase_admin
from backend.auth import _require_admin_access
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

from celery.result import AsyncResult # type: ignore
from celery_app import celery_app


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
from data_adapter import adapt_data, read_file
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender
from federated_learning import train_federated_collaborative_model
from issue_triage import triage_issue

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="3.0")

from src.evaluation.evaluation import run_evaluation

EVALUATION_HISTORY = deque(maxlen=20)

@app.get("/api/evaluate")
async def evaluate_models(
    k: int = 10,
    mode: str = "all",
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
):
    try:
        results = run_evaluation(
            k=k,
            mode=mode,
            weights={"alpha": alpha, "beta": beta, "gamma": gamma}
        )
        
        # Save to history
        run_record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "k": k,
            "mode": mode,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "results": results,
        }
        EVALUATION_HISTORY.appendleft(run_record)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluate/history")
async def evaluate_history(limit: int = 5):
    history = list(EVALUATION_HISTORY)[:limit]
    return {"runs": history}

# Register routers
app.include_router(recommend.router, prefix="/api")

# ── OpenAPI CSRF header dependency ────────────────────────────────────
# WHY a Depends() instead of just relying on the middleware?
#
# The CSRFMiddleware enforces the token at the ASGI level — it never
# touches the OpenAPI schema that FastAPI builds from route signatures.
# Swagger UI only renders parameters that appear in the schema, so the
# X-CSRF-Token field is invisible to users testing the API interactively.
#
# This dependency solves that purely at the documentation layer:
#   - It declares X-CSRF-Token as a required header parameter on every
#     route that includes Depends(csrf_header_dep).
#   - FastAPI adds it to the OpenAPI spec → Swagger UI renders the field.
#   - The function body does nothing (returns None) because the middleware
#     has already validated the token before the route handler runs.
#   - No double-validation, no logic duplication.
#
# The `alias="X-CSRF-Token"` preserves the canonical mixed-case header
# name in the OpenAPI spec so Swagger UI labels it correctly, even though
# Starlette lowercases all incoming headers internally.
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
    # The middleware has already validated the token before this runs.
    # This function exists solely to make the header visible in Swagger UI.

@app.on_event("startup")
def download_nltk_assets():
    """
    Ensures NLTK VADER assets are downloaded safely at startup.
    Gracefully intercepts infrastructure failures to activate degraded fallback loops.
    """
    global _model_degraded, _model_degraded_reason
    try:
        SentimentIntensityAnalyzer()
        logger.info("NLTK VADER lexicon verified successfully.")
    except Exception as e:
        logger.error(f"VADER framework load crash encountered. Triggering clean degraded fallback mode: {e}")
        _model_degraded = True
        _model_degraded_reason = f"NLTK VADER initialization failed: {str(e)}"


RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0
CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_SEARCH_QUERY_LENGTH = 120
MOCK_PRODUCTS = []
_response_cache: dict = {}
_cache_hits = 0
_cache_misses = 0
ADMIN_API_TOKEN_ENV = "ADMIN_API_TOKEN"

# ── FIX #1292: O(1) LRU RATE LIMIT METRICS GLOBALS ──────────────────
from collections import OrderedDict
_rate_limit_buckets = OrderedDict()
_rate_limit_lock = Lock()
MAX_RATE_LIMIT_IPS = 10000
CLEANUP_THRESHOLD = 10000   # run stale-bucket cleanup every N requests
_request_counter = 0

_cache_lock = Lock()
_model_lock = Lock()

# ── Redis client ──────────────────────────────────────────────────────
_redis_client = None

_redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
try:
    _redis_client = Redis.from_url(_redis_url, decode_responses=True, socket_connect_timeout=2)
    _redis_client.ping()
    logger.info("Redis connected at %s", _redis_url)
except Exception:
    _redis_client = None
    logger.warning("Redis unavailable at %s — falling back to in-memory cache.", _redis_url)


def csrf_header_dep(request: Request) -> None:
    """No-op dependency; actual CSRF validation is handled by CSRFMiddleware."""
    pass

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


def _get_slow_response_threshold_ms() -> float:
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS


def _cache_key(*parts: Any) -> str:
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
    """Single authoritative cache key for recommendation responses."""
    return _cache_key(
        "recommend",
        title,
        top_n,
        explain,
        user_id or "",
        target_catalog or "",
        model_version or "",
        strategy or "",
    )


def _get_cached_response(key: str):
    global _cache_hits, _cache_misses

    if _redis_client is not None:
        try:
            cached = _redis_client.get(key)
            if cached is not None:
                _cache_hits += 1
                return json.loads(cached)
        except (RedisError, json.JSONDecodeError):
            pass

    with _cache_lock:
        cached = _response_cache.get(key)

        if not cached:
            _cache_misses += 1
            return None
        cached = _response_cache.get(key)

        if not cached:
            _cache_misses += 1
            return None

        expires_at, value = cached

        if expires_at <= time.time():
            _response_cache.pop(key, None)
            _cache_misses += 1
            return None
        _cache_hits += 1
        return value
# ── FIX #1292: HIGH PERFORMANCE RATE LIMITER PATH ──


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

        global _request_counter
        _request_counter += 1
        if _request_counter >= CLEANUP_THRESHOLD:
            _request_counter = 0
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


class InteractionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, max_length=128)
    item_id: int = Field(..., gt=0)
    interaction_type: str = Field(
        ...,
        pattern=r"^(view|click|search)$"
    )


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

    redis_ok = False
    if _redis_client is not None:
        try:
            _redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    db_ok = False
    try:
        get_supabase()
        db_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": db_ok,
        "redis": redis_ok,
        "models_ready": models.get("ready", False),
    }

def _set_cached_response(key: str, value: Any) -> None:
    if _redis_client is not None:
        try:
            _redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
        except (RedisError, TypeError):
            pass

    with _cache_lock:
        _response_cache[key] = (
            time.time() + CACHE_TTL_SECONDS,
            value,
        )

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


def _build_tfidf_for_items(item_df):
    """Build and return a TF-IDF matrix and vectorizer for the given item_df."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = (item_df.get('combined') or item_df.get('title')).fillna('').astype(str).tolist()
    vec = TfidfVectorizer(max_features=16384, stop_words='english')
    matrix = vec.fit_transform(texts)
    return vec, matrix


def cold_start_recommendation(combined_text: str, top_n: int = 10, weights: tuple[float, float, float] = (0.6, 0.3, 0.1), target_catalog: Optional[str] = None):
    """Cold-start blending of content similarity (TF-IDF) and simple popularity/rating signals.

    Returns list of dicts with blended score and components.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    item_df = models.get('item_df')
    if item_df is None or item_df.empty:
        return []

    vec, matrix = _build_tfidf_for_items(item_df)
    try:
        qv = vec.transform([combined_text])
    except Exception:
        return []

    scores = cosine_similarity(qv, matrix).flatten()

    # Popularity normalization (review_count) and rating normalization
    review_counts = item_df.get('review_count', None)
    if review_counts is None or len(review_counts) == 0:
        pop_norm = np.zeros_like(scores)
    else:
        max_rc = float(max(1, int(review_counts.max())))
        pop_norm = (np.array(item_df.get('review_count').fillna(0).astype(float)) / max_rc)

    ratings = item_df.get('rating')
    if ratings is None or len(ratings) == 0:
        rating_norm = np.zeros_like(scores)
    else:
        rating_norm = (np.array(item_df.get('rating').fillna(0).astype(float)) / 5.0)

    alpha, beta, gamma = weights

    blended = alpha * scores + beta * pop_norm + gamma * rating_norm

    idxs = blended.argsort()[::-1]
    results = []
    seen = set()
    for idx in idxs:
        title = str(item_df.iloc[idx].get('title', ''))
        if not title or title in seen:
            continue
        if target_catalog and 'category' in item_df.columns:
            cat = str(item_df.iloc[idx].get('category', ''))
            if cat and cat.casefold() != target_catalog.casefold():
                continue
        seen.add(title)
        results.append({
            'title': title,
            'blended_score': float(blended[idx]),
            'content_score': float(scores[idx]),
            'popularity_score': float(pop_norm[idx]),
            'rating_norm': float(rating_norm[idx]),
        })
        if len(results) >= top_n:
            break

    return results

def _precompute_recommendation_cache(
    top_n: int = 10,
    explain: bool = False,
) -> int:
    if not models.get("ready") or models.get("item_df") is None:
        return 0

    count = 0
    item_df = models["item_df"]

    for title in item_df["title"].dropna().astype(str).unique():
        cache_key = _recommendation_cache_key(title, top_n, explain)

        recs = models["hybrid"].recommend(title, top_n=top_n, explain=explain)

        if not recs:
            continue

        payload = {
            "query_item": title,
            "recommendations": recs,
            "weights": models["hybrid"].get_weights(),
            "explain": explain,
            "target_catalog": None,
            "model_version": ACTIVE_MODEL_VERSION,
            "has_history": False,
            "cache_precomputed": True,
        }

        _set_cached_response(cache_key, payload)
        count += 1

    return count

# ── Search ────────────────────────────────────────────────────────────
def _normalize_search_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    if len(normalized) > MAX_SEARCH_QUERY_LENGTH:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Search query must be {MAX_SEARCH_QUERY_LENGTH} characters or fewer.")
    return normalized

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
    _placeholders = {"change-me-to-a-random-secret", "changeme_admin_token", "your-secret-admin-token-here"}
    if not expected_token or expected_token in _placeholders:
        # No real admin token configured — allow access (local dev mode)
        return

    provided_token = (
        request.headers.get("x-admin-token", "").strip()
        or _extract_bearer_token(request.headers.get("authorization"))
    )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=401, detail="Admin token required.")


def _admin_access_dep(request: Request) -> None:
    _require_admin_access(request)


def _get_feedback_storage_client():
    client = get_supabase_admin()
    if client is None:
        raise HTTPException(status_code=500, detail="Feedback storage is unavailable.")
    return client


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


class RealtimeConnectionHub:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

realtime_hub = RealtimeConnectionHub()

USER_INTERACTIONS = []


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

class InteractionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, max_length=128)
    item_id: int = Field(..., gt=0)
    interaction_type: str = Field(
        ...,
        pattern=r"^(view|click|search)$"
    )

class MergeHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    guest_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)


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
    product_count = 0
    try:
        sb = get_supabase()
        result = sb.table('products').select('id', count='exact').limit(0).execute()
        product_count = result.count or 0
    except Exception:
        pass
    return {
        "status": "healthy",
        "model_ready": models["ready"],
        "product_count": product_count,
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
                escaped_query = _escape_like_pattern(query.strip())
                result = sb.table('products') \
                    .select('id, title, description, category, rating, avg_sentiment, review_count, reviews') \
                    .ilike('title', f'%{escaped_query}%') \
                    .order('rating', desc=True) \
                    .limit(limit) \
                    .execute()
    
                products = result.data or []
    
            # 2. Fuzzy fallback
            if not products:
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
            if 'rank' not in p or p['rank'] is None:
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
        else:
            sentiment_value = (
                raw_sentiment
                if raw_sentiment != 0.0
                else "N/A"
            )

        results.append({
            'id': p.get('id'),
            'title': p.get('title'),
            'description': str(p.get('description',''))[:200],
            'category': p.get('category'),
            'price': _product_price(p),
            'rating': float(p.get('rating') or 0),
            'avg_sentiment': sentiment_value,
            'review_count': p.get('review_count', 0),
            'rank': p.get('rank', 0.0)
        })
  
    final_output = {
        'query': query,
        'limit': limit,
        'offset': offset,
        'total_found': len(results),
        'results': results
    }
    
    return final_output


# ── Feature: Paginated Recommendations ───────────────────────────────
@app.get("/api/recommend/paginated")
async def recommend_item(
    title: str = Query(..., min_length=1, description="Item title to base recommendations on"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    user_id: str = Query(default="", description="Optional user ID for personalised scoring"),
):
    """
    Returns paginated hybrid recommendations for the given item title.
    Supports limit/offset pagination with a pagination metadata block.
    """
    try:
        all_results: list[dict] = []

        # Attempt to use the hybrid model via Celery task
        try:
            from src.model.hybrid_model import HybridRecommender
            from src.model.content_model import ContentRecommender
            from src.model.collaborative_model import CollaborativeRecommender

            dm = getattr(sys.modules.get("__main__"), "_dataset_manager", None)
            if dm is None:
                from src.data.dataset_manager import DatasetManager
                dm = DatasetManager()
            item_df = getattr(dm, "_item_df", None) or getattr(dm, "item_df", None)
            interaction_df = getattr(dm, "_interaction_df", None) or getattr(dm, "interaction_df", None)

            if item_df is not None and not item_df.empty:
                content_model = ContentRecommender(item_df)
                
                import os
                from src.model.neural_collaborative_model import NeuralCollaborativeRecommender
                use_ncf = os.getenv("USE_NCF", "true").lower() == "true"
                
                if interaction_df is not None and not interaction_df.empty:
                    if use_ncf:
                        collab_model = NeuralCollaborativeRecommender(interaction_df)
                    else:
                        collab_model = CollaborativeRecommender(interaction_df)
                else:
                    collab_model = None
                
                hybrid = HybridRecommender(content_model, collab_model, item_df)
                all_results = hybrid.recommend(title=title, user_id=user_id or None, top_n=limit + offset)
        except Exception:
            logger.warning("Hybrid model unavailable, using fallback recommendations")

            # Fallback: return mock products sorted by title similarity
            if not all_results:
                query = title.lower()
                scored = []
                for p in MOCK_PRODUCTS:
                    score = sum(1 for w in query.split() if w in p["title"].lower())
                    if score > 0 or query in p["category"].lower():
                        scored.append({**p, "hybrid_score": score, "content_score": score, "collab_score": 0})
                scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
                all_results = scored if scored else [{
                    "title": p["title"],
                    "rating": p["rating"],
                    "hybrid_score": 0.5,
                    "content_score": 0.3,
                    "collab_score": 0.2,
                    "category": p["category"],
                    "review_count": p.get("review_count", 0),
                } for p in MOCK_PRODUCTS[:3]]

            total = len(all_results)
            paginated = all_results[offset:offset + limit]

            return {
                "recommendations": paginated,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if offset + limit < total else None,
                    "has_more": (offset + limit) < total,
                },
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations/{item_id}/explanation")
async def get_recommendation_explanation(item_id: str, user_id: str):
    """
    Return a score-level explanation for why item_id is recommended.

    Delegates to HybridRecommender.recommend(..., explain=True) and returns
    the explanation block attached to the first result whose title matches
    item_id.  If the models have not been built yet, or item_id is not found
    in the recommendation results, an appropriate HTTP error is raised.
    """
    query = _normalize_search_query(item_id)
    if not query:
        return {"results": [], "count": 0, "query": query}
    if not models.get("ready") or not models.get("hybrid"):
        raise HTTPException(status_code=400, detail="Models not built. Call /api/build first.")

    item_id = item_id.strip()
    if not item_id:
        raise HTTPException(status_code=422, detail="item_id must not be empty.")

    try:
        recs = models["hybrid"].recommend(
            title=item_id,
            user_id=user_id or None,
            top_n=10,
            explain=True,
        )
    except Exception as e:
        logger.error("Explanation generation failed for item '%s': %s", item_id, e)
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")

    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for item '{item_id}'.",
        )

    explanation_entry = next(
        (r for r in recs if r.get("explanation") is not None),
        None,
    )
    if explanation_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Explanation not available for item '{item_id}'.",
        )

    return {
        "item_id": item_id,
        "user_id": user_id or None,
        "explanation": explanation_entry["explanation"],
        "representative_recommendation": {
            k: v for k, v in explanation_entry.items() if k != "explanation"
        },
    }


def _validate_upload_bytes(filename: str, ext: str, contents: bytes) -> None:
    """Validate raw upload bytes: empty, size, binary, and content-type checks."""
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Uploaded file exceeds {MAX_UPLOAD_BYTES} bytes.")
    if b"\x00" in contents[:4096]:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be binary.")
    try:
        decoded = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 encoded.")

    stripped = decoded.strip()
    lowered_name = filename.lower()
    if ext == ".csv":
        lowered_sample = stripped[:128].lower()
        if lowered_sample.startswith(("{", "[", "<!doctype", "<html", "<?xml")):
            raise HTTPException(status_code=400, detail="CSV uploads must contain CSV content.")
        if not lowered_name.endswith(".csv"):
            raise HTTPException(status_code=400, detail="CSV uploads must use a .csv filename.")
    elif ext == ".json":
        if not (stripped.startswith("{") or stripped.startswith("[")):
            raise HTTPException(status_code=400, detail="JSON uploads must contain JSON content.")
        import json
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="JSON uploads must contain valid JSON.")


# ── Upload ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    admin=Depends(_require_admin_access),
    _csrf: None = Depends(csrf_header_dep),
):
    """Upload a CSV or JSON dataset and import into Supabase."""
    import math
    filename = file.filename or "data.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.csv', '.json'):
        raise HTTPException(400, "Only CSV and JSON files are supported.")
    try:
        contents = await file.read()
        _validate_upload_bytes(filename, ext, contents)
        buf = io.BytesIO(contents)
        raw_df = read_file(buf, file_format=ext.replace('.', ''))
        adapted_df, meta = adapt_data(raw_df)
        adapted_df = adapted_df.drop_duplicates(subset='title', keep='first')
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
# --- sanitize HTML tags ---
                title = bleach.clean(title, strip=True)[:500]

                description = str(row.get('description', ''))
                description = bleach.clean(description, strip=True)[:2000]

                rows.append({
                    'title': title,
                    'description': description,
                    'category': str(row.get('category', ''))[:200],
                    'rating': round(rating_val, 2),
                    'avg_sentiment': 0.0,
                    'review_count': 0,
                    'metadata': {},
                })
            if not rows:
                continue
            try:
                sb.table('products').upsert(rows, on_conflict='title', ignore_duplicates=True).execute()
                imported += len(rows)
            except Exception as e:
                errors.append(f"Batch {start}-{start+len(rows)}: {str(e)[:100]}")
        models["ready"] = False
        _clear_response_cache()
        result = {
            "message": f"Imported {imported:,} products from {filename}",
            "imported": imported, "total_rows": total,
            "meta": {"has_user_data": meta['has_user_data'], "has_reviews": meta['has_reviews']},
        }
        if errors:
            result["warnings"] = errors[:5]
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed for %s: %s", filename, e, exc_info=True)
        raise HTTPException(400, "Upload failed. Check file format and try again.")


# ── Build Models ──────────────────────────────────────────────────────
@app.post("/api/build")
def build_models(
    _csrf: None = Depends(csrf_header_dep),
    _admin: None = Depends(_admin_access_dep),
):
    global STAGING_MODEL_VERSION
    try:
       sb = get_supabase_admin()
    except RuntimeError:
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
    collab_model = None
    try:
        purchases_result = sb.table('purchases').select('user_id, product_id, rating').limit(50000).execute()
        purchases = purchases_result.data or []
        if len(purchases) > 10:
            product_title_map = {p['id']: p['title'] for p in all_products}
            interaction_rows = []
            for p in purchases:
                title = product_title_map.get(p['product_id'])
                if title:
                    interaction_rows.append({'user_id': p['user_id'], 'title': title, 'rating': p.get('rating', 3.0)})
            if len(interaction_rows) > 10:
                interaction_df = pd.DataFrame(interaction_rows)
                if interaction_df['user_id'].nunique() > 1:
                    collab_model = CollaborativeRecommender(interaction_df)
    except Exception as e:
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

def build_models():
    with _model_lock:
        sb = get_supabase()

        all_products = []
        page_size = 1000
        offset = 0

        while True:
            result = sb.table('products').select(
                'id, title, description, category, rating, avg_sentiment, review_count'
            ).range(offset, offset + page_size - 1).execute()

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

        collab_model = None
        with _model_lock:
            try:
                purchases_result = sb.table('purchases').select(
                    'user_id, product_id, rating'
                ).limit(50000).execute()

                purchases = purchases_result.data or []

                if len(purchases) > 10:
                    product_title_map = {
                        p['id']: p['title']
                        for p in all_products
                    }

                    interaction_rows = []

                    for p in purchases:
                        title = product_title_map.get(p['product_id'])

                        if title:
                            interaction_rows.append({
                                'user_id': p['user_id'],
                                'title': title,
                                'rating': p.get('rating', 3.0)
                            })

                    if len(interaction_rows) > 10:
                        interaction_df = pd.DataFrame(interaction_rows)

                        if interaction_df['user_id'].nunique() > 1:
                            collab_model = CollaborativeRecommender(interaction_df)

            except Exception as e:
                logger.warning(
                    "Collaborative model data load failed: %s",
                    e
                )

        hybrid_model = HybridRecommender(
            content_model,
            collab_model,
            item_df
        )

        build_time = round(time.time() - start_time, 2)

        models["content"] = content_model
        models["collab"] = collab_model
        models["hybrid"] = hybrid_model
        models["item_df"] = item_df
        models["ready"] = True
        models["build_time"] = build_time
        models["last_trained_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        _clear_response_cache()

        return {
            "message": "Models built successfully!",
            "items": len(item_df),
            "has_collaborative": collab_model is not None,
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
    top_n: Optional[int] = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
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

    query_title = title or item_title
    if not query_title:
        raise HTTPException(422, "Query parameter 'title' is required.")

    # ----- EDGE CASES SAFE CHECK -----
    # Agar model ready nahi hai ya database bilkul khali hai
    if not models or "ready" not in models or not models["ready"]:
        raise HTTPException(status_code=400, detail="Models not built or dynamic dataset is empty.")
    # ---------------------------------
    selected_models = models

    if model_version == "staging":
        if not STAGING_MODEL_VERSION:
            raise HTTPException(404, "No staging model available.")

        selected_models = MODEL_REGISTRY[STAGING_MODEL_VERSION]

    elif model_version:
        if model_version not in MODEL_REGISTRY:
            raise HTTPException(404, "Requested model version not found.")

        selected_models = MODEL_REGISTRY[model_version]

    effective_limit = top_n if top_n is not None else limit
    effective_offset = offset

    cache_key = _recommendation_cache_key(
        query_title,
        effective_limit,
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
        top_n=effective_limit + effective_offset,
        explain=explain,
        target_catalog=target_catalog
    )

    # Popularity fallback (existing behaviour)
    if not recs and strategy == "popularity" and models["collab"]:
        recs = models["collab"]._popularity_fallback(effective_limit + effective_offset)

    # Cold-start fallback: blend content similarity with popularity/rating
    if not recs and strategy == "cold":
        combined_text = query_title
        cold_recs = cold_start_recommendation(
            combined_text,
            top_n=effective_limit + effective_offset,
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

    total_found = len(recs)
    paginated_recs = recs[effective_offset : effective_offset + effective_limit]

    payload = {
        "query": query_title,
        "query_item": query_title,
        "count": len(paginated_recs),
        "results": paginated_recs,
        "recommendations": paginated_recs,
        "weights": hybrid_model.get_weights(),
        "explain": explain,
        "target_catalog": target_catalog,
        "model_version": model_version or ACTIVE_MODEL_VERSION,
        "has_history": has_history,
        "pagination": {
            "total": total_found,
            "limit": effective_limit,
            "offset": effective_offset,
            "next_offset": effective_offset + effective_limit if effective_offset + effective_limit < total_found else None,
            "has_more": (effective_offset + len(paginated_recs)) < total_found,
        }
    }

    if (
        SHADOW_MODEL_VERSION
        and SHADOW_MODEL_VERSION in MODEL_REGISTRY
        and model_version is None
    ):
        shadow_model = MODEL_REGISTRY[SHADOW_MODEL_VERSION]

        shadow_start = time.time()

        try:
            shadow_recommend_func = shadow_model["hybrid"].recommend
            shadow_sig = inspect.signature(shadow_recommend_func)
            shadow_kwargs = {"top_n": top_n}
            if "explain" in shadow_sig.parameters:
                shadow_kwargs["explain"] = explain
            if "target_catalog" in shadow_sig.parameters:
                shadow_kwargs["target_catalog"] = target_catalog
            if "user_id" in shadow_sig.parameters:
                shadow_kwargs["user_id"] = user_id
            shadow_recs = shadow_recommend_func(query_title, **shadow_kwargs)

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


@app.get("/api/recommendations")
@app.get("/api/recommendations/{item_title}")
def get_recommendations_alias(
    request: Request,
    response: Response,
    item_title: Optional[str] = None,
    title: Optional[str] = Query(None),
    top_n: Optional[int] = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    explain: bool = Query(False),
    user_id: Optional[str] = Query(None),
    target_catalog: Optional[str] = Query(None),
    model_version: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
):
    """Backward-compatible alias for clients calling /api/recommendations."""
    return get_recommendations(
        request=request,
        response=response,
        item_title=item_title,
        title=title,
        top_n=top_n,
        limit=limit,
        offset=offset,
        explain=explain,
        user_id=user_id,
        target_catalog=target_catalog,
        model_version=model_version,
        strategy=strategy,
    )



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
        raise HTTPException(400, "Models not built or no item catalog available.")

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
        raise HTTPException(400, "Provide at least one of title, description, category or tags.")

    weights = (float(alpha), float(beta), float(gamma))
    recs = cold_start_recommendation(combined_text, top_n=top_n, weights=weights, target_catalog=target_catalog)
    if not recs:
        raise HTTPException(404, "No cold-start recommendations available.")

    # Do not cache cold-start responses by default (content depends on input metadata)
    _set_cache_headers(response, "MISS")
    return {"query": combined_text, "recommendations": recs, "weights": {"alpha": weights[0], "beta": weights[1], "gamma": weights[2]}}


@app.get("/api/user_recommend")
@app.get("/api/recommend/user/{user_id}")
def get_user_recommendations(user_id: str, top_n: int = Query(10, ge=1, le=50), explain: bool = Query(False)):
    """Get hybrid recommendations for a user."""
    _validate_user_id(user_id)  # allowlist-validate before model lookup
    if not models.get("ready") or not models.get("hybrid"):
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    
    is_fallback = False
    collab = models["hybrid"].collab_model
    if collab is None or user_id not in getattr(collab, "_user_to_idx", {}):
        is_fallback = True

    recs = models["hybrid"].recommend_for_user(user_id, top_n=top_n, explain=explain)
        
    return {
        "query_user": user_id,
        "recommendations": recs,
        "fallback": is_fallback,
        "weights": models["hybrid"].get_weights(),
    }

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
                await websocket.send_json({
                    "type": "error",
                    "message": "Models not built yet."
                })
                continue

            recs = models["hybrid"].recommend(item_title, user_id=user_id, top_n=top_n, explain=explain)
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
        raise HTTPException(status_code=400, detail="Models not built yet. Train the models first.")

    recs = models["hybrid"].recommend(req.item_title, top_n=req.top_n, explain=req.explain)
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
    candidate_limit = top_n if requested_category is None else min(top_n * 5, 100)
    recs = models["hybrid"].recommend(source_title, top_n=candidate_limit, explain=explain)
    if requested_category is not None:
        recs = [r for r in recs if str(r.get("category", "")).casefold() == requested_category.casefold()]
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
    content_model = models["content"]
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
        return {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    return models["hybrid"].get_weights()


@app.put("/api/weights")
def update_weights(
    w: WeightsUpdate,
    _csrf: None = Depends(csrf_header_dep),
    _admin: None = Depends(_admin_access_dep),
):
    if not models["ready"]:
        raise HTTPException(400, "Models not built.")
    models["hybrid"].set_weights(w.alpha, w.beta, w.gamma)
    _clear_response_cache()
    return {"message": "Weights updated", "weights": models["hybrid"].get_weights()}


# ── Items ─────────────────────────────────────────────────────────────
@app.get("/api/items")
def list_items(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    sb = get_supabase()
    limit = per_page
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


@app.post("/api/interactions")
def log_interaction(data: InteractionCreate):
    USER_INTERACTIONS.append({
        "user_id": data.user_id,
        "item_id": data.item_id,
        "interaction_type": data.interaction_type,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "message": "Interaction logged successfully",
        "interaction": USER_INTERACTIONS[-1]
    }

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

    # Issue #1596: Perform micro-update of latent vectors in memory
    try:
        if "collab" in models and models["collab"]:
            collab_model = models["collab"]
            item_df = models.get("item_df")
            if item_df is not None:
                matched = item_df[item_df['id'] == data.product_id]
                if not matched.empty:
                    title = matched.iloc[0]['title']
                    collab_model.online_update(
                        user_id=data.user_id,
                        title=title,
                        rating=data.rating
                    )
                    logger.info("Micro-updated latent vectors for user %s, item %s", data.user_id, title)
    except Exception as e:
        logger.warning("Failed to perform online micro-update: %s", e)

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
    global TRENDING_CACHE

    # Cache for 1 hour
    now = datetime.utcnow()

    cache_key = (days, limit)
    if isinstance(TRENDING_CACHE, dict) and "data" in TRENDING_CACHE and TRENDING_CACHE["data"] is None:
        cached_val = None
    else:
        cached_val = TRENDING_CACHE.get(cache_key)

    if cached_val is not None:
        timestamp, cached_data = cached_val
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
        return {"results": []}

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
        avg_rating = (
            sum(data["ratings"]) / max(len(data["ratings"]), 1)
        )

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

    ranked.sort(
        key=lambda x: x["trending_score"],
        reverse=True
    )

    response = {
        "results": ranked[:limit],
        "days": days,
        "limit": limit,
    }

    if isinstance(TRENDING_CACHE, dict):
        TRENDING_CACHE.pop("data", None)
        TRENDING_CACHE.pop("timestamp", None)
        TRENDING_CACHE[cache_key] = (now, response)
    else:
        TRENDING_CACHE = {cache_key: (now, response)}

    return response

# ── Feedback ──────────────────────────────────────────────────────────
@app.post("/api/feedback")
def submit_feedback(
    data: FeedbackCreate,
    request: Request,
    response: Response,
    _csrf: None = Depends(csrf_header_dep),
):
    limited_response = _apply_rate_limit(
        request,
        response,
        scope="feedback",
        limit_env="RATE_LIMIT_FEEDBACK_PER_MIN",
        default_limit=20,
    )
    if limited_response is not None:
        return limited_response

    feedback_client = _get_feedback_storage_client()
    feedback_record = {
        "user_id": data.user_id,
        "item": data.item,
        "feedback": data.feedback,
        "metadata": {
            "source_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", ""),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/user-preferences/{user_id}")
def get_user_preferences(user_id: str):

    user_logs = [
        x for x in USER_INTERACTIONS
        if x["user_id"] == user_id
    ]

    interaction_count = {}

    for item in user_logs:
        key = item["interaction_type"]

        interaction_count[key] = (
            interaction_count.get(key, 0) + 1
        )

    return {
        "user_id": user_id,
        "preferences": interaction_count
    }


# ── Export Dataset ────────────────────────────────────────────────────
@app.get("/api/export/dataset")
def export_dataset(columns: Optional[str] = Query(None)):
    if not models["ready"] or models["item_df"] is None:
        raise HTTPException(400, "Models not built. Build first via /api/build.")
    import pandas as pd
    from fastapi.responses import StreamingResponse
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


# ── GitHub Webhook Triage ─────────────────────────────────────────────
import hmac
import hashlib

def _verify_github_signature(request_body: bytes, signature_header: str | None) -> None:
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=500, detail="GITHUB_WEBHOOK_SECRET is not configured.")
    if not signature_header:
        raise HTTPException(status_code=401, detail="Signature header X-Hub-Signature-256 missing.")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=400, detail="Invalid signature format.")
        
    expected_signature = hmac.new(
        secret.encode(),
        request_body,
        hashlib.sha256
    ).hexdigest()
    
    provided_signature = signature_header.partition("sha256=")[2].strip()
    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(status_code=403, detail="Invalid webhook signature.")


@app.post("/api/webhook/github")
async def github_webhook(request: Request, response: Response):
    limited_response = _apply_rate_limit(
        request,
        response,
        scope="github_webhook",
        limit_env="RATE_LIMIT_GITHUB_WEBHOOK_PER_MIN",
        default_limit=60,
    )
    if limited_response is not None:
        return limited_response

    body_bytes = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    _verify_github_signature(body_bytes, signature)
    
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
        
    event = request.headers.get("X-GitHub-Event")
    action = payload.get("action")
    
    issue_number = None
    title = None
    body = None
    repo_full_name = None
    should_triage = False
    
    if event == "issues" and action == "opened":
        issue = payload.get("issue", {})
        issue_number = issue.get("number")
        title = issue.get("title", "")
        body = issue.get("body", "")
        repo_full_name = payload.get("repository", {}).get("full_name")
        should_triage = True
        
    elif event == "issue_comment" and action == "created":
        comment = payload.get("comment", {})
        comment_body = comment.get("body", "").strip()
        if comment_body.startswith("!retriage"):
            issue = payload.get("issue", {})
            issue_number = issue.get("number")
            title = issue.get("title", "")
            body = issue.get("body", "")
            repo_full_name = payload.get("repository", {}).get("full_name")
            should_triage = True
            
    if should_triage and issue_number and repo_full_name:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        triage_res = await triage_issue(
            issue_number=issue_number,
            title=title,
            body=body,
            repo_full_name=repo_full_name,
            token=token
        )
        return {"status": "success", "action": "triaged", "details": triage_res}
        
    return {"status": "skipped", "reason": f"No triage actions required for event '{event}' action '{action}'."}

# ---------------------------------------------------------------------------
# Pydantic Schemas for Validation
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=MAX_SEARCH_QUERY_LENGTH)
    limit: Optional[int] = 5


    @app.get("/dashboard.html")
    def serve_dashboard():
        return FileResponse(os.path.join(frontend_dir, "dashboard.html"))

# ── CLEAR USER PREFERENCES & RESET CACHE ENDPOINT ───────────────────
@app.post("/api/v1/user/preferences/reset", dependencies=[Depends(csrf_header_dep)])
async def reset_user_preferences(request: Request):
    """
    Clears the user interaction weights/history from Supabase 
    and wipes the local/Redis recommendation cache for that user.
    """
    # 1. Authentic/Validate user (In production, replace with your real auth dependency)
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing x-user-id header.")
    
    # Securely validate the user ID format
    validated_user_id = _validate_user_id(user_id)
    
    try:
        # 2. Connect to Supabase and delete preference entries
        supabase = get_supabase_admin()
        
        # Adjust table name and column names to match your schema
        supabase.table("user_preferences").delete().eq("user_id", validated_user_id).execute()
        supabase.table("user_interactions").delete().eq("user_id", validated_user_id).execute()
        
        # 3. Wipe out the internal memory cache
        _clear_response_cache()


        
      
        if _redis_client is not None:
            try:
                # Find keys matching this user's recommendation cache pattern
                cache_pattern = f"*recommend*:{validated_user_id.lower()}*"
                keys_to_delete = _redis_client.keys(cache_pattern)
                if keys_to_delete:
                    _redis_client.delete(*keys_to_delete)
            except Exception as cache_err:
                logger.warning(f"Failed to clear Redis cache keys: {cache_err}")

        return {
            "status": "success",
            "message": "User preferences completely reset. Cache successfully evicted."
        }
        
    except Exception as e:
        logger.error(f"Error resetting preferences for user {validated_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reset user data.")

