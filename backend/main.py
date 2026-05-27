"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth,
and the improved hybrid model.
"""

import os
import sys
import io
import time
import math
import logging
import re

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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
    Depends,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Internal Modules
# ─────────────────────────────────────────────────────────────

from src.data.db import get_supabase, get_supabase_admin
from src.data.data_adapter import adapt_data, read_file
from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from celery_app import celery_app
from tasks import compute_recommendations
from src.evaluation.ab_testing import DEFAULT_EXPERIMENT_ID, run_recommendation_experiment


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid Recommender API",
    version="3.0",
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

RESPONSE_TIME_HEADER = "X-Response-Time-ms"

DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0

CACHE_TTL_SECONDS = 300

CACHE_CONTROL_VALUE = (
    f"public, max-age={CACHE_TTL_SECONDS}"
)

SLOW_RESPONSE_THRESHOLD_MS = 500.0

METRICS_SAMPLE_SIZE = 1000

TRENDING_CACHE_TTL = 60 * 60

MAX_FILE_SIZE = 5 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/json",
    "application/vnd.ms-excel",
    "application/octet-stream",
}

# ─────────────────────────────────────────────────────────────
# In-Memory Stores
# ─────────────────────────────────────────────────────────────

_response_cache: dict[str, tuple[float, Any]] = {}

_rate_limit_buckets: dict[str, list[float]] = {}

response_time_samples = deque(
    maxlen=METRICS_SAMPLE_SIZE
)

response_metrics = {
    "total_requests": 0,
    "error_requests": 0,
}

response_metrics_lock = Lock()

TRENDING_CACHE = {
    "data": None,
    "timestamp": None,
}

# ─────────────────────────────────────────────────────────────
# Mock Products
# ─────────────────────────────────────────────────────────────

MOCK_PRODUCTS = [
    {
        "id": 1,
        "title": "Acoustic Noise-Cancelling Headphones",
        "description": (
            "Immerse yourself in pure sound with "
            "these premium over-ear headphones "
            "featuring active noise cancellation."
        ),
        "category": "Electronics",
        "rating": 4.8,
        "avg_sentiment": 0.85,
        "review_count": 245,
        "image": (
            "https://images.unsplash.com/"
            "photo-1505740420928-5e560c06d30e"
            "?w=500&auto=format&fit=crop&q=60"
        ),
    },
    {
        "id": 2,
        "title": "Ergonomic Mechanical Keyboard",
        "description": (
            "Type in comfort all day with tactile "
            "brown switches, customizable RGB "
            "backlighting, and a plush wrist rest."
        ),
        "category": "Electronics",
        "rating": 4.5,
        "avg_sentiment": 0.65,
        "review_count": 189,
        "image": (
            "https://images.unsplash.com/"
            "photo-1587829741301-dc798b83add3"
            "?w=500&auto=format&fit=crop&q=60"
        ),
    },
]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_slow_response_threshold_ms() -> float:
    try:
        return float(
            os.environ.get(
                "RESPONSE_TIME_SLOW_MS",
                DEFAULT_SLOW_RESPONSE_THRESHOLD_MS,
            )
        )
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS


def _cache_key(*parts: Any) -> str:
    return ":".join(
        str(part).strip().lower()
        for part in parts
    )


def _get_cached_response(key: str):
    cached = _response_cache.get(key)

    if not cached:
        logger.info(
            "cache_miss",
            extra={"cache_key": key},
        )
        return None

    expires_at, value = cached

    if expires_at <= time.time():
        _response_cache.pop(key, None)

        logger.info(
            "cache_expired",
            extra={"cache_key": key},
        )

        return None

    logger.info(
        "cache_hit",
        extra={"cache_key": key},
    )

    return value


def _set_cached_response(
    key: str,
    value: Any,
):
    _response_cache[key] = (
        time.time() + CACHE_TTL_SECONDS,
        value,
    )


def _clear_response_cache():
    _response_cache.clear()


def _set_cache_headers(
    response: Response,
    status: str,
):
    response.headers["Cache-Control"] = (
        CACHE_CONTROL_VALUE
    )
    response.headers["X-Cache"] = status


def _percentile(values, percentile):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = (
        math.ceil(
            (percentile / 100) * len(sorted_values)
        )
        - 1
    )

    index = max(
        0,
        min(index, len(sorted_values) - 1),
    )

    return sorted_values[index]


# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8000,"
    "http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────

security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(
        security
    ),
):
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    try:
        sb = get_supabase()

        user = sb.auth.get_user(token)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    return token

# ─────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────

def record_response_metric(
    endpoint,
    method,
    status_code,
    response_time_ms,
):
    with response_metrics_lock:
        response_metrics["total_requests"] += 1

        if status_code >= 400:
            response_metrics["error_requests"] += 1

        response_time_samples.append(
            response_time_ms
        )

    log_level = (
        logging.WARNING
        if response_time_ms >
        SLOW_RESPONSE_THRESHOLD_MS
        else logging.INFO
    )

    logger.log(
        log_level,
        (
            "API request endpoint=%s "
            "method=%s status_code=%s "
            "response_time_ms=%.2f"
        ),
        endpoint,
        method,
        status_code,
        response_time_ms,
    )


@app.middleware("http")
async def response_time_middleware(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    response = None
    status_code = 500

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    finally:
        response_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if response is not None:
            response.headers[
                RESPONSE_TIME_HEADER
            ] = f"{response_time_ms:.2f}"

        record_response_metric(
            request.url.path,
            request.method,
            status_code,
            response_time_ms,
        )

# ─────────────────────────────────────────────────────────────
# Models State
# ─────────────────────────────────────────────────────────────

models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
    "last_trained_at": None,
}

# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# WebSocket Hub
# ─────────────────────────────────────────────────────────────

class RealtimeRecommendationHub:

    def __init__(self):
        self.active_connections = []

    async def connect(
        self,
        websocket: WebSocket,
    ):
        await websocket.accept()
        self.active_connections.append(
            websocket
        )

    def disconnect(
        self,
        websocket: WebSocket,
    ):
        if websocket in self.active_connections:
            self.active_connections.remove(
                websocket
            )

    async def broadcast(self, payload: dict):
        disconnected = []

        for websocket in self.active_connections:
            try:
                await websocket.send_json(
                    payload
                )
            except RuntimeError:
                disconnected.append(
                    websocket
                )

        for websocket in disconnected:
            self.disconnect(websocket)


realtime_hub = RealtimeRecommendationHub()

# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": os.getenv(
            "APP_VERSION",
            "1.0.0",
        ),
    }

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {
        "supabase_url": os.environ.get(
            "SUPABASE_URL",
            "",
        ),
        "supabase_anon_key": os.environ.get(
            "SUPABASE_ANON_KEY",
            "",
        ),
    }

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_api_metrics():

    samples = list(response_time_samples)

    avg_response_time = (
        sum(samples) / len(samples)
        if samples else 0.0
    )

    error_rate = (
        (
            response_metrics["error_requests"]
            /
            response_metrics["total_requests"]
        ) * 100
        if response_metrics["total_requests"]
        else 0.0
    )

    return {
        "avg_response_time": round(
            avg_response_time,
            2,
        ),
        "p95_response_time": round(
            _percentile(samples, 95),
            2,
        ),
        "total_requests":
            response_metrics["total_requests"],
        "error_rate":
            round(error_rate, 2),
    }

# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────

@app.get("/api/search")
def search_items(
    request: Request,
    response: Response,
    q: str = "",
    limit: int = 8,
    offset: int = 0,
):

    cache_key = _cache_key(
        "search",
        q,
        limit,
        offset,
    )

    cached = _get_cached_response(
        cache_key
    )

    if cached is not None:
        _set_cache_headers(
            response,
            "HIT",
        )
        return cached

    try:
        sb = get_supabase()

        query = q.strip()

        if query:
            result = (
                sb.table("products")
                .select(
                    "id, title, description, "
                    "category, rating, "
                    "avg_sentiment, review_count, image"
                )
                .ilike(
                    "title",
                    f"%{query}%"
                )
                .order(
                    "rating",
                    desc=True,
                )
                .limit(limit)
                .offset(offset)
                .execute()
            )

            products = result.data or []

        else:
            result = (
                sb.table("products")
                .select(
                    "id, title, description, "
                    "category, rating, "
                    "avg_sentiment, review_count, image"
                )
                .order(
                    "rating",
                    desc=True,
                )
                .limit(limit)
                .offset(offset)
                .execute()
            )

            products = result.data or []

        results = []

        for p in products:
            results.append({
                "id": p.get("id"),
                "title": p.get("title", ""),
                "description":
                    str(
                        p.get(
                            "description",
                            "",
                        )
                    )[:200],
                "category":
                    p.get("category", ""),
                "rating":
                    p.get("rating", 0.0),
                "avg_sentiment":
                    p.get(
                        "avg_sentiment",
                        0.0,
                    ),
                "review_count":
                    p.get(
                        "review_count",
                        0,
                    ),
                "rank": 0.0,
                "image":
                    p.get("image", ""),
            })

    except Exception as e:

        logger.warning(
            "Supabase search failed: %s",
            e,
        )

        query_clean = q.strip().lower()

        if query_clean:
            filtered = [
                p for p in MOCK_PRODUCTS
                if query_clean
                in p["title"].lower()
            ]
        else:
            filtered = MOCK_PRODUCTS

        paginated = filtered[
            offset: offset + limit
        ]

        results = paginated

    payload = {
        "results": results,
        "total": len(results),
        "query": q,
    }

    _set_cached_response(
        cache_key,
        payload,
    )

    _set_cache_headers(
        response,
        "MISS",
    )

    return payload

# ─────────────────────────────────────────────────────────────
# Build Models
# ─────────────────────────────────────────────────────────────

@app.post("/api/build")
def build_models(
    token: str = Depends(
        verify_token
    )
):

    try:
        sb = get_supabase()

        result = (
            sb.table("products")
            .select("*")
            .limit(5000)
            .execute()
        )

        all_products = (
            result.data or []
        )

    except Exception as e:
        logger.warning(
            "Fallback to mock products: %s",
            e,
        )

        all_products = MOCK_PRODUCTS

    if not all_products:
        raise HTTPException(
            400,
            "No products available.",
        )

    import pandas as pd

    item_df = pd.DataFrame(
        all_products
    )

    item_df["combined"] = (
        item_df["title"].astype(str)
        + " "
        + item_df["description"]
        .fillna("")
        .astype(str)
        + " "
        + item_df["category"]
        .fillna("")
        .astype(str)
    )

    start_time = time.time()

    content_model = ContentRecommender(
        item_df
    )

    hybrid_model = HybridRecommender(
        content_model,
        None,
        item_df,
    )

    build_time = round(
        time.time() - start_time,
        2,
    )

    models["content"] = content_model
    models["hybrid"] = hybrid_model
    models["item_df"] = item_df
    models["ready"] = True
    models["build_time"] = build_time

    models["last_trained_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    _clear_response_cache()

    return {
        "message":
            "Models built successfully!",
        "items": len(item_df),
        "build_time_seconds":
            build_time,
    }

# ─────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────

@app.get("/api/recommend")
def get_recommendations(
    title: str,
    top_n: int = 10,
):

    if not models["ready"]:
        raise HTTPException(
            400,
            "Models not built.",
        )

    recs = models["hybrid"].recommend(
        title,
        top_n=top_n,
    )

    if not recs:
        raise HTTPException(
            404,
            "No recommendations found.",
        )

    return {
        "query_item": title,
        "recommendations": recs,
    }

# ─────────────────────────────────────────────────────────────
# Frontend Serving
# ─────────────────────────────────────────────────────────────

frontend_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "frontend",
)

if os.path.isdir(frontend_dir):

    app.mount(
        "/static",
        StaticFiles(
            directory=frontend_dir
        ),
        name="frontend",
    )

    @app.get("/")
    def serve_frontend():
        return FileResponse(
            os.path.join(
                frontend_dir,
                "index.html",
            )
        )

    @app.get("/dashboard.html")
    def serve_dashboard():
        return FileResponse(
            os.path.join(
                frontend_dir,
                "dashboard.html",
            )
        )
