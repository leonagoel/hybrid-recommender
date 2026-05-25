"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""

import os
import sys
import io
import time
import math
import logging

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

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
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)

from db import get_supabase, get_supabase_admin
from src.data.data_adapter import adapt_data, read_file
from src.model.nlp_engine import (
    batch_analyze,
    aggregate_sentiment_by_item,
)

from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender

from celery.result import AsyncResult
from celery_app import celery_app
from tasks import compute_recommendations

from src.evaluation.ab_testing import (
    DEFAULT_EXPERIMENT_ID,
    run_recommendation_experiment,
)

# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid Recommender API",
    version="3.0",
)

RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0
CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"

_response_cache: dict = {}


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
        return None

    expires_at, value = cached

    if expires_at <= time.time():
        _response_cache.pop(key, None)
        return None

    return value


def _set_cached_response(key: str, value: Any) -> None:
    _response_cache[key] = (
        time.time() + CACHE_TTL_SECONDS,
        value,
    )


def _clear_response_cache() -> None:
    _response_cache.clear()


def _set_cache_headers(response: Response, status: str) -> None:
    response.headers["Cache-Control"] = CACHE_CONTROL_VALUE
    response.headers["X-Cache"] = status


# ── CORS ─────────────────────────────────────────────────────────────

allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response Time Monitoring ────────────────────────────────────────

SLOW_RESPONSE_THRESHOLD_MS = 500.0
METRICS_SAMPLE_SIZE = 1000

response_time_samples = deque(
    maxlen=METRICS_SAMPLE_SIZE
)

response_metrics = {
    "total_requests": 0,
    "error_requests": 0,
}

response_metrics_lock = Lock()


def _percentile(values, percentile):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = math.ceil(
        (percentile / 100) * len(sorted_values)
    ) - 1

    index = max(
        0,
        min(index, len(sorted_values) - 1)
    )

    return sorted_values[index]


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
        if response_time_ms > SLOW_RESPONSE_THRESHOLD_MS
        else logging.INFO
    )

    logger.log(
        log_level,
        "API request endpoint=%s method=%s status=%s time=%.2fms",
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

    avg_response_time = (
        sum(samples) / len(samples)
        if samples else 0.0
    )

    error_rate = (
        (error_requests / total_requests) * 100
        if total_requests else 0.0
    )

    return {
        "avg_response_time": round(avg_response_time, 2),
        "p95_response_time": round(
            _percentile(samples, 95),
            2,
        ),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 2),
    }


@app.middleware("http")
async def response_time_middleware(
    request,
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
                "X-Response-Time"
            ] = f"{response_time_ms:.2f}ms"

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


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_loaded": models["ready"],
    }


# ── API Metrics ──────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_api_metrics():
    return get_response_metrics_snapshot()


# ── Config ───────────────────────────────────────────────────────────

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


# ── Status ───────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "status": "healthy",
        "model_ready": models["ready"],
        "message": "Hybrid Recommender API running",
    }


# ── Build Models ─────────────────────────────────────────────────────

@app.post("/api/build")
def build_models():
    """Build recommendation models from Supabase data."""

    sb = get_supabase()

    all_products = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            sb.table("products")
            .select(
                "id, title, description, category, "
                "rating, avg_sentiment, review_count"
            )
            .range(
                offset,
                offset + page_size - 1,
            )
            .execute()
        )

        batch = result.data or []

        all_products.extend(batch)

        if len(batch) < page_size:
            break

        offset += page_size

    if not all_products:
        raise HTTPException(
            400,
            "No products in database. Upload data first.",
        )

    import pandas as pd

    item_df = pd.DataFrame(all_products)

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

    item_df["review_count"] = (
        item_df["review_count"]
        .fillna(0)
        .astype(int)
    )

    start_time = time.time()

    content_model = ContentRecommender(item_df)

    collab_model = None

    try:
        purchases_result = (
            sb.table("purchases")
            .select(
                "user_id, product_id, rating"
            )
            .limit(50000)
            .execute()
        )

        purchases = purchases_result.data or []

        if len(purchases) > 10:
            product_title_map = {
                p["id"]: p["title"]
                for p in all_products
            }

            interaction_rows = []

            for p in purchases:
                title = product_title_map.get(
                    p["product_id"]
                )

                if title:
                    interaction_rows.append({
                        "user_id": p["user_id"],
                        "title": title,
                        "rating": p.get(
                            "rating",
                            3.0,
                        ),
                    })

            if len(interaction_rows) > 10:
                interaction_df = pd.DataFrame(
                    interaction_rows
                )

                if (
                    interaction_df["user_id"]
                    .nunique() > 1
                ):
                    collab_model = (
                        CollaborativeRecommender(
                            interaction_df
                        )
                    )

    except Exception as e:
        logger.warning(
            "Collaborative model data load failed: %s",
            e,
        )

    hybrid_model = HybridRecommender(
        content_model,
        collab_model,
        item_df,
    )

    build_time = round(
        time.time() - start_time,
        2,
    )

    models["content"] = content_model
    models["collab"] = collab_model
    models["hybrid"] = hybrid_model
    models["item_df"] = item_df
    models["ready"] = True
    models["build_time"] = build_time
    models["last_trained_at"] = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    _clear_response_cache()

    return {
        "message": "Models built successfully!",
        "items": len(item_df),
        "has_collaborative": (
            collab_model is not None
        ),
        "build_time_seconds": build_time,
    }


# ── Frontend Serving ─────────────────────────────────────────────────

frontend_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "frontend",
)

if os.path.isdir(frontend_dir):

    app.mount(
        "/static",
        StaticFiles(directory=frontend_dir),
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


# ── Startup ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
