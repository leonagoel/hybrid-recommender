import os
import sys
import io
import time
import logging
import math
import pandas as pd

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Request,
    Response,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.diversity import (
    calculate_diversity_score,
    diversify_results,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)

logger = logging.getLogger("hybrid_recommender.api")

from src.data.db import get_supabase, get_supabase_admin
from data_adapter import adapt_data, read_file
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

app = FastAPI(title="Hybrid Recommender API", version="3.0")

RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0

CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"

_response_cache = {}

# ─────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────

def _cache_key(*parts: Any) -> str:
    return ":".join(str(part).strip().lower() for part in parts)


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


# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

allowed_origins = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# RESPONSE METRICS
# ─────────────────────────────────────────────────────────────

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

        response_time_samples.append(response_time_ms)

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
        if samples
        else 0.0
    )

    error_rate = (
        (error_requests / total_requests) * 100
        if total_requests
        else 0.0
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


# ─────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def response_time_middleware(
    request: Request,
    call_next,
):
    started_at = time.perf_counter()

    response = None

    try:
        response = await call_next(request)

        return response

    finally:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        status_code = (
            response.status_code
            if response is not None
            else 500
        )

        if response is not None:
            response.headers["X-Response-Time"] = (
                f"{duration_ms:.2f}ms"
            )

        record_response_metric(
            request.url.path,
            request.method,
            status_code,
            duration_ms,
        )


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────

models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
}


# ─────────────────────────────────────────────────────────────
# MODELS
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