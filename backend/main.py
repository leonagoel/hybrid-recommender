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

app = FastAPI(title="Hybrid Recommender API", version="3.0")

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
_rate_limit_buckets: dict = {}
_rate_limit_lock = Lock()
_cache_lock = Lock()

# Optional Redis client for distributed caching.  None when REDIS_URL is unset
# or the connection cannot be established at startup; the in-process dict cache
# is used as a fallback in both cases.
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
    """Single authoritative cache key for recommendation responses.

    Both the precomputation path (_precompute_recommendation_cache) and
    the request-serving path (get_recommendations) must use this function
    so that precomputed entries are always retrievable by the API.

    All optional parameters default to '' so that a plain item lookup
    produces the same key whether called from precomputation or from the
    API handler with all optional query params absent.

    Args:
        title (str): Reference item title acting as the core query.
        top_n (int, optional): Number of item suggestions requested. Defaults to 10.
        explain (bool, optional): Indicates if rationale metrics are attached. Defaults to False.
        user_id (str, optional): Target client identifier profile string. Defaults to "".
        target_catalog (str, optional): Scoped inventory isolation target namespace. Defaults to "".
        model_version (str, optional): Tracking signature tag of active pipeline. Defaults to "".
        strategy (str, optional): Structural algorithmic routing variant label. Defaults to "".

    Returns:
        str: Authoritative combined lowercase cache string key.
    """
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
    """Retrieve an item value from the distributed Redis cache or fallback dictionary layer.

    Checks the global external Redis storage pool first. If missing or disconnected, 
    acquires a thread lock and falls back to inspecting the in-process dictionary cache.

    Args:
        key (str): Target string lookup identifier matching cache keys.

    Returns:
        Any | None: Deserialized object data if cache is valid; otherwise None.
    """
    global _cache_hits, _cache_misses   # Move globals to the top

    if _redis_client is not None:
        try:
            cached = _redis_client.get(key)
            if cached is not None:
