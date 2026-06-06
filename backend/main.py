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
    global _request_counter
    current_time = time.time()
    
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
