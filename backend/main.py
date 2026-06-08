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

_cache_lock = Lock()

# ── Redis client ──────────────────────────────────────────────────────
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


    return ":".join(str(part).strip().lower() for part in parts)


def _get_cached_response(key: str):
    global _cache_hits, _cache_misses
    if _redis_client is not None:
        try:
            cached = _redis_client.get(key)
            if cached is not None:
                _cache_hits += 1
                return json.loads(cached)

    with _cache_lock:
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


# ── FIX #1292: HIGH PERFORMANCE RATE LIMITER PATH ─────────────────────
def _apply_rate_limit(*args, **kwargs):
    """
    Applies token-bucket rate limiting dynamically.
    Optimized to handle Algorithmic Complexity DoS scenarios.
    """
    current_time = time.time()
    allowed = False
    
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.get(ip_address)
        if bucket is None:
            bucket = {"tokens": 10.0, "last_updated": current_time}
            _rate_limit_buckets[ip_address] = bucket
        else:
            _rate_limit_buckets.move_to_end(ip_address)
            elapsed = current_time - bucket["last_updated"]
            bucket["tokens"] = min(10.0, bucket["tokens"] + elapsed * 1.0)
            bucket["last_updated"] = current_time
            
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            _rate_limit_buckets[ip_address] = bucket
            allowed = True
            

        global _request_counter
        _request_counter += 1
        if random.random() < 0.01 or _request_counter >= CLEANUP_THRESHOLD:
            _request_counter = 0
            # Evict stale buckets older than 1 hour to prevent memory leaks
            cutoff = current_time - 3600
            to_remove = [k for k, v in _rate_limit_buckets.items() if v["last_updated"] < cutoff]
            for k in to_remove:
                del _rate_limit_buckets[k]
                
    return allowed

# ---------------------------------------------------------------------------
# Pydantic Schemas for Validation
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=MAX_SEARCH_QUERY_LENGTH)
    limit: Optional[int] = 5


    item_title: str
    top_n: Optional[int] = 10
    explain: Optional[bool] = False
    strategy: Optional[str] = "hybrid"

class TuningWeights(BaseModel):
    alpha: float = Field(..., ge=0.0, le=1.0)
    beta: float = Field(..., ge=0.0, le=1.0)
    gamma: float = Field(..., ge=0.0, le=1.0)

# ---------------------------------------------------------------------------
# API Endpoints & Core Logic
# ---------------------------------------------------------------------------

@app.get("/api/csrf-token", response_model=CSRFTokenResponse)
def get_csrf_token_endpoint(request: Request, response: Response):
    """Generates a secure CSRF token and sets it in the user's browser cookies."""
    token = generate_csrf_token()
    set_csrf_cookie(response, token)
    return {"csrfToken": token}



    try:
        supabase = get_supabase()
        res = supabase.rpc("search_products_fts", {"search_query": cleaned_query, "row_limit": req.limit or 5}).execute()
        return success_response(res.data or [])
    except Exception as e:
        logger.error(f"Full-text search failure: {e}", exc_info=True)
        return error_response("Catalog search service is temporarily unavailable.")

@app.post("/api/v1/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dependencies: None = Depends(csrf_header_dep)
):
    """Processes, adapts, and validates custom CSV/JSON product file uploads."""
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File payload configuration exceeds maximum allowed limits.")

    try:
        file_extension = os.path.splitext(file.filename or "").lower()
        raw_df = read_file(io.BytesIO(contents), file_extension)
        adapted_df = adapt_data(raw_df)
        
        # Trigger asynchronous sentiment scoring block on incoming texts
        scored_df = await asyncio.to_thread(batch_analyze, adapted_df)
        records = scored_df.to_dict(orient="records")
        
        return success_response({
            "filename": file.filename,
            "total_records": len(records),
            "preview": records[:3]
        })
    except Exception as e:
        logger.error(f"Dataset upload pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process dataset file structure: {str(e)}")

@app.post("/api/v1/tune-weights")
async def update_hybrid_weights(weights: TuningWeights, dependencies: None = Depends(csrf_header_dep)):
    """Dynamically scales model weights for the multi-signal hybrid recommendation system."""
    total = weights.alpha + weights.beta + weights.gamma
    if total == 0:
        raise HTTPException(status_code=400, detail="Combined component weights cannot equal zero.")
    
    normalized = {
        "alpha": weights.alpha / total,
        "beta": weights.beta / total,
        "gamma": weights.gamma / total
    }
    return success_response({"message": "Weights updated successfully", "normalized_weights": normalized})

# ---------------------------------------------------------------------------
# Real-Time Asynchronous WebSockets
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass



frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(os.path.join(frontend_path, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

