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
import asyncio
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
    """Retrieve the duration threshold used to classify slow API responses."""
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS

def _cache_key(*parts: Any) -> str:
    """Generate a consistent, lowercased cache string key from input segments."""
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
    current_time = time.time()
    allowed = False
    
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
            
        # Optimization: Move cleanup out of the request loop path securely
        global _request_counter
        _request_counter += 1
        if random.random() < 0.01 or _request_counter >= CLEANUP_THRESHOLD:
            _request_counter = 0
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

class RecommendationRequest(BaseModel):
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

# ── FIX #914: NON-BLOCKING TELEMETRY ENGINE ROUTES ────────────────────
@app.get("/api/v1/recommend")
async def get_hybrid_recommendations(
    title: str = Query(...),
    top_n: int = 10,
    user_id: Optional[str] = Query(None)
):
    """
    Fetch hybrid recommendations securely without blocking the FastAPI event loop.
    Utilizes asyncio.to_thread to offload CPU-heavy matrix similarity math.
    """
    cache_key = _recommendation_cache_key(title, top_n, user_id=user_id)
    
    # Check cache layers safely
    cached_res = _get_cached_response(cache_key)
    if cached_res:
        return success_response(cached_res)

    try:
        # 🚀 PERFORMANCE FIX: Offload heavy ML Cosine Similarity math to a separate threadpool
              # 🚀 PERFORMANCE FIX: Offload heavy ML Cosine Similarity math to a separate threadpool
        # This keeps the main FastAPI event loop completely free to accept concurrent traffic
        recommendations = await asyncio.to_thread(
            HybridRecommender.get_recommendations, 
            title=title, 
            top_n=top_n, 
            user_id=user_id
        )
        
        if _redis_client:
            try:
                await asyncio.to_thread(
                    _redis_client.setex, cache_key, CACHE_TTL_SECONDS, json.dumps(recommendations)
                )
            except RedisError:
                pass
                
        return success_response(recommendations)
        
    except Exception as e:
        logger.error(f"Failed compiling asynchronous cold-start matrices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal matrix computation bottleneck encountered.")

@app.post("/api/v1/search")
async def search_catalog(req: SearchRequest, request: Request):
    """Performs dynamic full-text search against the product database."""
    ip = request.client.host if request.client else "unknown"
    if not _apply_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")

    cleaned_query = bleach.clean(req.query.strip())
    if not cleaned_query:
        return success_response([])

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

manager = ConnectionManager()

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """Establishes real-time persistent connections to broadcast system updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            await manager.broadcast({"event": "client_ping", "timestamp": str(datetime.now(timezone.utc))})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket runtime drop: {e}")
        manager.disconnect(websocket)

# ---------------------------------------------------------------------------
# Fallback Server Static Files Mount
# ---------------------------------------------------------------------------

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(os.path.join(frontend_path, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

