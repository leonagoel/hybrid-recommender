from __future__ import annotations

"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

from backend.csrf import CSRFMiddleware
from backend.core.metrics import response_time_middleware_logic
from backend.api.api_router import router as api_router

logger = logging.getLogger(__name__)

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

# CORS
allowed_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-CSRF-Token"],
)

app.add_middleware(CSRFMiddleware)

@app.middleware("http")
async def response_time_middleware(request, call_next):
    return await response_time_middleware_logic(request, call_next)

app.include_router(api_router)

# ── Frontend Serving ──────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/dashboard.html")
    def serve_dashboard():
        return FileResponse(os.path.join(frontend_dir, "dashboard.html"))
