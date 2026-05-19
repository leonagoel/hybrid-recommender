"""
Celery application configuration for the Hybrid Recommender System.
Uses Redis as both the message broker and the result backend.
"""
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "hybrid_recommender",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],  # task module to auto-discover
)

celery_app.conf.update(
    # Serialize tasks as JSON (safe, human-readable)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Results expire after 1 hour to prevent Redis memory bloat
    result_expires=3600,

    # Acknowledge task only after it completes (prevents data loss on crash)
    task_acks_late=True,

    # One task at a time per worker process (CPU-bound ML work)
    worker_prefetch_multiplier=1,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
