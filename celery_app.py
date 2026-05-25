"""
Celery application configuration for the Hybrid Recommender System.
Uses Redis as both the message broker and the result backend.
"""
import os
from celery import Celery
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app without connecting at import time
celery_app = Celery("hybrid_recommender")

# Configure broker/backend with broker_connection_retry_on_startup=False
# to prevent hanging if Redis is not available
celery_app.conf.broker_url = REDIS_URL
celery_app.conf.result_backend = REDIS_URL
celery_app.conf.broker_connection_retry_on_startup = False

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

# Auto-discover tasks
try:
    celery_app.autodiscover_tasks(["tasks"])
except Exception as e:
    logger.debug(f"Celery task discovery: {e}")
