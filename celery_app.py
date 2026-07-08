"""
Celery application configuration for the Hybrid Recommender System.
Uses Redis as both the message broker and the result backend.
Includes fallback mechanisms for synchronous local execution if Redis is offline.
"""
import os
import logging
from celery import Celery
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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

# Issue #1596: Schedule full SVD reconstruction to run as a cron job during off-peak hours
from celery.schedules import crontab
celery_app.conf.beat_schedule = {
    "rebuild-models-nightly": {
        "task": "tasks.rebuild_models_task",
        "schedule": crontab(hour=3, minute=0),
    },
}
celery_app.conf.worker_max_tasks_per_child = 5

def dispatch_task_safely(task, *args, **kwargs):
    """
    FIX FOR ISSUE #488: Verifies Redis broker health dynamically.
    Dispatches task asynchronously via Celery if cluster elements are healthy,
    otherwise routes execution along a graceful inline synchronous fallback path.
    """
    try:
        # Ping the Redis server with a strict 1-second connection timeout flag
        r = Redis.from_url(REDIS_URL, socket_connect_timeout=1.0)
        r.ping()
        
        # If connection succeeds, dispatch asynchronously to the Celery worker cluster
        logger.info(f"Redis cluster verified healthy. Dispatching task {task.__name__} asynchronously.")
        return task.delay(*args, **kwargs)
        
    except Exception as e:
        # If Redis is offline, degrade gracefully by executing the logic locally and synchronously
        logger.warning(
            f"Celery message broker/Redis offline ({e}). "
            f"Degrading gracefully to execute task {task.__name__} synchronously inline."
        )
        # .apply() tells Celery to run the function right now on the main execution thread
        return task.apply(args=args, kwargs=kwargs)
        celery_app.conf.worker_max_tasks_per_child = 5
