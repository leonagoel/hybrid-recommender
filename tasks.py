"""
Celery tasks for the Hybrid Recommender System.
Heavy recommendation computation is moved here so the API
thread returns immediately with a task_id.

Worker model isolation
----------------------
Celery workers run in separate OS processes and therefore do not share
memory with the API server process.  Importing ``models`` directly from
``backend.main`` would give each worker its own empty copy of that dict
(``ready=False``) that never reflects the API's in-memory state after
``POST /api/build`` completes.

Instead, workers maintain a *process-local* model cache that is built
from the same shared data source (Supabase) that the API uses.  The API
writes a lightweight version token to Redis after every successful build
or promotion (``hybrid_recommender:model_version``).  Workers compare
their locally cached version against that token before each task and
rebuild only when a newer version is detected, so:

* No worker restart is required after a model rebuild.
* Multiple worker processes each hold their own up-to-date copy.
* If Redis is unreachable, the worker falls back to its previously
  cached model (best-effort) rather than failing hard.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from functools import wraps

sys.path.insert(0, os.path.dirname(__file__))

from redis import Redis  # noqa: E402  (module-level for patchability in tests)

from celery_app import celery_app, REDIS_URL  # noqa: E402

logger = logging.getLogger(__name__)

# ── User Request Serialization Lock Registry ──────────────────────────────
_user_task_locks: dict[str, threading.Lock] = {}
_lock_registry_mutex: threading.Lock = threading.Lock()

def serialize_user_requests(func):
    """
    Decorator to prevent race conditions on concurrent tasks for the same user.
    Ensures that parallel weight matrix or recommendation generation requests
    from an identical user_id are executed sequentially.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        user_id = kwargs.get("user_id") or (args[0] if args else None)
        
        if not user_id:
            return func(*args, **kwargs)

        with _lock_registry_mutex:
            if user_id not in _user_task_locks:
                _user_task_locks[user_id] = threading.Lock()
            user_lock = _user_task_locks[user_id]
        
        with user_lock:
            return func(*args, **kwargs)
    return wrapper

# ── Redis coordination key (must match the constant in backend/main.py) ───────
REDIS_MODEL_VERSION_KEY = "hybrid_recommender:model_version"

# ── Worker-local model cache ──────────────────────────────────────────────────
# All three globals are guarded by _worker_lock when being updated.

_worker_lock: threading.Lock = threading.Lock()
_worker_model_version: str | None = None  # version string of the cached model
_worker_models: dict | None = None        # {"hybrid": ..., "content": ..., "ready": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_current_version() -> str | None:
    """
    Return the model version token written to Redis by the API process, or
    ``None`` when Redis is unavailable.

    Uses a short connection timeout (1 s) so that a Redis outage does not
    block task execution for long.
    """
    try:
        r = Redis.from_url(REDIS_URL, socket_connect_timeout=1)
        raw = r.get(REDIS_MODEL_VERSION_KEY)
        return raw.decode() if raw else None
    except Exception as exc:
        logger.warning("Worker: could not read model version from Redis: %s", exc)
        return None


def _build_models_for_worker() -> dict:
    """
    Build recommendation models from the shared Supabase catalog.

    This mirrors the logic in ``POST /api/build`` so workers always reflect
    the same training data and catalog snapshot that the API used.

    Returns a dict with keys ``content``, ``collab``, ``hybrid``, ``item_df``,
    and ``ready=True``.

    Raises ``RuntimeError`` when the database is unreachable or the catalog
    contains no products.
    """
    import pandas as pd
    from src.data.db import get_supabase
    from src.model.content_model import ContentRecommender
    from src.model.collaborative_model import CollaborativeRecommender
    from src.model.hybrid_model import HybridRecommender

    try:
        sb = get_supabase()
    except Exception as exc:
        raise RuntimeError(f"Worker: cannot connect to Supabase: {exc}") from exc

    # ── Fetch product catalog (paginated to match API behaviour) ─────────────
    all_products: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            sb.table("products")
            .select("id, title, description, category, rating, avg_sentiment, review_count")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        all_products.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not all_products:
        raise RuntimeError(
            "Worker: no products found in database. "
            "Ensure POST /api/build has been called at least once."
        )

    item_df = pd.DataFrame(all_products)
    item_df["combined"] = (
        item_df["title"].astype(str)
        + " "
        + item_df["description"].fillna("").astype(str)
        + " "
        + item_df["category"].fillna("").astype(str)
    )
    item_df["review_count"] = item_df["review_count"].fillna(0).astype(int)

    content_model = ContentRecommender(item_df)

    # ── Optional collaborative model ─────────────────────────────────────────
    collab_model = None
    try:
        purchases_result = (
            sb.table("purchases")
            .select("user_id, product_id, rating")
            .limit(50000)
            .execute()
        )
        purchases = purchases_result.data or []
        if len(purchases) > 10:
            product_title_map = {p["id"]: p["title"] for p in all_products}
            interaction_rows = [
                {
                    "user_id": p["user_id"],
                    "title": product_title_map[p["product_id"]],
                    "rating": p.get("rating", 3.0),
                }
                for p in purchases
                if p["product_id"] in product_title_map
            ]
            if len(interaction_rows) > 10:
                interaction_df = pd.DataFrame(interaction_rows)
                if interaction_df["user_id"].nunique() > 1:
                    collab_model = CollaborativeRecommender(interaction_df)
    except Exception as exc:
        logger.warning("Worker: collaborative model data load failed: %s", exc)

    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

    return {
        "content": content_model,
        "collab": collab_model,
        "hybrid": hybrid_model,
        "item_df": item_df,
        "ready": True,
    }


def _get_worker_models() -> dict:
    """
    Return the worker-local model dict, rebuilding when the API has published
    a newer version token to Redis.

    Thread-safe: a module-level lock prevents concurrent rebuilds inside a
    single worker process (relevant when Celery uses a threaded pool).

    Fallback behaviour
    ------------------
    * If Redis is unreachable and the worker already has a cached model, the
      cached model is returned (best-effort, potentially stale).
    * If Redis is unreachable and no cached model exists the function
      attempts a build anyway; if Supabase is also unreachable a
      ``RuntimeError`` propagates to the task, which will retry.
    """
    global _worker_model_version, _worker_models

    current_version = _read_current_version()

    if _worker_models is not None and _worker_models.get("ready"):
        if current_version is None or _worker_model_version == current_version:
            return _worker_models

    with _worker_lock:
        if _worker_models is not None and _worker_models.get("ready"):
            if current_version is None or _worker_model_version == current_version:
                return _worker_models

        logger.info(
            "Worker: model cache miss (local_version=%s, current_version=%s). "
            "Rebuilding from Supabase.",
            _worker_model_version,
            current_version,
        )
        _worker_models = _build_models_for_worker()
        _worker_model_version = current_version
        return _worker_models

# ── Celery Tasks ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.get_recommendations")
@serialize_user_requests
def get_recommendations(user_id: str, top_n: int = 10) -> list[dict]:
    """
    Generate hybrid recommendations for a user.
    """
    try:
        models = _get_worker_models()
        hybrid_model = models["hybrid"]
        recs = hybrid_model.recommend(user_id, top_n=top_n)
        return recs
    except Exception as exc:
        logger.error("Worker: recommendation generation failed for user %s: %s", user_id, exc)
        raise
