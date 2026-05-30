import time
import json
import os
from collections import OrderedDict
from threading import Lock
from typing import Any
from redis import Redis
from redis.exceptions import RedisError

from backend.core.config import CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS

class _BoundedTTLCache:
    def __init__(self, max_entries: int, ttl: int) -> None:
        self._store: OrderedDict = OrderedDict()
        self._max = max(1, max_entries)
        self._ttl = ttl
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.time():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time() + self._ttl, value)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

_response_cache = _BoundedTTLCache(CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS)

try:
    _redis_client = Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        decode_responses=True
    )
    _redis_client.ping()
except Exception:
    _redis_client = None

_metrics_lock = Lock()
_cache_hits = 0
_cache_misses = 0

def _get_cached_response(key: str):
    global _cache_hits, _cache_misses
    
    cached = _response_cache.get(key)
    if cached is not None:
        with _metrics_lock:
            _cache_hits += 1
        return cached

    if _redis_client is not None:
        try:
            cached_str = _redis_client.get(key)
            if cached_str is not None:
                parsed = json.loads(cached_str)
                _response_cache.set(key, parsed)
                with _metrics_lock:
                    _cache_hits += 1
                return parsed
        except (RedisError, json.JSONDecodeError):
            pass

    with _metrics_lock:
        _cache_misses += 1
    return None

def _set_cached_response(key: str, value: Any) -> None:
    _response_cache.set(key, value)
    
    if _redis_client is not None:
        try:
            _redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
        except (RedisError, TypeError):
            pass

def _clear_response_cache() -> None:
    global _cache_hits, _cache_misses
    _response_cache.clear()
    with _metrics_lock:
        _cache_hits = 0
        _cache_misses = 0

def get_cache_metrics_data():
    global _cache_hits, _cache_misses
    with _metrics_lock:
        hits = _cache_hits
        misses = _cache_misses
    return {
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "hits": hits,
        "misses": misses,
        "current_items": len(_response_cache),
    }
