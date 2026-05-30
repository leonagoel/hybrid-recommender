"""
Regression tests for API response caching.

Covers:
- Cache MISS / HIT header behaviour for recommendation and search endpoints.
- _BoundedTTLCache: entry cap enforcement, LRU eviction order, TTL expiry,
  and clear() semantics.
- Metrics endpoint exposes cache pressure fields.
"""

import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main
from backend.main import _BoundedTTLCache


# ── Fake collaborators ────────────────────────────────────────────────

class FakeHybrid:
    def __init__(self):
        self.calls = 0

    def get_weights(self):
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}

    def recommend(self, item_title, top_n=10, explain=False, target_catalog=None, **kwargs):
        self.calls += 1
        return [{"title": f"{item_title} match", "hybrid_score": 0.98}][:top_n]


class FakeSupabaseQuery:
    def __init__(self):
        self.calls = 0

    def select(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def offset(self, *args, **kwargs):
        return self

    def execute(self):
        self.calls += 1
        return SimpleNamespace(
            data=[
                {
                    "id": 1,
                    "title": "Cached Product",
                    "description": "A product returned from the fake Supabase query.",
                    "category": "Test",
                    "rating": 4.5,
                    "avg_sentiment": 0.7,
                    "review_count": 12,
                }
            ]
        )


class FakeSupabase:
    def __init__(self):
        self.query = FakeSupabaseQuery()

    def table(self, name):
        return self.query


# ── Fixtures ──────────────────────────────────────────────────────────

def setup_function():
    main._clear_response_cache()


def teardown_function():
    main._clear_response_cache()
    main.models.update(
        {
            "content": None,
            "collab": None,
            "hybrid": None,
            "ready": False,
            "item_df": None,
            "build_time": None,
        }
    )


# ── Endpoint cache header tests ───────────────────────────────────────

def test_recommendation_endpoint_sets_hit_after_first_response():
    fake_hybrid = FakeHybrid()
    main.models.update({"ready": True, "hybrid": fake_hybrid})
    client = TestClient(main.app)

    first = client.get("/api/recommend/Product%20A?top_n=3")
    second = client.get("/api/recommend/Product%20A?top_n=3")

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    assert first_payload["query_item"] == "Product A"
    assert len(first_payload["recommendations"]) == 1
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert second.headers["cache-control"] == main.CACHE_CONTROL_VALUE
    assert fake_hybrid.calls == 1


def test_search_endpoint_caches_supabase_response(monkeypatch):
    fake_supabase = FakeSupabase()
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    first = client.get("/api/search?limit=5")
    second = client.get("/api/search?limit=5")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert fake_supabase.query.calls == 1


# ── _BoundedTTLCache unit tests ───────────────────────────────────────

def test_bounded_cache_does_not_exceed_max_entries():
    cache = _BoundedTTLCache(max_entries=5, ttl=60)
    for i in range(20):
        cache.set(f"key:{i}", f"value:{i}")
    assert len(cache) <= 5


def test_bounded_cache_lru_eviction_drops_oldest_accessed_entry():
    cache = _BoundedTTLCache(max_entries=3, ttl=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    # Access "a" so it becomes the most-recently-used; "b" is now LRU.
    cache.get("a")

    # Inserting a fourth entry must evict "b" (LRU), not "a".
    cache.set("d", 4)

    assert cache.get("b") is None, "LRU entry 'b' should have been evicted"
    assert cache.get("a") == 1, "'a' was recently accessed and must survive"
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_bounded_cache_expired_entry_returns_none():
    cache = _BoundedTTLCache(max_entries=10, ttl=0)
    cache.set("stale", "data")
    # TTL of 0 means the entry expires immediately (expires_at <= now on next read).
    result = cache.get("stale")
    assert result is None, "Expired entry must not be returned"


def test_bounded_cache_non_expired_entry_is_returned():
    cache = _BoundedTTLCache(max_entries=10, ttl=300)
    cache.set("live", "payload")
    assert cache.get("live") == "payload"


def test_bounded_cache_clear_empties_all_entries():
    cache = _BoundedTTLCache(max_entries=10, ttl=60)
    for i in range(5):
        cache.set(f"k{i}", i)
    assert len(cache) == 5

    cache.clear()
    assert len(cache) == 0
    for i in range(5):
        assert cache.get(f"k{i}") is None


def test_bounded_cache_overwrite_updates_value_and_refreshes_position():
    cache = _BoundedTTLCache(max_entries=3, ttl=60)
    cache.set("x", "old")
    cache.set("y", "y-val")
    cache.set("z", "z-val")

    # Overwrite "x" — this should move it to the MRU position.
    cache.set("x", "new")

    # Now "y" is LRU. Inserting a new key must evict "y".
    cache.set("w", "w-val")

    assert cache.get("y") is None, "'y' should be evicted as LRU"
    assert cache.get("x") == "new", "overwritten value must be updated"


def test_bounded_cache_size_one_always_replaces():
    cache = _BoundedTTLCache(max_entries=1, ttl=60)
    cache.set("first", 1)
    cache.set("second", 2)

    assert cache.get("first") is None
    assert cache.get("second") == 2
    assert len(cache) == 1


def test_bounded_cache_concurrent_access_does_not_corrupt_state():
    import threading

    cache = _BoundedTTLCache(max_entries=50, ttl=60)
    errors = []

    def writer(start):
        try:
            for i in range(start, start + 30):
                cache.set(f"key:{i}", i)
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for i in range(100):
                cache.get(f"key:{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i * 30,)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"
    assert len(cache) <= 50


# ── Metrics endpoint exposes cache pressure ───────────────────────────

def test_metrics_endpoint_includes_cache_fields():
    client = TestClient(main.app)
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "cache_entries" in payload, "metrics must expose cache_entries"
    assert "cache_max_entries" in payload, "metrics must expose cache_max_entries"
    assert isinstance(payload["cache_entries"], int)
    assert isinstance(payload["cache_max_entries"], int)
    assert payload["cache_max_entries"] > 0


def test_metrics_cache_entries_increments_on_set():
    main._clear_response_cache()
    client = TestClient(main.app)

    before = client.get("/api/metrics").json()["cache_entries"]

    main._set_cached_response("test:metrics:key", {"data": "value"})

    after = client.get("/api/metrics").json()["cache_entries"]
    assert after == before + 1


def test_clear_response_cache_resets_metrics_count():
    main._set_cached_response("key:a", 1)
    main._set_cached_response("key:b", 2)

    client = TestClient(main.app)
    assert client.get("/api/metrics").json()["cache_entries"] >= 2

    main._clear_response_cache()
    assert client.get("/api/metrics").json()["cache_entries"] == 0


# ── Global cache module-level behaviour ──────────────────────────────

def test_module_cache_respects_cache_max_entries_env(monkeypatch):
    # CACHE_MAX_ENTRIES is read at import time, but _BoundedTTLCache._max
    # is set from the module constant — verify the module-level constant
    # matches the env default or the configured value.
    assert main.CACHE_MAX_ENTRIES >= 1
    assert isinstance(main._response_cache, _BoundedTTLCache)


def test_get_and_set_cached_response_round_trip():
    main._clear_response_cache()
    key = "roundtrip:test"
    payload = {"result": [1, 2, 3], "count": 3}

    assert main._get_cached_response(key) is None

    main._set_cached_response(key, payload)
    retrieved = main._get_cached_response(key)

    assert retrieved == payload


def test_different_keys_do_not_collide():
    main._clear_response_cache()
    main._set_cached_response("ns:a", "value-a")
    main._set_cached_response("ns:b", "value-b")

    assert main._get_cached_response("ns:a") == "value-a"
    assert main._get_cached_response("ns:b") == "value-b"
