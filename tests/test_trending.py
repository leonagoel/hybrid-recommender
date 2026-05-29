"""
Tests for the /api/trending endpoint.

Covers:
- Parameter validation bounds.
- Empty result set when no purchases exist in the window.
- Correct Bayesian ranking when purchases are present.
- Cache key is scoped to (days, limit) so different parameter
  combinations never overwrite each other.
- Hard row-cap fallback (TRENDING_FETCH_LIMIT) is applied when the RPC
  is unavailable.
- Graceful handling of a database error in the fallback path.
- RPC result path aggregates pre-summed rows correctly.
"""

import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main

client = TestClient(main.app)


# ── Fake Supabase plumbing ────────────────────────────────────────────

class FakeQuery:
    """Chainable fake for sb.table(...).select(...).gte(...).limit(...).execute()."""

    def __init__(self, data, should_fail=False):
        self._data = data
        self._should_fail = should_fail
        self.calls = []
        self.applied_limit = None

    def select(self, *args, **kwargs):
        self.calls.append(("select", args))
        return self

    def gte(self, *args, **kwargs):
        self.calls.append(("gte", args))
        return self

    def limit(self, n, *args, **kwargs):
        self.applied_limit = n
        self.calls.append(("limit", n))
        return self

    def execute(self):
        if self._should_fail:
            raise RuntimeError("Database query failed")
        return SimpleNamespace(data=self._data)


class FakeRPC:
    """Fake for sb.rpc(...).execute() that can succeed or fail."""

    def __init__(self, data=None, should_fail=False):
        self._data = data
        self._should_fail = should_fail

    def execute(self):
        if self._should_fail:
            raise RuntimeError("RPC unavailable")
        return SimpleNamespace(data=self._data)


class FakeSupabase:
    def __init__(self, table_query=None, rpc_response=None, rpc_fails=True):
        self._table_query = table_query
        self._rpc_response = rpc_response
        self._rpc_fails = rpc_fails

    def table(self, name):
        return self._table_query

    def rpc(self, name, params):
        if self._rpc_fails:
            raise RuntimeError("RPC unavailable")
        return self._rpc_response


# ── Helpers ───────────────────────────────────────────────────────────

def _make_purchase(product_id: int, rating: float, title: str = "Product",
                   category: str = "Cat", prod_rating: float = 4.0) -> dict:
    return {
        "product_id": product_id,
        "rating": rating,
        "products": {
            "id": product_id,
            "title": title,
            "category": category,
            "rating": prod_rating,
            "avg_sentiment": 0.5,
            "review_count": 10,
        },
    }


def _clear_trending_cache():
    """Remove all trending cache entries from the shared response cache."""
    main._clear_response_cache()


# ── Parameter validation ──────────────────────────────────────────────

def test_trending_rejects_days_below_minimum():
    response = client.get("/api/trending?days=0")
    assert response.status_code == 422


def test_trending_rejects_days_above_maximum():
    response = client.get("/api/trending?days=366")
    assert response.status_code == 422


def test_trending_rejects_limit_zero():
    response = client.get("/api/trending?limit=0")
    assert response.status_code == 422


def test_trending_rejects_limit_above_maximum():
    response = client.get("/api/trending?limit=101")
    assert response.status_code == 422


def test_trending_validation_bounds():
    assert client.get("/api/trending?days=-1").status_code == 422
    assert client.get("/api/trending?days=366").status_code == 422
    assert client.get("/api/trending?limit=0").status_code == 422
    assert client.get("/api/trending?limit=101").status_code == 422


# ── Empty result set ──────────────────────────────────────────────────

def test_trending_empty_db(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([])
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload
    assert payload["results"] == []


def test_trending_returns_empty_when_rpc_and_fallback_return_nothing(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([])
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending?days=30&limit=5")
    assert response.status_code == 200
    assert response.json()["results"] == []


# ── Correct ranking ───────────────────────────────────────────────────

def test_trending_success(monkeypatch):
    _clear_trending_cache()
    mock_purchases = [
        _make_purchase(101, 5.0, "Book A", "Books", 4.5),
        _make_purchase(202, 3.0, "Laptop", "Tech", 4.0),
    ]
    query_mock = FakeQuery(mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending?days=30&limit=5")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    assert results[0]["id"] == 101
    assert results[0]["title"] == "Book A"


def test_trending_ranked_by_trending_score_descending(monkeypatch):
    _clear_trending_cache()
    purchases = [
        _make_purchase(1, 5.0, "High", "A", 5.0),
        _make_purchase(2, 1.0, "Low", "B", 1.0),
        _make_purchase(1, 5.0, "High", "A", 5.0),  # second purchase — higher count
    ]
    query_mock = FakeQuery(purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending?days=7&limit=10")
    assert response.status_code == 200
    results = response.json()["results"]
    scores = [r["trending_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by trending_score DESC"


def test_trending_response_contains_required_fields(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([_make_purchase(1, 4.0, "Widget", "Tools")])
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload
    assert "days" in payload
    assert "limit" in payload
    item = payload["results"][0]
    for field in ("id", "title", "category", "rating", "interaction_count",
                  "bayesian_rating", "trending_score"):
        assert field in item, f"Missing field: {field}"


# ── Cache is scoped to (days, limit) ─────────────────────────────────

def test_trending_different_days_params_cached_independently(monkeypatch):
    _clear_trending_cache()
    call_count = {"n": 0}

    def make_supabase():
        call_count["n"] += 1
        purchases = [_make_purchase(call_count["n"], 4.0, f"Product {call_count['n']}")]
        return FakeSupabase(table_query=FakeQuery(purchases))

    monkeypatch.setattr(main, "get_supabase", make_supabase)

    r7 = client.get("/api/trending?days=7&limit=10").json()
    r30 = client.get("/api/trending?days=30&limit=10").json()

    # Two separate DB calls must have been made — different cache keys.
    assert call_count["n"] == 2, "Separate (days, limit) keys must each trigger a DB call"
    # The results should differ because each call seeded different data.
    assert r7["results"][0]["id"] != r30["results"][0]["id"]


def test_trending_different_limit_params_cached_independently(monkeypatch):
    _clear_trending_cache()
    call_count = {"n": 0}

    def make_supabase():
        call_count["n"] += 1
        purchases = [_make_purchase(call_count["n"], 4.0, f"Item {call_count['n']}")]
        return FakeSupabase(table_query=FakeQuery(purchases))

    monkeypatch.setattr(main, "get_supabase", make_supabase)

    client.get("/api/trending?days=7&limit=5")
    client.get("/api/trending?days=7&limit=10")

    assert call_count["n"] == 2, "Different limit values must produce separate cache entries"


def test_trending_same_params_uses_cache(monkeypatch):
    _clear_trending_cache()
    call_count = {"n": 0}

    def make_supabase():
        call_count["n"] += 1
        return FakeSupabase(table_query=FakeQuery([_make_purchase(1, 4.0, "Cached")]))

    monkeypatch.setattr(main, "get_supabase", make_supabase)

    client.get("/api/trending?days=7&limit=5")
    client.get("/api/trending?days=7&limit=5")

    assert call_count["n"] == 1, "Same (days, limit) must be served from cache on second call"


# ── Fallback row-cap behaviour ────────────────────────────────────────

def test_trending_fallback_applies_limit(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([_make_purchase(1, 4.0, "Widget")])
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    client.get("/api/trending")

    assert query_mock.applied_limit is not None, "Fallback query must have a .limit() applied"
    assert query_mock.applied_limit <= main.TRENDING_FETCH_LIMIT


def test_trending_fallback_db_error_returns_empty(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([], should_fail=True)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(table_query=query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    assert response.json()["results"] == []


# ── RPC aggregation path ──────────────────────────────────────────────

def test_trending_rpc_path_returns_ranked_results(monkeypatch):
    _clear_trending_cache()
    rpc_data = [
        {
            "product_id": 10,
            "purchase_count": 50,
            "avg_rating": 4.8,
            "title": "Popular Widget",
            "category": "Tools",
            "rating": 4.5,
            "avg_sentiment": 0.9,
            "review_count": 100,
        },
        {
            "product_id": 20,
            "purchase_count": 10,
            "avg_rating": 3.0,
            "title": "Niche Gadget",
            "category": "Misc",
            "rating": 3.0,
            "avg_sentiment": 0.4,
            "review_count": 12,
        },
    ]
    fake_rpc = FakeRPC(data=rpc_data, should_fail=False)
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_response=fake_rpc, rpc_fails=False)
    )

    response = client.get("/api/trending?days=7&limit=10")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] == 10, "Product with higher purchase_count must rank first"
    assert results[0]["title"] == "Popular Widget"


def test_trending_rpc_failure_falls_back_to_table_query(monkeypatch):
    _clear_trending_cache()
    query_mock = FakeQuery([_make_purchase(99, 4.5, "Fallback Product")])
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(table_query=query_mock, rpc_fails=True)
    )

    response = client.get("/api/trending?days=7&limit=10")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == 99


# ── Cache hit after RPC success ───────────────────────────────────────

def test_trending_rpc_result_is_cached(monkeypatch):
    _clear_trending_cache()
    rpc_call_count = {"n": 0}

    class CountingRPC:
        def execute(self):
            rpc_call_count["n"] += 1
            return SimpleNamespace(data=[{
                "product_id": 1,
                "purchase_count": 20,
                "avg_rating": 4.0,
                "title": "Counted",
                "category": "X",
                "rating": 4.0,
                "avg_sentiment": 0.5,
                "review_count": 20,
            }])

    class CountingSupabase:
        def table(self, _name):
            return FakeQuery([])

        def rpc(self, _name, _params):
            return CountingRPC()

    monkeypatch.setattr(main, "get_supabase", CountingSupabase)

    client.get("/api/trending?days=7&limit=5")
    client.get("/api/trending?days=7&limit=5")

    assert rpc_call_count["n"] == 1, "RPC must be called only once; second request served from cache"
    query_mock = FakeQuery(mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    # First call will populate cache because results are not empty
    first_response = client.get("/api/trending")
    assert first_response.status_code == 200
    
    # Second call should use cache even if get_supabase raises error
    def failing_supabase():
        raise RuntimeError("Should not be called because of cache!")
        
    monkeypatch.setattr(main, "get_supabase", failing_supabase)
    second_response = client.get("/api/trending")
    assert second_response.status_code == 200
    assert second_response.json() == first_response.json()


def test_trending_negative_ratings(monkeypatch):
    # Reset cache
    main.TRENDING_CACHE = {"data": None, "timestamp": None}
    
    mock_purchases = [
        {
            "product_id": 101,
            "rating": -1.0,
            "products": {
                "id": 101,
                "title": "Book A",
                "category": "Books",
                "rating": 4.5,
                "avg_sentiment": 0.8,
                "review_count": 10
            }
        }
    ]
    query_mock = FakeQuery(mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    assert results[0]["bayesian_rating"] < 3.0


def test_trending_missing_product_details(monkeypatch):
    # Reset cache
    main.TRENDING_CACHE = {"data": None, "timestamp": None}
    
    mock_purchases = [
        {
            "product_id": 101,
            "rating": 4.0,
            "products": {
                "id": 101,
                "title": "Book A"
                # Missing category, rating, sentiment, count
            }
        }
    ]
    query_mock = FakeQuery(mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    assert results[0]["category"] == ""
    assert results[0]["rating"] == 0

