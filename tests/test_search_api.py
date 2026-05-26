"""
Integration tests for the /api/search endpoint.
"""
import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main


PRODUCTS = [
    {
        "id": 1,
        "title": "Wireless Headphones",
        "description": "Noise cancelling headphones with a comfortable fit." * 8,
        "category": "Audio",
        "rating": 4.8,
        "avg_sentiment": 0.7,
        "review_count": 120,
        "rank": 0.91,
    },
    {
        "id": 2,
        "title": "Laptop Stand",
        "description": "Adjustable aluminum stand for desks.",
        "category": "Accessories",
        "rating": 4.4,
        "avg_sentiment": 0.4,
        "review_count": 80,
    },
]


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def select(self, *args, **kwargs):
        self.calls.append(("select", args, kwargs))
        return self

    def ilike(self, *args, **kwargs):
        self.calls.append(("ilike", args, kwargs))
        return self

    def order(self, *args, **kwargs):
        self.calls.append(("order", args, kwargs))
        return self

    def limit(self, *args, **kwargs):
        self.calls.append(("limit", args, kwargs))
        return self

    def offset(self, *args, **kwargs):
        self.calls.append(("offset", args, kwargs))
        return self

    def execute(self):
        self.calls.append(("execute", (), {}))
        return SimpleNamespace(data=list(self.data))


class FakeSupabase:
    def __init__(self, rpc_data=None, table_data=None, rpc_error=None):
        self.rpc_data = rpc_data if rpc_data is not None else PRODUCTS[:1]
        self.table_query = FakeQuery(table_data if table_data is not None else PRODUCTS)
        self.rpc_error = rpc_error
        self.rpc_calls = []

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        if self.rpc_error:
            raise self.rpc_error
        return FakeQuery(self.rpc_data)

    def table(self, name):
        assert name == "products"
        return self.table_query


def test_search_empty_query_returns_top_rated_products(monkeypatch):
    fake_supabase = FakeSupabase(table_data=PRODUCTS)
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search?limit=2&offset=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_fallback"] is True
    assert payload["query"] == ""
    assert payload["total"] == 2
    assert payload["results"][0]["title"] == "Wireless Headphones"
    assert len(payload["results"][0]["description"]) == 200
    assert ("offset", (5,), {}) in fake_supabase.table_query.calls


def test_search_query_uses_postgres_rpc(monkeypatch):
    fake_supabase = FakeSupabase(rpc_data=PRODUCTS[:1])
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search?q=headphones&limit=3&offset=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_fallback"] is False
    assert payload["query"] == "headphones"
    assert payload["results"][0]["rank"] == 0.91
    assert fake_supabase.rpc_calls == [
        (
            "search_products",
            {"query_text": "headphones", "match_count": 3, "offset_val": 2},
        )
    ]


def test_search_query_falls_back_to_title_match_when_rpc_fails(monkeypatch):
    fake_supabase = FakeSupabase(table_data=PRODUCTS[1:], rpc_error=RuntimeError("rpc unavailable"))
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search?q=stand&limit=4")

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["title"] == "Laptop Stand"
    assert payload["results"][0]["rank"] == 0.0
    assert ("ilike", ("title", "%stand%"), {}) in fake_supabase.table_query.calls


def test_search_rejects_oversized_query(monkeypatch):
    fake_supabase = FakeSupabase()
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search", params={"q": "a" * (main.MAX_SEARCH_QUERY_LENGTH + 1)})

    assert response.status_code == 400
    assert response.json()["detail"] == "Search query must be 120 characters or fewer."
    assert fake_supabase.rpc_calls == []


def test_search_normalizes_query_before_cache_and_rpc(monkeypatch):
    fake_supabase = FakeSupabase(rpc_data=PRODUCTS[:1])
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search", params={"q": "  wireless   headphones  "})

    assert response.status_code == 200
    assert response.json()["query"] == "wireless headphones"
    assert fake_supabase.rpc_calls[0][1]["query_text"] == "wireless headphones"


def test_search_fallback_escapes_like_wildcards(monkeypatch):
    fake_supabase = FakeSupabase(table_data=PRODUCTS[1:], rpc_error=RuntimeError("rpc unavailable"))
    monkeypatch.setattr(main, "get_supabase", lambda: fake_supabase)
    client = TestClient(main.app)

    response = client.get("/api/search", params={"q": r"50%_off\sale"})

    assert response.status_code == 200
    assert ("ilike", ("title", r"%50\%\_off\\sale%"), {}) in fake_supabase.table_query.calls
