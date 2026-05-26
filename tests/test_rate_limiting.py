"""
Regression tests for API rate limiting.
"""
import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main


class FakeSupabaseQuery:
    def select(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def offset(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=[{
            "id": 1,
            "title": "Rate Limited Product",
            "description": "Test product",
            "category": "Testing",
            "rating": 4.2,
            "avg_sentiment": 0.3,
            "review_count": 12,
        }])


class FakeSupabase:
    def table(self, name):
        assert name == "products"
        return FakeSupabaseQuery()


def setup_function():
    main._rate_limit_buckets.clear()


def teardown_function():
    main._rate_limit_buckets.clear()


def test_search_rate_limit_returns_headers_before_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_SEARCH_PER_MIN", "2")
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase())
    client = TestClient(main.app)

    response = client.get("/api/search")

    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "1"
    assert "x-ratelimit-reset" in response.headers


def test_search_rate_limit_rejects_excess_requests(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_SEARCH_PER_MIN", "2")
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase())
    client = TestClient(main.app)

    assert client.get("/api/search").status_code == 200
    assert client.get("/api/search").status_code == 200
    response = client.get("/api/search")

    assert response.status_code == 429
    assert response.json() == {
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
    }
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "0"


def test_non_limited_endpoint_does_not_emit_rate_limit_headers():
    client = TestClient(main.app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert "x-ratelimit-limit" not in response.headers
