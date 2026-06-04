"""
Regression tests for API response caching.
"""

import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main


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
