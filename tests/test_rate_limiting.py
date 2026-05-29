"""
Regression tests for API rate limiting.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

import pandas as pd
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
    main._clear_response_cache()


def teardown_function():
    main._rate_limit_buckets.clear()
    main._clear_response_cache()
    main.models.update(
        {
            "content": None,
            "collab": None,
            "hybrid": None,
            "ready": False,
            "item_df": None,
            "build_time": None,
            "last_trained_at": None,
        }
    )


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


class FakeFeedbackInsertResult:
    def __init__(self, data):
        self.data = data


class FakeFeedbackTable:
    def __init__(self):
        self.inserted = []

    def insert(self, payload):
        self.inserted.append(payload)
        return self

    def execute(self):
        return FakeFeedbackInsertResult([self.inserted[-1]])


class FakeFeedbackSupabase:
    def __init__(self):
        self.feedback_table = FakeFeedbackTable()

    def table(self, name):
        assert name == "feedback_submissions"
        return self.feedback_table


def test_feedback_endpoint_uses_rate_limit_scope(monkeypatch):
    calls = []
    fake_supabase = FakeFeedbackSupabase()

    def fake_apply_rate_limit(request, response, scope, limit_env, default_limit):
        calls.append({
            "scope": scope,
            "limit_env": limit_env,
            "default_limit": default_limit,
            "host": request.client.host if request.client else None,
        })
        response.headers["x-ratelimit-limit"] = str(default_limit)
        response.headers["x-ratelimit-remaining"] = str(default_limit - 1)
        response.headers["x-ratelimit-reset"] = "60"
        return None

    monkeypatch.setattr(main, "_apply_rate_limit", fake_apply_rate_limit)
    monkeypatch.setattr(main, "_get_feedback_storage_client", lambda: fake_supabase)

    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"user-agent": "pytest"})
    response = main.Response()
    result = main.submit_feedback(
        main.FeedbackCreate(user_id="user123", item="item1", feedback="Excellent"),
        request,
        response,
        None,
    )

    assert calls[0]["scope"] == "feedback"
    assert calls[0]["limit_env"] == "RATE_LIMIT_FEEDBACK_PER_MIN"
    assert calls[0]["default_limit"] == 20
    assert result["message"] == "Feedback submitted successfully"
    assert fake_supabase.feedback_table.inserted[0]["user_id"] == "user123"
    assert response.headers["x-ratelimit-limit"] == "20"


def test_github_webhook_uses_rate_limit_scope(monkeypatch):
    calls = []

    def fake_apply_rate_limit(request, response, scope, limit_env, default_limit):
        calls.append({
            "scope": scope,
            "limit_env": limit_env,
            "default_limit": default_limit,
            "host": request.client.host if request.client else None,
        })
        response.headers["x-ratelimit-limit"] = str(default_limit)
        response.headers["x-ratelimit-remaining"] = str(default_limit - 1)
        response.headers["x-ratelimit-reset"] = "60"
        return None

    monkeypatch.setattr(main, "_apply_rate_limit", fake_apply_rate_limit)
    monkeypatch.setattr(main, "_verify_github_signature", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "triage_issue", lambda *args, **kwargs: asyncio.sleep(0, result={"triaged": True}))

    class FakeWebhookRequest:
        def __init__(self):
            self.client = SimpleNamespace(host="203.0.113.10")
            self.headers = {
                "X-Hub-Signature-256": "sha256=fake",
                "X-GitHub-Event": "ping",
            }

        async def body(self):
            return b"{}"

        async def json(self):
            return {}

    response = main.Response()
    result = asyncio.run(main.github_webhook(FakeWebhookRequest(), response))

    assert calls[0]["scope"] == "github_webhook"
    assert calls[0]["limit_env"] == "RATE_LIMIT_GITHUB_WEBHOOK_PER_MIN"
    assert calls[0]["default_limit"] == 60
    assert result["status"] == "skipped"
    assert response.headers["x-ratelimit-limit"] == "60"


class FakeHybrid:
    def recommend(self, title, top_n=10, explain=False):
        return [{"title": f"{title} match", "hybrid_score": 0.9}][:top_n]

    def get_weights(self):
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}


def test_recommend_rate_limit_rejects_excess_requests(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RECOMMEND_PER_MIN", "2")
    main.models.update({"ready": True, "hybrid": FakeHybrid()})
    client = TestClient(main.app)

    assert client.get("/api/recommend/Product%20A").status_code == 200
    assert client.get("/api/recommend/Product%20A").status_code == 200
    response = client.get("/api/recommend/Product%20A")

    assert response.status_code == 429
    assert response.json()["error"] == "Rate limit exceeded"
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "0"


def test_similar_rate_limit_rejects_excess_requests(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_SIMILAR_PER_MIN", "2")
    item_df = pd.DataFrame({
        "id": [1],
        "title": ["Product A"],
        "category": ["Testing"],
    })
    main.models.update({
        "ready": True,
        "hybrid": FakeHybrid(),
        "item_df": item_df,
    })
    client = TestClient(main.app)

    assert client.get("/api/similar/1").status_code == 200
    assert client.get("/api/similar/1").status_code == 200
    response = client.get("/api/similar/1")

    assert response.status_code == 429
    assert response.json()["error"] == "Rate limit exceeded"
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "0"
