from fastapi.testclient import TestClient
from types import SimpleNamespace
from datetime import datetime
from backend import main

client = TestClient(main.app)


class FakeQuery:
    def __init__(self, data, should_fail=False):
        self.data = data
        self.should_fail = should_fail
        self.calls = []

    def select(self, *args, **kwargs):
        self.calls.append(("select", args, kwargs))
        return self

    def gte(self, *args, **kwargs):
        self.calls.append(("gte", args, kwargs))
        return self

    def limit(self, *args, **kwargs):
        self.calls.append(("limit", args, kwargs))
        return self

    def execute(self):
        if self.should_fail:
            raise RuntimeError("Database query failed")
        return SimpleNamespace(data=self.data)


class FakeSupabase:
    def __init__(self, table_query):
        self.table_query = table_query

    def table(self, name):
        assert name == "purchases"
        return self.table_query


def test_trending_validation_bounds():
    response = client.get("/api/trending?days=-1")
    assert response.status_code == 422

    response = client.get("/api/trending?days=366")
    assert response.status_code == 422

    response = client.get("/api/trending?limit=0")
    assert response.status_code == 422

    response = client.get("/api/trending?limit=101")
    assert response.status_code == 422


def test_trending_empty_db(monkeypatch):
    # Reset cache
    main.TRENDING_CACHE = {"data": None, "timestamp": None}
    
    query_mock = FakeQuery([])
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/trending")
    assert response.status_code == 200
    
    payload = response.json()
    assert "results" in payload
    assert payload["results"] == []


def test_trending_success(monkeypatch):
    # Reset cache
    main.TRENDING_CACHE = {"data": None, "timestamp": None}
    
    mock_purchases = [
        {
            "product_id": 101,
            "rating": 5.0,
            "products": {
                "id": 101,
                "title": "Book A",
                "category": "Books",
                "rating": 4.5,
                "avg_sentiment": 0.8,
                "review_count": 10
            }
        },
        {
            "product_id": 202,
            "rating": 3.0,
            "products": {
                "id": 202,
                "title": "Laptop",
                "category": "Tech",
                "rating": 4.0,
                "avg_sentiment": 0.5,
                "review_count": 5
            }
        }
    ]
    query_mock = FakeQuery(mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/trending?days=30&limit=5")
    assert response.status_code == 200
    
    payload = response.json()
    assert "results" in payload
    results = payload["results"]
    assert len(results) > 0
    # Book A has higher rating and count
    assert results[0]["id"] == 101
    assert results[0]["title"] == "Book A"


def test_trending_cache_hits(monkeypatch):
    # Reset cache
    main.TRENDING_CACHE = {"data": None, "timestamp": None}
    
    mock_purchases = [
        {
            "product_id": 101,
            "rating": 5.0,
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
