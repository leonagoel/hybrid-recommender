from fastapi.testclient import TestClient
from types import SimpleNamespace
from backend import main

client = TestClient(main.app)


class FakeTableQuery:
    def __init__(self, data, should_fail=False):
        self.data = data
        self.should_fail = should_fail
        self.calls = []

    def select(self, *args, **kwargs):
        self.calls.append(("select", args, kwargs))
        return self

    def ilike(self, *args, **kwargs):
        self.calls.append(("ilike", args, kwargs))
        return self

    def limit(self, *args, **kwargs):
        self.calls.append(("limit", args, kwargs))
        return self

    def execute(self):
        if self.should_fail:
            raise RuntimeError("Database query failed")
        return SimpleNamespace(data=self.data)


class FakeSupabase:
    def __init__(self, table_query=None):
        self.table_query = table_query or FakeTableQuery([])

    def table(self, name):
        assert name == "products"
        return self.table_query


def test_autocomplete_success(monkeypatch):
    mock_data = [
        {"title": "Anime Naruto"},
        {"title": "Naruto Shippuden"},
        {"title": "  Naruto  "},  # needs stripping
        {"title": "anime naruto"}  # case insensitivity duplicate
    ]
    query_mock = FakeTableQuery(mock_data)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/autocomplete?q=naruto&limit=3")
    assert response.status_code == 200
    
    payload = response.json()
    assert "suggestions" in payload
    # Stripped and deduplicated
    assert payload["suggestions"] == ["Anime Naruto", "Naruto Shippuden", "Naruto"]


def test_autocomplete_empty_query(monkeypatch):
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase())
    # If q is empty, min_length=1 constraint on Query will trigger 422 Unprocessable Entity
    response = client.get("/api/autocomplete?q=")
    assert response.status_code == 422


def test_autocomplete_oversized_query(monkeypatch):
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase())
    # max limit is MAX_SEARCH_QUERY_LENGTH (120)
    response = client.get("/api/autocomplete", params={"q": "a" * 121})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_autocomplete_db_failure_returns_500(monkeypatch):
    query_mock = FakeTableQuery([], should_fail=True)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.get("/api/autocomplete?q=naruto")
    assert response.status_code == 500
