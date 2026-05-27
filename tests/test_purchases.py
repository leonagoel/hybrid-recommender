from fastapi.testclient import TestClient
from types import SimpleNamespace
from backend import main

client = TestClient(main.app)


class FakeQuery:
    def __init__(self, data):
        self.data = data

    def insert(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)


class FakeSupabase:
    def __init__(self, table_query):
        self.table_query = table_query

    def table(self, name):
        assert name == "purchases"
        return self.table_query


def test_create_purchase_validation_failures():
    # Empty user_id should fail
    response = client.post("/api/purchases", json={"user_id": "", "product_id": 123})
    assert response.status_code == 422

    # Negative product_id should fail
    response = client.post("/api/purchases", json={"user_id": "user123", "product_id": -5})
    assert response.status_code == 422

    # Out of bounds rating should fail
    response = client.post("/api/purchases", json={"user_id": "user123", "product_id": 123, "rating": 6.0})
    assert response.status_code == 422


def test_create_purchase_success(monkeypatch):
    mock_response = [{"id": 1, "user_id": "user123", "product_id": 123, "rating": 4.5, "review_text": "Good!"}]
    query_mock = FakeQuery(mock_response)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    response = client.post(
        "/api/purchases",
        json={"user_id": "user123", "product_id": 123, "rating": 4.5, "review_text": "Good!"}
    )
    assert response.status_code == 200
    
    payload = response.json()
    assert "purchase" in payload
    assert payload["purchase"][0]["user_id"] == "user123"
