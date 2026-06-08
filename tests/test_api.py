from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert isinstance(data["model_loaded"], bool)


def test_categories_endpoint_handles_failures_gracefully(monkeypatch):
    from backend import main
    # Mock get_supabase to raise an error
    monkeypatch.setattr(main, "get_supabase", lambda: None)
    
    # We expect that if it fails completely, it raises AttributeError (since None has no table/rpc)
    # and get_categories catches it and returns {"categories": []}
    response = client.get("/api/categories")
    assert response.status_code == 200
    assert response.json() == {"categories": []}
def test_version_endpoint():
    response = client.get("/api/version")

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "version": "3.0",
        "service": "Hybrid Recommender API",
        "status": "running",
    }


def test_list_items_endpoint(monkeypatch):
    from backend import main

    class MockTable:
        def __init__(self, name):
            self.name = name
            self.data = []
            self.count = 100

        def select(self, fields, count=None):
            return self

        def order(self, field, desc=True):
            return self

        def range(self, start, end):
            self.data = [
                {
                    'id': i,
                    'title': f'Product {i}',
                    'category': 'Electronics',
                    'rating': 4.5,
                    'avg_sentiment': 0.8,
                    'description': 'Description'
                }
                for i in range(start, min(end + 1, 100))
            ]
            return self

        def limit(self, limit_val):
            return self

        def execute(self):
            class Result:
                def __init__(self, data, count):
                    self.data = data
                    self.count = count
            return Result(self.data, self.count)

    class MockSupabase:
        def table(self, name):
            return MockTable(name)

    monkeypatch.setattr(main, "get_supabase", lambda: MockSupabase())

    # Call with page=2 and per_page=10
    response = client.get("/api/items?page=2&per_page=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 10
    assert data["total"] == 100
    assert data["page"] == 2
    assert data["limit"] == 10
    assert data["has_more"] is True

