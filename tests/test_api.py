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
