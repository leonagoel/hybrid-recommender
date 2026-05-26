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


def test_version_endpoint():
    response = client.get("/api/version")

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "version": "3.0",
        "service": "Hybrid Recommender API",
        "status": "running",
    }
