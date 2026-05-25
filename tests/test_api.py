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