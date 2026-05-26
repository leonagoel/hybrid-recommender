from fastapi.testclient import TestClient
from backend.main import app, models


client = TestClient(app)


# ── Mock Hybrid Model ───────────────────────────────────────────────

class MockHybridModel:
    def recommend(self, item_title, top_n=10, explain=False):

        if item_title == "Unknown Product":
            return []

        return [
            {
                "title": "Related Product",
                "hybrid_score": 0.91,
                "content_score": 0.82,
                "collab_score": 0.75,
            }
        ]

    def get_weights(self):
        return {
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.2,
        }


# ── Setup Mock State ────────────────────────────────────────────────

models["ready"] = True
models["hybrid"] = MockHybridModel()


# ── Tests ───────────────────────────────────────────────────────────

def test_valid_product_returns_recommendations():
    response = client.get("/api/recommend/Test Product")

    assert response.status_code == 200

    data = response.json()

    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0


def test_unknown_product_returns_404():
    response = client.get("/api/recommend/Unknown Product")

    assert response.status_code == 404


def test_missing_path_parameter():
    response = client.get("/api/recommend/")

    assert response.status_code == 404


def test_recommendation_response_fields():
    response = client.get("/api/recommend/Test Product")

    assert response.status_code == 200

    data = response.json()

    recommendation = data["recommendations"][0]

    assert "title" in recommendation
    assert "hybrid_score" in recommendation
    assert "content_score" in recommendation
    assert "collab_score" in recommendation