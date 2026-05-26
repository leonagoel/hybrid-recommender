import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app, models


class FakeHybridModel:
    def __init__(self):
        self.last_title = None

    def recommend(self, title, top_n=10, explain=False):
        self.last_title = title
        return [{"title": "Related Item", "hybrid_score": 0.91}]

    def get_weights(self):
        return {"content": 0.5, "collaborative": 0.3, "sentiment": 0.2}


def test_recommend_accepts_reserved_characters_in_query_title():
    hybrid = FakeHybridModel()
    original_ready = models["ready"]
    original_hybrid = models["hybrid"]
    models["ready"] = True
    models["hybrid"] = hybrid

    try:
        client = TestClient(app)
        response = client.get("/api/recommend", params={"title": "AC/DC Greatest Hits? Deluxe + Café", "top_n": 12})
    finally:
        models["ready"] = original_ready
        models["hybrid"] = original_hybrid

    assert response.status_code == 200
    assert response.json()["query_item"] == "AC/DC Greatest Hits? Deluxe + Café"
    assert hybrid.last_title == "AC/DC Greatest Hits? Deluxe + Café"
