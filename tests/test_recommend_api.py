import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app, models


class FakeHybridModel:
    def __init__(self):
        self.last_title = None
        self.last_target_catalog = None

    def recommend(self, title, top_n=10, explain=False, target_catalog=None):
        self.last_title = title
        self.last_target_catalog = target_catalog
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


def test_recommend_supports_cross_domain_catalog_filtering():
    hybrid = FakeHybridModel()
    original_ready = models["ready"]
    original_hybrid = models["hybrid"]
    models["ready"] = True
    models["hybrid"] = hybrid

    try:
        client = TestClient(app)
        response = client.get("/api/recommend", params={
            "title": "Query Item",
            "top_n": 5,
            "target_catalog": "Movies"
        })
    finally:
        models["ready"] = original_ready
        models["hybrid"] = original_hybrid

    assert response.status_code == 200
    assert response.json()["target_catalog"] == "Movies"
    assert hybrid.last_target_catalog == "Movies"
