import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main

app = main.app
models = main.models


class FakeCollabModel:
    def __init__(self):
        self._user_to_idx = {"known_user": 0}

class FakeHybridModel:
    def __init__(self):
        self.last_title = None
        self.last_target_catalog = None
        self.collab_model = FakeCollabModel()

    def recommend(self, title, top_n=10, explain=False, target_catalog=None):
        self.last_title = title
        self.last_target_catalog = target_catalog
        return [{"title": "Related Item", "hybrid_score": 0.91}]

    def recommend_for_user(self, user_id, top_n=10, explain=False):
        if user_id == "known_user":
            return [{"title": "Collab Item", "hybrid_score": 0.85}]
        return [{"title": "Popular Item", "hybrid_score": 0.50}]

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
    payload = response.json()
    assert payload["query"] == "AC/DC Greatest Hits? Deluxe + Café"
    assert payload["query_item"] == "AC/DC Greatest Hits? Deluxe + Café"
    assert payload["count"] == 1
    assert payload["results"] == payload["recommendations"]
    assert hybrid.last_title == "AC/DC Greatest Hits? Deluxe + Café"
    main._clear_response_cache()


def test_user_recommendations_known_user():
    hybrid = FakeHybridModel()
    original_ready = models["ready"]
    original_hybrid = models["hybrid"]
    models["ready"] = True
    models["hybrid"] = hybrid

    try:
        client = TestClient(app)
        response = client.get("/api/recommend/user/known_user", params={"top_n": 5})
    finally:
        models["ready"] = original_ready
        models["hybrid"] = original_hybrid

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_user"] == "known_user"
    assert payload["fallback"] is False
    assert len(payload["recommendations"]) == 1
    assert payload["recommendations"][0]["title"] == "Collab Item"
    main._clear_response_cache()


def test_user_recommendations_unknown_user_fallback():
    hybrid = FakeHybridModel()
    original_ready = models["ready"]
    original_hybrid = models["hybrid"]
    models["ready"] = True
    models["hybrid"] = hybrid

    try:
        client = TestClient(app)
        response = client.get("/api/recommend/user/unknown_user", params={"top_n": 5})
    finally:
        models["ready"] = original_ready
        models["hybrid"] = original_hybrid

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_user"] == "unknown_user"
    assert payload["fallback"] is True
    assert len(payload["recommendations"]) == 1
    assert payload["recommendations"][0]["title"] == "Popular Item"
    main._clear_response_cache()


def test_user_recommendations_invalid_query():
    try:
        client = TestClient(app)
        response = client.get("/api/recommend/user/known_user", params={"top_n": 51})
    finally:
        pass
    
    assert response.status_code == 422
    main._clear_response_cache()


def teardown_function():
    main._clear_response_cache()
