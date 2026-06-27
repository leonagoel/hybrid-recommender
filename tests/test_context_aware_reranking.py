"""
Unit tests for context-aware recommendation reranking based on weather and time of day.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.hybrid_model import HybridRecommender
from backend import main


class MockModel:
    pass


def setup_function():
    main._clear_response_cache()


def teardown_function():
    main._clear_response_cache()
    main.models.update(
        {
            "content": None,
            "collab": None,
            "hybrid": None,
            "ready": False,
            "item_df": None,
            "build_time": None,
        }
    )


def test_context_rerank_weather():
    results = [
        {"title": "Rain Coat", "category": "Apparel", "hybrid_score": 0.8},
        {"title": "Beach Volleyball", "category": "Sports", "hybrid_score": 0.5},
        {"title": "Warm Hoodie", "category": "Apparel", "hybrid_score": 0.7},
    ]

    hybrid = HybridRecommender(MockModel(), None, None)

    # Rerank with sunny weather (sports and apparel boosted)
    reranked_sunny = hybrid._context_rerank(results, weather="sunny")
    # Rain Coat: Apparel gets +0.2 boost -> 0.8 * 1.2 = 0.96
    # Beach Volleyball: Sports gets +0.2 boost -> 0.5 * 1.2 = 0.60
    # Warm Hoodie: Apparel gets +0.2 boost -> 0.7 * 1.2 = 0.84
    assert reranked_sunny[0]["title"] == "Rain Coat"
    assert reranked_sunny[1]["title"] == "Warm Hoodie"
    assert reranked_sunny[2]["title"] == "Beach Volleyball"

    # Rerank with rainy weather (indoor/cozy boosted, and warm/cozy keywords)
    reranked_rainy = hybrid._context_rerank(results, weather="rainy")
    # Warm Hoodie gets +0.15 boost for 'warm' in title -> 0.7 * 1.15 = 0.805
    # Rain Coat gets 0.8
    # Beach Volleyball gets 0.5
    assert reranked_rainy[0]["title"] == "Warm Hoodie"
    assert reranked_rainy[1]["title"] == "Rain Coat"
    assert reranked_rainy[2]["title"] == "Beach Volleyball"


def test_api_recommendation_passes_context():
    class FakeHybrid:
        def __init__(self):
            self.weather = None
            self.time_of_day = None

        def get_weights(self):
            return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}

        def recommend(self, item_title, top_n=10, explain=False, target_catalog=None, **kwargs):
            self.weather = kwargs.get("weather")
            self.time_of_day = kwargs.get("time_of_day")
            return [{"title": "Match", "hybrid_score": 0.9, "category": "Electronics"}]

    fake_hybrid = FakeHybrid()
    main.models.update({"ready": True, "hybrid": fake_hybrid})
    client = TestClient(main.app)

    response = client.get("/api/recommend/Product%20A?weather=sunny&time_of_day=morning")
    print("RESPONSE STATUS:", response.status_code)
    print("RESPONSE HEADERS:", dict(response.headers))
    print("RESPONSE JSON:", response.json())
    print("WEATHER:", fake_hybrid.weather)
    print("TIME_OF_DAY:", fake_hybrid.time_of_day)
    assert response.status_code == 200
    assert fake_hybrid.weather == "sunny"
    assert fake_hybrid.time_of_day == "morning"
