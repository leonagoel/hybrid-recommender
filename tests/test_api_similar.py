"""
API tests for item-id based similar recommendations.
"""
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app, models
from content_model import ContentRecommender
from hybrid_model import HybridRecommender


@pytest.fixture
def api_client():
    item_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "title": ["Alpha", "Beta", "Gamma", "Desk Stand", "USB Hub"],
        "description": [
            "Wireless headphones with active noise cancellation",
            "Budget earbuds with balanced wireless audio",
            "Premium studio headphones for audio work",
            "Adjustable laptop stand for desks",
            "USB-C hub with HDMI and card reader",
        ],
        "category": ["Audio", "Audio", "Audio", "Accessories", "Accessories"],
        "rating": [4.5, 3.8, 4.9, 4.2, 3.5],
        "review_count": [120, 45, 200, 80, 30],
        "avg_sentiment": [0.6, 0.2, 0.8, 0.5, 0.1],
    })
    item_df["combined"] = (
        item_df["title"] + " " + item_df["description"] + " " + item_df["category"]
    )
    previous = models.copy()
    content_model = ContentRecommender(item_df)
    models.update({
        "content": content_model,
        "collab": None,
        "hybrid": HybridRecommender(content_model, None, item_df),
        "ready": True,
        "item_df": item_df,
        "build_time": 0.01,
    })
    try:
        yield TestClient(app)
    finally:
        models.clear()
        models.update(previous)


def test_similar_items_endpoint_returns_recommendations_for_item_id(api_client):
    response = api_client.get("/api/similar/1?top_n=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_item"]["id"] == 1
    assert payload["query_item"]["title"] == "Alpha"
    assert payload["total"] == 2
    assert all(item["title"] != "Alpha" for item in payload["recommendations"])


def test_similar_items_endpoint_filters_by_category_case_insensitive(api_client):
    response = api_client.get("/api/similar/1?top_n=3&category=audio")

    assert response.status_code == 200
    payload = response.json()
    assert payload["category_filter"] == "audio"
    assert payload["recommendations"]
    assert {item["category"] for item in payload["recommendations"]} == {"Audio"}


def test_similar_items_endpoint_returns_404_for_unknown_item(api_client):
    response = api_client.get("/api/similar/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found."


def test_similar_items_endpoint_requires_built_models(api_client):
    models["ready"] = False

    response = api_client.get("/api/similar/1")

    assert response.status_code == 400
    assert "Models not built" in response.json()["detail"]
