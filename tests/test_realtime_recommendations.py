"""
Tests for real-time recommendation transports.
"""
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app, models, realtime_hub
from src.model.content_model import ContentRecommender
from src.model.hybrid_model import HybridRecommender


@pytest.fixture
def realtime_client():
    item_df = pd.DataFrame({
        "title": ["Alpha", "Beta", "Gamma", "Desk Stand"],
        "description": [
            "Wireless headphones with active noise cancellation",
            "Budget earbuds with balanced wireless audio",
            "Premium studio headphones for audio work",
            "Adjustable laptop stand for desks",
        ],
        "category": ["Audio", "Audio", "Audio", "Accessories"],
        "rating": [4.5, 3.8, 4.9, 4.2],
        "review_count": [120, 45, 200, 80],
        "avg_sentiment": [0.6, 0.2, 0.8, 0.5],
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
    realtime_hub.active_connections.clear()
    try:
        yield TestClient(app)
    finally:
        models.clear()
        models.update(previous)
        realtime_hub.active_connections.clear()


def test_recommendations_websocket_streams_updates(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({"item_title": "Alpha", "top_n": 2})
        payload = websocket.receive_json()

    assert payload["type"] == "recommendations"
    assert payload["query_item"] == "Alpha"
    assert len(payload["recommendations"]) == 2
    assert all(item["title"] != "Alpha" for item in payload["recommendations"])

def test_recommendations_websocket_accepts_structured_event(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({
            "type": "recommendation_request",
            "payload": {"item_title": "Alpha", "top_n": 2},
        })
        payload = websocket.receive_json()

    assert payload["type"] == "recommendations"
    assert payload["payload"]["query_item"] == "Alpha"
    assert len(payload["payload"]["recommendations"]) == 2


def test_recommendations_websocket_rejects_invalid_payload(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({
            "type": "recommendation_request",
            "payload": {"top_n": 2},
        })
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "INVALID_PAYLOAD"


def test_recommendations_websocket_rejects_invalid_top_n(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({
            "type": "recommendation_request",
            "payload": {"item_title": "Alpha", "top_n": 500},
        })
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "INVALID_PAYLOAD"


def test_recommendations_websocket_ping_pong(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({"type": "ping", "payload": {}})
        payload = websocket.receive_json()

    assert payload["type"] == "pong"
    assert payload["payload"]["status"] == "ok"


def test_recommendations_websocket_rejects_unsupported_event(realtime_client):
    with realtime_client.websocket_connect("/ws/recommendations") as websocket:
        websocket.send_json({"type": "unknown_event", "payload": {}})
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "UNSUPPORTED_EVENT"

def post_with_csrf(client, url, payload):
    token = "a" * 64
    origin = "http://testserver"
    client.cookies.set("csrftoken", token)
    return client.post(
        url,
        json=payload,
        headers={
            "X-CSRF-Token": token,
            "Origin": origin,
            "Referer": f"{origin}/",
        },
    )


def test_realtime_behavior_endpoint_returns_http_fallback_payload(realtime_client):
    response = post_with_csrf(
        realtime_client,
        "/api/realtime/behavior",
        {"item_title": "Alpha", "top_n": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "recommendations"
    assert payload["query_item"] == "Alpha"
    assert len(payload["recommendations"]) == 2


def test_realtime_behavior_requires_built_models(realtime_client):
    models["ready"] = False

    response = post_with_csrf(
        realtime_client,
        "/api/realtime/behavior",
        {"item_title": "Alpha", "top_n": 2},
    )

    assert response.status_code == 400
    assert "Models not built" in response.json()["detail"]
