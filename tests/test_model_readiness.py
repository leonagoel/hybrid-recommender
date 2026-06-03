import os
import sys
import types

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

nlp_engine_stub = types.ModuleType("src.model.nlp_engine")
nlp_engine_stub.batch_analyze = lambda df, text_col="review_text": df
nlp_engine_stub.aggregate_sentiment_by_item = (
    lambda df, item_col="title": pd.DataFrame(
        {item_col: [], "avg_sentiment": [], "review_count": []}
    )
)
sys.modules.setdefault("src.model.nlp_engine", nlp_engine_stub)

from backend import main


class FakeHybridModel:
    def get_weights(self):
        return {"content": 0.5, "collaborative": 0.3, "sentiment": 0.2}


def complete_item_df():
    item_df = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Alpha", "Beta"],
            "description": ["Noise cancelling headphones", "Desk stand"],
            "category": ["Audio", "Accessories"],
            "rating": [4.7, 4.2],
            "review_count": [120, 80],
            "avg_sentiment": [0.7, 0.4],
        }
    )
    item_df["combined"] = (
        item_df["title"] + " " + item_df["description"] + " " + item_df["category"]
    )
    return item_df


@pytest.fixture
def readiness_client():
    previous_models = main.models.copy()
    previous_active_model_version = main.ACTIVE_MODEL_VERSION
    main.models.update(
        {
            "content": None,
            "collab": None,
            "hybrid": None,
            "ready": False,
            "item_df": None,
            "build_time": None,
            "last_trained_at": None,
        }
    )
    main.ACTIVE_MODEL_VERSION = None

    try:
        yield TestClient(main.app)
    finally:
        main.models.clear()
        main.models.update(previous_models)
        main.ACTIVE_MODEL_VERSION = previous_active_model_version


def test_model_readiness_reports_models_not_built(readiness_client):
    response = readiness_client.get("/api/model-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["active_model_version"] is None
    assert payload["last_trained_at"] is None
    assert payload["components"] == {
        "content": False,
        "collab": False,
        "hybrid": False,
        "item_df": False,
    }
    assert payload["dataset"]["available"] is False
    assert payload["dataset"]["shape"] == {"rows": 0, "columns": 0}
    assert payload["weights"] is None
    assert "Models have not been built yet." in payload["warnings"]


def test_model_readiness_reports_content_only_partial_readiness(readiness_client):
    main.models.update(
        {
            "content": object(),
            "item_df": complete_item_df(),
            "ready": True,
            "last_trained_at": "2026-06-01T10:00:00+00:00",
        }
    )

    response = readiness_client.get("/api/model-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["components"] == {
        "content": True,
        "collab": False,
        "hybrid": False,
        "item_df": True,
    }
    assert payload["dataset"]["available"] is True
    assert payload["dataset"]["shape"] == {"rows": 2, "columns": 8}
    assert all(payload["dataset"]["important_columns"].values())
    assert payload["weights"] is None
    assert any(
        "missing components: collab, hybrid" in warning
        for warning in payload["warnings"]
    )


def test_model_readiness_reports_full_hybrid_readiness(readiness_client):
    main.ACTIVE_MODEL_VERSION = "1.0.0-20260601100000"
    main.models.update(
        {
            "content": object(),
            "collab": object(),
            "hybrid": FakeHybridModel(),
            "item_df": complete_item_df(),
            "ready": True,
            "last_trained_at": "2026-06-01T10:00:00+00:00",
        }
    )

    response = readiness_client.get("/api/model-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["active_model_version"] == "1.0.0-20260601100000"
    assert payload["last_trained_at"] == "2026-06-01T10:00:00+00:00"
    assert payload["components"] == {
        "content": True,
        "collab": True,
        "hybrid": True,
        "item_df": True,
    }
    assert payload["weights"] == {
        "content": 0.5,
        "collaborative": 0.3,
        "sentiment": 0.2,
    }
    assert payload["warnings"] == []


def test_model_readiness_warns_when_ready_state_is_missing_dataset(readiness_client):
    main.models.update(
        {
            "content": object(),
            "collab": object(),
            "hybrid": FakeHybridModel(),
            "ready": True,
        }
    )

    response = readiness_client.get("/api/model-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["item_df"] is False
    assert payload["dataset"]["available"] is False
    assert payload["weights"] == {
        "content": 0.5,
        "collaborative": 0.3,
        "sentiment": 0.2,
    }
    assert any("missing components: item_df" in warning for warning in payload["warnings"])


def test_model_readiness_warns_for_missing_important_dataset_columns(readiness_client):
    main.models.update(
        {
            "content": object(),
            "collab": object(),
            "hybrid": FakeHybridModel(),
            "item_df": pd.DataFrame({"title": ["Alpha"], "rating": [4.7]}),
            "ready": True,
        }
    )

    response = readiness_client.get("/api/model-readiness")

    assert response.status_code == 200
    payload = response.json()
    important_columns = payload["dataset"]["important_columns"]
    assert important_columns["id"] is False
    assert important_columns["title"] is True
    assert important_columns["rating"] is True
    assert important_columns["description"] is False
    assert important_columns["combined"] is False
    assert any("missing important columns" in warning for warning in payload["warnings"])
