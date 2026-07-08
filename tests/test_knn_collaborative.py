import pytest
import pandas as pd
import numpy as np
from knn_collaborative import KNNCollaborativeRecommender

@pytest.fixture
def df():
    return pd.DataFrame([
        {"user_id": "u1", "title": "Item A", "rating": 5.0},
        {"user_id": "u1", "title": "Item B", "rating": 4.0},
        {"user_id": "u1", "title": "Item C", "rating": 3.0},
        {"user_id": "u2", "title": "Item A", "rating": 4.0},
        {"user_id": "u2", "title": "Item B", "rating": 5.0},
        {"user_id": "u2", "title": "Item D", "rating": 4.0},
        {"user_id": "u3", "title": "Item A", "rating": 3.0},
        {"user_id": "u3", "title": "Item C", "rating": 4.0},
        {"user_id": "u3", "title": "Item E", "rating": 5.0},
        {"user_id": "u4", "title": "Item B", "rating": 2.0},
        {"user_id": "u4", "title": "Item D", "rating": 5.0},
        {"user_id": "u4", "title": "Item E", "rating": 4.0},
    ])

@pytest.fixture
def model(df):
    return KNNCollaborativeRecommender(df, k=3)

def test_user_index_built(model):
    assert "u1" in model._user_to_idx and len(model._user_to_idx) == 4

def test_item_index_built(model):
    assert "Item A" in model._title_to_idx and len(model._title_to_idx) == 5

def test_matrix_shape(model):
    assert model._matrix.shape == (4, 5)

def test_missing_columns_raises():
    with pytest.raises(ValueError):
        KNNCollaborativeRecommender(pd.DataFrame([{"user_id": "u1", "title": "A"}]))

def test_returns_list(model):
    assert isinstance(model.recommend("Item A", top_n=3), list)

def test_top_n_respected(model):
    assert len(model.recommend("Item A", top_n=2)) <= 2

def test_rec_has_keys(model):
    for r in model.recommend("Item A", top_n=3):
        assert "title" in r and "collab_score" in r

def test_scores_in_range(model):
    for r in model.recommend("Item A", top_n=5):
        assert 0.0 <= r["collab_score"] <= 1.0

def test_query_item_excluded(model):
    assert "Item A" not in [r["title"] for r in model.recommend("Item A", top_n=5)]

def test_sorted_descending(model):
    scores = [r["collab_score"] for r in model.recommend("Item A", top_n=5)]
    assert scores == sorted(scores, reverse=True)

def test_unknown_item_returns_empty(model):
    assert model.recommend("Nonexistent", top_n=5) == []

def test_user_personalisation(model):
    titles = [r["title"] for r in model.recommend("Item A", top_n=5, user_id="u1")]
    assert "Item A" not in titles and "Item B" not in titles and "Item C" not in titles

def test_popularity_fallback(model):
    fb = model._popularity_fallback(top_n=3)
    assert len(fb) <= 3 and all("title" in r for r in fb)

def test_single_user():
    df = pd.DataFrame([{"user_id": "u1", "title": "A", "rating": 5.0}, {"user_id": "u1", "title": "B", "rating": 4.0}])
    assert isinstance(KNNCollaborativeRecommender(df, k=5).recommend("A", top_n=2), list)

def test_nan_ratings_handled():
    df = pd.DataFrame([
        {"user_id": "u1", "title": "A", "rating": None},
        {"user_id": "u1", "title": "B", "rating": 4.0},
        {"user_id": "u2", "title": "A", "rating": 3.0},
        {"user_id": "u2", "title": "B", "rating": 5.0},
    ])
    assert isinstance(KNNCollaborativeRecommender(df, k=2).recommend("A", top_n=2), list)
