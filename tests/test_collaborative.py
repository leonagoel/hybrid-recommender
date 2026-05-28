import pandas as pd
from src.model.collaborative_model import CollaborativeRecommender


def sample_data():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "title": [
                "Naruto",
                "One Piece",
                "Naruto",
                "Bleach",
                "Attack on Titan",
            ],
            "rating": [5, 4, 5, 3, 4],
        }
    )


def test_matrix_creation():
    df = sample_data()

    model = CollaborativeRecommender(df)

    assert model.user_item_sparse.shape[0] > 0
    assert model.user_item_sparse.shape[1] > 0


def test_svd_training():
    df = sample_data()

    model = CollaborativeRecommender(df)

    assert model.svd is not None
    assert model.user_factors is not None
    assert model.item_factors is not None


def test_prediction_output_format():
    df = sample_data()

    model = CollaborativeRecommender(df)

    results = model.recommend("Naruto", top_n=2)

    assert isinstance(results, list)

    if len(results) > 0:
        assert "title" in results[0]
        assert "collab_score" in results[0]


def test_cold_start_user():
    df = sample_data()

    model = CollaborativeRecommender(df)

    results = model.predict_for_user(999)

    assert results == []


def test_extreme_sparse_matrix():
    df = pd.DataFrame(
        {
            "user_id": [1],
            "title": ["Naruto"],
            "rating": [5],
        }
    )

    model = CollaborativeRecommender(df)
    assert model.svd is None
    assert model.user_factors.shape == (1, 1)
    assert model.item_factors.shape == (1, 1)
    assert model.predict_rating(1, "Naruto") == 1.0


def test_top_n_validation_in_collaborative():
    import pytest
    df = sample_data()
    model = CollaborativeRecommender(df)

    with pytest.raises(ValueError):
        model.recommend("Naruto", top_n=-1)

    with pytest.raises(ValueError):
        model.recommend("Naruto", top_n=0)

    with pytest.raises(ValueError):
        model.recommend("Naruto", top_n="five")

    with pytest.raises(ValueError):
        model.predict_for_user(1, top_n=-5)

    assert len(model.recommend("Naruto", top_n=999)) <= 100
    assert len(model.predict_for_user(1, top_n=999)) <= 100


def test_predict_for_user_top_n_limits():
    """Test top_n default behavior in predict_for_user."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    results = model.predict_for_user(1)
    assert len(results) <= 100


def test_predict_for_user_with_catalog():
    """Test catalog filtering in predict_for_user."""
    df_with_catalog = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "title": ["Item A", "Item B", "Item A", "Item C"],
        "rating": [5, 4, 3, 4],
        "catalog": ["books", "books", "movies", "movies"]
    })
    model = CollaborativeRecommender(df_with_catalog)
    results = model.predict_for_user(2, target_catalog="movies", top_n=10)
    assert isinstance(results, list)


def test_user_with_all_items_seen():
    """Test predict_for_user when user has seen all items."""
    df = pd.DataFrame({
        "user_id": [1, 1, 1],
        "title": ["Item A", "Item B", "Item C"],
        "rating": [5, 4, 3]
    })
    model = CollaborativeRecommender(df)
    results = model.predict_for_user(1, top_n=10)
    assert len(results) == 0
