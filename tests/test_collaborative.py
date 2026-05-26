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


def test_predict_rating_returns_float():
    """Test rating prediction for valid user-item pair."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    rating = model.predict_rating(1, "Naruto")
    assert isinstance(rating, float)


def test_predict_rating_invalid_user_returns_none():
    """Test with non-existent user returns None."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    rating = model.predict_rating(999, "Naruto")
    assert rating is None


def test_predict_rating_invalid_item_returns_none():
    """Test with non-existent item returns None."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    rating = model.predict_rating(1, "Nonexistent Item")
    assert rating is None


def test_recommend_excludes_self():
    """Verify query item is not in recommendations."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    results = model.recommend("Naruto", top_n=10)
    titles = [r["title"] for r in results]
    assert "Naruto" not in titles


def test_implicit_feedback_integration():
    """Test that views and purchases affect recommendations when use_implicit=True."""
    df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "title": ["Item A", "Item B", "Item C"],
        "rating": [3.0, 4.0, 5.0],
        "views": [10, 0, 5],
        "purchases": [2, 0, 1]
    })
    model_with_implicit = CollaborativeRecommender(df, use_implicit=True)
    model_without_implicit = CollaborativeRecommender(df, use_implicit=False)
    # Check that the matrix values are different when implicit feedback is used
    with_implicit_data = model_with_implicit.user_item_sparse.data.copy()
    without_implicit_data = model_without_implicit.user_item_sparse.data.copy()
    # With implicit feedback, ratings should be boosted by purchases and views
    assert any(with_implicit_data != without_implicit_data)


def test_recommend_returns_scores_in_valid_range():
    """Test that recommend scores are between -1 and 1 (cosine similarity range)."""
    df = sample_data()
    model = CollaborativeRecommender(df)
    results = model.recommend("Naruto", top_n=10)
    for r in results:
        assert -1.0 <= r["collab_score"] <= 1.0