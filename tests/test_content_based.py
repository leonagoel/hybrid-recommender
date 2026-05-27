import pandas as pd

from content_model import ContentRecommender


def sample_df():
    return pd.DataFrame({
        "title": [
            "Apple iPhone",
            "Samsung Galaxy",
            "Python Programming",
            "Gaming Laptop"
        ],
        "combined": [
            "apple iphone smartphone electronics",
            "samsung galaxy android electronics",
            "python programming coding books",
            "gaming laptop electronics computer"
        ],
        "category": [
            "electronics",
            "electronics",
            "books",
            "electronics"
        ],
        "description": [
            "Apple smartphone",
            "Samsung smartphone",
            "Learn Python",
            "Laptop for gaming"
        ],
        "item_id": [
            "1",
            "2",
            "3",
            "4"
        ],
        "top_reviews": [
            ["Excellent phone"],
            ["Great Android"],
            ["Very informative"],
            ["Amazing performance"]
        ]
    })


def test_basic_search_functionality():
    recommender = ContentRecommender(sample_df())

    results = recommender.search("iphone")

    assert len(results) > 0
    assert results[0]["title"] == "Apple iPhone"
    assert results[0]["score"] > 0


def test_search_returns_expected_fields():
    recommender = ContentRecommender(sample_df())

    results = recommender.search("python")

    assert len(results) > 0

    result = results[0]

    assert "title" in result
    assert "score" in result
    assert "item_id" in result
    assert "category" in result
    assert "description" in result
    assert "top_reviews" in result


def test_search_with_no_matches():
    recommender = ContentRecommender(sample_df())

    results = recommender.search("nonexistentquery")

    assert results == []


def test_search_scores_are_positive():
    recommender = ContentRecommender(sample_df())

    results = recommender.search("electronics")

    assert len(results) > 0

    for result in results:
        assert result["score"] > 0


def test_search_top_n_limit():
    recommender = ContentRecommender(sample_df())

    results = recommender.search("electronics", top_n=2)

    assert len(results) <= 2