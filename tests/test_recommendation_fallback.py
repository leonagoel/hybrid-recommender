import math
import logging
import pytest
import pandas as pd
import numpy as np

from src.model.validation import validate_recommendations
from src.model.collaborative_model import CollaborativeRecommender
from src.model.content_model import ContentRecommender
from src.model.hybrid_model import HybridRecommender


# Dummy SentenceTransformer to avoid network/model downloads
class DummySentenceTransformer:
    def __init__(self, model_name=None):
        pass
    def encode(self, texts, show_progress_bar=False):
        # Return simple mock 2D embeddings of dimension 2
        return np.ones((len(texts), 2), dtype=np.float32)


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Automatically mock SentenceTransformer in src.model.content_model."""
    import src.model.content_model
    monkeypatch.setattr(src.model.content_model, "SentenceTransformer", DummySentenceTransformer)


# ==============================================================================
# Unit Tests for validate_recommendations
# ==============================================================================

def test_validate_recommendations_filters_malformed():
    raw_input = [
        {"title": "Valid Item A", "score": 0.9},
        "Not a dictionary",
        {"no_title": "Invalid dict"},
        {"title": "   ", "score": 0.8}, # empty whitespace title
        {"title": None, "score": 0.7},
    ]
    results = validate_recommendations(raw_input, top_n=5, force_padding=False)
    assert len(results) == 1
    assert results[0]["title"] == "Valid Item A"


def test_validate_recommendations_filters_nan_and_inf(caplog):
    raw_input = [
        {"title": "Item A", "hybrid_score": 0.9},
        {"title": "Item B", "hybrid_score": float("nan")},
        {"title": "Item C", "hybrid_score": float("inf")},
        {"title": "Item D", "hybrid_score": None},
    ]
    with caplog.at_level(logging.WARNING):
        results = validate_recommendations(raw_input, top_n=5, force_padding=False)
    
    # Item A is the only valid one
    assert len(results) == 1
    assert results[0]["title"] == "Item A"
    
    # Should trigger NaN hybrid score logging warning
    assert any("Recommendation fallback triggered: NaN hybrid scores" in record.message for record in caplog.records)


def test_validate_recommendations_cascading_fallbacks():
    # 1. Test fallback_fn popularity trigger
    def mock_popularity_fallback(top_n):
        return [
            {"title": "Popular A", "predicted_score": 0.95},
            {"title": "Popular B", "predicted_score": 0.90},
        ]
        
    results = validate_recommendations(
        recommendations=[],
        fallback_fn=mock_popularity_fallback,
        top_n=2,
        force_padding=True
    )
    assert len(results) == 2
    assert results[0]["title"] == "Popular A"
    assert results[1]["title"] == "Popular B"
    assert all(r.get("fallback") is True for r in results)

    # 2. Test default_fallback_items top-rated items trigger when fallback_fn is insufficient
    results = validate_recommendations(
        recommendations=[],
        fallback_fn=lambda top_n: [],
        top_n=2,
        default_fallback_items=["Top Rated A", "Top Rated B"],
        force_padding=True
    )
    assert len(results) == 2
    assert results[0]["title"] == "Top Rated A"
    assert results[0]["fallback"] is True
    assert results[1]["title"] == "Top Rated B"
    assert results[1]["fallback"] is True

    # 3. Test hardcoded defaults + spacers when everything else is unavailable
    results = validate_recommendations(
        recommendations=[],
        fallback_fn=None,
        top_n=5,
        default_fallback_items=None,
        force_padding=True
    )
    assert len(results) == 5
    # First three should be static trending defaults
    assert results[0]["title"] == "Top Trending Item A"
    assert results[1]["title"] == "Top Trending Item B"
    assert results[2]["title"] == "Top Trending Item C"
    # Remaining two should be default fallback items (spacers)
    assert results[3]["title"] == "Default Fallback Item 4"
    assert results[4]["title"] == "Default Fallback Item 5"
    assert all(r.get("fallback") is True for r in results)


def test_validate_recommendations_logging_empty_triggers(caplog):
    # CF empty warning
    with caplog.at_level(logging.WARNING):
        validate_recommendations([], top_n=2, context="CF", force_padding=False)
    assert any("Recommendation fallback triggered: empty CF output" in record.message for record in caplog.records)

    caplog.clear()

    # Hybrid empty warning
    with caplog.at_level(logging.WARNING):
        validate_recommendations([], top_n=2, context="hybrid", force_padding=False)
    assert any("Recommendation fallback triggered: empty hybrid output" in record.message for record in caplog.records)


# ==============================================================================
# Integration / Model Tests
# ==============================================================================

def test_collaborative_model_nan_scores_fallback(caplog):
    interaction_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "title": ["Item A", "Item B", "Item A", "Item C", "Item B"],
        "rating": [5, 4, 5, 3, 4],
    })
    model = CollaborativeRecommender(interaction_df)
    
    # Inject an item with NaN factors or weights
    model.item_factors = np.full_like(model.item_factors, np.nan)
    
    with caplog.at_level(logging.WARNING):
        recs = model.recommend("Item A", top_n=3)
    
    # Should yield empty list since NaN scores are filtered out (force_padding=False at CF layer)
    assert len(recs) == 0
    # Warning should be logged about NaN collab scores (since predicted_score/collab_score calculations yielded NaN)
    assert any("Recommendation fallback triggered: NaN collab scores" in record.message for record in caplog.records)


def test_content_model_fallback(caplog):
    item_df = pd.DataFrame({
        "title": ["Book A", "Book B"],
        "combined": ["combined a", "combined b"],
        "rating": [4.0, 3.5]
    })
    model = ContentRecommender(item_df)
    
    # Recommend for a non-existent item (unknown title returns [] without validator wrapper)
    recs_unknown = model.recommend("Book Unknown", top_n=2)
    assert len(recs_unknown) == 0

    # Test search with no match (score <= 0) - should trigger popularity fallback in ContentRecommender
    # Force cosine similarity to return 0.0
    model.matrix = np.zeros_like(model.matrix)
    
    with caplog.at_level(logging.WARNING):
        search_recs = model.search("wizard", top_n=2)
        
    assert len(search_recs) > 0
    # It should have filtered to the popularity fallback
    assert search_recs[0]["title"] in ["Book A", "Book B"]


def test_hybrid_model_cascading_fallback(caplog):
    item_df = pd.DataFrame({
        "title": ["Product A", "Product B", "Product C"],
        "combined": ["A", "B", "C"],
        "rating": [4.5, 3.8, 4.9],
        "review_count": [10, 5, 20],
        "avg_sentiment": [0.6, 0.2, 0.8],
    })
    
    # Create empty content model
    class EmptyContent:
        def __init__(self, df):
            self.df = df
        def recommend(self, title, top_n=10, target_catalog=None):
            return []

    # Initialize HybridRecommender with no collaborative model
    model = HybridRecommender(
        content_model=EmptyContent(item_df),
        collab_model=None,
        item_df=item_df
    )
    
    # Recommending an unknown product should trigger popularity fallback
    with caplog.at_level(logging.WARNING):
        recs = model.recommend("Unknown", top_n=2)
        
    assert len(recs) == 2
    # Verify they are the popular ones (Product C then Product A based on review count/rating)
    assert recs[0]["title"] == "Product C"
    assert recs[1]["title"] == "Product A"
    assert recs[0]["fallback"] is True
    assert recs[1]["fallback"] is True
    
    # Empty hybrid warning should be triggered
    assert any("Recommendation fallback triggered: empty hybrid output" in record.message for record in caplog.records)


def test_hybrid_cold_start_and_sparse_dataset():
    # Cold start user: hybrid model falls back cleanly to popularity
    item_df = pd.DataFrame({
        "title": ["Item 1", "Item 2"],
        "combined": ["1", "2"],
        "rating": [4.0, 5.0],
        "review_count": [10, 20],
    })
    
    class MockContent:
        def __init__(self, df):
            self.df = df
        def recommend(self, title, top_n=10, target_catalog=None):
            return []
            
    model = HybridRecommender(
        content_model=MockContent(item_df),
        collab_model=None,
        item_df=item_df
    )
    
    # New user cold-start test
    recs = model.recommend_for_user(user_id="new_user_xyz", top_n=5)
    assert len(recs) == 5
    assert recs[0]["title"] == "Item 2"  # Item 2 is more popular
    assert recs[1]["title"] == "Item 1"
    assert all(r["fallback"] is True for r in recs)
