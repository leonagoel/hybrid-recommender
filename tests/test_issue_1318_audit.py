import pytest
import pandas as pd
import numpy as np
import logging
from src.model.hybrid_model import HybridRecommender
from src.model.collaborative_model import CollaborativeRecommender

class MockContent:
    def __init__(self, df):
        self.df = df
    def recommend(self, title, top_n=10, target_catalog=None):
        return []

@pytest.fixture
def base_data():
    item_df = pd.DataFrame({
        "title": ["Item A", "Item B", "Item C"],
        "category": ["cat1", "cat2", "cat1"],
        "rating": [5.0, 3.0, 4.0],
        "review_count": [100, 50, 200]
    })
    
    interact_df = pd.DataFrame({
        "user_id": [1, 2, 1],
        "title": ["Item A", "Item B", "Item C"],
        "rating": [5, 4, 3]
    })
    
    return item_df, interact_df

def test_audit_scenarios(base_data, caplog):
    item_df, interact_df = base_data
    
    collab = CollaborativeRecommender(interact_df)
    model = HybridRecommender(
        content_model=MockContent(item_df),
        collab_model=collab,
        item_df=item_df
    )
    
    with caplog.at_level(logging.WARNING):
        # 1. Existing user (int)
        recs = model.recommend_for_user(1, top_n=2)
        # Existing user should just return normal recommendations
        
        # 1b. Existing user (str mapping)
        recs_str = model.recommend_for_user("1", top_n=2)
        # Existing user (str mapping)
        
        # 2 & 3 & 4. Unknown user (str, int)
        recs_unk_int = model.recommend_for_user(999, top_n=2)
        recs_unk_str = model.recommend_for_user("999", top_n=2)
        assert all(r.get("fallback") for r in recs_unk_int)
        assert all(r.get("fallback") for r in recs_unk_str)
        # Should be sorted by review_count: Item C (200), Item A (100)
        assert recs_unk_int[0]["title"] == "Item C"
        assert recs_unk_int[1]["title"] == "Item A"
        
        # 8 & 9 & 10. Invalid, None, Float user IDs
        recs_none = model.recommend_for_user(None, top_n=2)
        recs_float = model.recommend_for_user(999.9, top_n=2)
        recs_empty = model.recommend_for_user("", top_n=2)
        assert all(r.get("fallback") for r in recs_none)
        assert all(r.get("fallback") for r in recs_float)
        assert all(r.get("fallback") for r in recs_empty)
        
def test_fallback_hierarchy():
    # Test Empty Dataset
    empty_df = pd.DataFrame()
    collab_empty = CollaborativeRecommender(pd.DataFrame({'user_id': [1], 'title': ['A'], 'rating': [5]}))
    collab_empty.df = empty_df # Force empty
    model_empty = HybridRecommender(content_model=MockContent(empty_df), collab_model=collab_empty, item_df=empty_df)
    
    recs = model_empty.recommend_for_user(999, top_n=2)
    # validate_recommendations will force padding since dataset is empty
    assert len(recs) == 2
    assert recs[0]["title"] == "Top Trending Item A"
    
    # Test Missing Ratings Data (only review_count)
    df_no_rating = pd.DataFrame({
        "title": ["Item X", "Item Y"],
        "review_count": [10, 50]
    })
    collab_no_rating = CollaborativeRecommender(pd.DataFrame({'user_id': [1], 'title': ['Item X'], 'rating': [5]}))
    model_no_rating = HybridRecommender(content_model=MockContent(df_no_rating), collab_model=collab_no_rating, item_df=df_no_rating)
    recs = model_no_rating.recommend_for_user(999, top_n=2)
    assert recs[0]["title"] == "Item Y" # highest review count
    
    # Test Missing Review Count (only rating)
    df_no_reviews = pd.DataFrame({
        "title": ["Item X", "Item Y"],
        "rating": [2.0, 5.0]
    })
    collab_no_reviews = CollaborativeRecommender(pd.DataFrame({'user_id': [1], 'title': ['Item X'], 'rating': [2.0]}))
    model_no_reviews = HybridRecommender(content_model=MockContent(df_no_reviews), collab_model=collab_no_reviews, item_df=df_no_reviews)
    recs = model_no_reviews.recommend_for_user(999, top_n=2)
    assert recs[0]["title"] == "Item Y" # highest rating
    
def test_predict_rating_safe():
    interact_df = pd.DataFrame({
        "user_id": [1],
        "title": ["Item A"],
        "rating": [5]
    })
    collab = CollaborativeRecommender(interact_df)
    
    # Normal
    assert isinstance(collab.predict_rating(1, "Item A"), float)
    # Type mapped
    assert isinstance(collab.predict_rating("1", "Item A"), float)
    # Unknown user
    assert collab.predict_rating(999, "Item A") is None
    # None user
    assert collab.predict_rating(None, "Item A") is None
    # Float user natively hashes the same as int in Python if the value matches
    assert isinstance(collab.predict_rating(1.0, "Item A"), float)
    assert collab.predict_rating(999.0, "Item A") is None

