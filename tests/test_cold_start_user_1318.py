"""
Regression tests for Issue #1318: cold-start / unknown user handling.

Ensures that recommendation requests for unknown or invalid user IDs never
crash the application and always return a safe fallback list.
"""
import pytest
import pandas as pd

from src.model.hybrid_model import HybridRecommender
from src.model.collaborative_model import CollaborativeRecommender


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def item_df():
    return pd.DataFrame({
        "title": ["Product A", "Product B", "Product C", "Product D"],
        "description": ["desc A", "desc B", "desc C", "desc D"],
        "category": ["Electronics", "Electronics", "Home", "Home"],
        "rating": [4.5, 3.8, 4.9, 3.2],
        "review_count": [120, 45, 200, 10],
        "avg_sentiment": [0.6, 0.2, 0.8, 0.1],
        "combined": [
            "Product A desc A Electronics",
            "Product B desc B Electronics",
            "Product C desc C Home",
            "Product D desc D Home",
        ],
    })


@pytest.fixture
def interaction_df():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "title": ["Product A", "Product B", "Product B", "Product C", "Product A"],
        "rating": [5.0, 3.0, 4.0, 5.0, 4.0],
    })


class _MockContentModel:
    """Minimal content model stub that avoids loading SentenceTransformer."""

    def __init__(self, df):
        self.df = df

    def recommend(self, title, top_n=10, target_catalog=None):
        rows = self.df[self.df["title"] != title].head(top_n)
        return [{"title": row["title"], "content_score": 0.5} for _, row in rows.iterrows()]

    def search(self, query, top_n=10):
        return []


@pytest.fixture
def hybrid_model(item_df, interaction_df):
    content = _MockContentModel(item_df)
    collab = CollaborativeRecommender(interaction_df)
    return HybridRecommender(content, collab_model=collab, item_df=item_df)


@pytest.fixture
def hybrid_model_no_collab(item_df):
    content = _MockContentModel(item_df)
    return HybridRecommender(content, collab_model=None, item_df=item_df)


# ---------------------------------------------------------------------------
# Test 1: Existing user receives normal personalized recommendations
# ---------------------------------------------------------------------------

def test_existing_user_gets_recommendations(hybrid_model):
    recs = hybrid_model.recommend_for_user("u1", top_n=3)
    assert isinstance(recs, list)
    assert len(recs) > 0
    required_keys = {"title", "hybrid_score", "collab_score", "content_score", "sentiment_score"}
    for r in recs:
        assert required_keys.issubset(r.keys()), f"Missing keys in {r}"


# ---------------------------------------------------------------------------
# Test 2: Non-existent user ID returns fallback recommendations, not an error
# ---------------------------------------------------------------------------

def test_nonexistent_user_returns_fallback(hybrid_model):
    recs = hybrid_model.recommend_for_user("user_999999999", top_n=3)
    assert isinstance(recs, list)
    assert len(recs) > 0, "Fallback must return at least one item when catalog data exists"


# ---------------------------------------------------------------------------
# Test 3: Unknown user does not raise KeyError
# ---------------------------------------------------------------------------

def test_unknown_user_no_keyerror(hybrid_model):
    try:
        hybrid_model.recommend_for_user("completely_unknown_xyz", top_n=3)
    except KeyError as exc:
        pytest.fail(f"KeyError raised for unknown user: {exc}")


# ---------------------------------------------------------------------------
# Test 4: Unknown user does not crash application flow
# ---------------------------------------------------------------------------

def test_unknown_user_no_crash(hybrid_model):
    try:
        recs = hybrid_model.recommend_for_user("no_such_user_abc", top_n=3)
        assert isinstance(recs, list)
    except Exception as exc:
        pytest.fail(f"Unexpected exception for unknown user: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 5: Fallback list is non-empty when catalog data exists
# ---------------------------------------------------------------------------

def test_fallback_nonempty_with_catalog(hybrid_model):
    recs = hybrid_model.get_fallback_recommendations(top_n=4)
    assert len(recs) > 0, "get_fallback_recommendations must return items when item_df is populated"
    for r in recs:
        assert "title" in r
        assert "hybrid_score" in r


# ---------------------------------------------------------------------------
# Test 6: Invalid user IDs are handled gracefully (None, negative int, empty string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_user_id", [None, -1, "", "   "])
def test_invalid_user_ids_handled_gracefully(hybrid_model, bad_user_id):
    try:
        recs = hybrid_model.recommend_for_user(bad_user_id, top_n=3)
        assert isinstance(recs, list), f"Expected list for user_id={bad_user_id!r}"
    except (KeyError, IndexError) as exc:
        pytest.fail(f"Crash ({type(exc).__name__}) for user_id={bad_user_id!r}: {exc}")


# ---------------------------------------------------------------------------
# Extra: no collab model still provides fallback for any user
# ---------------------------------------------------------------------------

def test_no_collab_model_fallback_for_any_user(hybrid_model_no_collab):
    recs = hybrid_model_no_collab.recommend_for_user("u1", top_n=3)
    assert isinstance(recs, list)
    assert len(recs) > 0
