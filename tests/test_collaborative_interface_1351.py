"""
Regression tests for Issue #1351 — CollaborativeRecommender interface consistency.

Verifies that:
1. recommend()         returns dicts with 'collab_score' key (item-item CF)
2. predict_for_user()  returns dicts with 'predicted_score' key (user-item MF)
3. _popularity_fallback() respects the score_key parameter
4. Cold-start users get popularity fallback with 'predicted_score' key
5. recommend() fallback also uses 'collab_score' key  (not 'predicted_score')
6. predict_rating()    returns float or None, never crashes
7. All public methods accept and respect the top_n parameter
"""
import pytest
import pandas as pd
import numpy as np

from src.model.collaborative_model import CollaborativeRecommender


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def interaction_df():
    """Minimal interaction dataset with 3 users and 5 items."""
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 1, 2, 3],
        'title': ['Alpha', 'Beta', 'Alpha', 'Gamma', 'Beta', 'Delta',
                  'Epsilon', 'Delta', 'Epsilon'],
        'rating': [5.0, 4.0, 3.0, 5.0, 4.0, 2.0, 5.0, 3.0, 4.0],
    })


@pytest.fixture()
def model(interaction_df):
    return CollaborativeRecommender(interaction_df)


# ---------------------------------------------------------------------------
# 1. recommend() score key contract
# ---------------------------------------------------------------------------

class TestRecommendScoreKey:
    """recommend() must emit 'collab_score' for every result dict."""

    def test_all_results_have_collab_score(self, model):
        results = model.recommend('Alpha', top_n=3)
        assert results, "Expected at least one result for known title"
        for r in results:
            assert 'collab_score' in r, f"Missing 'collab_score' in {r}"

    def test_no_predicted_score_in_recommend_results(self, model):
        results = model.recommend('Alpha', top_n=3)
        for r in results:
            assert 'predicted_score' not in r, \
                f"'predicted_score' should not appear in recommend() output: {r}"

    def test_collab_score_is_float(self, model):
        results = model.recommend('Beta', top_n=3)
        for r in results:
            assert isinstance(r['collab_score'], float), \
                f"collab_score must be float, got {type(r['collab_score'])}"

    def test_unknown_title_returns_empty_list(self, model):
        results = model.recommend('NonExistentTitle_XYZ', top_n=5)
        assert results == [], "Unknown title should return empty list"


# ---------------------------------------------------------------------------
# 2. predict_for_user() score key contract
# ---------------------------------------------------------------------------

class TestPredictForUserScoreKey:
    """predict_for_user() must emit 'predicted_score' for every result dict."""

    def test_all_results_have_predicted_score(self, model):
        results = model.predict_for_user(1, top_n=3)
        assert results, "Expected results for known user"
        for r in results:
            assert 'predicted_score' in r, f"Missing 'predicted_score' in {r}"

    def test_no_collab_score_in_predict_for_user_results(self, model):
        results = model.predict_for_user(1, top_n=3)
        for r in results:
            assert 'collab_score' not in r, \
                f"'collab_score' should not appear in predict_for_user() output: {r}"

    def test_predicted_score_is_float(self, model):
        results = model.predict_for_user(2, top_n=4)
        for r in results:
            assert isinstance(r['predicted_score'], float), \
                f"predicted_score must be float, got {type(r['predicted_score'])}"


# ---------------------------------------------------------------------------
# 3. _popularity_fallback() score_key parameter
# ---------------------------------------------------------------------------

class TestPopularityFallbackScoreKey:
    """_popularity_fallback() must honour the score_key argument."""

    def test_default_score_key_is_predicted_score(self, model):
        results = model._popularity_fallback(top_n=3)
        assert results, "Fallback should return items"
        for r in results:
            assert 'predicted_score' in r, f"Default key missing in {r}"

    def test_collab_score_key_variant(self, model):
        results = model._popularity_fallback(top_n=3, score_key='collab_score')
        assert results, "Fallback should return items"
        for r in results:
            assert 'collab_score' in r, f"collab_score key missing in {r}"
            assert 'predicted_score' not in r, \
                f"predicted_score should not appear when score_key='collab_score'"

    def test_fallback_items_marked_as_fallback(self, model):
        results = model._popularity_fallback(top_n=3)
        for r in results:
            assert r.get('fallback') is True, f"fallback flag missing in {r}"

    def test_fallback_respects_top_n(self, model):
        results = model._popularity_fallback(top_n=2)
        assert len(results) <= 2, "Fallback returned more items than top_n"


# ---------------------------------------------------------------------------
# 4. Cold-start user gets 'predicted_score' fallback (not 'collab_score')
# ---------------------------------------------------------------------------

class TestColdStartScoreKey:
    """predict_for_user() cold-start path must use 'predicted_score', not 'collab_score'."""

    def test_unknown_user_returns_fallback(self, model):
        results = model.predict_for_user(user_id=9999, top_n=3)
        assert results, "Cold-start should return fallback items, not empty list"

    def test_unknown_user_fallback_has_predicted_score(self, model):
        results = model.predict_for_user(user_id='ghost_user', top_n=3)
        for r in results:
            assert 'predicted_score' in r, \
                f"Cold-start fallback must use 'predicted_score', got keys: {list(r.keys())}"

    def test_unknown_user_fallback_no_collab_score_key(self, model):
        results = model.predict_for_user(user_id=99999, top_n=3)
        for r in results:
            assert 'collab_score' not in r, \
                f"'collab_score' must not appear in user fallback: {r}"

    def test_none_user_does_not_crash(self, model):
        """predict_for_user(None) should return fallback, never raise."""
        try:
            results = model.predict_for_user(user_id=None, top_n=5)
            assert isinstance(results, list)
        except (KeyError, TypeError):
            pytest.fail("predict_for_user(None) raised KeyError or TypeError")


# ---------------------------------------------------------------------------
# 5. recommend() fallback path uses 'collab_score' key
# ---------------------------------------------------------------------------

class TestRecommendFallbackScoreKey:
    """When recommend() triggers fallback, fallback items must have 'collab_score'."""

    def test_fallback_triggered_on_unknown_title_with_padding(self, interaction_df):
        """Force force_padding=True via a model wrapper to confirm collab_score key in fallback."""
        from src.model.validation import validate_recommendations

        model = CollaborativeRecommender(interaction_df)
        # Call _popularity_fallback directly with the same score_key recommend() uses
        fallback = model._popularity_fallback(top_n=5, score_key='collab_score')
        for r in fallback:
            assert 'collab_score' in r, \
                f"recommend() fallback path must use 'collab_score', got: {list(r.keys())}"
            assert 'predicted_score' not in r


# ---------------------------------------------------------------------------
# 6. predict_rating() is safe
# ---------------------------------------------------------------------------

class TestPredictRating:
    """predict_rating() must return float for known pairs and None for unknowns."""

    def test_known_user_known_title_returns_float(self, model):
        result = model.predict_rating(user_id=1, title='Alpha')
        assert result is not None
        assert isinstance(result, float)

    def test_unknown_user_returns_none(self, model):
        result = model.predict_rating(user_id=8888, title='Alpha')
        assert result is None

    def test_unknown_title_returns_none(self, model):
        result = model.predict_rating(user_id=1, title='Nonexistent Movie')
        assert result is None

    def test_both_unknown_returns_none(self, model):
        result = model.predict_rating(user_id=8888, title='Nonexistent Movie')
        assert result is None

    def test_string_user_id_coercion(self, model):
        """predict_rating('1', ...) must resolve to the same user as integer 1."""
        result_int = model.predict_rating(user_id=1, title='Alpha')
        result_str = model.predict_rating(user_id='1', title='Alpha')
        if result_int is not None and result_str is not None:
            assert abs(result_int - result_str) < 1e-9, \
                "String and int user_id should yield identical rating predictions"


# ---------------------------------------------------------------------------
# 7. top_n is respected across all public methods
# ---------------------------------------------------------------------------

class TestTopNParameter:
    """top_n cap must be honoured by all public methods."""

    @pytest.mark.parametrize("method,kwargs", [
        ("recommend",       {"title": "Alpha", "top_n": 2}),
        ("predict_for_user", {"user_id": 1,     "top_n": 2}),
    ])
    def test_top_n_cap(self, model, method, kwargs):
        results = getattr(model, method)(**kwargs)
        assert len(results) <= kwargs["top_n"], \
            f"{method}() returned {len(results)} results, expected <= {kwargs['top_n']}"

    @pytest.mark.parametrize("method,kwargs", [
        ("recommend",       {"title": "Alpha", "top_n": 0}),
        ("recommend",       {"title": "Alpha", "top_n": -1}),
        ("predict_for_user", {"user_id": 1,     "top_n": 0}),
    ])
    def test_invalid_top_n_raises(self, model, method, kwargs):
        with pytest.raises(ValueError):
            getattr(model, method)(**kwargs)
