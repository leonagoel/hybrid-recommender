"""
Unit tests for CollaborativeRecommender — GitHub Issue #655
Covers: item-item similarity, user-item scoring, predict_rating,
        popularity fallback, edge cases, and mathematical properties.
Run with: pytest tests/test_collaborative.py -v
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.model.collaborative_model import CollaborativeRecommender


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Standard 3-user / 4-item interaction DataFrame."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "title": ["Naruto", "One Piece", "Naruto", "Bleach", "Attack on Titan"],
            "rating": [5, 4, 5, 3, 4],
        }
    )


@pytest.fixture
def model(sample_df):
    return CollaborativeRecommender(sample_df)


@pytest.fixture
def implicit_df():
    """Interaction DataFrame that includes implicit feedback columns."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "title": ["Item A", "Item B", "Item A", "Item C", "Item B", "Item C"],
            "rating": [5.0, 3.0, 4.0, 5.0, 4.0, 3.5],
            "views": [10, 2, 8, 1, 5, 3],
            "purchases": [2, 0, 1, 3, 1, 0],
        }
    )


@pytest.fixture
def catalog_df():
    """Interaction DataFrame with a catalog column for filtering tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "title": ["Item A", "Item B", "Item A", "Item C", "Item B"],
            "rating": [5, 4, 3, 4, 5],
            "catalog": ["books", "movies", "books", "movies", "movies"],
        }
    )


@pytest.fixture
def identical_users_df():
    """Two users with identical rating patterns — their latent vectors should be close."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3"],
            "title": ["A", "B", "C", "A", "B", "C", "A"],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 1.0],
        }
    )


@pytest.fixture
def identical_items_df():
    """Two items rated identically by all users — cosine similarity should be ~1."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "title": ["Twin A", "Twin B", "Twin A", "Twin B", "Twin A", "Twin B"],
            "rating": [5.0, 5.0, 3.0, 3.0, 4.0, 4.0],
        }
    )


# ─── Matrix Construction ─────────────────────────────────────────────────────


class TestMatrixConstruction:
    def test_matrix_shape_matches_unique_users_and_items(self, sample_df):
        m = CollaborativeRecommender(sample_df)
        n_users = sample_df["user_id"].nunique()
        n_items = sample_df["title"].nunique()
        assert m.user_item_sparse.shape == (n_users, n_items)

    def test_matrix_is_nonzero(self, model):
        assert model.user_item_sparse.nnz > 0

    def test_implicit_feedback_increases_matrix_values(self, implicit_df):
        m_explicit = CollaborativeRecommender(implicit_df, use_implicit=False)
        m_implicit = CollaborativeRecommender(implicit_df, use_implicit=True)
        explicit_sum = m_explicit.user_item_sparse.sum()
        implicit_sum = m_implicit.user_item_sparse.sum()
        assert implicit_sum > explicit_sum

    def test_implicit_feedback_missing_columns_does_not_raise(self, sample_df):
        """use_implicit=True on a DataFrame without views/purchases must not crash."""
        m = CollaborativeRecommender(sample_df, use_implicit=True)
        assert m.user_item_sparse.nnz > 0

    def test_catalog_map_populated_when_column_present(self, catalog_df):
        m = CollaborativeRecommender(catalog_df)
        assert len(m._catalog_map) > 0

    def test_catalog_map_empty_without_column(self, sample_df):
        m = CollaborativeRecommender(sample_df)
        assert m._catalog_map == {}


# ─── SVD / Factor Shapes ─────────────────────────────────────────────────────


class TestSVDFactors:
    def test_svd_is_fitted_on_normal_data(self, model):
        assert model.svd is not None

    def test_user_factors_row_count_equals_n_users(self, sample_df, model):
        n_users = sample_df["user_id"].nunique()
        assert model.user_factors.shape[0] == n_users

    def test_item_factors_col_count_equals_n_items(self, sample_df, model):
        n_items = sample_df["title"].nunique()
        assert model.item_factors.shape[1] == n_items

    def test_user_and_item_factors_share_latent_dim(self, model):
        assert model.user_factors.shape[1] == model.item_factors.shape[0]

    def test_adaptive_factors_reduced_for_sparse_data(self):
        """Very sparse matrix should use fewer than default 50 components."""
        df = pd.DataFrame(
            {
                "user_id": list(range(25)),
                "title": [f"item_{i}" for i in range(25)],
                "rating": [3.0] * 25,
            }
        )
        m = CollaborativeRecommender(df, n_factors=50)
        if m.svd is not None:
            assert m.svd.n_components < 50

    def test_fallback_on_2x2_matrix(self):
        """min_dim <= 2 must skip SVD and use all-ones fallback factors."""
        df = pd.DataFrame(
            {"user_id": [1, 2], "title": ["A", "B"], "rating": [5.0, 3.0]}
        )
        m = CollaborativeRecommender(df)
        assert m.svd is None
        assert np.all(m.user_factors == 1.0)
        assert np.all(m.item_factors == 1.0)

    def test_fallback_on_single_user_single_item(self):
        df = pd.DataFrame({"user_id": [1], "title": ["Naruto"], "rating": [5]})
        m = CollaborativeRecommender(df)
        assert m.svd is None
        assert m.user_factors.shape == (1, 1)
        assert m.item_factors.shape == (1, 1)
        assert m.predict_rating(1, "Naruto") == 1.0


# ─── recommend() — Item-Item Similarity ──────────────────────────────────────


class TestRecommendItemItem:
    def test_returns_list_of_dicts(self, model):
        results = model.recommend("Naruto", top_n=2)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_result_has_required_keys(self, model):
        results = model.recommend("Naruto", top_n=2)
        for r in results:
            assert "title" in r
            assert "collab_score" in r

    def test_excludes_query_item(self, model):
        results = model.recommend("Naruto", top_n=10)
        titles = [r["title"] for r in results]
        assert "Naruto" not in titles

    def test_no_duplicate_titles(self, model):
        results = model.recommend("Naruto", top_n=10)
        titles = [r["title"] for r in results]
        assert len(titles) == len(set(titles))

    def test_top_n_respected(self, model):
        results = model.recommend("Naruto", top_n=2)
        assert len(results) <= 2

    def test_top_n_capped_at_100(self, model):
        results = model.recommend("Naruto", top_n=500)
        assert len(results) <= 100

    def test_scores_are_floats(self, model):
        results = model.recommend("Naruto", top_n=3)
        for r in results:
            assert isinstance(r["collab_score"], float)

    def test_scores_in_cosine_range(self, model):
        """Cosine similarity must be in [-1.0, 1.0]."""
        results = model.recommend("Naruto", top_n=10)
        for r in results:
            assert -1.0 <= r["collab_score"] <= 1.0

    def test_scores_sorted_descending(self, model):
        results = model.recommend("Naruto", top_n=10)
        scores = [r["collab_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_title_returns_empty_list(self, model):
        assert model.recommend("NonExistentTitle_XYZ", top_n=5) == []

    def test_invalid_top_n_zero_raises(self, model):
        with pytest.raises(ValueError):
            model.recommend("Naruto", top_n=0)

    def test_invalid_top_n_negative_raises(self, model):
        with pytest.raises(ValueError):
            model.recommend("Naruto", top_n=-1)

    def test_invalid_top_n_string_raises(self, model):
        with pytest.raises(ValueError):
            model.recommend("Naruto", top_n="five")

    def test_catalog_filter_restricts_results(self, catalog_df):
        m = CollaborativeRecommender(catalog_df)
        results = m.recommend("Item A", top_n=10, target_catalog="movies")
        for r in results:
            assert m._catalog_map.get(r["title"]) == "movies"

    def test_catalog_filter_no_match_returns_empty(self, catalog_df):
        m = CollaborativeRecommender(catalog_df)
        results = m.recommend("Item A", top_n=10, target_catalog="nonexistent_catalog")
        assert results == []


# ─── Item-Item Similarity — Mathematical Properties ──────────────────────────


class TestItemSimilarityMath:
    def test_self_similarity_is_one(self, model):
        """An item's cosine similarity with itself must be 1.0."""
        idx = model._title_to_idx["Naruto"]
        vec = model.item_factors[:, idx].reshape(1, -1)
        score = cosine_similarity(vec, vec).flatten()[0]
        assert math.isclose(score, 1.0, abs_tol=1e-6)

    def test_identical_items_have_max_similarity(self, identical_items_df):
        """Items with identical interaction patterns should score ~1.0."""
        m = CollaborativeRecommender(identical_items_df)
        results = m.recommend("Twin A", top_n=1)
        if results:
            assert results[0]["title"] == "Twin B"
            assert results[0]["collab_score"] > 0.99

    def test_scores_are_finite(self, model):
        results = model.recommend("Naruto", top_n=10)
        for r in results:
            assert math.isfinite(r["collab_score"])


# ─── predict_for_user() — User-Item Scoring ──────────────────────────────────


class TestPredictForUser:
    def test_returns_list_of_dicts(self, model):
        results = model.predict_for_user(1, top_n=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_result_has_required_keys(self, model):
        results = model.predict_for_user(1, top_n=3)
        for r in results:
            assert "title" in r
            assert "predicted_score" in r

    def test_excludes_already_seen_items(self, sample_df, model):
        seen = set(sample_df[sample_df["user_id"] == 1]["title"].tolist())
        results = model.predict_for_user(1, top_n=10)
        for r in results:
            assert r["title"] not in seen

    def test_top_n_respected(self, model):
        results = model.predict_for_user(1, top_n=2)
        assert len(results) <= 2

    def test_top_n_capped_at_100(self, model):
        results = model.predict_for_user(1, top_n=500)
        assert len(results) <= 100

    def test_scores_sorted_descending(self, model):
        results = model.predict_for_user(1, top_n=10)
        scores = [r["predicted_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_finite(self, model):
        results = model.predict_for_user(1, top_n=10)
        for r in results:
            assert math.isfinite(r["predicted_score"])

    def test_invalid_top_n_raises(self, model):
        with pytest.raises(ValueError):
            model.predict_for_user(1, top_n=0)

        with pytest.raises(ValueError):
            model.predict_for_user(1, top_n=-5)

    def test_cold_start_unknown_user_triggers_fallback(self, model):
        results = model.predict_for_user(999, top_n=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.get("fallback") is True for r in results)

    def test_cold_start_fallback_has_required_keys(self, model):
        results = model.predict_for_user(999, top_n=3)
        for r in results:
            assert "title" in r
            assert "predicted_score" in r

    def test_user_with_all_items_seen_returns_empty(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "title": ["A", "B", "C"],
                "rating": [5, 4, 3],
            }
        )
        m = CollaborativeRecommender(df)
        assert m.predict_for_user(1, top_n=10) == []

    def test_catalog_filter_applied(self, catalog_df):
        m = CollaborativeRecommender(catalog_df)
        results = m.predict_for_user(1, top_n=10, target_catalog="movies")
        for r in results:
            assert m._catalog_map.get(r["title"]) == "movies"

    def test_catalog_filter_no_match_returns_empty(self, catalog_df):
        m = CollaborativeRecommender(catalog_df)
        results = m.predict_for_user(1, top_n=10, target_catalog="nonexistent_catalog")
        assert results == []


# ─── User-User Similarity — Latent Space Properties ─────────────────────────


class TestUserUserSimilarity:
    def test_identical_users_have_close_latent_vectors(self, identical_users_df):
        """u1 and u2 have the same ratings — their user_factors should be nearly identical."""
        m = CollaborativeRecommender(identical_users_df)
        if m.svd is None:
            pytest.skip("SVD not fitted on this matrix size")
        u1_idx = m._user_to_idx["u1"]
        u2_idx = m._user_to_idx["u2"]
        vec1 = m.user_factors[u1_idx].reshape(1, -1)
        vec2 = m.user_factors[u2_idx].reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]
        assert sim > 0.95

    def test_dissimilar_users_have_lower_similarity(self, identical_users_df):
        """u1 (high ratings) and u3 (single low rating) should be less similar than u1/u2."""
        m = CollaborativeRecommender(identical_users_df)
        if m.svd is None:
            pytest.skip("SVD not fitted on this matrix size")
        u1 = m.user_factors[m._user_to_idx["u1"]].reshape(1, -1)
        u2 = m.user_factors[m._user_to_idx["u2"]].reshape(1, -1)
        u3 = m.user_factors[m._user_to_idx["u3"]].reshape(1, -1)
        sim_12 = cosine_similarity(u1, u2)[0][0]
        sim_13 = cosine_similarity(u1, u3)[0][0]
        assert sim_12 > sim_13

    def test_identical_users_get_same_predictions(self, identical_users_df):
        """Users with identical histories should receive the same top recommendation."""
        m = CollaborativeRecommender(identical_users_df)
        recs_u1 = m.predict_for_user("u1", top_n=1)
        recs_u2 = m.predict_for_user("u2", top_n=1)
        if recs_u1 and recs_u2:
            assert recs_u1[0]["title"] == recs_u2[0]["title"]


# ─── predict_rating() ────────────────────────────────────────────────────────


class TestPredictRating:
    def test_returns_float_for_known_user_and_item(self, model):
        result = model.predict_rating(1, "Naruto")
        assert result is not None
        assert isinstance(result, float)

    def test_returns_none_for_unknown_user(self, model):
        assert model.predict_rating(9999, "Naruto") is None

    def test_returns_none_for_unknown_item(self, model):
        assert model.predict_rating(1, "NonExistentItem_XYZ") is None

    def test_returns_none_when_both_unknown(self, model):
        assert model.predict_rating(9999, "NonExistentItem_XYZ") is None

    def test_result_is_finite(self, model):
        result = model.predict_rating(1, "Naruto")
        assert result is not None
        assert math.isfinite(result)

    def test_fallback_single_user_single_item(self):
        df = pd.DataFrame({"user_id": [1], "title": ["Naruto"], "rating": [5]})
        m = CollaborativeRecommender(df)
        assert m.predict_rating(1, "Naruto") == 1.0


# ─── _popularity_fallback() ──────────────────────────────────────────────────


class TestPopularityFallback:
    def test_returns_list(self, model):
        results = model._popularity_fallback(top_n=3)
        assert isinstance(results, list)

    def test_respects_top_n(self, model):
        results = model._popularity_fallback(top_n=2)
        assert len(results) <= 2

    def test_each_item_has_fallback_flag(self, model):
        results = model._popularity_fallback(top_n=3)
        for r in results:
            assert r.get("fallback") is True

    def test_each_item_has_required_keys(self, model):
        results = model._popularity_fallback(top_n=3)
        for r in results:
            assert "title" in r
            assert "predicted_score" in r

    def test_bayesian_score_is_finite(self, model):
        results = model._popularity_fallback(top_n=5)
        for r in results:
            assert math.isfinite(r["predicted_score"])

    def test_high_count_item_ranked_above_low_count(self):
        """Item with many high ratings and higher mean should rank above a rare item."""
        # Popular: 50 ratings of 5.0 — Rare: 1 rating of 1.0
        # Bayesian prior (m=5) cannot pull Popular below Rare here.
        df = pd.DataFrame(
            {
                "user_id": list(range(1, 52)),
                "title": ["Popular"] * 50 + ["Rare"],
                "rating": [5.0] * 50 + [1.0],
            }
        )
        m = CollaborativeRecommender(df)
        results = m._popularity_fallback(top_n=2)
        titles = [r["title"] for r in results]
        assert titles[0] == "Popular"


# ─── Legacy / Regression Tests (preserved from original test_collaborative.py) ─


def sample_data():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "title": ["Naruto", "One Piece", "Naruto", "Bleach", "Attack on Titan"],
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
    assert len(results) > 0
    assert all(r.get("fallback") is True for r in results)


def test_extreme_sparse_matrix():
    df = pd.DataFrame({"user_id": [1], "title": ["Naruto"], "rating": [5]})
    model = CollaborativeRecommender(df)
    assert model.svd is None
    assert model.user_factors.shape == (1, 1)
    assert model.item_factors.shape == (1, 1)
    assert model.predict_rating(1, "Naruto") == 1.0


def test_top_n_validation_in_collaborative():
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
    df = sample_data()
    model = CollaborativeRecommender(df)
    results = model.predict_for_user(1)
    assert len(results) <= 100


def test_predict_for_user_with_catalog():
    df_with_catalog = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "title": ["Item A", "Item B", "Item A", "Item C"],
            "rating": [5, 4, 3, 4],
            "catalog": ["books", "books", "movies", "movies"],
        }
    )
    model = CollaborativeRecommender(df_with_catalog)
    results = model.predict_for_user(2, target_catalog="movies", top_n=10)
    assert isinstance(results, list)


def test_user_with_all_items_seen():
    df = pd.DataFrame(
        {"user_id": [1, 1, 1], "title": ["Item A", "Item B", "Item C"], "rating": [5, 4, 3]}
    )
    model = CollaborativeRecommender(df)
    results = model.predict_for_user(1, top_n=10)
    assert len(results) == 0


def test_predict_rating_unknown_entities():
    df = sample_data()
    model = CollaborativeRecommender(df)
    assert model.predict_rating(999, "Naruto") is None
    assert model.predict_rating(1, "Unknown Item") is None


def test_collaborative_implicit_feedback():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "title": ["Item A", "Item B", "Item B", "Item C", "Item A"],
        "rating": [5, 4, 5, 3, 4],
        "views": [10, 2, 0, 5, 1],
        "purchases": [1, 0, 2, 0, 0]
    })
    model = CollaborativeRecommender(df, use_implicit=True)
    results = model.recommend("Item A", top_n=2)
    assert len(results) > 0
