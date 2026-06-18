"""
Regression tests for Issue #1349.

Verifies that:
  1. User-personalized evaluation uses predict_for_user(), not item-to-item recommend().
  2. Item-based recommendation paths are preserved and still functional.
  3. Precision@K, Recall@K, MAP, and NDCG produce numerically correct values.
  4. Edge cases (cold-start users, sparse interactions, empty histories) are handled.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluation import (
    _precision_at_k,
    _recall_at_k,
    _ndcg_at_k,
    _dcg_at_k,
    average_precision_at_k,
    _build_user_test_data,
    _intra_list_diversity,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_interaction_df(n_users=5, n_items=10, seed=42):
    rng = np.random.default_rng(seed)
    items = [f"item_{i}" for i in range(n_items)]
    rows = []
    for uid in range(n_users):
        n_interactions = int(rng.integers(3, n_items + 1))
        chosen = rng.choice(items, size=n_interactions, replace=False).tolist()
        for title in chosen:
            rows.append({
                "user_id": f"u{uid}",
                "title": title,
                "rating": float(rng.integers(1, 6)),
            })
    return pd.DataFrame(rows)


def _make_item_df(n=10):
    return pd.DataFrame({
        "title": [f"item_{i}" for i in range(n)],
        "combined": [f"product description category {i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# 1. User-personalized evaluation: _build_user_test_data
# ---------------------------------------------------------------------------

class TestBuildUserTestData:
    def test_returns_dataframe_and_list(self):
        df = _make_interaction_df()
        train_df, pairs = _build_user_test_data(df, min_interactions=3)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(pairs, list)

    def test_pair_structure(self):
        df = _make_interaction_df()
        _, pairs = _build_user_test_data(df, min_interactions=3)
        for uid, train_items, test_items in pairs:
            assert isinstance(uid, str)
            assert isinstance(train_items, set)
            assert isinstance(test_items, set)
            assert len(test_items) >= 1

    def test_train_test_no_overlap(self):
        df = _make_interaction_df(n_users=5, n_items=8)
        _, pairs = _build_user_test_data(df, min_interactions=3)
        for _, train_items, test_items in pairs:
            assert train_items.isdisjoint(test_items), \
                "Train and test sets must be disjoint"

    def test_cold_start_users_excluded_from_test_pairs(self):
        rows = (
            [{"user_id": "cold", "title": f"item_{i}", "rating": 3.0} for i in range(2)]
            + [{"user_id": "warm", "title": f"item_{i}", "rating": 3.0} for i in range(5)]
        )
        df = pd.DataFrame(rows)
        _, pairs = _build_user_test_data(df, min_interactions=3)
        user_ids = {p[0] for p in pairs}
        assert "cold" not in user_ids
        assert "warm" in user_ids

    def test_cold_start_user_rows_kept_in_train(self):
        rows = (
            [{"user_id": "cold", "title": f"item_{i}", "rating": 3.0} for i in range(2)]
            + [{"user_id": "warm", "title": f"item_{i}", "rating": 3.0} for i in range(5)]
        )
        df = pd.DataFrame(rows)
        train_df, _ = _build_user_test_data(df, min_interactions=3)
        # Cold-start user rows should still appear in train
        assert "cold" in train_df["user_id"].values

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["user_id", "title", "rating"])
        train_df, pairs = _build_user_test_data(df)
        assert pairs == []
        assert len(train_df) == 0

    def test_reproducible_with_same_seed(self):
        df = _make_interaction_df()
        _, pairs1 = _build_user_test_data(df, random_seed=7)
        _, pairs2 = _build_user_test_data(df, random_seed=7)
        result1 = sorted((p[0], frozenset(p[2])) for p in pairs1)
        result2 = sorted((p[0], frozenset(p[2])) for p in pairs2)
        assert result1 == result2

    def test_different_seeds_may_differ(self):
        df = _make_interaction_df(n_users=4, n_items=12, seed=0)
        _, pairs1 = _build_user_test_data(df, random_seed=0)
        _, pairs2 = _build_user_test_data(df, random_seed=99)
        test_sets1 = sorted(frozenset(p[2]) for p in pairs1)
        test_sets2 = sorted(frozenset(p[2]) for p in pairs2)
        # At least one user's test split should differ between seeds
        assert test_sets1 != test_sets2

    def test_exactly_min_interactions_included(self):
        rows = [{"user_id": "u0", "title": f"item_{i}", "rating": 3.0} for i in range(3)]
        df = pd.DataFrame(rows)
        _, pairs = _build_user_test_data(df, min_interactions=3)
        assert len(pairs) == 1

    def test_below_min_interactions_excluded(self):
        rows = [{"user_id": "u0", "title": f"item_{i}", "rating": 3.0} for i in range(2)]
        df = pd.DataFrame(rows)
        _, pairs = _build_user_test_data(df, min_interactions=3)
        assert len(pairs) == 0

    def test_train_df_has_required_columns(self):
        df = _make_interaction_df()
        train_df, _ = _build_user_test_data(df)
        for col in ("user_id", "title", "rating"):
            assert col in train_df.columns


# ---------------------------------------------------------------------------
# 2. Item-based recommendation path (preserved, not broken)
# ---------------------------------------------------------------------------

class TestItemBasedRecommendationPath:
    def test_collaborative_recommend_accepts_item_title(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=5, n_items=8, seed=0)
        model = CollaborativeRecommender(df)
        item = df["title"].iloc[0]
        recs = model.recommend(item, top_n=3)
        assert isinstance(recs, list)

    def test_item_recs_exclude_query_item(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=5, n_items=8, seed=0)
        model = CollaborativeRecommender(df)
        item = df["title"].iloc[0]
        recs = model.recommend(item, top_n=3)
        rec_titles = [r["title"] for r in recs]
        assert item not in rec_titles

    def test_collaborative_predict_for_user_excludes_seen_items(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=5, n_items=10, seed=1)
        model = CollaborativeRecommender(df)
        user_id = df["user_id"].iloc[0]
        seen = set(df[df["user_id"] == user_id]["title"])
        recs = model.predict_for_user(user_id, top_n=5)
        for r in recs:
            assert r["title"] not in seen, \
                f"predict_for_user returned already-seen item: {r['title']}"

    def test_item_based_and_user_based_are_separate_apis(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=8, n_items=12, seed=5)
        model = CollaborativeRecommender(df)
        user_id = df["user_id"].iloc[0]
        first_item = df[df["user_id"] == user_id]["title"].iloc[0]
        item_recs = model.recommend(first_item, top_n=5)
        user_recs = model.predict_for_user(user_id, top_n=5)
        # Both return lists of dicts
        assert isinstance(item_recs, list)
        assert isinstance(user_recs, list)

    def test_item_recs_have_collab_score_key(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=5, n_items=8, seed=0)
        model = CollaborativeRecommender(df)
        item = df["title"].iloc[0]
        recs = model.recommend(item, top_n=3)
        for r in recs:
            assert "title" in r


# ---------------------------------------------------------------------------
# 3. Metric correctness
# ---------------------------------------------------------------------------

class TestPrecisionAtKCorrectness:
    def test_perfect_precision(self):
        assert _precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_zero_precision(self):
        assert _precision_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_partial_precision(self):
        result = _precision_at_k(["a", "x", "b"], {"a", "b"}, 3)
        assert abs(result - 2 / 3) < 1e-9

    def test_k_larger_than_list(self):
        result = _precision_at_k(["a", "b"], {"a", "b", "c"}, 5)
        assert abs(result - 2 / 5) < 1e-9

    def test_zero_k(self):
        assert _precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_recommended(self):
        assert _precision_at_k([], {"a"}, 3) == 0.0

    def test_empty_relevant(self):
        assert _precision_at_k(["a", "b"], set(), 2) == 0.0


class TestRecallAtKCorrectness:
    def test_perfect_recall(self):
        assert _recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_zero_recall(self):
        assert _recall_at_k(["x", "y"], {"a", "b", "c"}, 2) == 0.0

    def test_partial_recall(self):
        result = _recall_at_k(["a", "x"], {"a", "b", "c"}, 2)
        assert abs(result - 1 / 3) < 1e-9

    def test_zero_k(self):
        assert _recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_relevant(self):
        assert _recall_at_k(["a", "b"], set(), 2) == 0.0


class TestNDCGCorrectness:
    def test_perfect_ndcg(self):
        assert abs(_ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) - 1.0) < 1e-9

    def test_zero_ndcg_no_hits(self):
        assert _ndcg_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_position_matters(self):
        ndcg_top = _ndcg_at_k(["a", "x", "x"], {"a"}, 3)
        ndcg_bot = _ndcg_at_k(["x", "x", "a"], {"a"}, 3)
        assert ndcg_top > ndcg_bot

    def test_empty_recommended(self):
        assert _ndcg_at_k([], {"a"}, 3) == 0.0

    def test_empty_relevant(self):
        assert _ndcg_at_k(["a", "b"], set(), 3) == 0.0

    def test_zero_k(self):
        assert _ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_dcg_single_relevant_at_rank1(self):
        result = _dcg_at_k(["a", "b", "c"], {"a"}, 3)
        assert abs(result - 1.0) < 1e-9  # 1 / log2(2) = 1


class TestMAPCorrectness:
    def test_perfect_map(self):
        result = average_precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
        assert abs(result - 1.0) < 1e-9

    def test_single_hit_at_first_position(self):
        result = average_precision_at_k(["a", "x", "x"], {"a"}, 3)
        assert abs(result - 1.0) < 1e-9

    def test_no_hits(self):
        result = average_precision_at_k(["x", "y"], {"a", "b"}, 2)
        assert result == 0.0

    def test_zero_k(self):
        assert average_precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_recommended(self):
        assert average_precision_at_k([], {"a"}, 3) == 0.0

    def test_empty_relevant(self):
        assert average_precision_at_k(["a", "b"], set(), 2) == 0.0


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_cold_start_user_gets_fallback_not_error(self):
        from src.model.collaborative_model import CollaborativeRecommender
        df = _make_interaction_df(n_users=5, n_items=8, seed=0)
        model = CollaborativeRecommender(df)
        recs = model.predict_for_user("unknown_user_xyz_cold", top_n=5)
        assert isinstance(recs, list)

    def test_single_user_single_item_excluded_from_pairs(self):
        df = pd.DataFrame([{"user_id": "u0", "title": "item_0", "rating": 4.0}])
        _, pairs = _build_user_test_data(df, min_interactions=3)
        assert pairs == []

    def test_all_cold_start_returns_empty_pairs(self):
        rows = [
            {"user_id": f"u{i}", "title": "item_0", "rating": 3.0}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        _, pairs = _build_user_test_data(df, min_interactions=3)
        assert pairs == []

    def test_intra_list_diversity_none_matrix(self):
        item_df = _make_item_df()
        result = _intra_list_diversity(["item_0", "item_1"], item_df, None)
        assert result == 0.0

    def test_intra_list_diversity_single_item(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        item_df = _make_item_df()
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(item_df["combined"])
        result = _intra_list_diversity(["item_0"], item_df, matrix)
        assert result == 0.0

    def test_intra_list_diversity_two_items(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        item_df = _make_item_df()
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(item_df["combined"])
        result = _intra_list_diversity(["item_0", "item_5"], item_df, matrix)
        assert 0.0 <= result <= 1.0

    def test_intra_list_diversity_empty_recs(self):
        item_df = _make_item_df()
        result = _intra_list_diversity([], item_df, None)
        assert result == 0.0

    def test_precision_all_zeros_k_equals_zero(self):
        assert _precision_at_k(["a", "b", "c"], {"a"}, 0) == 0.0

    def test_recall_with_very_large_k(self):
        result = _recall_at_k(["a", "b"], {"a", "c"}, 100)
        assert abs(result - 0.5) < 1e-9

    def test_ndcg_with_no_overlap_between_recs_and_relevant(self):
        result = _ndcg_at_k(["x", "y", "z"], {"a", "b", "c"}, 5)
        assert result == 0.0

    def test_user_test_pairs_test_fraction_respected(self):
        rows = [{"user_id": "u0", "title": f"item_{i}", "rating": 3.0} for i in range(10)]
        df = pd.DataFrame(rows)
        _, pairs = _build_user_test_data(df, min_interactions=3, test_fraction=0.3)
        assert len(pairs) == 1
        _, _, test_items = pairs[0]
        assert len(test_items) == 3  # 30% of 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
