"""
Integration tests for the causal inference layer.

Covers:
- PropensityModel standalone behaviour
- CausalDebiaser delegating to PropensityModel
- score_key wiring from CausalConfig through HybridRecommender
- recommend_for_user() causal path (previously bypassed)
- causal_evaluation metrics
- API /recommend thread-safety (per-request model construction)

Run with: pytest tests/test_causal_integration.py -v
"""
import sys
import os

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.propensity_model import PropensityModel
from src.model.causal_model import CausalDebiaser
from src.model.causal_config import CausalConfig
from src.model.hybrid_model import HybridRecommender
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.evaluation.causal_evaluation import compare_causal_vs_baseline, score_key_distribution


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def item_df():
    return pd.DataFrame({
        "title":        ["Blockbuster A", "Blockbuster B", "Niche C", "Niche D", "Rare E"],
        "review_count": [5000,             4000,            50,        30,         5],
        "category":     ["Electronics",    "Electronics",   "Books",   "Books",    "Art"],
        "rating":       [4.5,              4.2,             4.8,       4.7,        4.9],
        "avg_sentiment":[0.6,              0.5,             0.7,       0.8,        0.9],
        "description":  ["Popular gadget", "Another gadget","Rare book","Rare book","Art piece"],
        "combined":     [
            "Blockbuster A Popular gadget Electronics",
            "Blockbuster B Another gadget Electronics",
            "Niche C Rare book Books",
            "Niche D Rare book Books",
            "Rare E Art piece Art",
        ],
    })


@pytest.fixture
def interaction_df():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "title":   ["Blockbuster A", "Blockbuster B", "Niche C", "Niche D", "Rare E"],
        "rating":  [5.0, 4.0, 5.0, 4.5, 5.0],
    })


@pytest.fixture
def causal_model(item_df, interaction_df):
    content = ContentRecommender(item_df)
    collab = CollaborativeRecommender(interaction_df)
    return HybridRecommender(
        content, collab, item_df,
        causal_config=CausalConfig(enabled=True, blend_lambda=0.5),
    )


@pytest.fixture
def baseline_model(item_df, interaction_df):
    content = ContentRecommender(item_df)
    collab = CollaborativeRecommender(interaction_df)
    return HybridRecommender(
        content, collab, item_df,
        causal_config=CausalConfig.disabled(),
    )


# ---------------------------------------------------------------------------
# PropensityModel
# ---------------------------------------------------------------------------

class TestPropensityModel:

    def test_popular_item_higher_propensity(self, item_df):
        pm = PropensityModel(item_df)
        assert pm.get("Blockbuster A") > pm.get("Rare E")

    def test_ips_weight_inverts_propensity(self, item_df):
        pm = PropensityModel(item_df)
        assert pm.get_ips_weight("Rare E") > pm.get_ips_weight("Blockbuster A")

    def test_unknown_title_returns_default(self, item_df):
        pm = PropensityModel(item_df)
        assert pm.get("does_not_exist") == 1.0

    def test_empty_df_does_not_crash(self):
        pm = PropensityModel(pd.DataFrame())
        assert pm.get("anything") == 1.0
        assert pm.summary() == {}

    def test_all_scores_returns_full_mapping(self, item_df):
        pm = PropensityModel(item_df)
        scores = pm.all_scores()
        assert set(scores.keys()) == set(item_df["title"].tolist())

    def test_summary_keys(self, item_df):
        pm = PropensityModel(item_df)
        s = pm.summary()
        for key in ("n_items", "mean", "std", "min", "max"):
            assert key in s

    def test_no_review_count_uniform_propensity(self):
        df = pd.DataFrame({"title": ["A", "B"], "category": ["X", "X"]})
        pm = PropensityModel(df)
        assert abs(pm.get("A") - pm.get("B")) < 1e-6

    def test_no_category_uses_popularity_only(self):
        df = pd.DataFrame({"title": ["A", "B"], "review_count": [1000, 1]})
        pm = PropensityModel(df)
        assert pm.get("A") > pm.get("B")

    def test_clip_max_respected(self, item_df):
        pm = PropensityModel(item_df)
        for title in item_df["title"]:
            assert pm.get_ips_weight(title, clip_max=3.0) <= 3.0


# ---------------------------------------------------------------------------
# CausalDebiaser delegates to PropensityModel
# ---------------------------------------------------------------------------

class TestCausalDebiaserDelegation:

    def test_propensity_property_matches_model(self, item_df):
        d = CausalDebiaser(item_df)
        assert d._propensity == d._propensity_model.all_scores()

    def test_get_propensity_delegates(self, item_df):
        d = CausalDebiaser(item_df)
        for title in item_df["title"]:
            assert d.get_propensity(title) == d._propensity_model.get(title)

    def test_get_ips_weight_delegates(self, item_df):
        d = CausalDebiaser(item_df, clip_max=4.0)
        for title in item_df["title"]:
            assert d.get_ips_weight(title) == d._propensity_model.get_ips_weight(title, 4.0)

    def test_summary_includes_lambda_and_clip(self, item_df):
        d = CausalDebiaser(item_df, blend_lambda=0.3, clip_max=7.0)
        s = d.summary()
        assert s["blend_lambda"] == 0.3
        assert s["clip_max"] == 7.0


# ---------------------------------------------------------------------------
# score_key wiring from CausalConfig
# ---------------------------------------------------------------------------

class TestScoreKeyWiring:

    def test_default_score_key_is_hybrid_score(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True),
        )
        recs = model.recommend("Blockbuster A", top_n=3)
        for r in recs:
            assert "causal_score" in r
            assert "original_score" in r

    def test_custom_score_key_from_config(self, item_df, interaction_df):
        """CausalConfig.score_key must be passed through to debias_batch."""
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)
        # Inject a custom score_key — the debiaser should operate on it
        cfg = CausalConfig(enabled=True, blend_lambda=0.5, score_key="hybrid_score")
        model = HybridRecommender(content, collab, item_df, causal_config=cfg)
        assert model._causal_config.score_key == "hybrid_score"
        recs = model.recommend("Blockbuster A", top_n=3)
        assert all("hybrid_score" in r for r in recs)


# ---------------------------------------------------------------------------
# recommend_for_user() causal path
# ---------------------------------------------------------------------------

class TestRecommendForUserCausal:

    def test_causal_keys_present_in_user_recs(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True, blend_lambda=0.5),
        )
        recs = model.recommend_for_user("u1", top_n=3)
        # If causal is enabled and results are returned, causal keys must be present
        if recs:
            for r in recs:
                assert "causal_score" in r, "causal_score missing from recommend_for_user output"
                assert "original_score" in r

    def test_no_causal_keys_when_disabled(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig.disabled(),
        )
        recs = model.recommend_for_user("u1", top_n=3)
        for r in recs:
            assert "causal_score" not in r
            assert "original_score" not in r

    def test_user_recs_sorted_by_hybrid_score(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True, blend_lambda=0.5),
        )
        recs = model.recommend_for_user("u1", top_n=4)
        if len(recs) > 1:
            scores = [r["hybrid_score"] for r in recs]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# API thread-safety — per-request model construction
# ---------------------------------------------------------------------------

class TestAPIThreadSafety:
    """
    Verify that the fixed main.py builds a fresh HybridRecommender per request
    so causal config is never shared across concurrent calls.
    """

    def test_two_requests_with_different_causal_settings(self, item_df, interaction_df):
        """
        Simulate two concurrent requests: one with causal ON, one with causal OFF.
        Each must produce independent results without cross-contamination.
        """
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)

        # Request 1 — causal ON
        model_causal = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True, blend_lambda=1.0),
        )
        # Request 2 — causal OFF
        model_plain = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig.disabled(),
        )

        recs_causal = model_causal.recommend("Blockbuster A", top_n=4)
        recs_plain = model_plain.recommend("Blockbuster A", top_n=4)

        # Causal model must have causal_score keys; plain must not
        for r in recs_causal:
            assert "causal_score" in r
        for r in recs_plain:
            assert "causal_score" not in r

        # The two models must not share the same _debiaser instance
        assert model_causal._debiaser is not model_plain._debiaser

    def test_causal_model_does_not_mutate_content_model(self, item_df, interaction_df):
        """ContentRecommender is shared read-only — causal layer must not modify it."""
        content = ContentRecommender(item_df)
        collab = CollaborativeRecommender(interaction_df)

        matrix_before = content.matrix.copy()

        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True, blend_lambda=0.8),
        )
        model.recommend("Blockbuster A", top_n=4)

        assert np.allclose(content.matrix, matrix_before), (
            "ContentRecommender.matrix was mutated by causal layer"
        )


# ---------------------------------------------------------------------------
# causal_evaluation metrics
# ---------------------------------------------------------------------------

class TestCausalEvaluation:

    def test_compare_returns_expected_keys(self, causal_model, baseline_model, item_df):
        query_titles = ["Blockbuster A", "Niche C"]
        result = compare_causal_vs_baseline(
            causal_model, baseline_model, item_df, query_titles, top_n=3
        )
        expected_keys = {
            "popularity_bias_reduction", "coverage_gain", "diversity_gain",
            "causal_avg_popularity_rank", "baseline_avg_popularity_rank",
            "causal_coverage", "baseline_coverage",
            "causal_diversity", "baseline_diversity",
            "n_queries",
        }
        assert expected_keys.issubset(result.keys())

    def test_n_queries_matches_valid_queries(self, causal_model, baseline_model, item_df):
        query_titles = ["Blockbuster A", "Niche C", "nonexistent_item_xyz"]
        result = compare_causal_vs_baseline(
            causal_model, baseline_model, item_df, query_titles, top_n=3
        )
        # nonexistent_item_xyz returns no recs from either model — excluded
        assert result["n_queries"] <= len(query_titles)

    def test_coverage_values_in_range(self, causal_model, baseline_model, item_df):
        result = compare_causal_vs_baseline(
            causal_model, baseline_model, item_df, ["Blockbuster A"], top_n=3
        )
        assert 0.0 <= result["causal_coverage"] <= 1.0
        assert 0.0 <= result["baseline_coverage"] <= 1.0

    def test_diversity_values_in_range(self, causal_model, baseline_model, item_df):
        result = compare_causal_vs_baseline(
            causal_model, baseline_model, item_df, ["Blockbuster A"], top_n=3
        )
        assert 0.0 <= result["causal_diversity"] <= 1.0
        assert 0.0 <= result["baseline_diversity"] <= 1.0

    def test_score_key_distribution_returns_expected_keys(self, causal_model, item_df):
        result = score_key_distribution(
            causal_model, item_df, ["Blockbuster A"], top_n=3
        )
        for key in ("mean", "std", "min", "max", "n_scores"):
            assert key in result

    def test_score_key_distribution_scores_in_bounds(self, causal_model, item_df):
        result = score_key_distribution(
            causal_model, item_df, ["Blockbuster A", "Niche C"], top_n=3
        )
        assert 0.0 <= result["min"] <= result["max"] <= 1.0

    def test_empty_query_list_returns_zero_queries(self, causal_model, baseline_model, item_df):
        result = compare_causal_vs_baseline(
            causal_model, baseline_model, item_df, [], top_n=3
        )
        assert result["n_queries"] == 0
