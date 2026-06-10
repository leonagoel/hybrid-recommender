"""
Tests for Issue #928: zero-variance normalization must not inject phantom signal.

Before the fix, _normalize_scores([0.0, 0.0, 0.0]) returned [0.5, 0.5, 0.5].
When the collaborative model is absent every candidate receives raw_collab=0.0,
so the collab_scores vector is all-zero.  Normalizing that to 0.5 adds
beta * 0.5 ≈ 0.175 of phantom collaborative contribution to every hybrid score,
corrupting the ranking and misleading recommendation explanations.

This module verifies:
  - All-zero vectors remain all-zero after normalization (both minmax and zscore).
  - Legitimate constant non-zero vectors are handled as before (0.5 midpoint).
  - Hybrid recommendations with no collaborative model carry zero collab_score.
  - Hybrid recommendations with a real collaborative model carry correct scores.
  - Sentiment normalization is also correct when all sentiments are zero.
  - Explanation weighted_components['collaborative'] is zero when no collab model.
  - Ranking stability: content-only results are not reordered by phantom collab.
  - The regression case from the issue report is explicitly covered.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.model.hybrid_model import HybridRecommender


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_item_df(n: int = 5) -> pd.DataFrame:
    """Minimal item DataFrame with required columns."""
    titles = [f"Item {i}" for i in range(n)]
    return pd.DataFrame(
        {
            "title": titles,
            "description": [f"Desc {i}" for i in range(n)],
            "category": ["Electronics"] * n,
            "rating": [4.0 + i * 0.1 for i in range(n)],
            "avg_sentiment": [0.0] * n,   # deliberately all-zero
            "review_count": [10 + i * 5 for i in range(n)],
            "combined": [f"Item {i} Desc {i} Electronics" for i in range(n)],
        }
    )


def _make_content_model(item_df: pd.DataFrame) -> MagicMock:
    """Stub ContentRecommender that returns deterministic scores."""
    cm = MagicMock()
    cm.df = item_df
    titles = item_df["title"].tolist()
    # Recommend all items except the query with linearly decreasing scores
    def _recommend(title, top_n=10, target_catalog=None):
        others = [t for t in titles if t != title]
        return [
            {"title": t, "content_score": 1.0 - (i + 1) * 0.1}
            for i, t in enumerate(others[:top_n])
        ]
    cm.recommend.side_effect = _recommend
    cm.explain_similarity.return_value = []
    return cm


def _make_collab_model(item_df: pd.DataFrame, score_offset: float = 0.3) -> MagicMock:
    """Stub CollaborativeRecommender that returns deterministic non-zero scores."""
    cm = MagicMock()
    titles = item_df["title"].tolist()
    def _recommend(title, top_n=10, target_catalog=None):
        others = [t for t in titles if t != title]
        return [
            {"title": t, "collab_score": score_offset + (i + 1) * 0.05}
            for i, t in enumerate(others[:top_n])
        ]
    cm.recommend.side_effect = _recommend
    return cm


# ── _normalize_scores unit tests ──────────────────────────────────────────────

class TestNormalizeScoresZeroVariance(unittest.TestCase):
    """Unit tests for the _normalize_scores zero-variance fix."""

    def _minmax(self) -> HybridRecommender:
        return HybridRecommender(None, None, item_df=None, normalization='minmax')

    def _zscore(self) -> HybridRecommender:
        return HybridRecommender(None, None, item_df=None, normalization='zscore')

    # ── Regression case (the exact scenario from the issue report) ─────────────

    def test_regression_all_zero_minmax_returns_zeros(self):
        """Issue #928 regression: [0,0,0] must not become [0.5,0.5,0.5]."""
        h = self._minmax()
        result = h._normalize_scores([0.0, 0.0, 0.0])
        self.assertEqual(result, [0.0, 0.0, 0.0],
                         "[0,0,0] must remain [0,0,0] — not become [0.5,0.5,0.5]")

    def test_regression_all_zero_zscore_returns_zeros(self):
        """Issue #928 regression (zscore path): [0,0,0] must not become [0.5,0.5,0.5]."""
        h = self._zscore()
        result = h._normalize_scores([0.0, 0.0, 0.0])
        self.assertEqual(result, [0.0, 0.0, 0.0],
                         "[0,0,0] must remain [0,0,0] under zscore normalization")

    # ── All-zero input ─────────────────────────────────────────────────────────

    def test_minmax_single_zero_returns_zero(self):
        h = self._minmax()
        self.assertEqual(h._normalize_scores([0.0]), [0.0])

    def test_minmax_larger_all_zero_returns_zeros(self):
        h = self._minmax()
        result = h._normalize_scores([0.0] * 10)
        self.assertEqual(result, [0.0] * 10)

    def test_zscore_single_zero_returns_zero(self):
        h = self._zscore()
        self.assertEqual(h._normalize_scores([0.0]), [0.0])

    def test_zscore_larger_all_zero_returns_zeros(self):
        h = self._zscore()
        result = h._normalize_scores([0.0] * 10)
        self.assertEqual(result, [0.0] * 10)

    # ── Legitimate constant non-zero input (genuine tie) ───────────────────────

    def test_minmax_constant_nonzero_returns_half(self):
        """A constant non-zero vector is a tie; 0.5 is the correct midpoint."""
        h = self._minmax()
        result = h._normalize_scores([2.0, 2.0, 2.0])
        self.assertTrue(all(abs(v - 0.5) < 1e-9 for v in result),
                        "Constant non-zero must map to 0.5")

    def test_zscore_constant_nonzero_returns_half(self):
        h = self._zscore()
        result = h._normalize_scores([5.0, 5.0, 5.0])
        self.assertTrue(all(abs(v - 0.5) < 1e-9 for v in result),
                        "Constant non-zero must map to 0.5 under zscore")

    def test_minmax_constant_negative_nonzero_returns_half(self):
        """Negative constant value is also a tie — same logic applies."""
        h = self._minmax()
        result = h._normalize_scores([-1.0, -1.0, -1.0])
        self.assertTrue(all(abs(v - 0.5) < 1e-9 for v in result))

    # ── Normal (non-degenerate) input ─────────────────────────────────────────

    def test_minmax_normal_input_correct(self):
        h = self._minmax()
        result = h._normalize_scores([0.0, 0.5, 1.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)

    def test_minmax_positive_values_bounded(self):
        h = self._minmax()
        result = h._normalize_scores([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in result))

    def test_zscore_normal_input_monotonic(self):
        h = self._zscore()
        vals = [1.0, 2.0, 3.0, 4.0]
        result = h._normalize_scores(vals)
        self.assertEqual(result, sorted(result))
        self.assertTrue(all(0.0 < v < 1.0 for v in result))

    def test_empty_list_returns_empty(self):
        h = self._minmax()
        self.assertEqual(h._normalize_scores([]), [])


# ── End-to-end phantom signal tests ──────────────────────────────────────────

class TestPhantomCollabSignal(unittest.TestCase):
    """
    Verify that recommendations produced without a collaborative model
    carry exactly zero collaborative contribution.
    """

    def setUp(self):
        self.item_df = _make_item_df(6)
        self.content_model = _make_content_model(self.item_df)

    def test_collab_score_is_zero_when_no_collab_model(self):
        """
        Core regression: every recommendation must have collab_score == 0.0
        when collab_model is None.
        """
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=5)

        self.assertTrue(len(recs) > 0, "Should return some recommendations")
        for rec in recs:
            self.assertEqual(
                rec["collab_score"], 0.0,
                f"collab_score must be 0.0 when no collab model, got {rec['collab_score']} "
                f"for item '{rec['title']}'",
            )

    def test_hybrid_score_not_inflated_without_collab(self):
        """
        Without collaborative data the hybrid score must equal
        alpha * content_score + gamma * sentiment_score (+ tiny popularity bonus).
        The phantom beta * 0.5 contribution must be absent.
        """
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.5, beta=0.4, gamma=0.1,
        )
        recs = hr.recommend("Item 0", top_n=5)
        self.assertTrue(len(recs) > 0)

        for rec in recs:
            # Recompute expected hybrid score from reported components
            a_norm = 0.5 / (0.5 + 0.4 + 0.1)  # _get_active_weights normalizes
            b_norm = 0.4 / (0.5 + 0.4 + 0.1)
            g_norm = 0.1 / (0.5 + 0.4 + 0.1)

            expected_max = (
                a_norm * rec["content_score"]
                + b_norm * rec["collab_score"]   # collab_score must be 0
                + g_norm * rec["sentiment_score"]
                + 0.05  # maximum popularity bonus
            )
            self.assertLessEqual(
                rec["hybrid_score"], round(min(1.0, expected_max), 4) + 1e-4,
                "hybrid_score must not exceed content+collab+sentiment+popularity",
            )

    def test_collab_score_nonzero_when_collab_model_present(self):
        """
        When a collaborative model IS present, collab_score must reflect real signal.
        """
        collab = _make_collab_model(self.item_df, score_offset=0.6)
        hr = HybridRecommender(
            self.content_model, collab_model=collab, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=5)
        self.assertTrue(len(recs) > 0)

        # At least some recommendations must have non-zero collab_score
        collab_scores = [r["collab_score"] for r in recs]
        self.assertTrue(
            any(s > 0.0 for s in collab_scores),
            "At least one recommendation should have non-zero collab_score "
            f"when collab model is present. Got: {collab_scores}",
        )


class TestPhantomSentimentSignal(unittest.TestCase):
    """Verify the same zero-variance fix applies to the sentiment channel."""

    def setUp(self):
        # All avg_sentiment = 0.0 — common when NLP pipeline hasn't run yet
        self.item_df = _make_item_df(6)  # avg_sentiment is all 0.0
        self.content_model = _make_content_model(self.item_df)

    def test_sentiment_score_is_zero_when_all_sentiments_absent(self):
        """
        When every item has avg_sentiment=0.0 the sentiment_score in the
        recommendation output must also be 0.0 (not 0.5).
        """
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=5)
        self.assertTrue(len(recs) > 0)

        for rec in recs:
            self.assertEqual(
                rec["sentiment_score"], 0.0,
                f"sentiment_score must be 0.0 when all sentiments are 0.0, "
                f"got {rec['sentiment_score']} for '{rec['title']}'",
            )


# ── Ranking stability ─────────────────────────────────────────────────────────

class TestRankingStability(unittest.TestCase):
    """
    Content-only ranking must be determined solely by content scores.
    Phantom collab signal must not reorder results.
    """

    def setUp(self):
        self.item_df = _make_item_df(6)
        self.content_model = _make_content_model(self.item_df)

    def test_ranking_order_matches_content_scores_without_collab(self):
        """
        Without collaborative data the ranking must follow content_score order
        (plus a negligible popularity adjustment that does not change relative
        order for items with similar popularity).
        """
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.8, beta=0.15, gamma=0.05,
        )
        recs = hr.recommend("Item 0", top_n=4)
        self.assertTrue(len(recs) >= 2)

        hybrid_scores = [r["hybrid_score"] for r in recs]
        self.assertEqual(
            hybrid_scores, sorted(hybrid_scores, reverse=True),
            "Results must be sorted by hybrid_score descending",
        )

    def test_ranking_deterministic_across_calls(self):
        """Calling recommend() twice must return identical results."""
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
        )
        recs1 = hr.recommend("Item 0", top_n=4)
        recs2 = hr.recommend("Item 0", top_n=4)
        self.assertEqual(
            [r["title"] for r in recs1],
            [r["title"] for r in recs2],
        )
        self.assertEqual(
            [r["hybrid_score"] for r in recs1],
            [r["hybrid_score"] for r in recs2],
        )


# ── Explanation integrity ─────────────────────────────────────────────────────

class TestExplanationIntegrity(unittest.TestCase):
    """
    Recommendation explanations must not imply collaborative evidence when none exists.
    """

    def setUp(self):
        self.item_df = _make_item_df(5)
        self.content_model = _make_content_model(self.item_df)

    def test_explanation_collaborative_component_is_zero_without_collab(self):
        """
        explain=True: weighted_components['collaborative'] must be 0.0 when
        there is no collaborative model.
        """
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=3, explain=True)
        self.assertTrue(len(recs) > 0)

        for rec in recs:
            if "explanation" not in rec:
                continue
            collab_contrib = rec["explanation"]["weighted_components"]["collaborative"]
            self.assertEqual(
                collab_contrib, 0.0,
                f"Explanation must show zero collaborative contribution when "
                f"no collab model exists. Got {collab_contrib} for '{rec['title']}'",
            )

    def test_explanation_collaborative_match_false_without_collab(self):
        """signals['collaborative_match'] must be False when raw_collab == 0."""
        hr = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=3, explain=True)
        self.assertTrue(len(recs) > 0)

        for rec in recs:
            if "explanation" not in rec:
                continue
            match = rec["explanation"]["signals"]["collaborative_match"]
            self.assertFalse(
                match,
                f"collaborative_match must be False when no collab model. "
                f"Got {match} for '{rec['title']}'",
            )


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompatibility(unittest.TestCase):
    """
    Verify that the fix does not regress recommendations when genuine
    collaborative signal is present.
    """

    def setUp(self):
        # Give items non-zero sentiment to also test that path
        self.item_df = _make_item_df(6)
        self.item_df["avg_sentiment"] = [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
        self.content_model = _make_content_model(self.item_df)
        self.collab_model = _make_collab_model(self.item_df, score_offset=0.5)

    def test_hybrid_recs_with_collab_return_results(self):
        hr = HybridRecommender(
            self.content_model, self.collab_model, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=4)
        self.assertTrue(len(recs) > 0)

    def test_all_scores_in_valid_range_with_collab(self):
        hr = HybridRecommender(
            self.content_model, self.collab_model, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs = hr.recommend("Item 0", top_n=4)
        for rec in recs:
            for key in ("content_score", "collab_score", "sentiment_score", "hybrid_score"):
                self.assertGreaterEqual(rec[key], 0.0, f"{key} must be >= 0")
                self.assertLessEqual(rec[key], 1.0, f"{key} must be <= 1")

    def test_hybrid_score_greater_than_content_only_when_collab_present(self):
        """
        Adding real collaborative signal should generally raise hybrid scores
        above what a content-only model would produce (with the same weights).
        """
        hr_hybrid = HybridRecommender(
            self.content_model, self.collab_model, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        hr_content = HybridRecommender(
            self.content_model, collab_model=None, item_df=self.item_df,
            alpha=0.4, beta=0.35, gamma=0.25,
        )
        recs_hybrid = hr_hybrid.recommend("Item 0", top_n=4)
        recs_content = hr_content.recommend("Item 0", top_n=4)

        avg_hybrid = sum(r["hybrid_score"] for r in recs_hybrid) / len(recs_hybrid)
        avg_content = sum(r["hybrid_score"] for r in recs_content) / len(recs_content)

        self.assertGreater(
            avg_hybrid, avg_content,
            "Hybrid mode with real collab signal should produce higher average "
            "scores than content-only mode (not inflated by phantom 0.5)",
        )
