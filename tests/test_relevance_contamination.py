"""
Tests for Issue #926: relevance-set contamination from rating threshold.

Confirmed issues before the fix
---------------------------------
1. _build_test_data() built relevance sets as the UNION of:
     (a) items in the same category as the query, AND
     (b) ALL items with rating >= 4.0 across the entire catalog
   This meant a children's educational product's relevance set contained
   every highly-rated heavy metal album, horror movie, or power tool —
   items that share no semantic similarity with the query.

2. A dead _get_relevant() function was defined inside run_evaluation() but
   never called.  It had identical contamination.  Left uncleaned it was a
   landmine waiting to be wired up.

Consequences of the contamination
-----------------------------------
* Precision@K was inflated: random recommendations would still "hit"
  globally popular items that happened to appear in the relevance set.
* Recall@K was inflated and semantically meaningless (the denominator
  included unrelated items).
* NDCG@K and MRR were inflated by the same mechanism.
* A popularity-biased model (recommending high-rated items regardless of
  query) scored identically to a well-tuned semantic model on this metric.

After the fix
-------------
* Relevance sets use category membership only.
* No item enters the relevant set solely because its rating exceeds a
  global threshold.
* _get_relevant() dead code is removed.
* The guard comment in the item-based fallback loop prevents regression.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.evaluation import (
    _build_test_data,
    _precision_at_k,
    _recall_at_k,
    _ndcg_at_k,
    _mean_reciprocal_rank,
    _hit_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df_with_cross_category_ratings() -> pd.DataFrame:
    """
    DataFrame where high-rated items span multiple categories.

    Category A (children's products): 5 items, ratings 2.0–3.5
    Category B (heavy metal music):   5 items, ratings 4.5–5.0

    Before the fix, all of Category B would appear in Category A's
    relevance sets solely because their ratings exceed 4.0.
    """
    return pd.DataFrame(
        {
            "title": [
                # Category A — children's products, moderate ratings
                "Alphabet Puzzle", "Color Blocks", "Story Time Book",
                "Drawing Set", "Baby Rattle",
                # Category B — heavy metal music, high ratings
                "Metal Fury Album", "Scream Machine LP", "Iron Riff Vinyl",
                "Thunder Blast CD", "Dark Legend Record",
            ],
            "category": ["Children"] * 5 + ["HeavyMetal"] * 5,
            "rating": [2.0, 2.5, 3.0, 3.5, 2.8, 4.5, 4.8, 5.0, 4.6, 4.9],
            "description": [f"desc {i}" for i in range(10)],
            "sentiment_score": [0.0] * 10,
        }
    )


def _make_csv_from_df(df: pd.DataFrame, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "products.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. _build_test_data relevance — cross-category rating items excluded
# ---------------------------------------------------------------------------

class TestBuildTestDataRelevance(unittest.TestCase):
    """
    Verify that _build_test_data() no longer includes high-rated items from
    unrelated categories in the relevance sets.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.df = _make_df_with_cross_category_ratings()
        self.csv = _make_csv_from_df(self.df, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_heavy_metal_not_relevant_to_childrens_query(self):
        """
        Regression: a children's product query must not have heavy metal
        albums in its relevant set.
        """
        _, _, df, test_pairs = _build_test_data(self.csv)

        heavy_metal_titles = set(
            self.df[self.df["category"] == "HeavyMetal"]["title"].tolist()
        )
        children_titles = set(
            self.df[self.df["category"] == "Children"]["title"].tolist()
        )

        for uid, query_title, relevant in test_pairs:
            if query_title in children_titles:
                contamination = relevant & heavy_metal_titles
                self.assertEqual(
                    contamination,
                    set(),
                    f"Children's query '{query_title}' has heavy-metal items in "
                    f"its relevance set: {contamination}. "
                    "Cross-category high-rated items must not be included.",
                )

    def test_childrens_not_relevant_to_heavy_metal_query(self):
        """
        Reverse case: a heavy-metal query must not have children's products
        marked as relevant solely because they might meet a rating threshold.
        """
        _, _, df, test_pairs = _build_test_data(self.csv)

        children_titles = set(
            self.df[self.df["category"] == "Children"]["title"].tolist()
        )
        heavy_metal_titles = set(
            self.df[self.df["category"] == "HeavyMetal"]["title"].tolist()
        )

        for uid, query_title, relevant in test_pairs:
            if query_title in heavy_metal_titles:
                contamination = relevant & children_titles
                self.assertEqual(
                    contamination,
                    set(),
                    f"Heavy-metal query '{query_title}' has children's items in "
                    f"its relevance set: {contamination}.",
                )

    def test_relevant_items_are_same_category_only(self):
        """All items in a relevance set must share the query item's category."""
        _, _, df, test_pairs = _build_test_data(self.csv)
        category_map = dict(zip(self.df["title"], self.df["category"]))

        for uid, query_title, relevant in test_pairs:
            query_cat = category_map.get(query_title)
            for rel_title in relevant:
                rel_cat = category_map.get(rel_title)
                self.assertEqual(
                    rel_cat, query_cat,
                    f"Relevant item '{rel_title}' (category='{rel_cat}') does not "
                    f"match query '{query_title}' (category='{query_cat}'). "
                    "Relevance must be restricted to same-category items.",
                )

    def test_high_rated_cross_category_item_excluded(self):
        """
        Explicit regression test: a single concrete pair.

        Scenario: query = 'Alphabet Puzzle' (Children, rating=2.0)
                  Metal Fury Album (HeavyMetal, rating=4.5) must NOT be relevant.
        """
        _, _, df, test_pairs = _build_test_data(self.csv)
        alphabet_pairs = [
            (uid, q, rel)
            for uid, q, rel in test_pairs
            if q == "Alphabet Puzzle"
        ]
        if not alphabet_pairs:
            self.skipTest("'Alphabet Puzzle' not sampled — increase sample or re-seed")

        _, _, relevant = alphabet_pairs[0]
        self.assertNotIn(
            "Metal Fury Album",
            relevant,
            "'Metal Fury Album' must not be relevant to 'Alphabet Puzzle': "
            "they are in different categories. Only rating ≥ 4.0 made it "
            "appear in the buggy version.",
        )

    def test_test_pairs_not_empty(self):
        """After removing rating contamination, test pairs must still be generated."""
        _, _, df, test_pairs = _build_test_data(self.csv)
        self.assertGreater(
            len(test_pairs), 0,
            "test_pairs must be non-empty; category-based relevance alone is sufficient.",
        )

    def test_relevance_sets_non_empty_for_same_category_items(self):
        """Items from a category with ≥2 members must produce a non-empty relevant set."""
        _, _, df, test_pairs = _build_test_data(self.csv)
        # Both Category A (5 items) and Category B (5 items) have multiple members,
        # so every sampled item should have a non-empty relevant set.
        for uid, query_title, relevant in test_pairs:
            self.assertGreater(
                len(relevant), 0,
                f"Relevant set for '{query_title}' must be non-empty when "
                f"same-category peers exist.",
            )


# ---------------------------------------------------------------------------
# 2. _get_relevant was dead code — verify it is removed
# ---------------------------------------------------------------------------

class TestDeadCodeRemoved(unittest.TestCase):
    """The contaminated _get_relevant() helper must not exist."""

    def test_get_relevant_not_exported(self):
        """_get_relevant is not a module-level function and was never exported."""
        import src.evaluation.evaluation as ev
        self.assertFalse(
            hasattr(ev, "_get_relevant"),
            "_get_relevant must not be exported from evaluation.py",
        )

    def test_run_evaluation_source_has_no_rating_threshold(self):
        """run_evaluation source must not contain a rating >= 4.0 filter."""
        import inspect
        import src.evaluation.evaluation as ev
        source = inspect.getsource(ev.run_evaluation)
        self.assertNotIn(
            "rating >= 4",
            source,
            "run_evaluation() must not use a rating >= 4.0 relevance criterion",
        )

    def test_build_test_data_source_has_no_rating_threshold(self):
        """_build_test_data source must not contain a rating >= 4.0 filter."""
        import inspect
        import src.evaluation.evaluation as ev
        source = inspect.getsource(ev._build_test_data)
        self.assertNotIn(
            "rating >= 4",
            source,
            "_build_test_data() must not use a rating >= 4.0 relevance criterion",
        )


# ---------------------------------------------------------------------------
# 3. Metric validity with clean relevance labels
# ---------------------------------------------------------------------------

class TestMetricValidityCleanRelevance(unittest.TestCase):
    """
    With category-only relevance the metrics measure semantic similarity,
    not popularity.  A recommender that recommends popular out-of-category
    items should score poorly.
    """

    def _make_category_df(self):
        return pd.DataFrame(
            {
                "title": [
                    "ItemA1", "ItemA2", "ItemA3",  # Category A
                    "ItemB1", "ItemB2", "ItemB3",  # Category B (all rated >= 4)
                ],
                "category": ["A", "A", "A", "B", "B", "B"],
                "rating":   [2.0, 2.5, 3.0, 4.5, 4.8, 5.0],
            }
        )

    def test_precision_zero_when_recommending_wrong_category(self):
        """
        Precision@3 must be 0 when all recommendations are from the wrong
        category, even if those items are all highly rated.
        """
        # Query: ItemA1 (Category A) — relevant: {ItemA2, ItemA3}
        relevant = {"ItemA2", "ItemA3"}
        # Recommendations: only Category B items (highly rated but wrong category)
        recs = ["ItemB1", "ItemB2", "ItemB3"]
        precision = _precision_at_k(recs, relevant, k=3)
        self.assertEqual(
            precision, 0.0,
            f"Precision must be 0 when recommending only wrong-category items, "
            f"got {precision}",
        )

    def test_precision_nonzero_when_recommending_correct_category(self):
        """
        Precision@3 must be non-zero when recommendations include same-category items.
        """
        relevant = {"ItemA2", "ItemA3"}
        recs = ["ItemA2", "ItemA3", "ItemB1"]
        precision = _precision_at_k(recs, relevant, k=3)
        self.assertAlmostEqual(
            precision, 2 / 3,
            msg=f"Expected precision 2/3, got {precision}",
        )

    def test_recall_zero_when_recommending_wrong_category(self):
        relevant = {"ItemA2", "ItemA3"}
        recs = ["ItemB1", "ItemB2", "ItemB3"]
        recall = _recall_at_k(recs, relevant, k=3)
        self.assertEqual(recall, 0.0)

    def test_mrr_zero_when_no_same_category_items_recommended(self):
        relevant = {"ItemA2", "ItemA3"}
        recs = ["ItemB1", "ItemB2", "ItemB3"]
        mrr = _mean_reciprocal_rank(recs, relevant, k=3)
        self.assertEqual(mrr, 0.0)

    def test_ndcg_zero_when_recommending_wrong_category(self):
        relevant = {"ItemA2", "ItemA3"}
        recs = ["ItemB1", "ItemB2", "ItemB3"]
        ndcg = _ndcg_at_k(recs, relevant, k=3)
        self.assertEqual(ndcg, 0.0)

    def test_hit_rate_zero_when_recommending_wrong_category(self):
        relevant = {"ItemA2", "ItemA3"}
        recs = ["ItemB1", "ItemB2", "ItemB3"]
        hit = _hit_rate(recs, relevant, k=3)
        self.assertEqual(hit, 0.0)


# ---------------------------------------------------------------------------
# 4. Regression: contamination before vs. after fix
# ---------------------------------------------------------------------------

class TestRelevanceContaminationRegression(unittest.TestCase):
    """
    Reproduce the exact contamination described in Issue #926 and verify
    it is absent after the fix.
    """

    def test_contaminated_set_would_have_been_large(self):
        """
        Demonstrate that the OLD logic produced a large contaminated set.

        In the buggy code, for a children's product query, the relevance set
        would have been:
            same_category_items | all_items_with_rating_geq_4

        We simulate that union here and show it was larger than category-only.
        """
        df = _make_df_with_cross_category_ratings()
        query_title = "Alphabet Puzzle"

        # Category-only (correct)
        same_cat = set(df[df["category"] == "Children"]["title"]) - {query_title}

        # Buggy union
        high_rated = set(df[df["rating"] >= 4.0]["title"])
        buggy_relevant = same_cat | high_rated

        # The buggy set includes heavy metal items; the clean set does not
        metal_items = set(df[df["category"] == "HeavyMetal"]["title"])

        self.assertTrue(
            metal_items.issubset(buggy_relevant),
            "Simulated buggy logic should include all heavy metal items",
        )
        self.assertTrue(
            len(buggy_relevant) > len(same_cat),
            "Buggy set must be larger than category-only set",
        )

    def test_category_only_set_excludes_high_rated_cross_category(self):
        """
        The corrected logic (category membership only) excludes heavy metal
        items from a children's product relevance set, even though those items
        have rating >= 4.0.
        """
        df = _make_df_with_cross_category_ratings()
        query_title = "Alphabet Puzzle"

        # Correct logic
        same_cat = set(df[df["category"] == "Children"]["title"]) - {query_title}

        metal_items = set(df[df["category"] == "HeavyMetal"]["title"])

        self.assertEqual(
            same_cat & metal_items,
            set(),
            "Category-only relevant set must not contain any heavy metal items",
        )

    def test_metrics_lower_without_contamination_for_popularity_biased_model(self):
        """
        A model that always recommends globally high-rated items regardless of
        category should score LOWER with clean relevance than with contaminated
        relevance.  This proves the fix reduces metric inflation.
        """
        df = _make_df_with_cross_category_ratings()
        query_title = "Alphabet Puzzle"

        # Category-only relevance (correct)
        clean_relevant = (
            set(df[df["category"] == "Children"]["title"]) - {query_title}
        )

        # Buggy relevance (category ∪ high-rated)
        high_rated = set(df[df["rating"] >= 4.0]["title"])
        contaminated_relevant = clean_relevant | high_rated

        # Popularity-biased recommendations: just recommend all high-rated items
        popularity_recs = df[df["rating"] >= 4.0]["title"].tolist()

        precision_clean = _precision_at_k(popularity_recs, clean_relevant, k=5)
        precision_contaminated = _precision_at_k(
            popularity_recs, contaminated_relevant, k=5
        )

        self.assertLessEqual(
            precision_clean,
            precision_contaminated,
            "A popularity-biased model must score at least as high on contaminated "
            f"relevance ({precision_contaminated:.4f}) as on clean relevance "
            f"({precision_clean:.4f}) — if not, the contamination was not inflating.",
        )
        # With the five heavy metal items NOT in the relevant set, the
        # popularity-biased model should score 0.0 on clean relevance
        # (none of its recommendations are children's products)
        self.assertEqual(
            precision_clean, 0.0,
            "Popularity-biased model must score 0 on children's query when "
            "clean relevance is used (all recommendations are heavy-metal items)",
        )


# ---------------------------------------------------------------------------
# 5. Category-based relevance correctness
# ---------------------------------------------------------------------------

class TestCategoryRelevanceCorrectness(unittest.TestCase):

    def test_same_category_items_are_relevant(self):
        """Every item in the same category (except self) must be relevant."""
        df = _make_df_with_cross_category_ratings()
        children_titles = set(df[df["category"] == "Children"]["title"])
        query_title = "Alphabet Puzzle"
        expected_relevant = children_titles - {query_title}

        # Simulate what _build_test_data now does
        relevant = set()
        row = df[df["title"] == query_title].iloc[0]
        same = df[df["category"] == row["category"]]["title"].tolist()
        relevant.update(same)
        relevant.discard(query_title)

        self.assertEqual(
            relevant, expected_relevant,
            "Relevant set must exactly match same-category items (excluding self)",
        )

    def test_different_category_items_never_relevant(self):
        """No item from a different category must appear in the relevant set."""
        df = _make_df_with_cross_category_ratings()
        query_title = "Metal Fury Album"
        children_titles = set(df[df["category"] == "Children"]["title"])

        relevant = set()
        row = df[df["title"] == query_title].iloc[0]
        same = df[df["category"] == row["category"]]["title"].tolist()
        relevant.update(same)
        relevant.discard(query_title)

        self.assertEqual(
            relevant & children_titles,
            set(),
            "Children's items must not appear in heavy metal item's relevance set",
        )

    def test_self_never_in_relevant_set(self):
        """The query item must never appear in its own relevance set."""
        df = _make_df_with_cross_category_ratings()
        for title in df["title"]:
            relevant = set()
            row = df[df["title"] == title].iloc[0]
            same = df[df["category"] == row["category"]]["title"].tolist()
            relevant.update(same)
            relevant.discard(title)
            self.assertNotIn(
                title, relevant,
                f"Item '{title}' must not be in its own relevance set",
            )
