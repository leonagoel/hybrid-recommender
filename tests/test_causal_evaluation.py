"""
Unit tests for Causal Evaluation Metrics.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.causal_evaluation import (
    _build_popularity_rank,
    _build_category_map,
    _avg_popularity_rank,
    _intra_list_diversity,
    compare_causal_vs_baseline,
    score_key_distribution,
)


class MockModel:
    def __init__(self, recs):
        self.recs = recs

    def recommend(self, query, top_n=10):
        return self.recs.get(query, [])[:top_n]


def test_build_popularity_rank_empty():
    df = pd.DataFrame(columns=["title", "review_count"])
    res = _build_popularity_rank(df)
    assert res == {}


def test_build_popularity_rank_missing_column():
    df = pd.DataFrame({"title": ["A", "B"]})
    res = _build_popularity_rank(df)
    assert res == {"A": 0.5, "B": 0.5}


def test_build_popularity_rank_normal():
    df = pd.DataFrame({"title": ["A", "B", "C"], "review_count": [10, 0, 50]})
    res = _build_popularity_rank(df)
    assert res["A"] == 0.2
    assert res["B"] == 0.0
    assert res["C"] == 1.0


def test_build_category_map_missing():
    df = pd.DataFrame({"title": ["A"]})
    res = _build_category_map(df)
    assert res == {}


def test_build_category_map_normal():
    df = pd.DataFrame({"title": ["A", "B"], "category": ["SciFi", "Fantasy"]})
    res = _build_category_map(df)
    assert res == {"A": "SciFi", "B": "Fantasy"}


def test_avg_popularity_rank():
    pop_rank = {"A": 0.2, "B": 0.8}
    assert _avg_popularity_rank([], pop_rank) == 0.0
    # C is missing, should get 0.5
    assert np.allclose(_avg_popularity_rank(["A", "C"], pop_rank), 0.35)


def test_intra_list_diversity():
    cat_map = {"A": "SciFi", "B": "SciFi", "C": "Fantasy"}
    # Less than or equal to 1 item
    assert _intra_list_diversity([], cat_map) == 0.0
    assert _intra_list_diversity(["A"], cat_map) == 0.0
    # Same categories: dominant is 100%
    assert _intra_list_diversity(["A", "B"], cat_map) == 0.0
    # Mixed categories: dominant (SciFi) count = 2 out of 3 -> diversity = 1 - 2/3 = 1/3
    assert np.allclose(_intra_list_diversity(["A", "B", "C"], cat_map), 1.0 / 3.0)


def test_compare_causal_vs_baseline():
    item_df = pd.DataFrame({
        "title": ["A", "B", "C", "D"],
        "review_count": [100, 50, 10, 5],
        "category": ["SciFi", "SciFi", "Fantasy", "Fantasy"]
    })
    
    # Causal model recommends niche (less popular) items
    causal_recs = {
        "query1": [{"title": "C", "hybrid_score": 0.9}, {"title": "D", "hybrid_score": 0.8}]
    }
    # Baseline model recommends popular items
    baseline_recs = {
        "query1": [{"title": "A", "hybrid_score": 0.95}, {"title": "B", "hybrid_score": 0.85}]
    }
    
    causal_model = MockModel(causal_recs)
    baseline_model = MockModel(baseline_recs)
    
    results = compare_causal_vs_baseline(
        causal_model=causal_model,
        baseline_model=baseline_model,
        item_df=item_df,
        query_titles=["query1"],
        top_n=2
    )
    
    assert results["n_queries"] == 1
    # Niche items have lower rank, so popularity rank is lower, meaning bias reduction is positive
    assert results["popularity_bias_reduction"] > 0
    assert "causal_coverage" in results
    assert "baseline_coverage" in results


def test_score_key_distribution():
    recs = {
        "q": [{"title": "A", "val": 0.5}, {"title": "B", "val": 1.0}]
    }
    model = MockModel(recs)
    item_df = pd.DataFrame({"title": ["A", "B"]})
    res = score_key_distribution(model, item_df, ["q"], top_n=2, score_key="val")
    assert res["mean"] == 0.75
    assert res["min"] == 0.5
    assert res["max"] == 1.0
    assert res["n_scores"] == 2
