"""
test_ranking_stability.py — Automated Ranking Regression Tests
==============================================================
Validates that recommendation rankings remain stable across changes.
Integrates with pytest and supports configurable thresholds.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from src.evaluation.ranking_stability import (
    calculate_top_k_overlap,
    calculate_kendall_tau,
    calculate_position_changes,
    compare_rankings,
    build_recommender,
    FIXED_QUERIES
)

# ---------------------------------------------------------------------------
# Unit tests for comparison metrics
# ---------------------------------------------------------------------------

def test_top_k_overlap_perfect_match():
    list1 = ["item1", "item2", "item3"]
    list2 = ["item1", "item2", "item3"]
    assert calculate_top_k_overlap(list1, list2, k=3) == 1.0

def test_top_k_overlap_partial_match():
    list1 = ["item1", "item2", "item3"]
    list2 = ["item1", "item4", "item3"]
    # 2 items overlapping ("item1", "item3") out of k=3
    assert calculate_top_k_overlap(list1, list2, k=3) == pytest.approx(2/3)

def test_top_k_overlap_no_match():
    list1 = ["item1", "item2"]
    list2 = ["item3", "item4"]
    assert calculate_top_k_overlap(list1, list2, k=2) == 0.0

def test_kendall_tau_perfect_correlation():
    list1 = ["a", "b", "c"]
    list2 = ["a", "b", "c"]
    # Perfect alignment
    assert calculate_kendall_tau(list1, list2) == pytest.approx(1.0)

def test_kendall_tau_inverse_correlation():
    list1 = ["a", "b", "c"]
    list2 = ["c", "b", "a"]
    # Exactly reversed ranking
    assert calculate_kendall_tau(list1, list2) == pytest.approx(-1.0)

def test_kendall_tau_empty_input():
    assert calculate_kendall_tau([], []) == 1.0
    assert calculate_kendall_tau(["a"], []) == 0.0

def test_position_changes():
    list1 = ["a", "b", "c"]
    list2 = ["b", "a", "c"]
    res = calculate_position_changes(list1, list2)
    # "a" moved by 1 (0 -> 1)
    # "b" moved by 1 (1 -> 0)
    # "c" moved by 0 (2 -> 2)
    assert res["max_change"] == 1.0
    assert res["avg_change"] == pytest.approx(2/3)

# ---------------------------------------------------------------------------
# Regression Test using Golden Reference Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recommender():
    """Build recommender model on the sample dataset."""
    data_path = Path(__file__).resolve().parent.parent / "datasets" / "sample_products.csv"
    assert data_path.exists(), f"Sample dataset not found at {data_path}"
    return build_recommender(str(data_path))

@pytest.fixture(scope="module")
def golden_fixture():
    """Load the golden reference rankings from tests/fixtures."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "ranking_golden.json"
    assert fixture_path.exists(), f"Golden ranking fixture not found at {fixture_path}. Run ranking_stability.py --generate-golden"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_ranking_regression_against_golden(recommender, golden_fixture):
    """
    Generate recommendations for the fixed queries and verify they do not
    drift significantly from the golden reference rankings.
    """
    # Read configurable thresholds from environment variables with default fallback
    threshold_overlap = float(os.getenv("RANKING_MIN_OVERLAP", "0.8"))
    threshold_tau = float(os.getenv("RANKING_MIN_KENDALL_TAU", "0.7"))
    
    k = golden_fixture.get("k", 10)
    golden_recs = golden_fixture.get("recommendations", {})
    
    failures = []
    
    print(f"\nEvaluating ranking stability against golden references (K={k})...")
    print(f"Thresholds: min_overlap={threshold_overlap}, min_kendall_tau={threshold_tau}")
    
    for query in FIXED_QUERIES:
        if query not in golden_recs:
            print(f"⚠️ Query '{query}' has no golden reference; skipping.")
            continue
            
        old_list = golden_recs[query]
        
        # Get new recommendations
        new_recs = recommender.recommend(query, top_n=k)
        new_list = [r["title"] for r in new_recs]
        
        res = compare_rankings(
            old_list, 
            new_list, 
            k=k, 
            threshold_overlap=threshold_overlap, 
            threshold_tau=threshold_tau
        )
        
        print(f"Query: {query}")
        print(f" - Top-K Overlap: {res['overlap_ratio']:.4f} (passed: {res['overlap_ratio'] >= threshold_overlap})")
        print(f" - Kendall Tau: {res['kendall_tau']:.4f} (passed: {res['kendall_tau'] >= threshold_tau})")
        print(f" - Avg Pos Change: {res['avg_position_change']:.2f}")
        
        if not res["passed"]:
            failures.append((query, res))
            
    assert len(failures) == 0, f"Drift threshold exceeded for {len(failures)} query(ies): {failures}"
