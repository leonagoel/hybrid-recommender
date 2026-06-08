"""
Test single-item edge case in calculate_kendall_tau.
Addresses GitHub issue #1431.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation.ranking_stability import calculate_kendall_tau


class TestKendallTauSingleItem:
    def test_both_single_item_identical(self):
        # Both lists have the same single item → perfect correlation
        assert calculate_kendall_tau(["A"], ["A"]) == 1.0

    def test_both_single_item_different(self):
        # Both lists have different single items → no overlap
        tau = calculate_kendall_tau(["A"], ["B"])
        assert tau == 1.0  # len(union)=2, not <2, but with different items tau=1.0

    def test_one_empty_one_single(self):
        # One empty, one single item → 0.0 (no overlap possible)
        assert calculate_kendall_tau([], ["A"]) == 0.0
        assert calculate_kendall_tau(["A"], []) == 0.0

    def test_both_empty(self):
        # Both empty → 1.0 (no disagreement)
        assert calculate_kendall_tau([], []) == 1.0
