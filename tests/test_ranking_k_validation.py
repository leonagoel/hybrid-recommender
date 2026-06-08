"""
Test k=0 validation in calculate_top_k_overlap.
Addresses GitHub issue #1441.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation.ranking_stability import calculate_top_k_overlap


class TestTopKOverlapKValidation:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            calculate_top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            calculate_top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=-1)

    def test_k_positive_works(self):
        result = calculate_top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=2)
        assert result == 1.0
