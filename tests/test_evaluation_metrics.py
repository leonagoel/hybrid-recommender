"""
Unit tests for evaluation metric functions.
Tests edge cases for precision, recall, DCG, and NDCG calculations.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.evaluation import (
    _precision_at_k,
    _recall_at_k,
    _dcg_at_k,
    _ndcg_at_k,
)


class TestPrecisionAtK:
    """Test _precision_at_k function edge cases."""

    def test_empty_relevant_set_returns_zero(self):
        """When relevant set is empty, precision is 0."""
        result = _precision_at_k(["a", "b", "c"], set(), 3)
        assert result == 0.0

    def test_zero_k_returns_zero(self):
        """When k is 0, precision is 0."""
        result = _precision_at_k(["a", "b", "c"], {"a"}, 0)
        assert result == 0.0

    def test_k_greater_than_recommended_length(self):
        """When k is greater than recommended length, precision uses k as denominator."""
        result = _precision_at_k(["a", "b"], {"a", "b", "c"}, 5)
        assert result == 0.4  # 2 hits / 5

    def test_all_relevant_in_top_k(self):
        """When all top-K items are relevant."""
        result = _precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
        assert result == 1.0

    def test_none_relevant_in_top_k(self):
        """When none of top-K items are relevant."""
        result = _precision_at_k(["a", "b", "c"], {"x", "y", "z"}, 3)
        assert result == 0.0


class TestRecallAtK:
    """Test _recall_at_k function edge cases."""

    def test_empty_relevant_set_returns_zero(self):
        """When relevant set is empty, recall is 0."""
        result = _recall_at_k(["a", "b", "c"], set(), 3)
        assert result == 0.0

    def test_zero_k_returns_zero(self):
        """When k is 0, recall is 0."""
        result = _recall_at_k(["a", "b", "c"], {"a"}, 0)
        assert result == 0.0

    def test_some_relevant_found(self):
        """When some relevant items are found in top-K."""
        result = _recall_at_k(["a", "b"], {"a", "b", "c"}, 2)
        # both a and b are in relevant, so 2 hits / 3 relevant = 0.666
        assert abs(result - 0.666) < 0.01

    def test_all_relevant_found(self):
        """When all relevant items are found in top-K."""
        result = _recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
        assert result == 1.0

    def test_none_relevant_found(self):
        """When none of the relevant items are found."""
        result = _recall_at_k(["x", "y"], {"a", "b", "c"}, 2)
        assert result == 0.0


class TestDCGAtK:
    """Test _dcg_at_k function edge cases."""

    def test_empty_recommended_returns_zero(self):
        """When recommended list is empty, DCG is 0."""
        result = _dcg_at_k([], {"a", "b"}, 5)
        assert result == 0.0

    def test_empty_relevant_returns_zero(self):
        """When relevant set is empty, DCG is 0."""
        result = _dcg_at_k(["a", "b", "c"], set(), 3)
        assert result == 0.0

    def test_single_relevant_at_top(self):
        """When the single relevant item is at position 1."""
        result = _dcg_at_k(["a", "b", "c"], {"a"}, 3)
        assert result == 1.0  # 1 / log2(2) = 1

    def test_relevant_at_position_2(self):
        """When relevant item is at position 2."""
        result = _dcg_at_k(["x", "a", "b"], {"a"}, 3)
        expected = 1.0 / 1.58496  # approximately 0.63
        assert abs(result - expected) < 0.01


class TestNDCGAtK:
    """Test _ndcg_at_k function edge cases."""

    def test_empty_relevant_returns_zero(self):
        """When relevant set is empty, NDCG is 0."""
        result = _ndcg_at_k(["a", "b", "c"], set(), 3)
        assert result == 0.0

    def test_zero_k_returns_zero(self):
        """When k is 0, NDCG is 0."""
        result = _ndcg_at_k(["a", "b"], {"a"}, 0)
        assert result == 0.0

    def test_perfect_ranking_returns_one(self):
        """When recommended matches ideal ranking."""
        result = _ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
        assert result == 1.0

    def test_empty_recommended_returns_zero(self):
        """When recommended list is empty but relevant is not."""
        result = _ndcg_at_k([], {"a", "b", "c"}, 3)
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])