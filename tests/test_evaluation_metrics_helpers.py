"""
Unit tests for evaluation metric helper functions.
Tests _precision_at_k, _recall_at_k, _dcg_at_k, and _ndcg_at_k.
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
    """Test _precision_at_k function."""

    def test_standard_case(self):
        """Test with standard input."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert _precision_at_k(recommended, relevant, 3) == 1.0

    def test_partial_hits(self):
        """Test with some hits."""
        recommended = ["a", "x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert _precision_at_k(recommended, relevant, 3) == 1.0 / 3.0

    def test_empty_relevant_set(self):
        """Test with empty relevant set."""
        recommended = ["a", "b", "c"]
        relevant = set()
        assert _precision_at_k(recommended, relevant, 3) == 0.0

    def test_k_equals_zero(self):
        """Test with k=0."""
        recommended = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert _precision_at_k(recommended, relevant, 0) == 0.0

    def test_k_exceeds_recommended_length(self):
        """Test when k exceeds length of recommended list."""
        recommended = ["a", "b"]
        relevant = {"a", "b", "c"}
        result = _precision_at_k(recommended, relevant, 5)
        assert result == 0.4  # 2 hits / 5 = 0.4


class TestRecallAtK:
    """Test _recall_at_k function."""

    def test_standard_case(self):
        """Test with standard input."""
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = _recall_at_k(recommended, relevant, 3)
        assert abs(result - 1.0) < 0.001

    def test_partial_hits(self):
        """Test with 1 hit out of 3."""
        recommended = ["a"]
        relevant = {"a", "b", "c"}
        result = _recall_at_k(recommended, relevant, 1)
        assert abs(result - 1.0 / 3.0) < 0.001

    def test_empty_relevant_set(self):
        """Test with empty relevant set."""
        recommended = ["a", "b"]
        relevant = set()
        assert _recall_at_k(recommended, relevant, 2) == 0.0

    def test_k_equals_zero(self):
        """Test with k=0."""
        recommended = ["a", "b"]
        relevant = {"a"}
        assert _recall_at_k(recommended, relevant, 0) == 0.0


class TestDCGAtK:
    """Test _dcg_at_k function."""

    def test_all_relevant_at_top(self):
        """Test DCG when all relevant items are at top positions."""
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = _dcg_at_k(recommended, relevant, 3)
        assert result > 0

    def test_no_relevant_items(self):
        """Test DCG when no relevant items in top-k."""
        recommended = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert _dcg_at_k(recommended, relevant, 3) == 0.0

    def test_k_less_than_recommended(self):
        """Test DCG with k less than recommended length."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b"}
        result = _dcg_at_k(recommended, relevant, 3)
        assert result > 0


class TestNDCGAtK:
    """Test _ndcg_at_k function."""

    def test_perfect_retrieval(self):
        """Test NDCG when recommended matches relevant exactly."""
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = _ndcg_at_k(recommended, relevant, 3)
        assert abs(result - 1.0) < 0.001

    def test_no_relevant_items(self):
        """Test NDCG when no relevant items are retrieved."""
        recommended = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert _ndcg_at_k(recommended, relevant, 3) == 0.0

    def test_partial_retrieval(self):
        """Test NDCG with partial retrieval."""
        recommended = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        result = _ndcg_at_k(recommended, relevant, 3)
        assert 0 < result < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])