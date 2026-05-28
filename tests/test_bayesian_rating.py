"""
Unit tests for the bayesian_rating function in hybrid_model module.
Tests rating bias prevention using Bayesian smoothing.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.hybrid_model import bayesian_rating


class TestBayesianRating:
    """Test bayesian_rating function."""

    def test_items_with_many_votes_retain_actual_rating(self):
        """Test that items with many votes retain most of their actual rating."""
        rating = 4.5
        review_count = 100
        result = bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10)
        assert 4.0 < result < 4.7

    def test_items_with_few_votes_pulled_toward_global_average(self):
        """Test that items with few votes are pulled toward global average."""
        rating = 5.0
        review_count = 1
        result = bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10)
        assert 3.0 < result < 5.0

    def test_items_with_zero_votes_get_global_average(self):
        """Test that items with zero votes get global average."""
        result = bayesian_rating(5.0, 0, global_avg=3.0, min_votes=10)
        assert result == 3.0

    def test_items_with_few_votes_near_threshold(self):
        """Test items with few votes near the threshold."""
        rating = 4.0
        review_count = 5
        result = bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10)
        assert 3.0 < result < 4.0

    def test_global_average_influence_with_min_votes(self):
        """Test that global average influence is correctly calculated."""
        rating = 2.0
        review_count = 5
        m = 10
        C = 3.0
        v = review_count
        expected = (v / (v + m)) * rating + (m / (v + m)) * C
        result = bayesian_rating(rating, review_count, global_avg=C, min_votes=m)
        assert abs(result - expected) < 0.001

    def test_extreme_rating_values(self):
        """Test with extreme rating values (1.0 and 5.0)."""
        result_low = bayesian_rating(1.0, 5, global_avg=3.0, min_votes=10)
        result_high = bayesian_rating(5.0, 5, global_avg=3.0, min_votes=10)
        assert 1.0 < result_low < 3.0
        assert 3.0 < result_high < 5.0

    def test_very_high_review_count(self):
        """Test with very high review count."""
        rating = 3.5
        review_count = 10000
        result = bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10)
        assert abs(result - 3.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])