"""
Unit tests for TrendingRecommender in src/model/trending_model.py.
Run with: pytest tests/test_trending_model.py -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model.trending_model import TrendingRecommender


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(15)],
        "title": [f"Product {i}" for i in range(15)],
        "rating": [4.0] * 15,
        "views": [100] * 15,
        "purchases": [50] * 15,
    })


@pytest.fixture
def rec(sample_df):
    return TrendingRecommender(df=sample_df)


class TestTopNValidation:
    """Test top_n parameter validation in TrendingRecommender."""

    def test_invalid_top_n_zero_raises(self, rec):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            rec.get_trending_products(top_n=0)

    def test_invalid_top_n_negative_raises(self, rec):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            rec.get_trending_products(top_n=-5)

    def test_invalid_top_n_non_integer_raises(self, rec):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            rec.get_trending_products(top_n=2.5)

    def test_invalid_top_n_none_raises(self, rec):
        with pytest.raises((ValueError, TypeError)):
            rec.get_trending_products(top_n=None)
