"""
Unit tests for TrendingRecommender in src/model/trending_model.py.
Run with: pytest tests/test_trending_model.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.trending_model import TrendingRecommender


class TestTrendingRecommenderDefaults:
    """Test default top_n behavior."""

    def test_default_top_n_is_10(self):
        df = pd.DataFrame({
            "item_id": [f"item_{i}" for i in range(15)],
            "title": [f"Product {i}" for i in range(15)],
            "rating": [4.0] * 15,
            "views": [100] * 15,
            "purchases": [50] * 15,
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products()
        assert len(results) == 10

    def test_custom_top_n_5(self):
        df = pd.DataFrame({
            "item_id": [f"item_{i}" for i in range(15)],
            "title": [f"Product {i}" for i in range(15)],
            "rating": [4.0] * 15,
            "views": [100] * 15,
            "purchases": [50] * 15,
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products(top_n=5)
        assert len(results) == 5

    def test_custom_top_n_20_exceeds_available(self):
        df = pd.DataFrame({
            "item_id": [f"item_{i}" for i in range(12)],
            "title": [f"Product {i}" for i in range(12)],
            "rating": [4.0] * 12,
            "views": [100] * 12,
            "purchases": [50] * 12,
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products(top_n=20)
        assert len(results) == 12


class TestTrendingRecommenderMissingColumns:
    """Test behavior when DataFrame is missing views/purchases/rating columns."""

    def test_missing_views_column_uses_zero(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["Product A", "Product B"],
            "rating": [5.0, 3.0],
            "purchases": [100, 50],
        })
        rec = TrendingRecommender(df=df)
        # Should not raise — uses 0 for missing views
        results = rec.get_trending_products(top_n=2)
        assert len(results) == 2
        # Product A has higher rating, should rank first
        assert results[0]["title"] == "Product A"

    def test_missing_purchases_column_uses_zero(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["Product A", "Product B"],
            "rating": [5.0, 3.0],
            "views": [100, 50],
        })
        rec = TrendingRecommender(df=df)
        # Should not raise
        results = rec.get_trending_products(top_n=2)
        assert len(results) == 2

    def test_missing_rating_column(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["Product A", "Product B"],
            "views": [100, 50],
            "purchases": [100, 50],
        })
        rec = TrendingRecommender(df=df)
        # Should not raise
        results = rec.get_trending_products(top_n=2)
        assert len(results) == 2


class TestTrendingRecommenderEdgeCases:
    """Test edge case DataFrames."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["item_id", "title", "rating", "views", "purchases"])
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products()
        assert results == []

    def test_single_row_dataframe(self):
        df = pd.DataFrame({
            "item_id": ["i1"],
            "title": ["Only Product"],
            "rating": [4.5],
            "views": [100],
            "purchases": [50],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products()
        assert len(results) == 1
        assert results[0]["title"] == "Only Product"

    def test_nan_values_in_rating(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2", "i3"],
            "title": ["Product A", "Product B", "Product C"],
            "rating": [np.nan, 4.0, 3.0],
            "views": [100, 50, 75],
            "purchases": [50, 25, 40],
        })
        rec = TrendingRecommender(df=df)
        # Should not raise — NaN rating handled
        results = rec.get_trending_products(top_n=3)
        assert len(results) == 3
        # Product A with NaN rating should still appear (views+purchases contribute)
        titles = [r["title"] for r in results]
        assert "Product A" in titles

    def test_nan_values_in_views(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["Product A", "Product B"],
            "rating": [4.0, 4.0],
            "views": [np.nan, 50],
            "purchases": [50, 50],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products(top_n=2)
        assert len(results) == 2

    def test_all_nan_views_and_purchases(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["Product A", "Product B"],
            "rating": [5.0, 3.0],
            "views": [np.nan, np.nan],
            "purchases": [np.nan, np.nan],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products(top_n=2)
        assert len(results) == 2


class TestTrendingScoreCalculation:
    """Test trending_score formula: 0.5*purchases + 0.3*views + 0.2*avg_rating."""

    def test_score_calculation_correct(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["A", "B"],
            "rating": [5.0, 5.0],
            "views": [0, 100],
            "purchases": [100, 0],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products(top_n=2)
        # Both have same avg_rating=5.0
        # A: 0.5*100 + 0.3*0 + 0.2*5 = 50 + 0 + 1 = 51
        # B: 0.5*0 + 0.3*100 + 0.2*5 = 0 + 30 + 1 = 31
        # A should rank first
        assert results[0]["title"] == "A"
        assert results[1]["title"] == "B"

    def test_purchases_weight_is_dominant(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["A", "B"],
            "rating": [5.0, 5.0],
            "views": [0, 0],
            "purchases": [100, 50],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products()
        # A has more purchases
        assert results[0]["title"] == "A"
        assert results[1]["title"] == "B"

    def test_views_weight_matters(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "title": ["A", "B"],
            "rating": [5.0, 5.0],
            "views": [100, 50],
            "purchases": [0, 0],
        })
        rec = TrendingRecommender(df=df)
        results = rec.get_trending_products()
        # A has more views
        assert results[0]["title"] == "A"


class TestTrendingRecommenderInvalidTopN:
    """Test validation for invalid top_n values."""

    def test_top_n_zero_raises(self):
        df = pd.DataFrame({
            "item_id": ["i1"],
            "title": ["Product"],
            "rating": [4.0],
            "views": [10],
            "purchases": [5],
        })
        rec = TrendingRecommender(df=df)
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            rec.get_trending_products(top_n=0)

    def test_top_n_negative_raises(self):
        df = pd.DataFrame({
            "item_id": ["i1"],
            "title": ["Product"],
            "rating": [4.0],
            "views": [10],
            "purchases": [5],
        })
        rec = TrendingRecommender(df=df)
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            rec.get_trending_products(top_n=-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])