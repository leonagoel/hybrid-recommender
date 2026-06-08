"""
Unit tests for RecommendationHistory in src/model/recommendation_history.py.
Run with: pytest tests/test_recommendation_history.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.recommendation_history import RecommendationHistory


class TestRecommendationHistoryEmpty:
    """Test empty history behavior."""

    def test_empty_history_returns_empty_set(self):
        tracker = RecommendationHistory()
        result = tracker.get_recent_titles("unknown_user")
        assert result == set()

    def test_empty_history_get_history(self):
        tracker = RecommendationHistory()
        result = tracker.get_history("unknown_user")
        assert result == []


class TestRecommendationHistoryBasic:
    """Test basic add/get functionality."""

    def test_add_single_recommendation(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        result = tracker.get_history("u1")
        assert len(result) == 1
        assert result[0]["title"] == "Product A"

    def test_get_recent_titles_returns_set(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        tracker.add_recommendation("u1", "Product B")
        result = tracker.get_recent_titles("u1")
        assert isinstance(result, set)
        assert "Product A" in result
        assert "Product B" in result

    def test_duplicate_titles_deduplicated_in_set(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        tracker.add_recommendation("u1", "Product A")
        result = tracker.get_recent_titles("u1")
        assert len(result) == 1

    def test_add_recommendation_stores_timestamp(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        history = tracker.get_history("u1")
        assert len(history) == 1
        from datetime import datetime
        assert isinstance(history[0]["timestamp"], datetime)

    def test_history_returns_in_insertion_order(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        tracker.add_recommendation("u1", "Product B")
        tracker.add_recommendation("u1", "Product C")
        history = tracker.get_history("u1")
        titles = [h["title"] for h in history]
        assert titles == ["Product A", "Product B", "Product C"]

    def test_history_limits_to_last_50(self):
        tracker = RecommendationHistory()
        for i in range(60):
            tracker.add_recommendation("u1", f"Product {i}")
        history = tracker.get_history("u1")
        assert len(history) == 50
        # First item should be Product 10 (60-50=10)
        assert history[0]["title"] == "Product 10"
        # Last item should be Product 59
        assert history[-1]["title"] == "Product 59"

    def test_unknown_user_returns_empty_history(self):
        tracker = RecommendationHistory()
        tracker.add_recommendation("u1", "Product A")
        result = tracker.get_history("u2")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])