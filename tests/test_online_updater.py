"""
Unit tests for OnlineUpdater.ingest in src/model/online_updater.py.
Run with: pytest tests/test_online_updater.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.online_updater import OnlineUpdater


class TestOnlineUpdaterIngest:
    """Test OnlineUpdater.ingest edge cases."""

    def test_none_user_id(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id=None, item_title="Product A")
        assert result is True

    def test_none_item_title(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id="u1", item_title=None)
        assert result is True

    def test_none_rating(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id="u1", item_title="Product A", rating=None)
        assert result is True

    def test_none_sentiment(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id="u1", item_title="Product A", sentiment=None)
        assert result is True

    def test_none_recommender_returns_true(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id="u1", item_title="Product A", rating=4.0, recommender=None)
        assert result is True

    def test_all_none_parameters(self):
        updater = OnlineUpdater()
        result = updater.ingest(user_id=None, item_title=None, rating=None, sentiment=None, recommender=None)
        assert result is True

    def test_buffer_append(self):
        updater = OnlineUpdater()
        updater.ingest(user_id="u1", item_title="Product A", rating=4.0)
        assert len(updater.buffer) == 1
        assert updater.buffer[0]["user_id"] == "u1"
        assert updater.buffer[0]["title"] == "Product A"
        assert updater.buffer[0]["rating"] == 4.0

    def test_with_valid_recommender_updates_maps(self):
        # Create a minimal mock recommender
        class MockRecommender:
            def __init__(self):
                self._review_count_map = {}
                self._popularity_map = {}
                self._rating_map = {}
                self._sentiment_map = {}

        updater = OnlineUpdater()
        recommender = MockRecommender()

        result = updater.ingest(
            user_id="u1",
            item_title="Product A",
            rating=4.5,
            sentiment=0.8,
            recommender=recommender,
        )

        assert result is True
        assert recommender._review_count_map["Product A"] == 1
        assert "Product A" in recommender._popularity_map

    def test_multiple_ingests_increment_review_count(self):
        class MockRecommender:
            def __init__(self):
                self._review_count_map = {}
                self._popularity_map = {}
                self._rating_map = {}
                self._sentiment_map = {}
                self.collab_model = None

        updater = OnlineUpdater()
        recommender = MockRecommender()

        updater.ingest(user_id="u1", item_title="Product A", rating=4.0, recommender=recommender)
        updater.ingest(user_id="u2", item_title="Product A", rating=5.0, recommender=recommender)

        assert recommender._review_count_map["Product A"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])