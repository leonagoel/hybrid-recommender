"""
Unit tests specifically for Fairness-Aware Re-ranking in HybridRecommender.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.model.hybrid_model import HybridRecommender


class TestFairnessAwareSpec:

    @pytest.fixture
    def mock_recommender(self):
        # Create a mock recommender with minimal properties needed for _fair_rerank
        content_model = MagicMock()
        model = HybridRecommender(content_model=content_model, collab_model=None, item_df=None)
        return model

    def test_fairness_get_set_defaults(self, mock_recommender):
        # Default states initialized in constructor
        assert mock_recommender.fairness_enabled is False
        assert mock_recommender.fairness_key == 'category'
        assert mock_recommender.fairness_max_share == 1.0

        # Verify get_fairness returns expected dict
        d = mock_recommender.get_fairness()
        assert d['enabled'] is False
        assert d['key'] == 'category'
        assert d['max_share'] == 1.0

    def test_fairness_set_fairness(self, mock_recommender):
        mock_recommender.set_fairness(enabled=True, key='catalog', max_share=0.25)
        assert mock_recommender.fairness_enabled is True
        assert mock_recommender.fairness_key == 'catalog'
        assert mock_recommender.fairness_max_share == 0.25

        d = mock_recommender.get_fairness()
        assert d['enabled'] is True
        assert d['key'] == 'catalog'
        assert d['max_share'] == 0.25

    def test_fair_rerank_empty(self, mock_recommender):
        # Empty results should return empty list gracefully
        res = mock_recommender._fair_rerank([], top_n=5, key='category', max_share=0.5)
        assert res == []

    def test_fair_rerank_single_item(self, mock_recommender):
        results = [{'title': 'A', 'category': 'Tech'}]
        res = mock_recommender._fair_rerank(results, top_n=1, key='category', max_share=0.5)
        assert res == results

    def test_fair_rerank_invalid_max_share(self, mock_recommender):
        results = [
            {'title': 'A', 'category': 'Tech'},
            {'title': 'B', 'category': 'Tech'}
        ]
        # max_share <= 0 or > 1 or invalid type fallback to 1.0
        res = mock_recommender._fair_rerank(results, top_n=2, key='category', max_share=-0.5)
        assert len(res) == 2

        res2 = mock_recommender._fair_rerank(results, top_n=2, key='category', max_share='invalid-type')
        assert len(res2) == 2

    def test_fair_rerank_reallocates_overflow(self, mock_recommender):
        # 4 candidate recommendations from dominant category 'Tech', 1 from 'Art'
        # Sorted originally by score
        results = [
            {'title': 'Tech 1', 'category': 'Tech', 'score': 0.9},
            {'title': 'Tech 2', 'category': 'Tech', 'score': 0.8},
            {'title': 'Tech 3', 'category': 'Tech', 'score': 0.7},
            {'title': 'Art 1', 'category': 'Art', 'score': 0.6},
            {'title': 'Tech 4', 'category': 'Tech', 'score': 0.5},
        ]
        # top_n = 4, max_share = 0.5 -> max items per category = max(1, 2) = 2
        # Expected selection order:
        # 1. Tech 1 (first Tech - count=1)
        # 2. Tech 2 (second Tech - count=2)
        # 3. Tech 3 (third Tech - overflow, skipped initially)
        # 4. Art 1 (first Art - count=1)
        # At this point selected list has 3 items: [Tech 1, Tech 2, Art 1]
        # We need 4 items. Selected is filled with overflow: Tech 3.
        # Expected final list of size 4: [Tech 1, Tech 2, Art 1, Tech 3]
        res = mock_recommender._fair_rerank(results, top_n=4, key='category', max_share=0.5)
        assert len(res) == 4
        assert res[0]['title'] == 'Tech 1'
        assert res[1]['title'] == 'Tech 2'
        assert res[2]['title'] == 'Art 1'
        assert res[3]['title'] == 'Tech 3'

    def test_fair_rerank_missing_category_keys(self, mock_recommender):
        # Items missing the category key or key is None
        results = [
            {'title': 'A'},
            {'title': 'B', 'category': None},
            {'title': 'C', 'category': 'Tech'}
        ]
        # Should execute successfully without throwing KeyError
        res = mock_recommender._fair_rerank(results, top_n=3, key='category', max_share=0.5)
        assert len(res) == 3
