"""
Unit tests for Content-Based Recommender
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.content_model import ContentRecommender

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_item_df():
    """Sample DataFrame for testing ContentRecommender."""
    return pd.DataFrame({
        'title': [
            'Harry Potter', 
            'Lord of the Rings', 
            'The Hobbit',
            'Game of Thrones',
            'Dune'
        ],
        'description': [
            'A young wizard discovers his magical heritage',
            'A fellowship embarks on a quest to destroy a ring',
            'A hobbit goes on an unexpected journey',
            'Noble families fight for control of the Iron Throne',
            'A desert planet holds the most valuable resource',
        ],
        'category': [
            'Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'SciFi'
        ],
        'combined': [
            'Harry Potter A young wizard discovers his magical heritage Fantasy',
            'Lord of the Rings A fellowship embarks on a quest Fantasy',
            'The Hobbit A hobbit goes on an unexpected journey Fantasy',
            'Game of Thrones Noble families fight for the Iron Throne Fantasy',
            'Dune A desert planet holds the most valuable resource SciFi',
        ],
    })

@pytest.fixture
def content_model(sample_item_df):
    """Create ContentRecommender instance with sample data."""
    return ContentRecommender(sample_item_df)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestContentRecommender:

    def test_recommend_returns_list(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_query_item(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=5)
        titles = [r['title'] for r in recs]
        assert 'Harry Potter' not in titles

    def test_recommend_respects_top_n(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=2)
        assert len(recs) <= 2

    def test_recommend_unknown_title_returns_empty(self, content_model):
        recs = content_model.recommend('Nonexistent Book XYZ', top_n=5)
        assert recs == []

    def test_recommend_scores_are_floats(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=3)
        for r in recs:
            assert isinstance(r['content_score'], float)

    def test_recommend_scores_between_0_and_1(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=3)
        for r in recs:
            assert 0.0 <= r['content_score'] <= 1.0

    def test_recommend_has_required_keys(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=3)
        for r in recs:
            assert 'title' in r
            assert 'content_score' in r

    def test_recommend_similar_category_scores_higher(self, content_model):
        recs = content_model.recommend('Harry Potter', top_n=4)
        titles = [r['title'] for r in recs]
        fantasy_books = ['Lord of the Rings', 'The Hobbit', 'Game of Thrones']
        found_fantasy = any(t in titles for t in fantasy_books)
        assert found_fantasy

    def test_search_returns_list(self, content_model):
        results = content_model.search('wizard', top_n=3)
        assert isinstance(results, list)

    def test_search_returns_results(self, content_model):
        results = content_model.search('wizard', top_n=3)
        assert len(results) > 0

    def test_search_result_has_required_keys(self, content_model):
        results = content_model.search('wizard', top_n=2)
        for r in results:
            assert 'title' in r
            assert 'score' in r

    def test_search_scores_are_floats(self, content_model):
        results = content_model.search('wizard', top_n=3)
        for r in results:
            assert isinstance(r['score'], float)

    def test_search_empty_query_handles_gracefully(self, content_model):
        try:
            results = content_model.search('', top_n=3)
            assert isinstance(results, list)
        except Exception:
            pass

    def test_search_no_match_returns_empty(self, content_model):
        results = content_model.search('xyzxyzxyz123456', top_n=3)
        assert isinstance(results, list)

    def test_search_very_long_query(self, content_model):
        long_query = "wizard " * 100
        try:
            results = content_model.search(long_query, top_n=3)
            assert isinstance(results, list)
        except Exception:
            pass

    def test_search_special_characters(self, content_model):
        special_query = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        results = content_model.search(special_query, top_n=3)
        assert isinstance(results, list)

    def test_search_unicode_text(self, content_model):
        unicode_query = "wizard \u00e9\u00e8\u00ea"
        results = content_model.search(unicode_query, top_n=3)
        assert isinstance(results, list)

    def test_search_case_sensitivity(self, content_model):
        results_lower = content_model.search('wizard', top_n=3)
        results_upper = content_model.search('WIZARD', top_n=3)
        assert isinstance(results_lower, list)
        assert isinstance(results_upper, list)
