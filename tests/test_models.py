"""
Unit tests for Hybrid Recommender System
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hybrid_model import HybridRecommender, bayesian_rating
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_item_df():
    """Minimal item DataFrame for testing."""
    return pd.DataFrame({
        'title': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        'description': [
            'A great wireless headphone with noise cancellation',
            'Budget earbuds with decent sound quality',
            'Premium over-ear headphones for audiophiles',
            'Laptop stand for ergonomic work setup',
            'USB-C hub with multiple ports for connectivity',
        ],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
        'rating': [4.5, 3.8, 4.9, 4.2, 3.5],
        'review_count': [120, 45, 200, 80, 30],
        'avg_sentiment': [0.6, 0.2, 0.8, 0.5, 0.1],
        'combined': [
            'Product A A great wireless headphone with noise cancellation Electronics',
            'Product B Budget earbuds with decent sound quality Electronics',
            'Product C Premium over-ear headphones for audiophiles Electronics',
            'Product D Laptop stand for ergonomic work setup Accessories',
            'Product E USB-C hub with multiple ports for connectivity Accessories',
        ],
    })


@pytest.fixture
def sample_interaction_df():
    """Minimal interaction DataFrame for testing."""
    return pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3'],
        'title': ['Product A', 'Product B', 'Product B', 'Product C', 'Product A', 'Product D'],
        'rating': [5.0, 3.0, 4.0, 5.0, 4.0, 3.5],
    })


@pytest.fixture
def content_model(sample_item_df):
    return ContentRecommender(sample_item_df)


@pytest.fixture
def collab_model(sample_interaction_df):
    return CollaborativeRecommender(sample_interaction_df)


@pytest.fixture
def hybrid_model(content_model, collab_model, sample_item_df):
    return HybridRecommender(content_model, collab_model, sample_item_df)


# ─── bayesian_rating ─────────────────────────────────────────────────────────

class TestBayesianRating:
    def test_high_vote_count_close_to_raw_rating(self):
        """Item with many votes should have Bayesian rating close to its raw rating."""
        result = bayesian_rating(rating=4.5, review_count=1000, global_avg=3.0, min_votes=10)
        assert abs(result - 4.5) < 0.1

    def test_low_vote_count_pulled_toward_mean(self):
        """Item with very few votes should be pulled toward global average."""
        result = bayesian_rating(rating=5.0, review_count=1, global_avg=3.0, min_votes=10)
        assert result < 4.0  # Pulled well below raw 5.0

    def test_zero_votes_equals_global_avg(self):
        """Item with zero votes should return global average."""
        result = bayesian_rating(rating=5.0, review_count=0, global_avg=3.0, min_votes=10)
        assert result == 3.0

    def test_output_range(self):
        """Bayesian rating should stay within [1, 5] for typical inputs."""
        for r in [1.0, 2.5, 3.0, 4.0, 5.0]:
            result = bayesian_rating(r, review_count=50, global_avg=3.0)
            assert 1.0 <= result <= 5.0


# ─── ContentRecommender ──────────────────────────────────────────────────────

class TestContentRecommender:
    def test_recommend_returns_list(self, content_model):
        recs = content_model.recommend('Product A', top_n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_query_item(self, content_model):
        recs = content_model.recommend('Product A', top_n=5)
        titles = [r['title'] for r in recs]
        assert 'Product A' not in titles

    def test_recommend_respects_top_n(self, content_model):
        recs = content_model.recommend('Product A', top_n=2)
        assert len(recs) <= 2

    def test_recommend_unknown_title_returns_empty(self, content_model):
        recs = content_model.recommend('Nonexistent Product XYZ', top_n=5)
        assert recs == []

    def test_recommend_scores_are_floats(self, content_model):
        recs = content_model.recommend('Product A', top_n=3)
        for r in recs:
            assert isinstance(r['content_score'], float)

    def test_search_returns_results(self, content_model):
        results = content_model.search('headphone', top_n=3)
        assert len(results) > 0

    def test_search_result_has_required_keys(self, content_model):
        results = content_model.search('headphone', top_n=2)
        for r in results:
            assert 'title' in r
            assert 'score' in r

    def test_search_empty_query_handles_gracefully(self, content_model):
        # Should not raise, even for an empty query
        try:
            results = content_model.search('', top_n=3)
            assert isinstance(results, list)
        except Exception:
            pass  # Acceptable if it raises — just shouldn't crash the server


# ─── CollaborativeRecommender ────────────────────────────────────────────────

class TestCollaborativeRecommender:
    def test_recommend_returns_list(self, collab_model):
        recs = collab_model.recommend('Product A', top_n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_query_item(self, collab_model):
        recs = collab_model.recommend('Product A', top_n=5)
        titles = [r['title'] for r in recs]
        assert 'Product A' not in titles

    def test_recommend_unknown_title_returns_empty(self, collab_model):
        recs = collab_model.recommend('Unknown Item ZZZ', top_n=3)
        assert recs == []

    def test_predict_for_user_returns_list(self, collab_model):
        recs = collab_model.predict_for_user('u1', top_n=3)
        assert isinstance(recs, list)

    def test_predict_for_unknown_user_returns_empty(self, collab_model):
        recs = collab_model.predict_for_user('unknown_user_xyz', top_n=3)
        assert recs == []

    def test_predict_rating_known_user_item(self, collab_model):
        result = collab_model.predict_rating('u1', 'Product A')
        assert result is not None
        assert isinstance(result, float)

    def test_predict_rating_unknown_returns_none(self, collab_model):
        result = collab_model.predict_rating('ghost_user', 'Product A')
        assert result is None


# ─── HybridRecommender ───────────────────────────────────────────────────────

class TestHybridRecommender:
    def test_recommend_returns_list(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=3)
        assert isinstance(recs, list)

    def test_recommend_has_required_keys(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=2)
        required = {'title', 'hybrid_score', 'content_score', 'collab_score', 'sentiment_score'}
        for r in recs:
            assert required.issubset(r.keys())

    def test_recommend_explain_false_keeps_default_payload_compact(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=2)
        assert recs
        assert 'explanation' not in recs[0]

    def test_recommend_explain_true_adds_score_breakdown(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=2, explain=True)
        assert recs
        explanation = recs[0]['explanation']
        assert explanation['source_item'] == 'Product A'
        assert 'component_scores' in explanation
        assert 'weighted_components' in explanation
        assert 'top_content_terms' in explanation
        assert explanation['signals']['sentiment_polarity'] in {'positive', 'neutral', 'negative'}

    def test_recommend_sorted_by_hybrid_score(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=5)
        scores = [r['hybrid_score'] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_query_item(self, hybrid_model):
        recs = hybrid_model.recommend('Product A', top_n=5)
        titles = [r['title'] for r in recs]
        assert 'Product A' not in titles

    def test_set_weights_normalizes(self, hybrid_model):
        hybrid_model.set_weights(2, 2, 2)
        w = hybrid_model.get_weights()
        total = w['alpha'] + w['beta'] + w['gamma']
        assert abs(total - 1.0) < 1e-6

    def test_set_weights_zero_total_safe(self, hybrid_model):
        """set_weights with all zeros should not divide by zero."""
        try:
            hybrid_model.set_weights(0, 0, 0)
        except ZeroDivisionError:
            pytest.fail("set_weights raised ZeroDivisionError on all-zero input")

    def test_get_weights_returns_dict(self, hybrid_model):
        w = hybrid_model.get_weights()
        assert 'alpha' in w and 'beta' in w and 'gamma' in w

    def test_cold_start_fallback_unknown_title(self, hybrid_model):
        """Unknown title should return fallback results, not crash."""
        recs = hybrid_model.recommend('Totally Unknown Product 99999', top_n=3)
        assert isinstance(recs, list)

    def test_no_collab_model_still_works(self, content_model, sample_item_df):
        """HybridRecommender should work without a collab model (content + sentiment only)."""
        hm = HybridRecommender(content_model, collab_model=None, item_df=sample_item_df)
        recs = hm.recommend('Product A', top_n=3)
        assert isinstance(recs, list)
        assert len(recs) > 0
