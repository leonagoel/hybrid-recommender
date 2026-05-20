"""
Unit tests for the A/B Testing Module.
Run with: pytest tests/test_ab_testing.py -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ab_testing import ABTestRunner
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender


@pytest.fixture
def sample_item_df():
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
def test_pairs():
    return [
        ('u1', 'Product A', ['Product A', 'Product B', 'Product C']),
        ('u2', 'Product B', ['Product B', 'Product C']),
        ('u3', 'Product A', ['Product A', 'Product D']),
    ]


@pytest.fixture
def runner(content_model, collab_model, sample_item_df):
    return ABTestRunner(
        content_model, collab_model, sample_item_df,
        config_a={"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
        config_b={"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
        k=3,
    )


# ── ABTestRunner Tests ──────────────────────────────────────────────

class TestABTestRunner:
    def test_run_returns_results(self, runner, test_pairs):
        results = runner.run(test_pairs)
        assert isinstance(results, dict)
        assert 'config_a' in results
        assert 'config_b' in results
        assert 'winner' in results

    def test_winner_is_a_or_b(self, runner, test_pairs):
        results = runner.run(test_pairs)
        assert results['winner'] in ('A', 'B')

    def test_metrics_are_floats(self, runner, test_pairs):
        results = runner.run(test_pairs)
        for config_key in ('config_a', 'config_b'):
            config = results[config_key]
            assert isinstance(config['precision'], float)
            assert isinstance(config['recall'], float)
            assert isinstance(config['ndcg'], float)
            assert isinstance(config['diversity'], float)

    def test_metrics_in_valid_range(self, runner, test_pairs):
        results = runner.run(test_pairs)
        for config_key in ('config_a', 'config_b'):
            config = results[config_key]
            assert 0.0 <= config['precision'] <= 1.0
            assert 0.0 <= config['recall'] <= 1.0
            assert 0.0 <= config['ndcg'] <= 1.0
            assert 0.0 <= config['diversity'] <= 1.0

    def test_test_users_count(self, runner, test_pairs):
        results = runner.run(test_pairs)
        assert results['test_users'] == len(test_pairs)

    def test_k_value_stored(self, runner, test_pairs):
        results = runner.run(test_pairs)
        assert results['k'] == 3

    def test_empty_test_pairs_returns_empty(self, runner):
        results = runner.run([])
        assert results == {}

    def test_print_results_no_crash(self, runner, test_pairs):
        runner.run(test_pairs)
        runner.print_results()  # Should not raise

    def test_print_results_before_run_no_crash(self, runner):
        runner.print_results()  # Should not raise

    def test_write_results_md(self, runner, test_pairs, tmp_path):
        runner.run(test_pairs)
        filepath = str(tmp_path / "test_results.md")
        runner.write_results_md(filepath)
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
        assert "A/B Test Results" in content
        assert "Config A" in content
        assert "Config B" in content

    def test_write_results_before_run_no_crash(self, runner, tmp_path):
        filepath = str(tmp_path / "empty_results.md")
        runner.write_results_md(filepath)
        assert not os.path.exists(filepath)  # Should not write if no results


class TestDiversityScore:
    def test_all_same_category(self):
        recs = [{"category": "Electronics"}] * 5
        score = ABTestRunner.diversity_score(recs)
        assert score == 1 / 5

    def test_all_different_categories(self):
        recs = [
            {"category": "Electronics"},
            {"category": "Books"},
            {"category": "Clothing"},
        ]
        score = ABTestRunner.diversity_score(recs)
        assert score == 1.0

    def test_empty_list(self):
        assert ABTestRunner.diversity_score([]) == 0.0

    def test_mixed_categories(self):
        recs = [
            {"category": "A"},
            {"category": "A"},
            {"category": "B"},
            {"category": "C"},
        ]
        score = ABTestRunner.diversity_score(recs)
        assert score == 3 / 4
