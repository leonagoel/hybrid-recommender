"""
Unit tests specifically for PropensityModel.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from src.model.propensity_model import PropensityModel


class TestPropensityModelSpec:

    @pytest.fixture
    def sample_catalog(self):
        return pd.DataFrame({
            'title': ['A', 'B', 'C', 'D'],
            'review_count': [1000, 100, 10, 1],
            'category': ['Tech', 'Tech', 'Art', 'Art']
        })

    def test_propensity_init_empty(self):
        pm = PropensityModel(pd.DataFrame())
        assert pm.all_scores() == {}
        assert pm.summary() == {}

    def test_propensity_init_none(self):
        pm = PropensityModel(None)
        assert pm.all_scores() == {}
        assert pm.summary() == {}

    def test_propensity_uniform_review_count(self):
        df = pd.DataFrame({
            'title': ['A', 'B'],
            'review_count': [10, 10],
            'category': ['X', 'X']
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        assert abs(scores['A'] - scores['B']) < 1e-6

    def test_propensity_no_category(self):
        df = pd.DataFrame({
            'title': ['A', 'B'],
            'review_count': [100, 10]
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        # Item A is more popular, so should have higher propensity
        assert scores['A'] > scores['B']

    def test_propensity_no_review_count(self):
        df = pd.DataFrame({
            'title': ['A', 'B', 'C'],
            'category': ['X', 'X', 'Y']
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        # X category is dominant, so A and B should have higher propensity than C
        assert scores['A'] > scores['C']
        assert scores['B'] > scores['C']

    def test_propensity_conformal(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        scores = pm.all_scores()
        assert scores['A'] > scores['D']  # A is blockbuster, D is niche

        w_a = pm.get_ips_weight('A')
        w_d = pm.get_ips_weight('D')
        assert w_d > w_a  # Niche gets boosted more

    def test_propensity_get_nonexistent(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        assert pm.get('Nonexistent') == 1.0
        assert pm.get('Nonexistent', default=0.5) == 0.5

    def test_propensity_get_ips_weight_nonexistent(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        assert pm.get_ips_weight('Nonexistent') == 1.0
        assert pm.get_ips_weight('Nonexistent', clip_max=10.0) == 1.0

    def test_propensity_summary(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        summary = pm.summary()
        assert summary['n_items'] == 4
        assert summary['mean'] > 0
        assert summary['min'] <= summary['max']

    def test_propensity_nan_values(self):
        df = pd.DataFrame({
            'title': ['A', 'B', 'C'],
            'review_count': [100, None, np.nan],
            'category': ['Tech', None, np.nan]
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        assert len(scores) == 3
        # Should not raise errors and have positive scores
        assert scores['A'] > 0
        assert scores['B'] > 0
        assert scores['C'] > 0

    def test_propensity_empty_strings(self):
        df = pd.DataFrame({
            'title': ['A', 'B', ''],
            'review_count': [10, 20, 30],
            'category': ['', '   ', 'Tech']
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        assert scores['A'] > 0
        assert scores['B'] > 0

    def test_propensity_negative_reviews(self):
        df = pd.DataFrame({
            'title': ['A', 'B'],
            'review_count': [-100, 0],
            'category': ['Tech', 'Art']
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        # Non-positive review counts are clipped to min=0/positive values
        assert scores['A'] > 0
        assert scores['B'] > 0

    def test_propensity_extreme_reviews(self):
        df = pd.DataFrame({
            'title': ['A', 'B'],
            'review_count': [1000000000, 1],
            'category': ['Tech', 'Art']
        })
        pm = PropensityModel(df)
        scores = pm.all_scores()
        # Item with 1 billion reviews should not break normalization and have vastly higher score
        assert scores['A'] > scores['B']
        assert pm.get_ips_weight('A') < pm.get_ips_weight('B')

    def test_propensity_ips_weight_clip_max_boundary(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        # clip_max set to an extremely small positive bound like 0.1
        w = pm.get_ips_weight('D', clip_max=0.1)
        assert w == 0.1

    def test_propensity_ips_weight_extreme_small_propensity(self, sample_catalog):
        pm = PropensityModel(sample_catalog)
        # Mock propensity score of 'A' to exactly 0 to test division by zero safety
        pm._scores['A'] = 0.0
        w = pm.get_ips_weight('A', clip_max=5.0)
        # Minimum propensity is 1e-6, so weight is 1.0 / 1e-6 = 1000000, capped at clip_max (5.0)
        assert w == 5.0

