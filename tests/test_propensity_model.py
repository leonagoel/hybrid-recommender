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
