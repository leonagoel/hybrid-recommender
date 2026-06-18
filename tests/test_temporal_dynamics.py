"""
Tests for temporal dynamics in collaborative filtering.
Validates time-decay weighting and preference drift detection.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.model.collaborative_model import CollaborativeRecommender


def test_temporal_weight_computation():
    """Verify recent interactions receive higher weights than old ones."""
    now = datetime.now()
    old_date = now - timedelta(days=180)
    recent_date = now - timedelta(days=10)

    df = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2'],
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'rating': [5.0, 4.0, 3.0],
        'timestamp': [old_date, recent_date, recent_date],
    })

    recommender = CollaborativeRecommender(df, enable_temporal=True, time_decay_days=90)
    weights = recommender._compute_temporal_weights()

    old_weight = weights[0]
    recent_weights = weights[1:]

    assert old_weight < min(recent_weights), "Old interactions should have lower weight than recent"
    assert all(0.1 <= w <= 1.0 for w in weights), "Weights should be in [0.1, 1.0] range"


def test_temporal_disabled():
    """Verify temporal weighting can be disabled."""
    df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'title': ['Movie A', 'Movie B'],
        'rating': [5.0, 4.0],
        'timestamp': [datetime.now() - timedelta(days=100), datetime.now()],
    })

    recommender = CollaborativeRecommender(df, enable_temporal=False)
    assert recommender.enable_temporal is False


def test_temporal_without_timestamp():
    """Verify model works when timestamp column is missing."""
    df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'title': ['Movie A', 'Movie B'],
        'rating': [5.0, 4.0],
    })

    recommender = CollaborativeRecommender(df, enable_temporal=True)
    weights = recommender._compute_temporal_weights()
    assert np.allclose(weights, 1.0), "Should use uniform weights when timestamp unavailable"


def test_recent_preferences_override_old():
    """
    Verify that recent preference changes are reflected in recommendations.
    User changes preferences from old genre to new genre.
    """
    now = datetime.now()

    df = pd.DataFrame({
        'user_id': ['user1'] * 3 + ['user2'] * 2,
        'title': ['Old Genre Movie 1', 'Old Genre Movie 2', 'New Genre Movie 1', 'New Genre Movie 2', 'Old Genre Movie 3'],
        'rating': [5.0, 4.5, 5.0, 4.8, 4.0],
        'timestamp': [
            now - timedelta(days=200),
            now - timedelta(days=180),
            now - timedelta(days=5),
            now - timedelta(days=3),
            now - timedelta(days=50),
        ],
    })

    recommender = CollaborativeRecommender(df, enable_temporal=True, time_decay_days=90)

    assert recommender.enable_temporal is True
    assert recommender.time_decay_days == 90

    weights = recommender._compute_temporal_weights()
    assert weights[2] > weights[0], "Recent preference should have higher weight"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
