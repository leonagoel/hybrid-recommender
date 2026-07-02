"""
Tests for cold-start recommendation handling.
Validates new user recommendations and onboarding flow.
"""

import pytest
import pandas as pd
from src.model.cold_start_handler import ColdStartHandler


@pytest.fixture
def sample_data():
    """Create sample item and interaction data."""
    items = pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': ['action', 'drama', 'action', 'comedy', 'drama'],
        'popularity': [100, 80, 90, 50, 70],
        'avg_rating': [8.5, 7.8, 8.2, 6.5, 7.9],
    })

    interactions = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2', 'user2'],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'rating': [5.0, 4.0, 4.5, 3.0, 4.8],
    })

    return items, interactions


def test_cold_start_handler_initialization(sample_data):
    """Verify cold-start handler initializes correctly."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    assert len(handler.popular_items) > 0
    assert handler.item_df is not None


def test_onboarding_survey_generation(sample_data):
    """Verify onboarding survey is generated correctly."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    survey = handler.get_onboarding_survey()

    assert 'survey_type' in survey
    assert survey['survey_type'] == 'onboarding'
    assert 'genres' in survey
    assert 'sample_items' in survey
    assert len(survey['sample_items']) > 0


def test_popular_items_retrieval(sample_data):
    """Verify popular items are returned correctly."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    popular = handler.get_popular_items(limit=3)

    assert len(popular) <= 3
    assert all('title' in item for item in popular)
    assert all('popularity' in item for item in popular)

    sorted_popular = sorted(popular, key=lambda x: x['popularity'], reverse=True)
    assert popular == sorted_popular, "Items should be sorted by popularity"


def test_demographic_recommendations(sample_data):
    """Verify demographic-based recommendations."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    user_prefs = {'genres': ['action']}
    recommendations = handler.get_demographic_recommendations(user_prefs, limit=5)

    assert len(recommendations) > 0
    assert all('title' in item for item in recommendations)
    assert all('cold_start_score' in item for item in recommendations)


def test_user_profile_bootstrap(sample_data):
    """Verify user profile can be bootstrapped from onboarding ratings."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    initial_interaction_count = len(handler.interaction_df)

    onboarding_ratings = {
        'Movie A': 5.0,
        'Movie B': 4.0,
        'Movie C': 3.5,
    }

    handler.bootstrap_user_profile('new_user', onboarding_ratings)

    assert len(handler.interaction_df) == initial_interaction_count + len(onboarding_ratings)

    new_user_interactions = handler.interaction_df[handler.interaction_df['user_id'] == 'new_user']
    assert len(new_user_interactions) == len(onboarding_ratings)


def test_hybrid_recommendations_for_new_user(sample_data):
    """Verify hybrid recommendations combine multiple signals."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    user_prefs = {'genres': ['action', 'drama']}
    recommendations = handler.recommend_for_new_user(user_prefs, limit=10)

    assert len(recommendations) > 0
    assert all('title' in item for item in recommendations)
    assert all('reason' in item for item in recommendations)
    assert all('weight' in item for item in recommendations)

    weights = [item['weight'] for item in recommendations]
    assert sum(weights) > 0, "At least one recommendation should have non-zero weight"


def test_random_exploration(sample_data):
    """Verify random exploration recommendations."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    random_items = handler._get_random_exploration(limit=2)

    assert len(random_items) <= 2
    assert all('title' in item for item in random_items)
    assert all(item['weight'] == 0.1 for item in random_items)


def test_empty_onboarding_ratings(sample_data):
    """Verify graceful handling of empty onboarding ratings."""
    items, interactions = sample_data
    handler = ColdStartHandler(items, interactions)

    initial_count = len(handler.interaction_df)
    handler.bootstrap_user_profile('user_no_ratings', {})

    assert len(handler.interaction_df) == initial_count, "No interactions should be added"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
