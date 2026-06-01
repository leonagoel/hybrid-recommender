"""
conftest.py — Shared pytest fixtures for the hybrid-recommender test suite.

These fixtures are automatically discovered by pytest and available to every
test file in the tests/ directory without an explicit import.
"""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_item_df():
    """
    Minimal item DataFrame covering two categories.
    Used by ContentRecommender, HybridRecommender, and related tests.
    Session-scoped so the same object is reused across all tests that need it.
    """
    return pd.DataFrame(
        {
            "title": [
                "Product A",
                "Product B",
                "Product C",
                "Product D",
                "Product E",
            ],
            "description": [
                "A great wireless headphone with noise cancellation",
                "Budget earbuds with decent sound quality",
                "Premium over-ear headphones for audiophiles",
                "Laptop stand for ergonomic work setup",
                "USB-C hub with multiple ports for connectivity",
            ],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Accessories",
                "Accessories",
            ],
            "rating": [4.5, 3.8, 4.9, 4.2, 3.5],
            "review_count": [120, 45, 200, 80, 30],
            "avg_sentiment": [0.6, 0.2, 0.8, 0.5, 0.1],
            "combined": [
                "Product A A great wireless headphone with noise cancellation Electronics",
                "Product B Budget earbuds with decent sound quality Electronics",
                "Product C Premium over-ear headphones for audiophiles Electronics",
                "Product D Laptop stand for ergonomic work setup Accessories",
                "Product E USB-C hub with multiple ports for connectivity Accessories",
            ],
        }
    )


@pytest.fixture(scope="session")
def sample_interaction_df():
    """
    Minimal user-item interaction DataFrame.
    Used by CollaborativeRecommender and HybridRecommender tests.
    """
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "title": [
                "Product A",
                "Product B",
                "Product B",
                "Product C",
                "Product A",
                "Product D",
            ],
            "rating": [5.0, 3.0, 4.0, 5.0, 4.0, 3.5],
        }
    )


# ---------------------------------------------------------------------------
# Model fixtures  (function-scoped so tests can mutate them independently)
# ---------------------------------------------------------------------------

@pytest.fixture()
def collab_model(sample_interaction_df):
    """
    A fitted CollaborativeRecommender instance.
    Function-scoped so each test gets an isolated copy.
    """
    from src.model.collaborative_model import CollaborativeRecommender

    return CollaborativeRecommender(sample_interaction_df)
