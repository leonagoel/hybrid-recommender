"""
Tests for recommendation A/B testing helpers.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ab_testing import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_VARIANTS,
    ExperimentVariant,
    assign_variant,
    run_recommendation_experiment,
    summarize_variant_metrics,
)
from collaborative_model import CollaborativeRecommender
from content_model import ContentRecommender
from hybrid_model import HybridRecommender


@pytest.fixture
def sample_item_df():
    return pd.DataFrame({
        "title": ["Product A", "Product B", "Product C", "Product D"],
        "description": [
            "Wireless headphones with active noise cancellation",
            "Budget earbuds with balanced audio",
            "Premium studio headphones",
            "Portable USB-C hub with multiple ports",
        ],
        "category": ["Electronics", "Electronics", "Electronics", "Accessories"],
        "rating": [4.5, 3.8, 4.9, 4.2],
        "review_count": [120, 45, 200, 80],
        "avg_sentiment": [0.6, 0.2, 0.8, 0.4],
        "combined": [
            "Product A Wireless headphones with active noise cancellation Electronics",
            "Product B Budget earbuds with balanced audio Electronics",
            "Product C Premium studio headphones Electronics",
            "Product D Portable USB-C hub with multiple ports Accessories",
        ],
    })


@pytest.fixture
def sample_interaction_df():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "title": ["Product A", "Product B", "Product B", "Product C", "Product D"],
        "rating": [5.0, 3.0, 4.0, 5.0, 3.5],
    })


@pytest.fixture
def hybrid_model(sample_item_df, sample_interaction_df):
    return HybridRecommender(
        ContentRecommender(sample_item_df),
        CollaborativeRecommender(sample_interaction_df),
        sample_item_df,
    )


def test_assign_variant_is_stable_for_same_user():
    first = assign_variant("user-123", experiment_id=DEFAULT_EXPERIMENT_ID)
    second = assign_variant("user-123", experiment_id=DEFAULT_EXPERIMENT_ID)

    assert first == second
    assert first.name in {variant.name for variant in DEFAULT_VARIANTS}


def test_assign_variant_respects_weighted_traffic():
    variants = (
        ExperimentVariant(
            name="off",
            description="No traffic",
            weights={"alpha": 0.4, "beta": 0.35, "gamma": 0.25},
            traffic=0,
        ),
        ExperimentVariant(
            name="on",
            description="All traffic",
            weights={"alpha": 0.6, "beta": 0.25, "gamma": 0.15},
            traffic=100,
        ),
    )

    assert assign_variant("any-user", variants=variants).name == "on"


def test_run_recommendation_experiment_restores_original_weights(hybrid_model):
    original_weights = hybrid_model.get_weights()

    result = run_recommendation_experiment(
        hybrid_model,
        "Product A",
        user_key="stable-user",
        top_n=2,
        explain=True,
    )

    assert result["experiment"]["variant"] in {variant.name for variant in DEFAULT_VARIANTS}
    assert result["recommendations"]
    assert "explanation" in result["recommendations"][0]
    assert hybrid_model.get_weights() == original_weights


def test_run_recommendation_experiment_uses_assigned_variant_weights(hybrid_model):
    variant = ExperimentVariant(
        name="sentiment_only",
        description="Uses sentiment-heavy ranking for the test.",
        weights={"alpha": 0, "beta": 0, "gamma": 1},
    )

    result = run_recommendation_experiment(
        hybrid_model,
        "Product A",
        user_key="stable-user",
        top_n=2,
        variants=(variant,),
    )

    assert result["experiment"]["variant"] == "sentiment_only"
    assert result["experiment"]["weights"] == {"alpha": 0.0, "beta": 0.0, "gamma": 1.0}


def test_summarize_variant_metrics_returns_average_by_variant():
    summary = summarize_variant_metrics(
        [
            {"variant": "control", "clicked": 1},
            {"variant": "control", "clicked": 0},
            {"variant": "content_heavy", "clicked": 1},
        ],
        metric_name="clicked",
    )

    assert summary == [
        {"variant": "content_heavy", "count": 1, "total": 1.0, "average": 1.0},
        {"variant": "control", "count": 2, "total": 1.0, "average": 0.5},
    ]
