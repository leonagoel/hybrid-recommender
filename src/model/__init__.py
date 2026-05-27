"""
src.model — Public surface of the recommender model package.

Importing from this package gives access to all model classes and the
causal inference layer without needing to know internal module paths.
"""

from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.causal_model import CausalDebiaser
from src.model.causal_config import CausalConfig
from src.model.propensity_model import PropensityModel

__all__ = [
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "CausalDebiaser",
    "CausalConfig",
    "PropensityModel",
]
