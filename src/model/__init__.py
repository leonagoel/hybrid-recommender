"""
src.model — Public surface of the recommender model package.

Importing from this package gives access to all model classes and the
causal inference layer without needing to know internal module paths.
"""

import importlib.util

def _load_safe(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name, None)
    except Exception:
        return None

ContentRecommender = _load_safe("src.model.content_model", "ContentRecommender")
CollaborativeRecommender = _load_safe("src.model.collaborative_model", "CollaborativeRecommender")
HybridRecommender = _load_safe("src.model.hybrid_model", "HybridRecommender")
CausalDebiaser = _load_safe("src.model.causal_model", "CausalDebiaser")
CausalConfig = _load_safe("src.model.causal_config", "CausalConfig")
PropensityModel = _load_safe("src.model.propensity_model", "PropensityModel")

__all__ = [
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "CausalDebiaser",
    "CausalConfig",
    "PropensityModel",
]