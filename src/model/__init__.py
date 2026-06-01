"""
src.model — Public surface of the recommender model package.

Importing from this package gives access to all model classes and the
causal inference layer without needing to know internal module paths.

Imports are intentionally lazy so that unit tests that only need
lightweight modules (e.g. CollaborativeRecommender, nlp_engine) do not
pay the cost of loading sentence-transformers / torch transitively.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily import model classes on first access."""
    _map = {
        "ContentRecommender": ("src.model.content_model", "ContentRecommender"),
        "CollaborativeRecommender": ("src.model.collaborative_model", "CollaborativeRecommender"),
        "HybridRecommender": ("src.model.hybrid_model", "HybridRecommender"),
        "CausalDebiaser": ("src.model.causal_model", "CausalDebiaser"),
        "CausalConfig": ("src.model.causal_config", "CausalConfig"),
        "PropensityModel": ("src.model.propensity_model", "PropensityModel"),
    }
    if name in _map:
        module_path, attr = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'src.model' has no attribute {name!r}")


__all__ = [
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "CausalDebiaser",
    "CausalConfig",
    "PropensityModel",
]
