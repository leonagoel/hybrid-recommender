"""
Validation schemas and parameter contracts for the recommendation models.
Enforces strict type checking, bounds, and defaults using Pydantic.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class ModelHyperparametersSchema(BaseModel):
    """
    Strict configuration validation schema contract for model hyperparameters.
    Ensures structural parameters conform to training requirements.
    """
    n_factors: int = Field(default=50, ge=1, description="Number of latent factors for SVD matrix factorization.")
    use_implicit: bool = Field(default=True, description="Whether to treat interactions implicitly.")

    class Config:
        frozen = True
        extra = "forbid"


class HybridWeightsSchema(BaseModel):
    """
    Strict parameter verification schema contract for hybrid blending weights.
    Validates type bounds and dynamically enforces normalization limits.
    """
    alpha: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for content-based score blending component.")
    beta: float = Field(default=0.35, ge=0.0, le=1.0, description="Weight for collaborative filtering score blending component.")
    gamma: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for NLP sentiment score blending component.")

    @model_validator(mode="after")
    def validate_weights_normalization(self) -> "HybridWeightsSchema":
        """Ensures that blending parameters don't break normalization baselines."""
        total = self.alpha + self.beta + self.gamma
        if total <= 0.0:
            raise ValueError("The cumulative summation of blending weights (alpha, beta, gamma) must be greater than 0.")
        return self

    class Config:
        frozen = True
        extra = "forbid"