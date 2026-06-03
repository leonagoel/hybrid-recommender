"""
CausalConfig — Configuration dataclass for the causal inference layer.
======================================================================

Centralises every tuning knob for CausalDebiaser so callers never pass
raw floats scattered across constructor signatures.

Usage
-----
    from src.model.causal_config import CausalConfig

    # Sensible production defaults — 50 % causal correction, clip at 5×
    cfg = CausalConfig()

    # Aggressive debiasing for a heavily skewed catalog
    cfg = CausalConfig(blend_lambda=0.8, clip_max=8.0)

    # Disable debiasing entirely (useful for A/B control arm)
    cfg = CausalConfig(enabled=False)

    # Pass directly to HybridRecommender
    model = HybridRecommender(content, collab, item_df, causal_config=cfg)

Design notes
------------
- Pure dataclass — no logic, no imports beyond stdlib.
- All fields have defaults so `CausalConfig()` is always valid.
- `validate()` raises ValueError early rather than letting bad values
  propagate silently into numpy operations.
"""

from dataclasses import dataclass, field

from dataclasses import dataclass


@dataclass
class CausalConfig:
    """
    Configuration for the IPS causal debiasing layer.

    Attributes
    ----------
    enabled      : Master switch. False = causal layer is completely bypassed.
                   Equivalent to blend_lambda=0 but skips all computation.
    blend_lambda : λ — fraction of causal correction blended into the final score.
                   0.0 → pure correlation score (no debiasing).
                   1.0 → full IPS reweighting (maximum debiasing).
                   Default 0.5 balances exploration vs. ranking stability.
    clip_max     : Hard cap on IPS weights to prevent variance explosion from
                   extremely rare items. A value of 5.0 means a niche item can
                   receive at most a 5× score multiplier. Default 5.0.
    score_key    : Which field in each result dict to debias.
                   Default 'hybrid_score' matches the existing pipeline output.
    """

    enabled: bool = True
    blend_lambda: float = 0.5
    clip_max: float = 5.0
    score_key: str = 'hybrid_score'

    def validate(self) -> 'CausalConfig':
        """
        Validate all fields and return self for chaining.
        Raises ValueError on any invalid value.
        """
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not 0.0 <= self.blend_lambda <= 1.0:
            raise ValueError(f"blend_lambda must be in [0.0, 1.0], got {self.blend_lambda}")
        if self.clip_max <= 0:
            raise ValueError(f"clip_max must be positive, got {self.clip_max}")
        if not self.score_key:
            raise ValueError("score_key must be a non-empty string")
        return self

    def to_dict(self) -> dict:
        """Serialise to a plain dict — useful for API responses and logging."""
        return {
            'enabled': self.enabled,
            'blend_lambda': self.blend_lambda,
            'clip_max': self.clip_max,
            'score_key': self.score_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CausalConfig':
        """Deserialise from a plain dict — useful for loading from .env or JSON config."""
        return cls(
            enabled=bool(d.get('enabled', True)),
            blend_lambda=float(d.get('blend_lambda', 0.5)),
            clip_max=float(d.get('clip_max', 5.0)),
            score_key=str(d.get('score_key', 'hybrid_score')),
        ).validate()

    # ── Preset factory methods ────────────────────────────────────────────────

    @classmethod
    def disabled(cls) -> 'CausalConfig':
        """Return a config that completely disables causal debiasing."""
        return cls(enabled=False)

    @classmethod
    def conservative(cls) -> 'CausalConfig':
        """Light debiasing — safe default for production with unknown catalog skew."""
        return cls(enabled=True, blend_lambda=0.3, clip_max=3.0)

    @classmethod
    def aggressive(cls) -> 'CausalConfig':
        """Strong debiasing — use when catalog has severe popularity concentration."""
        return cls(enabled=True, blend_lambda=0.8, clip_max=8.0)