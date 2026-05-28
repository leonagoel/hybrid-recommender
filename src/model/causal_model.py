"""
Causal Inference Layer — Inverse Propensity Scoring (IPS) Debiaser
===================================================================

Problem this solves
-------------------
The existing hybrid pipeline scores items by *correlation*:
  - Popular items dominate collaborative scores because many users rated them
    (exposure bias — users can only rate what they were shown).
  - Category-dominant items dominate content scores because the embedding
    space is skewed toward frequent categories.
  - High-sentiment items may simply be cheap/impulse products that attract
    many reviews, not genuinely better recommendations.

None of these signals answer the causal question:
  "Would this user have liked this item *if* they had been shown it?"

Causal approach: Inverse Propensity Scoring (IPS)
-------------------------------------------------
IPS is a standard technique from causal inference / counterfactual learning.

1. Estimate propensity P(item shown) for each item — the probability it
   appeared in the training data due to confounders (popularity, category
   dominance), NOT because it was genuinely relevant.

2. Reweight each score by 1 / P(item shown).
   Items that were over-exposed get *downweighted*; niche items that were
   rarely shown but scored well get *upweighted*.

3. Blend the IPS-debiased score with the original correlation score using
   a configurable lambda (λ) to control how aggressively we debias.

Mathematical formulation
------------------------
Let s_i  = raw hybrid score for item i  ∈ [0, 1]
Let p_i  = propensity of item i being shown (estimated from popularity + category)
Let w_i  = IPS weight = clip(1 / p_i, max=clip_max)

Debiased score:
    s_i^causal = λ · (s_i · w_i / E[w]) + (1 - λ) · s_i

Where:
    E[w]  = mean IPS weight across all candidates (normalizes scale)
    λ     = blend factor (default 0.5) — 0 = pure correlation, 1 = pure causal
    clip  = max IPS weight cap to prevent variance explosion from rare items

The final score is clipped to [0, 1] to preserve the existing contract.

Final recommendation formula (full pipeline)
--------------------------------------------
    hybrid_base   = α·content + β·collab + γ·sentiment
    hybrid        = min(1.0, hybrid_base + 0.05·popularity)   ← existing
    causal_score  = λ·(hybrid · w_i / E[w]) + (1-λ)·hybrid   ← new (this file)

Where w_i is the IPS weight derived from PropensityModel.

Design decisions
----------------
- No new dependencies: uses only numpy + pandas (already in requirements.txt).
- Opt-in: HybridRecommender accepts use_causal_debiasing=False by default.
- Stateless per-call: debias_batch() is pure — it does not mutate the model.
- Transparent: each result dict gets a 'causal_score' key alongside 'hybrid_score'
  so the UI/API can show both for debugging.
- Propensity estimation is delegated to PropensityModel for independent
  testability and future extensibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.causal_config import CausalConfig
from src.model.propensity_model import PropensityModel


class CausalDebiaser:
    """
    Applies IPS reweighting to debias hybrid recommendation scores.

    Propensity estimation is handled by PropensityModel.
    This class is responsible only for the IPS reweighting math.
    """

    def __init__(
        self,
        item_df: pd.DataFrame,
        blend_lambda: float = 0.5,
        clip_max: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        item_df      : The adapted item DataFrame (output of adapt_data).
                       Must contain 'title'. Optionally 'review_count', 'category'.
        blend_lambda : λ — how much causal correction to apply.
                       0.0 = no debiasing (pure correlation score).
                       1.0 = full IPS reweighting.
                       Default 0.5 balances exploration vs. stability.
        clip_max     : Maximum IPS weight. Caps variance from very rare items.
                       Default 5.0 means a niche item can at most 5× its score.
        """
        if not 0.0 <= blend_lambda <= 1.0:
            raise ValueError("blend_lambda must be in [0.0, 1.0]")
        if clip_max <= 0:
            raise ValueError("clip_max must be positive")

        self.blend_lambda = blend_lambda
        self.clip_max = clip_max
        self._propensity_model = PropensityModel(item_df)

    # ------------------------------------------------------------------
    # Backward-compatible propensity accessors
    # ------------------------------------------------------------------

    @property
    def _propensity(self) -> dict[str, float]:
        """Backward-compatible access to the raw propensity dict."""
        return self._propensity_model.all_scores()

    # ------------------------------------------------------------------
    # Public debiasing API
    # ------------------------------------------------------------------

    def debias(self, title: str, score: float) -> float:
        """
        Apply IPS reweighting to a single item score.

        Formula:
            w_i           = clip(1 / p_i, max=clip_max)
            causal_score  = λ · (score · w_i) + (1 - λ) · score
                          = score · [λ · w_i + (1 - λ)]

        The IPS weight is NOT divided by E[w] here because single-item
        calls have no batch context. Use debias_batch() for proper
        batch normalization.

        Parameters
        ----------
        title : Item title (must match keys in item_df).
        score : Original hybrid score in [0, 1].

        Returns
        -------
        Debiased score clipped to [0, 1].
        """
        w_i = self._propensity_model.get_ips_weight(title, self.clip_max)
        debiased = score * (self.blend_lambda * w_i + (1.0 - self.blend_lambda))
        return float(np.clip(debiased, 0.0, 1.0))

    def debias_batch(
        self,
        items: list[dict],
        score_key: str = "hybrid_score",
    ) -> list[dict]:
        """
        Apply IPS reweighting to a batch of result dicts with proper
        batch-level normalization of IPS weights.

        Batch normalization divides each w_i by E[w] across the candidate
        set, which prevents the overall score distribution from shifting
        up or down after debiasing.

        Formula (batch-normalized):
            w_i           = clip(1 / p_i, max=clip_max)
            E[w]          = mean(w_i) across all candidates
            causal_score  = λ · (score · w_i / E[w]) + (1 - λ) · score

        Each result dict is updated in-place with:
            - score_key         : replaced with the debiased score
            - 'causal_score'    : the debiased score (for transparency)
            - 'original_score'  : the pre-debiasing score (for debugging)

        Parameters
        ----------
        items     : List of recommendation result dicts (from hybrid_model.recommend).
        score_key : Which score field to debias. Default 'hybrid_score'.

        Returns
        -------
        The same list with scores updated in-place.
        """
        if not items:
            return items

        # Collect raw IPS weights for all candidates
        raw_weights = [
            self._propensity_model.get_ips_weight(
                item.get("title", ""), self.clip_max
            )
            for item in items
        ]
        weights_arr = np.array(raw_weights, dtype=float)

        # Batch-level normalization: divide by mean so E[w_normalized] = 1
        mean_w = weights_arr.mean()
        normalized_weights = weights_arr / mean_w if mean_w > 0 else np.ones(len(raw_weights))

        for i, item in enumerate(items):
            original = float(item.get(score_key, 0.0))
            w_norm = float(normalized_weights[i])

            causal = original * (self.blend_lambda * w_norm + (1.0 - self.blend_lambda))
            causal = float(np.clip(causal, 0.0, 1.0))

            item["original_score"] = round(original, 4)
            item["causal_score"] = round(causal, 4)
            item[score_key] = round(causal, 4)

        return items

    # ------------------------------------------------------------------
    # Factory — construct from a CausalConfig object
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, item_df: pd.DataFrame, config: CausalConfig) -> "CausalDebiaser":
        """
        Construct a CausalDebiaser from a CausalConfig instance.

        This is the preferred construction path when using the configuration
        system, because it validates all parameters before building the
        propensity model.

        Parameters
        ----------
        item_df : Adapted item DataFrame (output of adapt_data).
        config  : Validated CausalConfig instance.

        Returns
        -------
        CausalDebiaser ready to call debias_batch().

        Example
        -------
            cfg = CausalConfig(blend_lambda=0.6, clip_max=4.0)
            debiaser = CausalDebiaser.from_config(item_df, cfg)
        """
        config.validate()
        return cls(
            item_df=item_df,
            blend_lambda=config.blend_lambda,
            clip_max=config.clip_max,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_propensity(self, title: str) -> float:
        """Return the estimated propensity score for a given item title."""
        return self._propensity_model.get(title)

    def get_ips_weight(self, title: str) -> float:
        """Return the clipped IPS weight for a given item title."""
        return self._propensity_model.get_ips_weight(title, self.clip_max)

    def summary(self) -> dict:
        """
        Return a summary of the propensity distribution for diagnostics.
        Useful for logging and the /evaluate endpoint.
        """
        prop_summary = self._propensity_model.summary()
        if not prop_summary:
            return {}
        return {
            "n_items": prop_summary["n_items"],
            "propensity_mean": prop_summary["mean"],
            "propensity_std": prop_summary["std"],
            "propensity_min": prop_summary["min"],
            "propensity_max": prop_summary["max"],
            "blend_lambda": self.blend_lambda,
            "clip_max": self.clip_max,
        }
