"""
PropensityModel — Item exposure propensity estimator.
======================================================

Extracted from CausalDebiaser._fit() so propensity estimation is independently
testable, replaceable, and extensible without touching the IPS reweighting logic.

Propensity P(item shown) models the probability that an item appeared in
training data due to confounders (popularity, category dominance) rather than
genuine relevance.  A higher propensity means the item was over-exposed; IPS
reweighting divides by propensity to correct for this.

Estimation strategy
-------------------
Two observable confounders are combined via geometric mean:

1. Popularity signal  — normalized review_count in (0, 1].
   Items with more reviews were shown more often in training data.

2. Category dominance — relative frequency of the item's category in the
   catalog.  Items in over-represented categories appear more in training
   interactions.

Combined propensity = geometric_mean(popularity, category_freq), then
re-centered around 1.0 so the average IPS weight across the catalog is ~1
(no net scale shift after reweighting).

Extending this model
--------------------
Replace or subclass _popularity_signal() / _category_signal() to plug in
richer estimators (e.g. time-decay, user-level exposure, logistic regression)
without changing CausalDebiaser or HybridRecommender.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PropensityModel:
    """
    Estimates item propensity scores from observational catalog data.

    Parameters
    ----------
    df : pd.DataFrame
        Adapted item DataFrame (output of adapt_data).
        Must contain 'title'. Optionally 'review_count', 'category'.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # title → normalized propensity score
        self._scores: dict[str, float] = {}
        if df is not None and not df.empty:
            self._fit(df)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        """Estimate and store propensity scores for all items in df."""
        n = len(df)

        pop_signal = self._popularity_signal(df, n)
        cat_signal = self._category_signal(df, n)

        # Geometric mean — preferred over arithmetic because both signals
        # are multiplicative proxies for exposure probability.
        raw = np.sqrt(pop_signal * cat_signal)

        # Re-center: divide by mean so average IPS weight ≈ 1.
        mean_raw = raw.mean()
        normalized = raw / mean_raw if mean_raw > 0 else np.ones(n)

        for i, row in enumerate(df.itertuples(index=False)):
            title = getattr(row, "title", None)
            if title is not None:
                self._scores[str(title)] = float(normalized[i])

    @staticmethod
    def _popularity_signal(df: pd.DataFrame, n: int) -> np.ndarray:
        """Normalize review_count to (0, 1]. Falls back to uniform if absent."""
        if "review_count" in df.columns:
            counts = df["review_count"].fillna(0).astype(float).values
        else:
            counts = np.ones(n)
        max_count = float(np.clip(counts.max(), 1, None))
        # Add epsilon so no item has zero propensity.
        return (counts / max_count) + 1e-6

    @staticmethod
    def _category_signal(df: pd.DataFrame, n: int) -> np.ndarray:
        """Relative category frequency in catalog. Falls back to uniform if absent."""
        if "category" in df.columns:
            freq_map = (
                df["category"].fillna("").value_counts(normalize=True).to_dict()
            )
            return (
                df["category"].fillna("").map(freq_map).fillna(1.0 / n).values
            )
        return np.ones(n) / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, title: str, default: float = 1.0) -> float:
        """Return the propensity score for a title. Unknown titles return default."""
        return self._scores.get(title, default)

    def get_ips_weight(self, title: str, clip_max: float = 5.0) -> float:
        """
        Return the clipped IPS weight for a title.

        w = clip(1 / p, max=clip_max)

        A higher weight means the item was under-exposed and should be
        boosted; a lower weight means it was over-exposed and should be
        downweighted.
        """
        p = self._scores.get(title, 1.0)
        return float(min(1.0 / max(p, 1e-6), clip_max))

    def all_scores(self) -> dict[str, float]:
        """Return a copy of the full title → propensity mapping."""
        return dict(self._scores)

    def summary(self) -> dict:
        """Descriptive statistics over the propensity distribution."""
        if not self._scores:
            return {}
        values = np.array(list(self._scores.values()))
        return {
            "n_items": len(values),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }
