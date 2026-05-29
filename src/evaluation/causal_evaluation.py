"""
causal_evaluation.py — Causal Layer Quality Metrics
====================================================

Measures the effect of IPS debiasing on recommendation quality beyond
standard Precision/Recall/NDCG, which do not capture bias reduction.

Three complementary metrics
---------------------------

1. popularity_bias_reduction
   Measures how much the causal layer reduces the average popularity rank
   of recommended items.  A positive value means the causal list surfaces
   less popular (more niche) items — the primary goal of IPS debiasing.

   Formula:
       avg_pop_rank(list) = mean(popularity_rank[item] for item in list)
       reduction = avg_pop_rank(causal) - avg_pop_rank(baseline)
       (positive = causal list is less popular-biased)

2. catalog_coverage_gain
   Fraction of the catalog covered by causal recommendations minus the
   fraction covered by baseline recommendations, across all queries.
   Higher coverage means the causal layer surfaces a broader item set.

3. intra_list_diversity_gain
   Average pairwise category diversity within each recommendation list.
   Diversity = 1 - (items_in_dominant_category / list_length).
   Gain = diversity(causal) - diversity(baseline).

Usage
-----
    from src.evaluation.causal_evaluation import compare_causal_vs_baseline

    results = compare_causal_vs_baseline(
        causal_model=causal_hybrid,
        baseline_model=plain_hybrid,
        item_df=item_df,
        query_titles=sample_titles,
        top_n=10,
    )
    # results is a dict with keys:
    #   popularity_bias_reduction, coverage_gain, diversity_gain,
    #   causal_avg_popularity_rank, baseline_avg_popularity_rank,
    #   causal_coverage, baseline_coverage,
    #   causal_diversity, baseline_diversity,
    #   n_queries
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_popularity_rank(item_df: pd.DataFrame) -> dict[str, float]:
    """
    Map each title to a normalized popularity rank in [0, 1].
    Rank 1.0 = most popular, 0.0 = least popular.
    Uses review_count if available, otherwise uniform.
    """
    if "review_count" not in item_df.columns or item_df.empty:
        titles = item_df["title"].tolist() if "title" in item_df.columns else []
        return {t: 0.5 for t in titles}

    df = item_df[["title", "review_count"]].copy()
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    max_count = df["review_count"].max()
    if max_count == 0:
        return {row["title"]: 0.0 for _, row in df.iterrows()}
    return {
        row["title"]: float(row["review_count"] / max_count)
        for _, row in df.iterrows()
    }


def _build_category_map(item_df: pd.DataFrame) -> dict[str, str]:
    """Map title → category string."""
    if "category" not in item_df.columns:
        return {}
    return dict(zip(item_df["title"].astype(str), item_df["category"].fillna("").astype(str)))


def _avg_popularity_rank(titles: list[str], pop_rank: dict[str, float]) -> float:
    """Mean popularity rank for a list of titles. Unknown titles get 0.5."""
    if not titles:
        return 0.0
    return float(np.mean([pop_rank.get(t, 0.5) for t in titles]))


def _intra_list_diversity(titles: list[str], category_map: dict[str, str]) -> float:
    """
    Diversity = 1 - (count of dominant category / list length).
    Returns 0.0 for empty or single-item lists.
    """
    if len(titles) <= 1:
        return 0.0
    cats = [category_map.get(t, "") for t in titles]
    most_common_count = Counter(cats).most_common(1)[0][1]
    return 1.0 - (most_common_count / len(titles))


def _get_rec_titles(model, query: str, top_n: int) -> list[str]:
    """Call model.recommend() and extract title strings safely."""
    try:
        recs = model.recommend(query, top_n=top_n)
        return [r["title"] for r in recs if "title" in r]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_causal_vs_baseline(
    causal_model,
    baseline_model,
    item_df: pd.DataFrame,
    query_titles: list[str],
    top_n: int = 10,
) -> dict:
    """
    Compare causal vs. baseline recommendation lists across a set of queries.

    Parameters
    ----------
    causal_model   : HybridRecommender with causal debiasing enabled.
    baseline_model : HybridRecommender with causal debiasing disabled.
    item_df        : Adapted item DataFrame (used to build popularity + category maps).
    query_titles   : List of item titles to use as recommendation seeds.
    top_n          : Number of recommendations per query.

    Returns
    -------
    dict with keys:
        popularity_bias_reduction  : float — positive means causal is less biased
        coverage_gain              : float — positive means causal covers more catalog
        diversity_gain             : float — positive means causal lists are more diverse
        causal_avg_popularity_rank : float
        baseline_avg_popularity_rank : float
        causal_coverage            : float
        baseline_coverage          : float
        causal_diversity           : float
        baseline_diversity         : float
        n_queries                  : int — number of queries that returned results
    """
    pop_rank = _build_popularity_rank(item_df)
    category_map = _build_category_map(item_df)
    catalog_size = len(item_df) if not item_df.empty else 1

    causal_pop_ranks: list[float] = []
    baseline_pop_ranks: list[float] = []
    causal_covered: set[str] = set()
    baseline_covered: set[str] = set()
    causal_diversities: list[float] = []
    baseline_diversities: list[float] = []
    n_queries = 0

    for query in query_titles:
        causal_titles = _get_rec_titles(causal_model, query, top_n)
        baseline_titles = _get_rec_titles(baseline_model, query, top_n)

        if not causal_titles and not baseline_titles:
            continue

        n_queries += 1
        causal_pop_ranks.append(_avg_popularity_rank(causal_titles, pop_rank))
        baseline_pop_ranks.append(_avg_popularity_rank(baseline_titles, pop_rank))
        causal_covered.update(causal_titles)
        baseline_covered.update(baseline_titles)
        causal_diversities.append(_intra_list_diversity(causal_titles, category_map))
        baseline_diversities.append(_intra_list_diversity(baseline_titles, category_map))

    causal_avg_pop = float(np.mean(causal_pop_ranks)) if causal_pop_ranks else 0.0
    baseline_avg_pop = float(np.mean(baseline_pop_ranks)) if baseline_pop_ranks else 0.0
    causal_cov = len(causal_covered) / catalog_size
    baseline_cov = len(baseline_covered) / catalog_size
    causal_div = float(np.mean(causal_diversities)) if causal_diversities else 0.0
    baseline_div = float(np.mean(baseline_diversities)) if baseline_diversities else 0.0

    return {
        # Primary metrics — positive = causal layer is better
        "popularity_bias_reduction": round(baseline_avg_pop - causal_avg_pop, 4),
        "coverage_gain": round(causal_cov - baseline_cov, 4),
        "diversity_gain": round(causal_div - baseline_div, 4),
        # Raw values for inspection
        "causal_avg_popularity_rank": round(causal_avg_pop, 4),
        "baseline_avg_popularity_rank": round(baseline_avg_pop, 4),
        "causal_coverage": round(causal_cov, 4),
        "baseline_coverage": round(baseline_cov, 4),
        "causal_diversity": round(causal_div, 4),
        "baseline_diversity": round(baseline_div, 4),
        "n_queries": n_queries,
    }


def score_key_distribution(
    model,
    item_df: pd.DataFrame,
    query_titles: list[str],
    top_n: int = 10,
    score_key: str = "hybrid_score",
) -> dict:
    """
    Compute descriptive statistics of a score field across all recommendation
    lists.  Useful for verifying that causal scores stay in [0, 1] and that
    the distribution is not degenerate (e.g. all zeros or all ones).

    Returns
    -------
    dict with mean, std, min, max, and n_scores.
    """
    all_scores: list[float] = []
    for query in query_titles:
        try:
            recs = model.recommend(query, top_n=top_n)
            all_scores.extend(
                float(r[score_key]) for r in recs if score_key in r
            )
        except Exception:
            continue

    if not all_scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_scores": 0}

    arr = np.array(all_scores)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "n_scores": len(all_scores),
    }
