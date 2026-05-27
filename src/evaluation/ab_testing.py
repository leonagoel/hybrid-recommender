"""
A/B testing helpers for recommendation ranking experiments.

The framework keeps user assignment deterministic and temporarily applies
variant weights while preserving the recommender's configured defaults.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


DEFAULT_EXPERIMENT_ID = "recommendation-ranking-v1"


@dataclass(frozen=True)
class ExperimentVariant:
    """One recommendation ranking variant in an experiment."""

    name: str
    description: str
    weights: Mapping[str, float]
    traffic: int = 1


DEFAULT_VARIANTS = (
    ExperimentVariant(
        name="control",
        description="Balanced production ranking weights.",
        weights={"alpha": 0.4, "beta": 0.35, "gamma": 0.25},
    ),
    ExperimentVariant(
        name="content_heavy",
        description="Prioritizes item metadata similarity.",
        weights={"alpha": 0.6, "beta": 0.25, "gamma": 0.15},
    ),
    ExperimentVariant(
        name="collaborative_heavy",
        description="Prioritizes co-purchase and user behavior signals.",
        weights={"alpha": 0.25, "beta": 0.6, "gamma": 0.15},
    ),
    ExperimentVariant(
        name="sentiment_heavy",
        description="Prioritizes review sentiment and satisfaction signals.",
        weights={"alpha": 0.3, "beta": 0.25, "gamma": 0.45},
    ),
)


def _stable_bucket(experiment_id: str, user_key: str) -> int:
    digest = hashlib.sha256(f"{experiment_id}:{user_key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def assign_variant(
    user_key: str,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    variants: Iterable[ExperimentVariant] = DEFAULT_VARIANTS,
) -> ExperimentVariant:
    """Assign a user to a variant with deterministic weighted bucketing."""
    variant_list = list(variants)
    if not variant_list:
        raise ValueError("At least one experiment variant is required.")

    total_traffic = sum(max(0, variant.traffic) for variant in variant_list)
    if total_traffic <= 0:
        raise ValueError("At least one experiment variant must receive traffic.")

    bucket = _stable_bucket(experiment_id, str(user_key)) % total_traffic
    cumulative = 0
    for variant in variant_list:
        cumulative += max(0, variant.traffic)
        if bucket < cumulative:
            return variant

    return variant_list[-1]


@contextmanager
def temporary_weights(recommender, weights: Mapping[str, float]):
    """Apply weights for one recommendation call, then restore originals."""
    original_weights = recommender.get_weights()
    recommender.set_weights(
        weights["alpha"],
        weights["beta"],
        weights["gamma"],
    )
    try:
        yield recommender.get_weights()
    finally:
        recommender.set_weights(
            original_weights["alpha"],
            original_weights["beta"],
            original_weights["gamma"],
        )


def run_recommendation_experiment(
    recommender,
    title: str,
    user_key: str,
    top_n: int = 10,
    explain: bool = False,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    variants: Iterable[ExperimentVariant] = DEFAULT_VARIANTS,
) -> Dict[str, object]:
    """Return recommendations plus A/B metadata for an opt-in user request."""
    variant = assign_variant(user_key, experiment_id=experiment_id, variants=variants)
    recommendations = recommender.recommend(
        title, top_n=top_n, explain=explain, weights=variant.weights
    )

    return {
        "experiment": {
            "id": experiment_id,
            "user_key": str(user_key),
            "variant": variant.name,
            "description": variant.description,
            "weights": variant.weights,
        },
        "recommendations": recommendations,
    }


def summarize_variant_metrics(
    events: Iterable[Mapping[str, object]],
    metric_name: str = "clicked",
) -> List[Dict[str, object]]:
    """
    Aggregate simple binary/number metrics by experiment variant.

    Events are intentionally plain dictionaries so API, cron, or notebook code
    can reuse the same helper before wiring a permanent analytics store.
    """
    aggregates: Dict[str, Dict[str, float]] = {}
    for event in events:
        variant = str(event.get("variant", "unknown"))
        value = float(event.get(metric_name, 0) or 0)
        row = aggregates.setdefault(variant, {"count": 0, "total": 0.0})
        row["count"] += 1
        row["total"] += value

    summary = []
    for variant, row in sorted(aggregates.items()):
        count = int(row["count"])
        total = row["total"]
        summary.append(
            {
                "variant": variant,
                "count": count,
                "total": round(total, 4),
                "average": round(total / count, 4) if count else 0.0,
            }
        )
    return summary
