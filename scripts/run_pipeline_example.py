#!/usr/bin/env python3
"""Lightweight smoke-test runner for the repository's recommender pipeline.

Loads the pinned local example dataset, adapts and preprocesses it through the
existing repository pipeline, builds the content + collaborative + hybrid
recommenders, and prints a deterministic top-5 recommendation list.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_manager import DatasetManager
from src.model.collaborative_model import CollaborativeRecommender
from src.model.content_model import ContentRecommender
from src.model.hybrid_model import HybridRecommender


def build_example_pipeline(dataset_path: Path):
    """Load and prepare the pinned example dataset using repo APIs."""
    manager = DatasetManager()
    manager.load_csv(str(dataset_path), name=dataset_path.name, catalog="example_small")
    interaction_df, item_df = manager.merge_all()

    if item_df.empty or interaction_df.empty:
        raise ValueError("Example dataset did not load into a usable DataFrame.")

    if "review_count" not in item_df.columns:
        item_df["review_count"] = interaction_df.groupby("title").size().reindex(item_df["title"]).fillna(0).astype(int).values
    if "avg_sentiment" not in item_df.columns:
        item_df["avg_sentiment"] = 0.0

    content_model = ContentRecommender(item_df)
    collab_df = interaction_df[["user_id", "title", "rating"]].copy()
    collab_model = CollaborativeRecommender(collab_df)
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

    return interaction_df, item_df, content_model, collab_model, hybrid_model


def main() -> int:
    random.seed(42)
    np.random.seed(42)

    dataset_path = ROOT / "datasets" / "example_small.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Example dataset not found: {dataset_path}")

    _, _, content_model, collab_model, hybrid_model = build_example_pipeline(dataset_path)

    source_title = "Aurora Wireless Headphones"
    user_id = "user_02"

    content_recs = content_model.recommend(source_title, top_n=5)
    collab_recs = collab_model.recommend(source_title, top_n=5)
    hybrid_recs = hybrid_model.recommend(source_title, user_id=user_id, top_n=5)

    if not hybrid_recs:
        raise RuntimeError("Hybrid recommender returned no recommendations for the example dataset.")

    print("Hybrid recommender smoke test")
    print(f"Dataset: {dataset_path.name}")
    print(f"Query title: {source_title}")
    print(f"User: {user_id}")
    print(f"Content candidates: {len(content_recs)}")
    print(f"Collaborative candidates: {len(collab_recs)}")
    print("Top recommendations:")
    for i, rec in enumerate(hybrid_recs, start=1):
        print(
            f"  {i}. {rec['title']} | "
            f"hybrid={rec['hybrid_score']:.4f} | "
            f"content={rec.get('content_score', 0.0):.4f} | "
            f"collab={rec.get('collab_score', 0.0):.4f}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - intentional CLI failure path
        print(f"Pipeline example failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
