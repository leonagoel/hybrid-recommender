"""
ranking_stability.py — Ranking Stability Regression Utility
=============================================================
Provides ranking comparison functions (Top-K Overlap, Kendall Tau,
Position Change) and a CLI/helper to generate a golden reference.
"""

import os
import json
import argparse
import numpy as np
import scipy.stats as stats
import pandas as pd
from pathlib import Path

# Fixed set of query items to monitor for ranking regression
FIXED_QUERIES = [
    "Premium Journal Lite",
    "Wireless Pulse Oximeter Lite",
    "Advanced Headphones One",
    "Smart Sprinkler 360",
    "Premium Plush Toy 360",
    "Ultra Shorts Pro",
    "Wireless Rug Plus",
    "Pro Dumbbell V2",
    "Essential Jump Rope V2",
    "Classic Tool Kit Lite"
]

def calculate_top_k_overlap(list1: list, list2: list, k: int) -> float:
    """Fraction of top-K elements shared between list1 and list2."""
    if k <= 0:
        return 0.0
    sub1 = list1[:k]
    sub2 = list2[:k]
    if not sub1 or not sub2:
        return 0.0
    overlap = set(sub1) & set(sub2)
    return len(overlap) / k

def calculate_kendall_tau(list1: list, list2: list) -> float:
    """
    Computes Kendall's Tau correlation on the rank alignment of items.
    Handles disjoint elements by mapping the union of items to a standard rank vector,
    where missing items are assigned a default rank of K + 1.
    """
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0

    union_items = list(set(list1) | set(list2))
    if len(union_items) < 2:
        return 1.0

    ranks1 = []
    ranks2 = []
    for item in union_items:
        r1 = list1.index(item) if item in list1 else len(list1)
        r2 = list2.index(item) if item in list2 else len(list2)
        ranks1.append(r1)
        ranks2.append(r2)

    tau, _ = stats.kendalltau(ranks1, ranks2)
    if np.isnan(tau):
        return 0.0
    return float(tau)

def calculate_position_changes(list1: list, list2: list) -> dict:
    """Analyzes absolute rank differences for items appearing in both lists."""
    common = set(list1) & set(list2)
    if not common:
        return {"avg_change": 0.0, "max_change": 0.0, "details": {}}

    changes = {}
    for item in common:
        pos1 = list1.index(item)
        pos2 = list2.index(item)
        changes[item] = abs(pos1 - pos2)

    vals = list(changes.values())
    return {
        "avg_change": float(np.mean(vals)),
        "max_change": float(np.max(vals)),
        "details": changes
    }

def compare_rankings(
    old_ranking: list,
    new_ranking: list,
    k: int = 10,
    threshold_overlap: float = 0.8,
    threshold_tau: float = 0.7
) -> dict:
    """
    Compares two rankings and checks if drift remains within threshold limits.
    """
    overlap = calculate_top_k_overlap(old_ranking, new_ranking, k)
    tau = calculate_kendall_tau(old_ranking, new_ranking)
    pos_changes = calculate_position_changes(old_ranking, new_ranking)

    passed = (overlap >= threshold_overlap) and (tau >= threshold_tau)
    return {
        "passed": bool(passed),
        "overlap_ratio": round(overlap, 4),
        "kendall_tau": round(tau, 4),
        "avg_position_change": round(pos_changes["avg_change"], 4),
        "max_position_change": round(pos_changes["max_change"], 4)
    }

def build_recommender(data_path: str):
    """Initializes Content, Collaborative, and Hybrid recommenders deterministically."""
    from src.data.dataset_manager import DatasetManager
    from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
    from src.model.content_model import ContentRecommender
    from src.model.collaborative_model import CollaborativeRecommender
    from src.model.hybrid_model import HybridRecommender

    dm = DatasetManager()
    dm.load_csv(data_path)
    interaction_df, item_df = dm.merge_all()

    # Pre-process sentiment
    interaction_df = batch_analyze(interaction_df, 'review_text')
    sa = aggregate_sentiment_by_item(interaction_df)
    item_df = item_df.merge(sa, on='title', how='left')
    item_df['avg_sentiment'] = item_df['avg_sentiment'].fillna(0)

    cm = ContentRecommender(item_df)
    collab = CollaborativeRecommender(interaction_df)
    hm = HybridRecommender(cm, collab, item_df)
    return hm

def generate_golden_fixture(data_path: str, output_path: str, k: int = 10):
    """Generates golden ranking recommendations for query titles and saves them to a file."""
    print(f"Loading data from {data_path} and building models...")
    hm = build_recommender(data_path)
    
    golden_recs = {}
    print(f"Generating top-{k} recommendations for fixed queries...")
    for query in FIXED_QUERIES:
        try:
            recs = hm.recommend(query, top_n=k)
            golden_recs[query] = [r["title"] for r in recs]
            print(f" - {query}: {[r['title'] for r in recs][:3]}...")
        except Exception as e:
            print(f" Error generating for '{query}': {e}")
            
    payload = {
        "dataset": data_path,
        "k": k,
        "recommendations": golden_recs
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Golden rankings written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking Stability Regression Utility")
    parser.add_argument("--generate-golden", action="store_true", help="Generate golden reference fixture file")
    parser.add_argument("--data-path", type=str, default="datasets/sample_products.csv", help="Path to evaluation dataset")
    parser.add_argument("--output-path", type=str, default="tests/fixtures/ranking_golden.json", help="Path to golden reference JSON")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to record")
    args = parser.parse_args()

    if args.generate_golden:
        generate_golden_fixture(args.data_path, args.output_path, args.k)
    else:
        parser.print_help()
