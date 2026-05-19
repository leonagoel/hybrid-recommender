"""
Evaluation Script — Precision@K, Recall@K, NDCG@K
Compares: content-only, collaborative-only, and hybrid (different weight configs).
"""
import os
import sys
import numpy as np
import pandas as pd
from math import log2

sys.path.insert(0, os.path.dirname(__file__))

from dataset_manager import DatasetManager
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender


def precision_at_k(recommended, relevant, k):
    """Proportion of top-K recommendations that are relevant."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0


def recall_at_k(recommended, relevant, k):
    """Proportion of relevant items found in top-K recommendations."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0


def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain @ K."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / log2(i + 2)  # i+2 because log2(1) = 0
    # Ideal DCG
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0


def evaluate():
    """Run the full evaluation pipeline."""
    # 1. Load data
    dm = DatasetManager()
    data_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    
    # Try to load all user-provided datasets first
    datasets_to_load = ['books.csv', 'booksdata.csv', 'ratings.csv']
    loaded_any = False
    
    for filename in datasets_to_load:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading dataset: {filename}...")
            dm.load_csv(filepath)
            loaded_any = True
            
    # Fallback to sample data if no user datasets found
    if not loaded_any:
        sample_file = os.path.join(data_dir, 'sample_products.csv')
        if not os.path.exists(sample_file):
            print("ERROR: datasets not found. Run: python scripts/generate_sample_data.py")
            return
        print("Loading sample_products.csv...")
        dm.load_csv(sample_file)

    interaction_df, item_df = dm.merge_all()
    # Create synthetic users if dataset has only one user
    if interaction_df['user_id'].nunique() <= 1:
     print("Generating synthetic users for evaluation...")

    interaction_df = interaction_df.copy()

    synthetic_users = []
    num_fake_users = 50

    for i in range(num_fake_users):
        temp = interaction_df.sample(
            min(40, len(interaction_df)),
            replace=True,
            random_state=i
        ).copy()

        temp['user_id'] = f"user_{i}"
        synthetic_users.append(temp)

    interaction_df = pd.concat(synthetic_users, ignore_index=True)

    print("Synthetic users created:",
          interaction_df['user_id'].nunique())
    print("Running NLP Sentiment Analysis on reviews...")
    interaction_df = batch_analyze(interaction_df.head(2000), 'review_text')
    sentiment_agg = aggregate_sentiment_by_item(interaction_df, 'title')
    item_df = item_df.merge(sentiment_agg, on='title', how='left')
    item_df['avg_sentiment'] = item_df['avg_sentiment'].fillna(0.0)

    # 2. Train-test split (leave-one-out per user)
    # For each user, hold out their highest-rated item as "ground truth"
    user_groups = interaction_df.groupby('user_id')
    test_pairs = []
    for user_id, group in user_groups:
        if len(group) < 2:
            continue
        top_item = group.sort_values('rating', ascending=False).iloc[0]['title']
        # Relevant items = items this user rated >= 3
        relevant = group[group['rating'] >= 3]['title'].tolist()
        if relevant:
            test_pairs.append((user_id, top_item, relevant))
        print("Interaction rows:", len(interaction_df))
        print("Unique users:", interaction_df['user_id'].nunique())

        user_counts = interaction_df.groupby('user_id').size()
        print(user_counts.describe())

        if not test_pairs:
            print("Not enough data for evaluation.")
            return  

    # 3. Build models
    content_model = ContentRecommender(item_df)
    collab_model = CollaborativeRecommender(interaction_df)

    configs = [
    ("Alpha 0.3", 0.3, 0.7, 0.0),
    ("Alpha 0.5", 0.5, 0.5, 0.0),
    ("Alpha 0.7", 0.7, 0.3, 0.0),
    ]

    K = 10

    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT — Precision@{K}, Recall@{K}, NDCG@{K}")
    print(f"  Test cases: {len(test_pairs)} users")
    print(f"{'='*70}\n")

    results_table = []

    for config_name, a, b, g in configs:
        hybrid = HybridRecommender(content_model, collab_model, item_df, a, b, g)

        precisions, recalls, ndcgs = [], [], []

        for user_id, query_item, relevant_items in test_pairs:
            recs_raw = hybrid.recommend(query_item, top_n=K)
            rec_titles = [r['title'] for r in recs_raw]

            precisions.append(precision_at_k(rec_titles, relevant_items, K))
            recalls.append(recall_at_k(rec_titles, relevant_items, K))
            ndcgs.append(ndcg_at_k(rec_titles, relevant_items, K))

        avg_p = np.mean(precisions)
        avg_r = np.mean(recalls)
        avg_n = np.mean(ndcgs)

        results_table.append((config_name, avg_p, avg_r, avg_n))
        print(f"  {config_name:30s}  P@{K}: {avg_p:.4f}  R@{K}: {avg_r:.4f}  NDCG@{K}: {avg_n:.4f}")

    print(f"\n{'='*70}")

    # Find best config
    best = max(results_table, key=lambda x: x[3])  # best NDCG
    print(f"\n  ★ Best config (by NDCG@{K}): {best[0]}")
    print(f"    Precision: {best[1]:.4f}  Recall: {best[2]:.4f}  NDCG: {best[3]:.4f}\n")


if __name__ == '__main__':
    evaluate()
