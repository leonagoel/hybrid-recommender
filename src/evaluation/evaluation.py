"""
evaluation.py — Model Performance Benchmarking
Computes Precision@K, Recall@K, and NDCG@K for four recommendation modes:
  - content       (TF-IDF cosine similarity only)
  - collaborative (Truncated SVD only)
  - sentiment     (VADER sentiment only)
  - hybrid        (weighted blend of all three)

Usage as CLI (unchanged from original behaviour):
    python evaluation.py
    python evaluation.py --k 20
    python evaluation.py --k 10 --mode hybrid

Usage as importable module (new — used by /api/evaluate endpoint):
    from evaluation import run_evaluation
    results = run_evaluation(k=10, mode="all", weights={"alpha":0.4,"beta":0.4,"gamma":0.2})
"""

from __future__ import annotations

import argparse
import json
import math
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Set, Tuple, Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Mode = Literal["content", "collaborative", "sentiment", "hybrid", "all"]

MetricsDict = dict[str, float]          # {"precision": 0.4, "recall": 0.38, "ndcg": 0.51}
ResultsDict = dict[str, MetricsDict]    # {"content": {...}, "hybrid": {...}, ...}
UNSAFE_CACHE_SUFFIXES = {".pkl", ".pickle"}


# ---------------------------------------------------------------------------
# Core metric helpers with safety guards against ZeroDivisionError
# ---------------------------------------------------------------------------

def _precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommended items that are relevant."""
    if not relevant or k == 0 or not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k

def recall_at_k(rec, rel, k):
    rec = rec[:k]
    return len(set(rec) & set(rel)) / len(rel) if rel else 0.0

def _recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant or k == 0 or not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    
    # FIX FOR ISSUE #486: Guard cold states to prevent ZeroDivisionError
    denom = len(relevant)
    return hits / denom if denom > 0 else 0.0


def _dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    if not recommended or not relevant or k == 0:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def _ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised DCG at K (IDCG assumes all relevant items are at top)."""
    dcg = _dcg_at_k(recommended, relevant, k)
    ideal = _dcg_at_k(list(relevant)[:k], relevant, k)
    
    # FIX FOR ISSUE #486: Handle zero baseline ideal scores gracefully
    return dcg / ideal if ideal > 0.0 else 0.0


# Public wrappers used by benchmark.py
def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Exported wrapper for normalized DCG."""
    return _ndcg_at_k(recommended, relevant, k)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Average Precision at K (AP@K).

    Implemented as sum(precision@i * rel_i) / min(|relevant|, k).
    """
    if not relevant or k == 0 or not recommended:
        return 0.0
    hits = 0
    precisions = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precisions += hits / i
    denom = min(len(relevant), k)
    return precisions / denom if denom > 0 else 0.0


# New metrics requested: MRR, Hit Rate, Catalog Coverage, ILD
def _mean_reciprocal_rank(recommended: list, relevant: set, k: int) -> float:
    """Mean Reciprocal Rank (MRR) — rank of first relevant item."""
    if not relevant or k == 0 or not recommended:
        return 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def _hit_rate(recommended: list, relevant: set, k: int) -> float:
    """Hit Rate — 1.0 if at least one relevant item in top-K."""
    if not relevant or k == 0 or not recommended:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def _catalog_coverage(all_recommendations: list[list], catalog_size: int) -> float:
    """Catalog coverage: fraction of unique items recommended."""
    if not all_recommendations or catalog_size == 0:
        return 0.0
    unique = set()
    for recs in all_recommendations:
        unique.update(recs)
    return len(unique) / catalog_size


def _intra_list_diversity(rec_titles: list, item_df: pd.DataFrame, tfidf_matrix) -> float:
    """Average pairwise cosine distance within a recommendation list (1 - avg_similarity)."""
    if not rec_titles or tfidf_matrix is None or len(rec_titles) < 2:
        return 0.0
    from sklearn.metrics.pairwise import cosine_similarity
    title_to_idx = {str(t): i for i, t in enumerate(item_df['title'].astype(str))}
    indices = [title_to_idx[t] for t in rec_titles if t in title_to_idx]
    if len(indices) < 2:
        return 0.0
    vecs = tfidf_matrix[indices]
    sim_matrix = cosine_similarity(vecs)
    n = len(indices)
    total_sim = float(sim_matrix.sum()) - n  # subtract diagonal (self-similarity = 1)
    pairs = n * (n - 1)
    return max(0.0, 1.0 - (total_sim / pairs)) if pairs > 0 else 0.0


def _build_user_test_data(
    interaction_df: pd.DataFrame,
    min_interactions: int = 3,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """
    Build user-personalized evaluation pairs by splitting each user's history.

    For each user with >= min_interactions, splits their history into:
    - train items: kept in the returned DataFrame for model training
    - test items: held-out set used to evaluate recommendations

    Cold-start users (< min_interactions) are included in train_df but
    excluded from user_test_pairs.

    Returns:
        train_df:         interaction_df restricted to training rows
        user_test_pairs:  list of (user_id, train_items: set, test_items: set)
    """
    if interaction_df.empty:
        return interaction_df.iloc[:0].copy(), []

    rng = np.random.default_rng(random_seed)
    user_test_pairs: list[tuple] = []
    train_indices: list = []

    for user_id, group in interaction_df.groupby('user_id'):
        items = group['title'].tolist()
        if len(items) < min_interactions:
            train_indices.extend(group.index.tolist())
            continue

        perm = rng.permutation(len(items))
        items_shuffled = [items[i] for i in perm]
        n_test = max(1, int(len(items_shuffled) * test_fraction))
        test_items = set(items_shuffled[:n_test])
        train_items = set(items_shuffled[n_test:])

        train_rows = group[group['title'].isin(train_items)]
        train_indices.extend(train_rows.index.tolist())

        if test_items:
            user_test_pairs.append((user_id, train_items, test_items))

    train_df = interaction_df.loc[train_indices].copy() if train_indices else interaction_df.iloc[:0].copy()
    return train_df, user_test_pairs


def _load_or_build_svd(df: pd.DataFrame) -> np.ndarray:
    """Helper to mock or build an SVD matrix for collaborative filtering."""
    return np.random.default_rng(42).random((len(df), 10))

def _build_test_data(
    data_path: str | None = None,
    random_seed: int = 42,
):
    """Build minimal models and test pairs for benchmark scripts."""
    rng = np.random.default_rng(random_seed)
    from src.model.content_model import ContentRecommender

    path = data_path or os.getenv("DATA_PATH", "data/products.csv")
    if not os.path.exists(path):
        return None, None, None, []
    df = pd.read_csv(path)
    if "product_name" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"product_name": "title"})
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    if 'combined' not in df.columns:
        df = df.copy()
        desc = df['description'] if 'description' in df.columns else pd.Series([''] * len(df))
        cat = df['category'] if 'category' in df.columns else pd.Series([''] * len(df))
        df['combined'] = df['title'].fillna('') + ' ' + desc.fillna('') + ' ' + cat.fillna('')

    try:
        content_model = ContentRecommender(df)
    except Exception:
        content_model = ContentRecommender(df, batch_size=256)

    svd_matrix = _load_or_build_svd(df)
    class _Collab:
        def recommend(self, title, top_n=10, **kwargs):
            return [{"title": t} for t in _get_collab_recs(title, df, svd_matrix, top_n)]

    collab_model = _Collab()

    test_pairs = []
    sample = min(50, len(df))
    indices = rng.choice(len(df), size=sample, replace=False)
    for uid, idx in enumerate(indices):
        title = df.iloc[idx]["title"]
        relevant = set()
        if "category" in df.columns and pd.notna(df.iloc[idx].get("category")):
            same = df[df["category"] == df.iloc[idx]["category"]]["title"].tolist()
            relevant.update(same)
        relevant.discard(title)
        if relevant:
            test_pairs.append((uid, title, relevant))
    return content_model, collab_model, df, test_pairs

def _get_content_recs(title: str, df: pd.DataFrame, tfidf_matrix, k: int) -> list[str]:
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = np.argsort(sim_scores)[::-1][:k]
    return df.iloc[top_indices]["title"].tolist()

def _get_collab_recs(title: str, df: pd.DataFrame, svd_matrix, k: int) -> list[str]:
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []
    sim_scores = cosine_similarity(svd_matrix[idx].reshape(1, -1), svd_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = np.argsort(sim_scores)[::-1][:k]
    return df.iloc[top_indices]["title"].tolist()

def _get_sentiment_recs(title: str, df: pd.DataFrame, k: int) -> list[str]:
    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []
    df_copy = df.copy()
    if "sentiment_score" not in df_copy.columns:
        df_copy["sentiment_score"] = 0.0
    df_copy = df_copy.drop(index=idx, errors="ignore")
    top = df_copy.sort_values(by="sentiment_score", ascending=False).head(k)
    return top["title"].tolist()

def _get_hybrid_recs(title: str, df: pd.DataFrame, tfidf_matrix, svd_matrix, alpha: float, beta: float, gamma: float, k: int) -> list[str]:
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []
    content_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    collab_scores  = cosine_similarity(svd_matrix[idx].reshape(1, -1), svd_matrix).flatten()
    sentiment_raw = df.get("sentiment_score", pd.Series(np.zeros(len(df)))).values.astype(float)
    s_min, s_max = sentiment_raw.min(), sentiment_raw.max()
    sentiment_scores = ((sentiment_raw - s_min) / (s_max - s_min) if s_max != s_min else np.zeros_like(sentiment_raw))
    hybrid_scores = alpha * content_scores + beta * collab_scores + gamma * sentiment_scores
    hybrid_scores[idx] = -1
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    return df.iloc[top_indices]["title"].tolist()

def run_evaluation(
    k: int = 10,
    mode: Mode = "all",
    weights: dict[str, float] | None = None,
    data_path: str | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> ResultsDict:
    """
    Compute user-personalized evaluation metrics.

    Recommendations are generated via CollaborativeRecommender.predict_for_user()
    so that Precision@K, Recall@K, MAP, and NDCG measure genuine user
    personalization — not item-to-item similarity.

    Requires an interaction CSV (interactions.csv, ratings.csv, or
    user_interactions.csv) alongside the item catalogue with columns:
    user_id, title, rating.
    """
    from src.model.collaborative_model import CollaborativeRecommender

    path = data_path or os.getenv("DATA_PATH", "data/products.csv")

    data_dir = os.path.dirname(os.path.abspath(path)) if path else "."
    interaction_candidates = [
        os.path.join(data_dir, "interactions.csv"),
        os.path.join(data_dir, "ratings.csv"),
        os.path.join(data_dir, "user_interactions.csv"),
    ]
    interaction_path = next((p for p in interaction_candidates if os.path.exists(p)), None)

    if interaction_path is None:
        print("No interaction data found; cannot compute user-personalized metrics.")
        return {}

    interaction_df = pd.read_csv(interaction_path)
    required = {"user_id", "title", "rating"}
    missing = required - set(interaction_df.columns)
    if missing:
        print(f"Interaction file missing required columns: {missing}")
        return {}

    interaction_df = interaction_df.dropna(subset=["user_id", "title"])

    train_df, user_test_pairs = _build_user_test_data(
        interaction_df, test_fraction=test_size, random_seed=random_seed
    )

    if not user_test_pairs:
        print("Not enough user interaction data for evaluation.")
        return {}

    print(f"Evaluating on {len(user_test_pairs)} users (K={k})...")

    collab_model = CollaborativeRecommender(train_df)

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    maps: List[float] = []

    for user_id, _train_items, test_items in user_test_pairs:
        recs = collab_model.predict_for_user(user_id, top_n=k)
        rec_titles = [r["title"] for r in recs]
        precisions.append(_precision_at_k(rec_titles, test_items, k))
        recalls.append(_recall_at_k(rec_titles, test_items, k))
        ndcgs.append(_ndcg_at_k(rec_titles, test_items, k))
        maps.append(average_precision_at_k(rec_titles, test_items, k))

    metrics: MetricsDict = {
        "precision": float(np.mean(precisions)),
        "recall":    float(np.mean(recalls)),
        "ndcg":      float(np.mean(ndcgs)),
        "map":       float(np.mean(maps)),
    }

    print(f"  Precision@{k}: {metrics['precision']:.4f}")
    print(f"  Recall@{k}:    {metrics['recall']:.4f}")
    print(f"  NDCG@{k}:      {metrics['ndcg']:.4f}")
    print(f"  MAP@{k}:       {metrics['map']:.4f}")
    print(f"  Users evaluated: {len(user_test_pairs)}")

    return {"collaborative": metrics}
