"""
evaluation.py — Model Performance Benchmarking
===============================================
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
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Mode = Literal["content", "collaborative", "sentiment", "hybrid", "all"]

MetricsDict = dict[str, float]          # {"precision": 0.4, "recall": 0.38, "ndcg": 0.51}
ResultsDict = dict[str, MetricsDict]    # {"content": {...}, "hybrid": {...}, ...}
UNSAFE_CACHE_SUFFIXES = {".pkl", ".pickle"}


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommended items that are relevant."""
    if not relevant or k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def _recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant or k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def _dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def _ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised DCG at K (IDCG assumes all relevant items are at top)."""
    dcg = _dcg_at_k(recommended, relevant, k)
    ideal = _dcg_at_k(list(relevant)[:k], relevant, k)
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Recommendation engine wrappers
# ---------------------------------------------------------------------------

def _get_content_recs(title: str, df: pd.DataFrame, tfidf_matrix, k: int) -> list[str]:
    """Return top-K titles using content-based (TF-IDF cosine) similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1  # exclude self
    top_indices = np.argsort(sim_scores)[::-1][:k]
    return df.iloc[top_indices]["title"].tolist()


def _get_collab_recs(title: str, df: pd.DataFrame, svd_matrix, k: int) -> list[str]:
    """Return top-K titles using collaborative filtering (SVD) similarity."""
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
    """Return top-K titles sorted by VADER sentiment score (descending)."""
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


def _get_hybrid_recs(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix,
    svd_matrix,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
) -> list[str]:
    """Return top-K titles using weighted hybrid score (α·content + β·collab + γ·sentiment)."""
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        idx = df[df["title"] == title].index[0]
    except IndexError:
        return []

    content_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    collab_scores  = cosine_similarity(svd_matrix[idx].reshape(1, -1), svd_matrix).flatten()

    # Normalise sentiment scores to [0, 1]
    sentiment_raw = df.get("sentiment_score", pd.Series(np.zeros(len(df)))).values.astype(float)
    s_min, s_max = sentiment_raw.min(), sentiment_raw.max()
    sentiment_scores = (
        (sentiment_raw - s_min) / (s_max - s_min)
        if s_max != s_min
        else np.zeros_like(sentiment_raw)
    )

    hybrid_scores = alpha * content_scores + beta * collab_scores + gamma * sentiment_scores
    hybrid_scores[idx] = -1  # exclude self

    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    return df.iloc[top_indices]["title"].tolist()


# ---------------------------------------------------------------------------
# Main evaluation function — importable by the FastAPI endpoint
# ---------------------------------------------------------------------------

def run_evaluation(
    k: int = 10,
    mode: Mode = "all",
    weights: dict[str, float] | None = None,
    data_path: str | None = None,
) -> ResultsDict:
    """
    Run Precision@K, Recall@K, NDCG@K evaluation for the requested mode(s).

    Args:
        k:          Number of recommendations to evaluate against.
        mode:       One of "content", "collaborative", "sentiment", "hybrid", "all".
        weights:    Dict with keys "alpha", "beta", "gamma" (used for hybrid mode).
                    Defaults to {"alpha": 0.4, "beta": 0.4, "gamma": 0.2}.
        data_path:  Path to the dataset CSV. Defaults to DATA_PATH env var or
                    "data/products.csv".

    Returns:
        Dict mapping mode name(s) to {"precision": float, "recall": float, "ndcg": float}.

    Raises:
        RuntimeError: If models have not been built yet (matrices not found).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # --- resolve weights ---
    w = {"alpha": 0.4, "beta": 0.4, "gamma": 0.2}
    if weights:
        w.update(weights)

    # --- load data ---
    path = data_path or os.getenv("DATA_PATH", "data/products.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"Dataset not found at '{path}'. Upload a dataset first.")

    df = pd.read_csv(path)

    # Normalise column names — support both "title"/"product_name"
    if "product_name" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"product_name": "title"})

    required = {"title"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["title"]).reset_index(drop=True)

    # --- auto-analyze sentiment if missing ---
    if "sentiment_score" not in df.columns:
        from nlp_engine import batch_analyze
        text_col = "description" if "description" in df.columns else ("review_text" if "review_text" in df.columns else "title")
        df = batch_analyze(df, text_col=text_col)

    # --- build/load matrices ---
    # Try to load pre-built matrices from disk; fall back to building on-the-fly
    tfidf_matrix = _load_or_build_tfidf(df)
    svd_matrix   = _load_or_build_svd(df)

    # --- build relevance sets from rating data ---
    # "relevant" items for a given title = items in same category OR rating >= 4.0
# --- FIXED: True User-Personalization Leave-One-Out Evaluation (#375) ---
    modes_to_run = (
        ["content", "collaborative", "sentiment", "hybrid"]
        if mode == "all"
        else [mode]
    )

    results: ResultsDict = {}
    
    # Check out if user interaction signals exist in the dataset
    has_user_data = "user_id" in df.columns and len(df["user_id"].dropna().unique()) > 1

    if has_user_data:
        # User-based Evaluation Profile
        unique_users = df["user_id"].dropna().unique()
        sample_users = np.random.choice(unique_users, size=min(100, len(unique_users)), replace=False)
    else:
        # Fallback to item index sample if explicit user tracking columns aren't present
        sample_indices = np.random.choice(len(df), size=min(100, len(df)), replace=False)

    for m in modes_to_run:
        precisions, recalls, ndcgs = [], [], []

        if has_user_data:
            # ----------------------------------------------------
            # USER-BASED PERSONALIZATION LOOP (Core Fix)
            # ----------------------------------------------------
            for current_user in sample_users:
                # User ki poori consumption profiles fetch karna
                user_profile = df[df["user_id"] == current_user].reset_index(drop=True)
                if len(user_profile) < 2:
                    continue  # Leave-one-out needs at least 2 items (1 history, 1 held-out)

                # Hold out the last item as the evaluation truth target
                query_item = user_profile.iloc[-1]["title"]
                relevant = {query_item}
                
                # Baki bache items user history seed banenge
                user_history = user_profile.iloc[:-1]["title"].tolist()

                all_recs = {}
                # Extract up to 5 interaction points for high-fidelity evaluation profiling
                for seed_title in user_history[:5]:
                    try:
                        if m == "content":
                            recs_raw = _get_content_recs(seed_title, df, tfidf_matrix, k)
                        elif m == "collaborative":
                            recs_raw = _get_collab_recs(seed_title, df, svd_matrix, k)
                        elif m == "sentiment":
                            recs_raw = _get_sentiment_recs(seed_title, df, k)
                        else:  # hybrid
                            recs_raw = _get_hybrid_recs(
                                seed_title, df, tfidf_matrix, svd_matrix,
                                w["alpha"], w["beta"], w["gamma"], k,
                            )
                        
                        # Blend recommendation confidence arrays
                        for idx_rank, item_name in enumerate(recs_raw):
                            score = 1.0 / (idx_rank + 1)  # Rank-based reciprocal pooling fallback
                            all_recs[item_name] = max(all_recs.get(item_name, 0), score)
                    except Exception:
                        continue

                # Sort aggregated items and filter out historical elements
                sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
                final_recs = [item[0] for item in sorted_recs if item[0] not in user_history][:k]

                if final_recs:
                    precisions.append(_precision_at_k(final_recs, relevant, k))
                    recalls.append(_recall_at_k(final_recs, relevant, k))
                    ndcgs.append(_ndcg_at_k(final_recs, relevant, k))
        else:
            # ----------------------------------------------------
            # FALLBACK: Item similarity processing if dataset is flat
            # ----------------------------------------------------
            for idx in sample_indices:
                title = df.iloc[idx]["title"]
                
                # Establish pseudo-relevance via category boundaries
                relevant = set()
                if "category" in df.columns and pd.notna(df.iloc[idx].get("category")):
                    relevant.update(df[df["category"] == df.iloc[idx]["category"]]["title"].tolist())
                relevant.discard(title)

                if not relevant:
                    continue

                if m == "content":
                    recs = _get_content_recs(title, df, tfidf_matrix, k)
                elif m == "collaborative":
                    recs = _get_collab_recs(title, df, svd_matrix, k)
                elif m == "sentiment":
                    recs = _get_sentiment_recs(title, df, k)
                else:
                    recs = _get_hybrid_recs(
                        title, df, tfidf_matrix, svd_matrix,
                        w["alpha"], w["beta"], w["gamma"], k,
                    )

                precisions.append(_precision_at_k(recs, relevant, k))
                recalls.append(_recall_at_k(recs, relevant, k))
                ndcgs.append(_ndcg_at_k(recs, relevant, k))

        results[m] = {
            "precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall":    round(float(np.mean(recalls)),    4) if recalls    else 0.0,
            "ndcg":      round(float(np.mean(ndcgs)),      4) if ndcgs      else 0.0,
        }

        results[m] = {
            "precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall":    round(float(np.mean(recalls)),    4) if recalls    else 0.0,
            "ndcg":      round(float(np.mean(ndcgs)),      4) if ndcgs      else 0.0,
        }

    return results


# ---------------------------------------------------------------------------
# Matrix helpers — load pre-built or build on-the-fly
# ---------------------------------------------------------------------------

def _load_or_build_tfidf(df: pd.DataFrame):
    """Load TF-IDF matrix from disk if available, else build from scratch."""
    cache_path = Path(os.getenv("TFIDF_CACHE", "models/tfidf_matrix.npz"))
    if cache_path.exists():
        _reject_unsafe_cache(cache_path)
        if cache_path.suffix != ".npz":
            raise RuntimeError("TF-IDF cache must use the safe .npz sparse matrix format.")
        from scipy import sparse
        return sparse.load_npz(cache_path)

    # Build on-the-fly using title + category as text
    text_col = "title"
    if "category" in df.columns:
        df = df.copy()
        df["_text"] = df["title"].fillna("") + " " + df["category"].fillna("")
        text_col = "_text"

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    return vectorizer.fit_transform(df[text_col].fillna(""))


def _load_or_build_svd(df: pd.DataFrame):
    """Load SVD matrix from disk if available, else build from scratch."""
    cache_path = Path(os.getenv("SVD_CACHE", "models/svd_matrix.npy"))
    if cache_path.exists():
        _reject_unsafe_cache(cache_path)
        if cache_path.suffix != ".npy":
            raise RuntimeError("SVD cache must use the safe .npy array format.")
        return np.load(cache_path, allow_pickle=False)

    # Build rating matrix and decompose
    from sklearn.decomposition import TruncatedSVD

    tfidf = _load_or_build_tfidf(df)
    svd = TruncatedSVD(n_components=min(50, tfidf.shape[1] - 1), random_state=42)
    return svd.fit_transform(tfidf)


def _reject_unsafe_cache(cache_path: Path) -> None:
    if cache_path.suffix.lower() in UNSAFE_CACHE_SUFFIXES:
        raise RuntimeError(
            f"Refusing to load unsafe pickle model cache '{cache_path}'. "
            "Use .npz for TF-IDF caches or .npy for SVD caches."
        )


# ---------------------------------------------------------------------------
# CLI entry point — original behaviour preserved
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid recommender models.")
    parser.add_argument("--k",    type=int,   default=10,   help="Number of recommendations (default: 10)")
    parser.add_argument("--mode", type=str,   default="all",
                        choices=["content", "collaborative", "sentiment", "hybrid", "all"],
                        help="Which model(s) to evaluate (default: all)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Content weight (default: 0.4)")
    parser.add_argument("--beta",  type=float, default=0.4, help="Collaborative weight (default: 0.4)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Sentiment weight (default: 0.2)")
    args = parser.parse_args()

    print(f"\n📊 Running evaluation — mode={args.mode}, k={args.k}")
    print(f"   Weights: α={args.alpha} β={args.beta} γ={args.gamma}\n")

    try:
        results = run_evaluation(
            k=args.k,
            mode=args.mode,
            weights={"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        )
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        return

    # Pretty-print results table
    header = f"{'Mode':<16} {'Precision@K':>12} {'Recall@K':>10} {'NDCG@K':>10}"
    print(header)
    print("-" * len(header))
    for mode_name, metrics in results.items():
        print(
            f"{mode_name:<16} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['ndcg']:>10.4f}"
        )
    print()


if __name__ == "__main__":
    _cli()
