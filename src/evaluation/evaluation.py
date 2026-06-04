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
# Core metric helpers with safety guards against ZeroDivisionError
# ---------------------------------------------------------------------------

def _precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommended items that are relevant."""
    if not relevant or k == 0 or not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


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


def _intra_list_diversity(
    recommended: list[str], df: pd.DataFrame, tfidf_matrix
) -> float:
    """Intra-List Diversity (ILD) using TF-IDF cosine dissimilarity.

    Returns 1 - average_pairwise_similarity. If TF-IDF matrix is not
    available or list too small, returns 0.0.
    """
    if tfidf_matrix is None or len(recommended) <= 1:
        return 0.0
    from sklearn.metrics.pairwise import cosine_similarity

    indices = []
    for title in recommended:
        try:
            idx = df[df["title"] == title].index[0]
            indices.append(idx)
        except IndexError:
            continue
    if len(indices) <= 1:
        return 0.0
    sims = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = cosine_similarity(
                tfidf_matrix[indices[i]], tfidf_matrix[indices[j]]
            )[0, 0]
            sims.append(sim)
    avg_sim = float(np.mean(sims)) if sims else 0.0
    return 1.0 - avg_sim


# Small helper used by benchmark to build models/test pairs
def _build_test_data(
    data_path: str | None = None,
    random_seed: int = 42,
):
    """Build minimal models and test pairs for benchmark scripts.

    Uses a fixed ``random_seed`` so that repeated calls with the same dataset
    produce the same test pairs, making benchmark comparisons stable.

    Returns (content_model, collab_model, df, test_pairs).
    """
    from src.model.content_model import ContentRecommender

    path = data_path or os.getenv("DATA_PATH", "data/products.csv")
    if not os.path.exists(path):
        return None, None, None, []
    df = pd.read_csv(path)
    if "product_name" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"product_name": "title"})
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    # ContentRecommender expects a 'combined' text column (title+desc+category).
    # Ensure it's present so the constructor can encode texts.
    if 'combined' not in df.columns:
        df = df.copy()
        desc = df['description'] if 'description' in df.columns else pd.Series([''] * len(df))
        cat = df['category'] if 'category' in df.columns else pd.Series([''] * len(df))
        df['combined'] = df['title'].fillna('') + ' ' + desc.fillna('') + ' ' + cat.fillna('')

    # ContentRecommender expects the item dataframe and builds
    # its own embedding matrix internally.
    try:
        content_model = ContentRecommender(df)
    except Exception:
        # Fallback: if constructor signature differs, pass both
        content_model = ContentRecommender(df, batch_size=256)

    svd_matrix = _load_or_build_svd(df)
    class _Collab:
        def recommend(self, title, top_n=10, **kwargs):
            return [{"title": t} for t in _get_collab_recs(title, df, svd_matrix, top_n)]

    collab_model = _Collab()

    # build simple test pairs — deterministic via random_seed
    rng = np.random.default_rng(random_seed)
    test_pairs = []
    sample = min(50, len(df))
    indices = rng.choice(len(df), size=sample, replace=False)
    for uid, idx in enumerate(indices):
        title = df.iloc[idx]["title"]
        relevant = set()
        if "category" in df.columns and pd.notna(df.iloc[idx].get("category")):
            same = df[df["category"] == df.iloc[idx]["category"]]["title"].tolist()
            relevant.update(same)
        if "rating" in df.columns:
            relevant.update(df[df["rating"] >= 4.0]["title"].tolist())
        relevant.discard(title)
        if relevant:
            test_pairs.append((uid, title, relevant))
    return content_model, collab_model, df, test_pairs


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
    # Fixed indentation for evaluation workflow as requested
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
    test_size: float = 0.2,
    random_seed: int = 42,
) -> ResultsDict:
    """
    Run Precision@K, Recall@K, NDCG@K evaluation for the requested mode(s).

    Train / test separation
    -----------------------
    The dataset is split into a training portion (1 - test_size) and a
    held-out test portion (test_size).  TF-IDF vocabulary / IDF weights and
    the SVD latent factors are fitted **only on the training split**.  Test
    items are then *transformed* (not fitted) with those trained artifacts,
    exactly mirroring how the system would treat unseen items in production.

    Recommendations for each test query are drawn from the training catalog,
    so no test item can appear as both a query and a candidate used during
    training, eliminating train / test leakage.

    Reproducibility
    ---------------
    All random operations use ``np.random.default_rng(random_seed)``
    (PCG64 generator).  Identical inputs always produce identical results.

    Parameters
    ----------
    k           : number of recommendations per query
    mode        : 'content', 'collaborative', 'sentiment', 'hybrid', or 'all'
    weights     : override hybrid weights {alpha, beta, gamma}
    data_path   : path to CSV dataset; falls back to DATA_PATH env var
    test_size   : fraction of rows held out for evaluation (default 0.2).
                  Must be strictly between 0 and 1.
    random_seed : seed for all random operations (default 42)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

    # --- validate parameters ---
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0 and 1.")

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

    if len(df) < 5:
        raise RuntimeError(
            "Dataset is too small for evaluation (need at least 5 items)."
        )

    # --- auto-analyze sentiment if missing ---
    if "sentiment_score" not in df.columns:
        try:
            from src.model.nlp_engine import batch_analyze
        except ModuleNotFoundError:
            # nltk or nlp dependencies not installed — fall back to neutral scores
            df["sentiment_score"] = 0.0
        else:
            text_col = (
                "description"
                if "description" in df.columns
                else ("review_text" if "review_text" in df.columns else "title")
            )
            df = batch_analyze(df, text_col=text_col)

    # -----------------------------------------------------------------------
    # Train / test split
    # -----------------------------------------------------------------------
    # Attempt stratification by category so both splits share the same
    # category distribution.  Stratification fails if any category has
    # fewer than 2 members; fall back to a plain random split in that case.
    try:
        stratify = (
            df["category"].fillna("__none__")
            if "category" in df.columns
            else None
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
        )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Build text feature column (title + category) for both splits
    # -----------------------------------------------------------------------
    def _add_text_col(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "category" in frame.columns:
            frame["_eval_text"] = (
                frame["title"].fillna("") + " " + frame["category"].fillna("")
            )
        else:
            frame["_eval_text"] = frame["title"].fillna("")
        return frame

    train_df = _add_text_col(train_df)
    test_df  = _add_text_col(test_df)

    # -----------------------------------------------------------------------
    # TF-IDF: fit on training split only, transform test split separately
    # -----------------------------------------------------------------------
    # We do NOT use _load_or_build_tfidf() here because those cached matrices
    # were fitted on the full dataset and would re-introduce leakage.
    tfidf_vec   = TfidfVectorizer(stop_words="english", max_features=5000)
    train_tfidf = tfidf_vec.fit_transform(train_df["_eval_text"])
    test_tfidf  = tfidf_vec.transform(test_df["_eval_text"])

    # -----------------------------------------------------------------------
    # SVD: fit on training TF-IDF only, transform test TF-IDF separately
    # -----------------------------------------------------------------------
    n_components = max(
        1, min(50, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1)
    )
    svd_model   = TruncatedSVD(n_components=n_components, random_state=random_seed)
    train_svd   = svd_model.fit_transform(train_tfidf)
    test_svd    = svd_model.transform(test_tfidf)

    # -----------------------------------------------------------------------
    # Deterministic RNG — all sampling goes through this single generator
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(random_seed)

    # -----------------------------------------------------------------------
    # Relevance sets are built exclusively from training items.
    # Test items are never included in the candidate or relevance pools.
    # -----------------------------------------------------------------------
    def _get_relevant_train(test_row: pd.Series) -> set[str]:
        """Relevance for a test item derived from the training catalog."""
        relevant: set[str] = set()
        if "category" in train_df.columns and pd.notna(test_row.get("category")):
            same_cat = train_df[
                train_df["category"] == test_row["category"]
            ]["title"].tolist()
            relevant.update(same_cat)
        if "rating" in train_df.columns:
            high_rated = train_df[train_df["rating"] >= 4.0]["title"].tolist()
            relevant.update(high_rated)
        return relevant

    # -----------------------------------------------------------------------
    # Local recommendation helpers: query vector → top-K from training split
    # -----------------------------------------------------------------------

    def _content_recs_vec(q_vec) -> list[str]:
        sims = _cosine_sim(q_vec, train_tfidf).flatten()
        top  = np.argsort(sims)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    def _collab_recs_vec(q_svd_vec) -> list[str]:
        sims = _cosine_sim(q_svd_vec.reshape(1, -1), train_svd).flatten()
        top  = np.argsort(sims)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    def _sentiment_recs_train() -> list[str]:
        frame = train_df.copy()
        if "sentiment_score" not in frame.columns:
            frame["sentiment_score"] = 0.0
        return frame.sort_values("sentiment_score", ascending=False).head(k)[
            "title"
        ].tolist()

    def _hybrid_recs_vec(q_tfidf_vec, q_svd_vec) -> list[str]:
        content_s = _cosine_sim(q_tfidf_vec, train_tfidf).flatten()
        collab_s  = _cosine_sim(q_svd_vec.reshape(1, -1), train_svd).flatten()
        sentiment_raw = (
            train_df.get("sentiment_score", pd.Series(np.zeros(len(train_df))))
            .values.astype(float)
        )
        s_min, s_max = sentiment_raw.min(), sentiment_raw.max()
        sentiment_s = (
            (sentiment_raw - s_min) / (s_max - s_min)
            if s_max != s_min
            else np.zeros_like(sentiment_raw)
        )
        hybrid_s = (
            w["alpha"] * content_s
            + w["beta"]  * collab_s
            + w["gamma"] * sentiment_s
        )
        top = np.argsort(hybrid_s)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    # -----------------------------------------------------------------------
    # Helpers for the user-based path: query by title from the training set
    # -----------------------------------------------------------------------

    def _content_recs_train_title(title: str) -> list[str]:
        matches = train_df[train_df["title"] == title]
        if matches.empty:
            return []
        idx  = matches.index[0]
        sims = _cosine_sim(train_tfidf[idx], train_tfidf).flatten()
        sims[idx] = -1
        top  = np.argsort(sims)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    def _collab_recs_train_title(title: str) -> list[str]:
        matches = train_df[train_df["title"] == title]
        if matches.empty:
            return []
        idx  = matches.index[0]
        sims = _cosine_sim(train_svd[idx].reshape(1, -1), train_svd).flatten()
        sims[idx] = -1
        top  = np.argsort(sims)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    def _sentiment_recs_train_title(title: str) -> list[str]:
        frame = train_df.copy()
        if "sentiment_score" not in frame.columns:
            frame["sentiment_score"] = 0.0
        matches = frame[frame["title"] == title]
        if not matches.empty:
            frame = frame.drop(index=matches.index[0], errors="ignore")
        return frame.sort_values("sentiment_score", ascending=False).head(k)[
            "title"
        ].tolist()

    def _hybrid_recs_train_title(title: str) -> list[str]:
        matches = train_df[train_df["title"] == title]
        if matches.empty:
            return []
        idx       = matches.index[0]
        content_s = _cosine_sim(train_tfidf[idx], train_tfidf).flatten()
        collab_s  = _cosine_sim(train_svd[idx].reshape(1, -1), train_svd).flatten()
        sentiment_raw = (
            train_df.get("sentiment_score", pd.Series(np.zeros(len(train_df))))
            .values.astype(float)
        )
        s_min, s_max = sentiment_raw.min(), sentiment_raw.max()
        sentiment_s = (
            (sentiment_raw - s_min) / (s_max - s_min)
            if s_max != s_min
            else np.zeros_like(sentiment_raw)
        )
        hybrid_s = (
            w["alpha"] * content_s
            + w["beta"]  * collab_s
            + w["gamma"] * sentiment_s
        )
        hybrid_s[idx] = -1
        top = np.argsort(hybrid_s)[::-1][:k]
        return train_df.iloc[top]["title"].tolist()

    # -----------------------------------------------------------------------
    # Determine evaluation strategy
    # -----------------------------------------------------------------------
    modes_to_run = (
        ["content", "collaborative", "sentiment", "hybrid"]
        if mode == "all"
        else [mode]
    )

    results: ResultsDict = {}

    has_user_data = (
        "user_id" in df.columns
        and len(df["user_id"].dropna().unique()) > 1
    )

    if has_user_data:
        unique_users = df["user_id"].dropna().unique()
        sample_users = rng.choice(
            unique_users, size=min(100, len(unique_users)), replace=False
        )
    else:
        sample_size    = min(200, len(test_df))
        sample_indices = rng.choice(len(test_df), size=sample_size, replace=False)

    for m in modes_to_run:
        precisions, recalls, ndcgs = [], [], []
        mrrs, hits, ilds = [], [], []
        all_recs = []

        if has_user_data:
            # ------------------------------------------------------------------
            # User-based leave-one-out evaluation.
            #
            # For each sampled user we hold out their last interaction as the
            # evaluation target.  Recommendations are generated from the
            # training split using the user's remaining interaction history as
            # seed items.  The training TF-IDF/SVD artifacts are used for all
            # similarity computations, so the held-out item never influences
            # the model fit.
            # ------------------------------------------------------------------
            for current_user in sample_users:
                user_profile = (
                    df[df["user_id"] == current_user].reset_index(drop=True)
                )
                if len(user_profile) < 2:
                    continue  # leave-one-out requires at least 2 interactions

                query_item   = user_profile.iloc[-1]["title"]
                relevant     = {query_item}
                user_history = user_profile.iloc[:-1]["title"].tolist()

                agg_recs: dict[str, float] = {}
                for seed_title in user_history[:5]:
                    try:
                        if m == "content":
                            recs_raw = _content_recs_train_title(seed_title)
                        elif m == "collaborative":
                            recs_raw = _collab_recs_train_title(seed_title)
                        elif m == "sentiment":
                            recs_raw = _sentiment_recs_train_title(seed_title)
                        else:
                            recs_raw = _hybrid_recs_train_title(seed_title)

                        for rank, item_name in enumerate(recs_raw):
                            score = 1.0 / (rank + 1)
                            agg_recs[item_name] = max(
                                agg_recs.get(item_name, 0.0), score
                            )
                    except Exception:
                        continue

                sorted_recs = sorted(
                    agg_recs.items(), key=lambda x: x[1], reverse=True
                )
                final_recs = [
                    item for item, _ in sorted_recs
                    if item not in user_history
                ][:k]

                if final_recs:
                    precisions.append(_precision_at_k(final_recs, relevant, k))
                    recalls.append(_recall_at_k(final_recs, relevant, k))
                    ndcgs.append(_ndcg_at_k(final_recs, relevant, k))
                    mrrs.append(_mean_reciprocal_rank(final_recs, relevant, k))
                    hits.append(_hit_rate(final_recs, relevant, k))
                    ilds.append(
                        _intra_list_diversity(final_recs, train_df, train_tfidf)
                    )
                    all_recs.append(final_recs)
        else:
            # ------------------------------------------------------------------
            # Item-based evaluation on the held-out test split.
            #
            # Each test item is transformed with the train-fitted TF-IDF / SVD
            # model (never used during fitting).  Recommendations come from the
            # training catalog, so there is no self-similarity inflation.
            # Relevance sets are also built from training items only.
            # ------------------------------------------------------------------
            for test_idx in sample_indices:
                relevant = _get_relevant_train(test_df.iloc[test_idx])
                if not relevant:
                    continue

                if m == "content":
                    recs = _content_recs_vec(test_tfidf[test_idx])
                elif m == "collaborative":
                    recs = _collab_recs_vec(test_svd[test_idx])
                elif m == "sentiment":
                    recs = _sentiment_recs_train()
                else:
                    recs = _hybrid_recs_vec(
                        test_tfidf[test_idx], test_svd[test_idx]
                    )

                precisions.append(_precision_at_k(recs, relevant, k))
                recalls.append(_recall_at_k(recs, relevant, k))
                ndcgs.append(_ndcg_at_k(recs, relevant, k))
                mrrs.append(_mean_reciprocal_rank(recs, relevant, k))
                hits.append(_hit_rate(recs, relevant, k))
                ilds.append(_intra_list_diversity(recs, train_df, train_tfidf))
                all_recs.append(recs)

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall    = float(np.mean(recalls))    if recalls    else 0.0
        avg_ndcg      = float(np.mean(ndcgs))      if ndcgs      else 0.0
        avg_mrr       = float(np.mean(mrrs))        if mrrs       else 0.0
        avg_hit       = float(np.mean(hits))        if hits        else 0.0
        avg_ild       = float(np.mean(ilds))        if ilds        else 0.0
        cov = _catalog_coverage(all_recs, len(train_df)) if all_recs else 0.0

        results[m] = {
            "precision":            round(avg_precision, 4),
            "recall":               round(avg_recall,    4),
            "ndcg":                 round(avg_ndcg,      4),
            "mrr":                  round(avg_mrr,       4),
            "hit_rate":             round(avg_hit,       4),
            "catalog_coverage":     round(cov,           4),
            "intra_list_diversity": round(avg_ild,       4),
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
    parser.add_argument("--alpha",       type=float, default=0.4,  help="Content weight (default: 0.4)")
    parser.add_argument("--beta",        type=float, default=0.4,  help="Collaborative weight (default: 0.4)")
    parser.add_argument("--gamma",       type=float, default=0.2,  help="Sentiment weight (default: 0.2)")
    parser.add_argument("--test-size",   type=float, default=0.2,
                        help="Fraction of data held out for evaluation (default: 0.2)")
    parser.add_argument("--random-seed", type=int,   default=42,
                        help="Random seed for reproducible evaluation (default: 42)")
    args = parser.parse_args()

    print(f"\n📊 Running evaluation — mode={args.mode}, k={args.k}")
    print(f"   Weights: α={args.alpha} β={args.beta} γ={args.gamma}")
    print(f"   Test split: {args.test_size:.0%}  seed: {args.random_seed}\n")

    try:
        results = run_evaluation(
            k=args.k,
            mode=args.mode,
            weights={"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
            test_size=args.test_size,
            random_seed=args.random_seed,
        )
    except (RuntimeError, ValueError) as e:
        print(f"❌ Error: {e}")
        return

    # Pretty-print results table
    header = (
        f"{'Mode':<16} {'Precision@K':>12} {'Recall@K':>10} {'NDCG@K':>10} "
        f"{'MRR@K':>8} {'Hit@K':>8} {'Coverage':>9} {'ILD':>8}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, metrics in results.items():
        print(
            f"{mode_name:<16} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['ndcg']:>10.4f} "
            f"{metrics.get('mrr', 0.0):>8.4f} "
            f"{metrics.get('hit_rate', 0.0):>8.4f} "
            f"{metrics.get('catalog_coverage', 0.0):>9.4f} "
            f"{metrics.get('intra_list_diversity', 0.0):>8.4f}"
        )
    print()


if __name__ == "__main__":
    _cli()