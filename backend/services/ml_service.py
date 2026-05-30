from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.state import models, ACTIVE_MODEL_VERSION
from backend.core.cache import _cache_key, _set_cached_response

def _build_tfidf_for_items(item_df):
    """Build and return a TF-IDF matrix and vectorizer for the given item_df."""
    texts = (item_df.get('combined') or item_df.get('title')).fillna('').astype(str).tolist()
    vec = TfidfVectorizer(max_features=16384, stop_words='english')
    matrix = vec.fit_transform(texts)
    return vec, matrix

def cold_start_recommendation(combined_text: str, top_n: int = 10, weights: tuple[float, float, float] = (0.6, 0.3, 0.1), target_catalog: Optional[str] = None):
    """Cold-start blending of content similarity (TF-IDF) and simple popularity/rating signals.
    Returns list of dicts with blended score and components.
    """
    item_df = models.get('item_df')
    if item_df is None or item_df.empty:
        return []

    vec, matrix = _build_tfidf_for_items(item_df)
    try:
        qv = vec.transform([combined_text])
    except Exception:
        return []

    scores = cosine_similarity(qv, matrix).flatten()

    review_counts = item_df.get('review_count', None)
    if review_counts is None or len(review_counts) == 0:
        pop_norm = np.zeros_like(scores)
    else:
        max_rc = float(max(1, int(review_counts.max())))
        pop_norm = (np.array(item_df.get('review_count').fillna(0).astype(float)) / max_rc)

    ratings = item_df.get('rating')
    if ratings is None or len(ratings) == 0:
        rating_norm = np.zeros_like(scores)
    else:
        rating_norm = (np.array(item_df.get('rating').fillna(0).astype(float)) / 5.0)

    alpha, beta, gamma = weights
    blended = alpha * scores + beta * pop_norm + gamma * rating_norm

    idxs = blended.argsort()[::-1]
    results = []
    seen = set()
    for idx in idxs:
        title = str(item_df.iloc[idx].get('title', ''))
        if not title or title in seen:
            continue
        if target_catalog and 'category' in item_df.columns:
            cat = str(item_df.iloc[idx].get('category', ''))
            if cat and cat.casefold() != target_catalog.casefold():
                continue
        seen.add(title)
        results.append({
            'title': title,
            'blended_score': float(blended[idx]),
            'content_score': float(scores[idx]),
            'popularity_score': float(pop_norm[idx]),
            'rating_norm': float(rating_norm[idx]),
        })
        if len(results) >= top_n:
            break
    return results

def _precompute_recommendation_cache(top_n: int = 10, explain: bool = False) -> int:
    if not models.get("ready") or models.get("item_df") is None:
        return 0

    count = 0
    item_df = models["item_df"]

    for title in item_df["title"].dropna().astype(str).unique():
        cache_key = _cache_key("recommend", title, top_n, explain, "")
        recs = models["hybrid"].recommend(title, top_n=top_n, explain=explain)

        if not recs:
            continue

        payload = {
            "query_item": title,
            "recommendations": recs,
            "weights": models["hybrid"].get_weights(),
            "explain": explain,
            "target_catalog": None,
            "model_version": ACTIVE_MODEL_VERSION,
            "has_history": False,
            "cache_precomputed": True,
        }
        _set_cached_response(cache_key, payload)
        count += 1
    return count
