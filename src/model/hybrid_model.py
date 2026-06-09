"""src.model.hybrid_model

HybridRecommender: combines content-based, collaborative, and sentiment signals
into a single weighted score.

This file was previously left in a merge-conflict/broken state.
It has been rewritten to be syntactically valid and to support
"recommendation explanations" via `explain=True`.

Returned recommendation dict keys (when available):
- title
- content_score
- collab_score
- sentiment_score
- hybrid_score
- rating
- category
- description
- top_reviews
- explanation (string, when explain=True)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Optional

import numpy as np

from src.model.causal_config import CausalConfig
from src.model.causal_model import CausalDebiaser

logger = logging.getLogger(__name__)


def bayesian_rating(rating: float, review_count: int, global_avg: float = 3.0, min_votes: int = 10) -> float:
    """Bayesian average: smooths ratings toward the global mean."""
    v = float(review_count)
    m = float(min_votes)
    C = float(global_avg)
    return (v / (v + m)) * float(rating) + (m / (v + m)) * C


class HybridRecommender:
    def __init__(
        self,
        content_model,
        collab_model=None,
        item_df=None,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        normalization: str = "minmax",
        weight_matrix: Optional[dict[str, Any]] = None,
        use_causal_debiasing: bool = False,
        causal_lambda: float = 0.5,
        causal_clip: float = 5.0,
        causal_config: Optional[CausalConfig] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Optional knowledge-graph component not currently used by hybrid formula
        self.kg_model = None
        self.delta = 0.0

        # Backward compatible knob; earlier code attempted to reference model_kwargs
        self.model_kwargs = model_kwargs or {}

        # Weight context overrides
        self.normalization = normalization
        self.weight_matrix = weight_matrix or {}

        # Fairness controls
        self.fairness_enabled = False
        self.fairness_key = "category"
        self.fairness_max_share = 1.0

        # Causal debiasing
        if causal_config is not None:
            causal_config.validate()
            self.use_causal_debiasing = bool(causal_config.enabled)
            self._debiaser: CausalDebiaser | None = (
                CausalDebiaser.from_config(item_df, causal_config)
                if causal_config.enabled and item_df is not None
                else None
            )
            self._causal_config: CausalConfig | None = causal_config
        else:
            self.use_causal_debiasing = bool(use_causal_debiasing)
            self._debiaser = (
                CausalDebiaser(item_df, blend_lambda=causal_lambda, clip_max=causal_clip)
                if use_causal_debiasing and item_df is not None
                else None
            )
            self._causal_config = None

        # Build lookup maps for explanation and metadata
        self._sentiment_map: dict[str, float] = {}
        self._rating_map: dict[str, float] = {}
        self._review_count_map: dict[str, int] = {}
        self._category_map: dict[str, str] = {}
        self._popularity_map: dict[str, float] = {}
        self._catalog_map: dict[str, str] = {}

        self.online_updater = None

        if item_df is not None:
            # Sensible global default
            global_avg = float(item_df["rating"].mean()) if "rating" in item_df.columns else 3.0

            for _, row in item_df.iterrows():
                title = row.get("title")
                if title is None or (isinstance(title, float) and math.isnan(title)):
                    continue
                title = str(title)

                if "avg_sentiment" in item_df.columns:
                    self._sentiment_map[title] = float(row.get("avg_sentiment") or 0.0)

                raw_rating = float(row.get("rating") or 0.0)
                review_count = row.get("review_count")
                if review_count is None or (isinstance(review_count, float) and math.isnan(review_count)):
                    review_count = 0
                review_count = int(review_count)

                self._review_count_map[title] = review_count
                self._rating_map[title] = bayesian_rating(raw_rating, review_count, global_avg=global_avg)
                self._category_map[title] = str(row.get("category") or "")
                self._catalog_map[title] = str(row.get("catalog") or "")

            if "review_count" in item_df.columns and len(item_df) > 0:
                max_reviews = float(item_df["review_count"].max() or 0.0)
                if max_reviews > 0:
                    for _, row in item_df.iterrows():
                        t = str(row.get("title"))
                        rc = int(row.get("review_count") or 0)
                        self._popularity_map[t] = rc / max_reviews

    # ------------------------- weight API -------------------------
    def set_weights(self, alpha: float, beta: float, gamma: float):
        if any(math.isnan(w) for w in [alpha, beta, gamma]):
            raise ValueError("Weights must be finite numbers")
        if any(w < 0 for w in [alpha, beta, gamma]):
            raise ValueError("Weights must be non-negative")
        total = float(alpha + beta + gamma)
        if total <= 0:
            total = 1.0
        self.alpha = float(alpha) / total
        self.beta = float(beta) / total
        self.gamma = float(gamma) / total

    def get_weights(self):
        # UI/tests expect these exact keys
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

    # ------------------------- fairness helpers -------------------------
    def set_fairness(self, enabled=None, key=None, max_share=None):
        if enabled is not None:
            self.fairness_enabled = bool(enabled)
        if key is not None:
            self.fairness_key = key or "category"
        if max_share is not None:
            try:
                self.fairness_max_share = float(max_share)
            except Exception:
                self.fairness_max_share = 1.0

    def _fair_rerank(self, results: list[dict[str, Any]], top_n: int, key: str, max_share: float):
        if not results or top_n <= 1:
            return results[:top_n]

        try:
            max_share = float(max_share)
        except Exception:
            max_share = 1.0

        if not (0 < max_share <= 1):
            max_share = 1.0

        max_per_group = max(1, int(math.ceil(max_share * top_n)))
        key = key or "category"

        group_counts: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []

        for item in results:
            group = str(item.get(key, "") or "").strip().casefold() or "unknown"
            current = group_counts.get(group, 0)
            if current < max_per_group:
                selected.append(item)
                group_counts[group] = current + 1
                if len(selected) >= top_n:
                    break
            else:
                overflow.append(item)

        if len(selected) < top_n:
            selected.extend(overflow[: (top_n - len(selected))])

        return selected

    # ------------------------- normalization -------------------------
    def _normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return scores

        arr = np.array(scores, dtype=float)

        if self.normalization == "zscore":
            mu = float(np.nanmean(arr))
            sigma = float(np.nanstd(arr))
            if sigma == 0 or math.isnan(sigma):
                return [0.5] * len(arr)
            z = (arr - mu) / sigma
            cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
            return [float(v) for v in cdf]

        # default: minmax
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx - mn == 0 or math.isnan(mn) or math.isnan(mx):
            return [0.5] * len(arr)
        return [float((v - mn) / (mx - mn)) for v in arr]

    def _resolve_active_weights(self, candidate_titles: list[str] | None = None, user_id: str | None = None):
        a, b, g = float(self.alpha), float(self.beta), float(self.gamma)

        # default override
        if "default" in self.weight_matrix:
            da, db, dg = self.weight_matrix["default"]
            a, b, g = float(da), float(db), float(dg)

        # category override (use modal category from candidates)
        if candidate_titles and self.item_df is not None:
            try:
                df = self.item_df
                if "title" in df.columns and "category" in df.columns:
                    cats = df[df["title"].isin(candidate_titles)]["category"].dropna().astype(str).tolist()
                    if cats:
                        top_cat = Counter(cats).most_common(1)[0][0]
                        key = f"category:{top_cat}"
                        if key in self.weight_matrix:
                            da, db, dg = self.weight_matrix[key]
                            a, b, g = float(da), float(db), float(dg)
            except Exception:
                logger.warning("Weight matrix category override failed", exc_info=True)

        # Normalize
        total = a + b + g
        if total <= 0:
            return float(self.alpha), float(self.beta), float(self.gamma)
        return a / total, b / total, g / total

    # ------------------------- main recommend -------------------------
    def recommend(
        self,
        title: str,
        user_id: str | None = None,
        top_n: int = 10,
        explain: bool = False,
        target_catalog: str | None = None,
        weights: dict[str, float] | None = None,
        fairness: bool | None = None,
        fairness_key: str | None = None,
        fairness_max_share: float | None = None,
        diversity: float = 0.0,
        serendipity: float = 0.0,
    ):
        # 1) collect candidates and raw component scores
        content_recs = self.content_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)

        candidates: dict[str, dict[str, Any]] = {}
        for r in content_recs:
            if not isinstance(r, dict):
                continue
            ctitle = r.get("title")
            if not ctitle:
                continue
            candidates[str(ctitle)] = {
                "title": str(ctitle),
                "raw_content": float(r.get("content_score", r.get("score", 0.0)) or 0.0),
                "raw_collab": 0.0,
                "raw_sentiment": float(self._sentiment_map.get(str(ctitle), 0.0) or 0.0),
            }

        if self.collab_model:
            collab_recs = self.collab_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)
            for r in collab_recs:
                if not isinstance(r, dict):
                    continue
                ctitle = r.get("title")
                if not ctitle:
                    continue
                key = str(ctitle)
                if key not in candidates:
                    candidates[key] = {
                        "title": key,
                        "raw_content": 0.0,
                        "raw_collab": float(r.get("collab_score", 0.0) or 0.0),
                        "raw_sentiment": float(self._sentiment_map.get(key, 0.0) or 0.0),
                    }
                else:
                    candidates[key]["raw_collab"] = float(r.get("collab_score", 0.0) or 0.0)

        if not candidates:
            return self._cold_start_fallback(title, top_n, target_catalog=target_catalog)

        items = list(candidates.values())

        # 2) normalize component scores
        content_scores = self._normalize_scores([it["raw_content"] for it in items])
        collab_scores = self._normalize_scores([it["raw_collab"] for it in items])
        sentiment_scores = self._normalize_scores([it["raw_sentiment"] for it in items])

        # 3) resolve weights
        if weights is not None:
            a = float(weights.get("alpha", self.alpha))
            b = float(weights.get("beta", self.beta))
            g = float(weights.get("gamma", self.gamma))
            tot = a + b + g
            if tot > 0:
                a, b, g = a / tot, b / tot, g / tot
        else:
            candidate_titles = [it["title"] for it in items]
            a, b, g = self._resolve_active_weights(candidate_titles=candidate_titles, user_id=user_id)

        # 4) compute hybrid scores
        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            hybrid_base = a * content_scores[i] + b * collab_scores[i] + g * sentiment_scores[i]

            # small popularity bonus for ranking stability
            popularity = float(self._popularity_map.get(item["title"], 0.5) or 0.5)
            hybrid = min(1.0, hybrid_base + 0.05 * popularity)

            # metadata for UI
            row_data = None
            description = ""
            top_reviews: list[Any] = []
            if hasattr(self.content_model, "df") and self.content_model.df is not None:
                df = self.content_model.df
                try:
                    row_data = df[df["title"] == item["title"]]
                except Exception:
                    row_data = None

            avg_rating = float(self._rating_map.get(item["title"], 0.0) or 0.0)
            category = self._category_map.get(item["title"], "")

            if row_data is not None and len(row_data) > 0:
                try:
                    description = str(row_data.iloc[0].get("description", "") or "")[:200]
                    tp = row_data.iloc[0].get("top_reviews", [])
                    top_reviews = tp if isinstance(tp, list) else []
                except Exception:
                    pass

            result: dict[str, Any] = {
                "title": item["title"],
                "content_score": round(float(content_scores[i]), 4),
                "collab_score": round(float(collab_scores[i]), 4),
                "sentiment_score": round(float(sentiment_scores[i]), 4),
                "hybrid_score": round(float(hybrid), 4),
                "rating": round(float(avg_rating), 2),
                "category": category,
                "description": description,
                "top_reviews": top_reviews,
            }

            if explain:
                result["explanation"] = self._build_explanation(
                    source_title=title,
                    candidate_title=item["title"],
                    content_score=content_scores[i],
                    collab_score=collab_scores[i],
                    sentiment_score=sentiment_scores[i],
                    popularity=popularity,
                    alpha=a,
                    beta=b,
                    gamma=g,
                )

            results.append(result)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # causal debiasing (if configured)
        if self.use_causal_debiasing and self._debiaser is not None and results:
            score_key = self._causal_config.score_key if self._causal_config is not None else "hybrid_score"
            results = self._debiaser.debias_batch(results, score_key=score_key)
            results.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)

        # diversity/serendipity are optional; keep ordering stable if not enabled
        # (existing project may have dedicated rankers; we keep it minimal here)

        apply_fairness = self.fairness_enabled if fairness is None else bool(fairness)
        if apply_fairness:
            key = fairness_key or self.fairness_key
            max_share = self.fairness_max_share if fairness_max_share is None else fairness_max_share
            results = self._fair_rerank(results, top_n, key=key, max_share=max_share)

        return results[:top_n]

    def recommend_for_user(self, user_id: str, top_n: int = 10, explain: bool = False):
        # Keep behavior similar to earlier: if no collab, fallback to popularity
        if self.collab_model is None or not hasattr(self.collab_model, "predict_for_user"):
            return self.get_popular_fallback_items(top_n=top_n)

        if not hasattr(self.collab_model, "_user_to_idx") or user_id not in getattr(self.collab_model, "_user_to_idx", {}):
            return self.get_popular_fallback_items(top_n=top_n)

        collab_recs = self.collab_model.predict_for_user(user_id, top_n=top_n * 3)

        results: list[dict[str, Any]] = []
        for r in collab_recs[:top_n]:
            item_title = r.get("title")
            if not item_title:
                continue
            item_title = str(item_title)

            row_data = None
            description = ""
            top_reviews: list[Any] = []
            if hasattr(self.content_model, "df") and self.content_model.df is not None:
                df = self.content_model.df
                try:
                    row_data = df[df["title"] == item_title]
                except Exception:
                    row_data = None

            if row_data is not None and len(row_data) > 0:
                try:
                    description = str(row_data.iloc[0].get("description", "") or "")[:200]
                    tp = row_data.iloc[0].get("top_reviews", [])
                    top_reviews = tp if isinstance(tp, list) else []
                except Exception:
                    pass

            hybrid_score = float(r.get("predicted_score", r.get("hybrid_score", 0.0)) or 0.0)
            rating = float(self._rating_map.get(item_title, 0.0) or 0.0)

            results.append(
                {
                    "title": item_title,
                    "content_score": 0.0,
                    "collab_score": round(float(hybrid_score), 4),
                    "sentiment_score": round((float(self._sentiment_map.get(item_title, 0.0) or 0.0) + 1.0) / 2.0, 4),
                    "hybrid_score": round(float(hybrid_score), 4),
                    "rating": round(rating, 2),
                    "category": self._category_map.get(item_title, ""),
                    "description": description,
                    "top_reviews": top_reviews,
                }
            )

        # No explanation for user path (frontend currently doesn't request it in these endpoints)
        return results[:top_n]

    # ------------------------- explanations -------------------------
    def _sentiment_label(self, score: float) -> str:
        if score > 0.2:
            return "positive"
        if score < -0.2:
            return "negative"
        return "neutral"

    def _build_explanation(
        self,
        source_title: str,
        candidate_title: str,
        content_score: float,
        collab_score: float,
        sentiment_score: float,
        popularity: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> str:
        # Weighted component contributions (match the hybrid formula)
        weighted = {
            "content": alpha * float(content_score),
            "collaborative": beta * float(collab_score),
            "sentiment": gamma * float(sentiment_score),
        }

        strongest = max(weighted, key=weighted.get)

        if strongest == "content":
            return f"Recommended because of similar product description/category to '{source_title}'."
        if strongest == "collaborative":
            return "Recommended because users with similar preferences interacted with this item."
        if strongest == "sentiment":
            label = self._sentiment_label(float(sentiment_score) - 0.5)  # rough mapping; sentiment_score is normalized
            # Provide human-friendly generic text
            return "Recommended because this item has strong customer sentiment and ratings." if label != "neutral" else "Recommended because of positive sentiment signals."

        return "Recommended based on hybrid signals."

    # ------------------------- cold start / popular fallback -------------------------
    def _cold_start_fallback(self, title: str, top_n: int, target_catalog: str | None = None):
        if self.item_df is None:
            return []

        df = self.item_df
        if target_catalog and "catalog" in df.columns:
            df = df[df["catalog"].astype(str).str.lower() == target_catalog.lower()]

        # category warm start
        target_cat = self._category_map.get(title, "")
        if target_cat and "category" in df.columns:
            cat_items = df[df["category"] == target_cat]
            if len(cat_items) >= top_n:
                df = cat_items

        return self.get_popular_fallback_items(top_n=top_n, source_df=df, exclude_title=title)

    def get_popular_fallback_items(self, top_n: int = 5, source_df=None, exclude_title: str | None = None):
        if self.item_df is None and source_df is None:
            return []

        df = source_df if source_df is not None else self.item_df
        if df is None or len(df) == 0:
            return []

        df = df.copy()
        global_avg = 3.0

        if exclude_title is not None and "title" in df.columns:
            df = df[df["title"] != exclude_title]

        if "rating" in df.columns and "review_count" in df.columns:
            df["_bayesian"] = df.apply(lambda r: bayesian_rating(r["rating"], int(r.get("review_count", 0) or 0), global_avg=global_avg), axis=1)
            df = df.sort_values(["_bayesian", "review_count"], ascending=[False, False])
        elif "rating" in df.columns:
            df = df.sort_values("rating", ascending=False)
        elif "review_count" in df.columns:
            df = df.sort_values("review_count", ascending=False)

        results: list[dict[str, Any]] = []
        for _, row in df.head(top_n).iterrows():
            t = str(row.get("title"))
            results.append(
                {
                    "title": t,
                    "content_score": 0.0,
                    "collab_score": 0.0,
                    "sentiment_score": (float(row.get("avg_sentiment", 0.0) or 0.0) + 1.0) / 2.0,
                    "hybrid_score": round(float(self._rating_map.get(t, 0.0) or 0.0) / 5.0, 4),
                    "rating": round(float(row.get("rating", 0.0) or 0.0), 2),
                    "category": row.get("category", "") or "",
                    "description": str(row.get("description", "") or "")[:200],
                    "top_reviews": [],
                }
            )

        return results

    # ------------------------- online updates -------------------------
    def set_online_updater(self, updater):
        self.online_updater = updater

    def apply_interaction(
        self,
        user_id: str,
        item_title: str,
        rating: float | None = None,
        sentiment: float | None = None,
        timestamp=None,
    ):
        if self.online_updater is not None:
            try:
                self.online_updater.ingest(
                    user_id=user_id,
                    item_title=item_title,
                    rating=rating,
                    sentiment=sentiment,
                    timestamp=timestamp,
                    recommender=self,
                )
                return True
            except Exception:
                pass

        try:
            prev = int(self._review_count_map.get(item_title, 0) or 0)
            new_count = prev + 1
            self._review_count_map[item_title] = new_count

            try:
                max_reviews = max(self._review_count_map.values())
            except Exception:
                max_reviews = new_count
            self._popularity_map[item_title] = (new_count / max_reviews) if max_reviews > 0 else 0.0

            if rating is not None:
                prev_rating = float(self._rating_map.get(item_title, 0.0) or 0.0)
                global_avg = float(np.mean(list(self._rating_map.values()))) if self._rating_map else 3.0
                self._rating_map[item_title] = bayesian_rating(global_avg, new_count, global_avg=global_avg)

            if sentiment is not None:
                prev_sent = self._sentiment_map.get(item_title)
                if prev_sent is None:
                    self._sentiment_map[item_title] = float(sentiment)
                else:
                    self._sentiment_map[item_title] = (float(prev_sent) * prev + float(sentiment)) / (prev + 1)

            return True
        except Exception:
            return False

