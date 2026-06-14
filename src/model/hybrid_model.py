"""src.model.hybrid_model

HybridRecommender: combines content-based, collaborative, and sentiment signals
into a single weighted score.

This module was previously left in a merge-conflict/broken state. It has been
rewritten to be syntactically valid and to support recommendation explanations
via `explain=True`.

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

The public API is intentionally kept compatible with the rest of the project.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Optional

import numpy as np

from src.model.causal_config import CausalConfig
from src.model.causal_model import CausalDebiaser
from src.model.recommendation_history import history_tracker

logger = logging.getLogger(__name__)


def bayesian_rating(
    rating: float,
    review_count: int,
    global_avg: float = 3.0,
    min_votes: int = 10,
) -> float:
    """Bayesian average: smooths ratings toward the global mean."""
    v = float(review_count)
    m = float(min_votes)
    C = float(global_avg)
    rating = float(rating)
    return (v / (v + m)) * rating + (m / (v + m)) * C


class HybridRecommender:
    def __init__(self, content_model, collab_model=None, item_df=None,
                 alpha=0.4, beta=0.35, gamma=0.25,
                 normalization='minmax', weight_matrix=None,
                 use_causal_debiasing=False, causal_lambda=0.5, causal_clip=5.0,
                 causal_config=None, model_kwargs=None,
                 kg_model=None, delta=0.05):
        """
        content_model:        ContentRecommender instance
        collab_model:         CollaborativeRecommender instance (optional)
        item_df:              DataFrame with 'avg_sentiment', 'rating', 'review_count' columns
        alpha:                weight for content-based score
        beta:                 weight for collaborative score
        gamma:                weight for sentiment score
        use_causal_debiasing: Enable IPS-based causal debiasing on the final hybrid score.
                              When True, a CausalDebiaser is built from item_df and applied
                              after the weighted blend, before final ranking.
        causal_lambda:        Blend factor λ for causal correction (0.0–1.0).
                              0.0 = no debiasing, 1.0 = full IPS reweighting. Default 0.5.
        causal_clip:          Max IPS weight cap to prevent variance explosion. Default 5.0.
        causal_config:        Optional CausalConfig instance. When provided, takes precedence
                              over use_causal_debiasing / causal_lambda / causal_clip.
                              Use this for structured configuration management.
        """
    """Hybrid recommender combining content + collaborative + sentiment."""

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
        # Optional KG hooks (some legacy variants included them)
        kg_model=None,
        delta: float = 0.0,
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.normalization = normalization
        self.weight_matrix = weight_matrix or {}

        # optional knowledge graph
        self.kg_model = kg_model
        self.delta = float(delta)

        self.model_kwargs = model_kwargs or {}

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

        # Bandit exploration
        self.epsilon = 0.1
        self.bandit_arms = [(self.alpha, self.beta, self.gamma)]
        self.arm_rewards = {0: 0.0}
        self.arm_counts = {0: 0}

        if item_df is not None:
            global_avg = float(item_df["rating"].mean()) if "rating" in item_df.columns else 3.0

            # sentiment / rating / category / popularity
            if "title" in item_df.columns:
                for _, row in item_df.iterrows():
                    t = row.get("title")
                    if t is None or (isinstance(t, float) and math.isnan(t)):
                        continue
                    title = str(t)

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

                if "review_count" in item_df.columns:
                    max_reviews = item_df["review_count"].max() or 0
                    if max_reviews > 0:
                        for _, row in item_df.iterrows():
                            title = str(row.get("title"))
                            rc = int(row.get("review_count") or 0)
                            self._popularity_map[title] = rc / float(max_reviews)

    # ------------------------- weight API -------------------------
    def set_weights(self, alpha: float, beta: float, gamma: float):
        """Update the scoring weights. Normalized to sum to 1."""
        for w in (alpha, beta, gamma):
            if math.isnan(float(w)):
                raise ValueError("Weights must be finite numbers")
        if any(w < 0 for w in (alpha, beta, gamma)):
            raise ValueError("Weights must be non-negative")
        total = float(alpha + beta + gamma)
        if total <= 0:
            total = 1.0
        self.alpha = float(alpha) / total
        self.beta = float(beta) / total
        self.gamma = float(gamma) / total

    def get_weights(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }


    def select_bandit_arm(self):
        import random
        if random.random() < self.epsilon:
            return random.randint(0, len(self.bandit_arms) - 1)
        best_arm = max(
            self.arm_rewards,
            key=lambda x: self.arm_rewards[x] / max(self.arm_counts[x], 1)
        )

        return best_arm

    def update_bandit_reward(self, arm_id, reward):
        self.arm_counts[arm_id] += 1
        self.arm_rewards[arm_id] += reward


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

    def _get_active_weights(
        self,
        base_a: float,
        base_b: float,
        base_g: float,
        base_d: float = 0.0,
        user_id: str | None = None,
        candidate_titles: list[str] | None = None,
    ) -> tuple[float, float, float, float]:
        """Resolve active weights using weight_matrix and runtime signals."""

        a, b, g, d = float(base_a), float(base_b), float(base_g), float(base_d)

        def unpack_weights(val, default_d=0.0):
            if isinstance(val, (list, tuple)):
                if len(val) >= 4:
                    return float(val[0]), float(val[1]), float(val[2]), float(val[3])
                if len(val) == 3:
                    return float(val[0]), float(val[1]), float(val[2]), default_d
                if len(val) == 2:
                    return float(val[0]), float(val[1]), 0.0, default_d
            return None

        if "default" in self.weight_matrix:
            w = unpack_weights(self.weight_matrix["default"], d)
            if w is not None:
                a, b, g, d = w

        # category override
        if candidate_titles and self.item_df is not None and {"title", "category"}.issubset(self.item_df.columns):
            try:
                cats = (
                    self.item_df[self.item_df["title"].isin(candidate_titles)]["category"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                if cats:
                    top_cat = Counter(cats).most_common(1)[0][0]
                    key = f"category:{top_cat}"
                    if key in self.weight_matrix:
                        w = unpack_weights(self.weight_matrix[key], d)
                        if w is not None:
                            a, b, g, d = w
            except Exception:
                logger.warning("weight_matrix category override failed", exc_info=True)

        # user signals
        if user_id and self.collab_model and hasattr(self.collab_model, 'df'):
            try:
                user_interacts = int(len(self.collab_model.df[self.collab_model.df['user_id'] == user_id]))
                if 'warm_user' in self.weight_matrix and user_interacts > 10:
                    w = unpack_weights(self.weight_matrix['warm_user'], d)
                    if w is not None:
                        a, b, g, d = w
                if 'cold_user' in self.weight_matrix and user_interacts < 3:
                    w = unpack_weights(self.weight_matrix['cold_user'], d)
                    if w is not None:
                        a, b, g, d = w
            except Exception:
                pass

        # feature absence overrides
        if self.collab_model is None and "no_collab" in self.weight_matrix:
            w = unpack_weights(self.weight_matrix["no_collab"], d)
            if w is not None:
                a, b, g, d = w

        if not self._sentiment_map and "no_sentiment" in self.weight_matrix:
            w = unpack_weights(self.weight_matrix["no_sentiment"], d)
            if w is not None:
                a, b, g, d = w

        total = a + b + g + d
        if total <= 0:
            return base_a, base_b, base_g, base_d
        return a / total, b / total, g / total, d / total

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
        content_recs = self.content_model.recommend(
            title, top_n=top_n * 3, target_catalog=target_catalog
        )

        candidates: dict[str, dict[str, Any]] = {}
        for r in content_recs or []:
            if not isinstance(r, dict):
                continue
            ctitle = r.get("title")
            if not ctitle:
                continue
            ctitle = str(ctitle)
            candidates[ctitle] = {
                "title": ctitle,
                "raw_content": float(r.get("content_score", r.get("score", 0.0)) or 0.0),
                "raw_collab": 0.0,
                "raw_sentiment": float(self._sentiment_map.get(ctitle, 0.0) or 0.0),
            }

        if self.collab_model:
            collab_recs = self.collab_model.recommend(
                title, top_n=top_n * 3, target_catalog=target_catalog
            )
            for r in collab_recs or []:
                if not isinstance(r, dict):
                    continue
                ct = r.get("title")
                if not ct:
                    continue
                ct = str(ct)
                if ct not in candidates:
                    candidates[ct] = {
                        "title": ct,
                        "raw_content": 0.0,
                        "raw_collab": float(r.get("collab_score", 0.0) or 0.0),
                        "raw_sentiment": float(self._sentiment_map.get(ct, 0.0) or 0.0),
                    }
                else:
                    candidates[ct]["raw_collab"] = float(r.get("collab_score", 0.0) or 0.0)

        kg_scores_by_title: dict[str, float] = {}
        if self.kg_model:
            kg_recs = self.kg_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)
            for r in kg_recs or []:
                if not isinstance(r, dict):
                    continue
                ct = r.get("title")
                if not ct:
                    continue
                kg_scores_by_title[str(ct)] = float(r.get("kg_score", r.get("score", 0.0)) or 0.0)

                if str(ct) not in candidates:
                    candidates[str(ct)] = {
                        "title": str(ct),
                        "raw_content": 0.0,
                        "raw_collab": 0.0,
                        "raw_sentiment": float(self._sentiment_map.get(str(ct), 0.0) or 0.0),
                    }

        if not candidates:
            return self._cold_start_fallback(title, top_n, target_catalog=target_catalog)

        items = list(candidates.values())

        # 2) normalize component scores
        content_scores = self._normalize_scores([it["raw_content"] for it in items])
        collab_scores = self._normalize_scores([it["raw_collab"] for it in items])
        sentiment_scores = self._normalize_scores([it["raw_sentiment"] for it in items])

        kg_raws = [kg_scores_by_title.get(it["title"], 0.0) for it in items]
        kg_scores = self._normalize_scores(kg_raws) if self.kg_model else [0.0] * len(items)

        # 3) resolve weights
        if weights is not None:
            a = float(weights.get("alpha", self.alpha))
            b = float(weights.get("beta", self.beta))
            g = float(weights.get("gamma", self.gamma))
            d = float(weights.get("delta", self.delta))
            total = a + b + g + d
            if total > 0:
                a, b, g, d = a / total, b / total, g / total, d / total
            else:
                a, b, g, d = self.alpha, self.beta, self.gamma, 0.0
        else:
            arm_id = self.select_bandit_arm()
            a, b, g = getattr(self, 'bandit_arms', [(self.alpha, self.beta, self.gamma)])[arm_id]

            a, b, g, d = self._get_active_weights(
                a, b, g, getattr(self, 'delta', 0.0),
                user_id=user_id,
            )
            d = self.delta if self.kg_model else 0.0

        # 4) compute hybrid scores
        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            hybrid_base = a * content_scores[i] + b * collab_scores[i] + g * sentiment_scores[i] + d * kg_scores[i]

            popularity = float(self._popularity_map.get(item["title"], 0.5) or 0.5)
            popularity_bonus = 0.05 * popularity
            hybrid = min(1.0, hybrid_base + popularity_bonus)

            # metadata
            description = ""
            top_reviews: list[Any] = []
            if hasattr(self.content_model, "df") and self.content_model.df is not None:
                try:
                    df = self.content_model.df
                    row_data = df[df["title"] == item["title"]]
                    if len(row_data) > 0:
                        description = str(row_data.iloc[0].get("description", "") or "")[:200]
                        tr = row_data.iloc[0].get("top_reviews", [])
                        top_reviews = tr if isinstance(tr, list) else []
                except Exception:
                    pass

            avg_rating = float(self._rating_map.get(item["title"], 0.0) or 0.0)
            category = self._category_map.get(item["title"], "")

            # Popularity as a proper fourth component weighted by delta.
            # Weights a, b, g are re-normalized here to leave room for delta,
            # guaranteeing hybrid_score in [0, 1].
            popularity = self._popularity_map.get(item['title'], 0.5)
            weight_sum = a + b + g + self.delta
            if weight_sum <= 0:
                weight_sum = 1.0
            hybrid = (
                (a * content_scores[i] +
                 b * collab_scores[i] +
                 g * sentiment_scores[i] +
                 self.delta * popularity) / weight_sum
            )
            popularity_bonus = 0.05 * popularity
            
            # Enforce strict upper bound limit check
            hybrid = min(1.0, hybrid_base + popularity_bonus)

            # Lookup info from content model's df
            row_data = self.content_model.df[
                self.content_model.df['title'] == item['title']
            ]
            avg_rating = self._rating_map.get(item['title'], 0.0)
            category = self._category_map.get(item['title'], '')
            description = ''
            top_reviews = []
            if len(row_data) > 0:
                description = str(row_data.iloc[0].get('description', ''))[:200]
                tp = row_data.iloc[0].get('top_reviews', [])
                top_reviews = tp if isinstance(tp, list) else []

            result = {
                'title': item['title'],
                'content_score': round(content_scores[i], 4),
                'collab_score': round(collab_scores[i], 4),
                'sentiment_score': round(sentiment_scores[i], 4),
                'hybrid_score': round(hybrid, 4),
                'rating': round(avg_rating, 2),
                'category': category,
                'description': description,
                'top_reviews': top_reviews,
            }
            if explain:
                result['explanation'] = self._build_explanation(
                    title,
                    item['title'],
                    content_scores[i],
                    collab_scores[i],
                    sentiment_scores[i],
                    popularity,
                    a,
                    b,
                    g,
                    item,
                )
            results.append(result)

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        if not results:
            return self.get_popular_fallback_items(top_n=top_n, exclude_title=title)

        # 7. Optional causal debiasing — applied after sorting so the debiaser
        #    sees the full candidate set for proper batch-level IPS normalization,
        #    then we re-sort by the updated causal score.
        if self.use_causal_debiasing and self._debiaser is not None:
            score_key = (
                self._causal_config.score_key
                if self._causal_config is not None
                else 'hybrid_score'
            )
            results = self._debiaser.debias_batch(results, score_key=score_key)
            results.sort(key=lambda x: x[score_key], reverse=True)

        # 8. Apply diversity and serendipity controls
        if diversity > 0.0 or serendipity > 0.0:
            results = self._diversity_rerank(
                results, top_n,
                diversity=diversity,
                serendipity=serendipity
            )

        apply_fairness = self.fairness_enabled if fairness is None else bool(fairness)
        if apply_fairness:
            key = fairness_key or self.fairness_key
            max_share = self.fairness_max_share if fairness_max_share is None else fairness_max_share
            return self._fair_rerank(results, top_n, key, max_share)

        return results[:top_n]
    
    def recommend_for_user(self, user_id, top_n=10, explain=False):
        """
        Get recommendations for a specific user.
        If the user is new (or no collab model exists), fallback to popular items.
        """
        if self.collab_model is None or user_id not in self.collab_model._user_to_idx:
            # Cold start fallback for new user
            return self._cold_start_fallback(title=None, top_n=top_n)

        collab_recs = self.collab_model.predict_for_user(user_id, top_n=top_n * 3)
        
        results = []
        for r in collab_recs[:top_n]:
            item_title = r['title']

            row_data = self.content_model.df[self.content_model.df['title'] == item_title]
            category = self._category_map.get(item_title, '')
            description = ''
            top_reviews = []
            if len(row_data) > 0:
                description = str(row_data.iloc[0].get('description', ''))[:200]
                tp = row_data.iloc[0].get('top_reviews', [])
                top_reviews = tp if isinstance(tp, list) else []

            hybrid_score = r.get('predicted_score', 0.0)
            rating = self._rating_map.get(item_title, 0.0)

            result = {
                'title': item_title,
                'content_score': 0.0,
                'collab_score': round(hybrid_score, 4),
                'sentiment_score': round((self._sentiment_map.get(item_title, 0.0) + 1) / 2, 4),
                'hybrid_score': round(hybrid_score, 4),
                'rating': round(rating, 2),
                'category': category,
                'description': description,
                'top_reviews': top_reviews,
            }
            results.append(result)

        # Apply causal debiasing on the user path as well, consistent with
        # the item-based recommend() path.
        if self.use_causal_debiasing and self._debiaser is not None:
            score_key = (
                self._causal_config.score_key
                if self._causal_config is not None
                else 'hybrid_score'
            )
            results = self._debiaser.debias_batch(results, score_key=score_key)
            results.sort(key=lambda x: x[score_key], reverse=True)

        for item in results:
            history_tracker.add_recommendation(
                user_id,
                item["title"]
            )
        return results

    def _build_explanation(
        self,
        source_title,
        candidate_title,
        content_score,
        collab_score,
        sentiment_score,
        popularity,
        alpha,
        beta,
        gamma,
        raw_item,
    ):
        content_terms = []
        if hasattr(self.content_model, 'explain_similarity'):
            content_terms = self.content_model.explain_similarity(source_title, candidate_title)

        weighted_components = {
            'content': round(alpha * content_score, 4),
            'collaborative': round(beta * collab_score, 4),
            'sentiment': round(gamma * sentiment_score, 4),
            'popularity_bonus': round(self.delta * popularity, 4),
        }
        strongest = max(weighted_components, key=weighted_components.get)

        return {
            'source_item': source_title,
            'candidate_item': candidate_title,
            'active_weights': {
                'alpha': round(alpha, 4),
                'beta': round(beta, 4),
                'gamma': round(gamma, 4),
            },
            'component_scores': {
                'content': round(content_score, 4),
                'collaborative': round(collab_score, 4),
                'sentiment': round(sentiment_score, 4),
                'raw_content': round(raw_item['raw_content'], 4),
                'raw_collaborative': round(raw_item['raw_collab'], 4),
                'raw_sentiment': round(raw_item['raw_sentiment'], 4),
            },
            'weighted_components': weighted_components,
            'top_content_terms': content_terms,
            'signals': {
                'strongest_component': strongest,
                'collaborative_match': raw_item['raw_collab'] > 0,
                'sentiment_polarity': self._sentiment_label(raw_item['raw_sentiment']),
                'popularity': round(popularity, 4),
            },
        }

    @staticmethod
    def _sentiment_label(score):
        if score > 0.2:
            return 'positive'
        if score < -0.2:
            return 'negative'
        return 'neutral'

    def set_online_updater(self, updater):
        """Attach an optional OnlineUpdater-like object exposing `ingest(...)`.

        This method only stores the reference; behaviour remains unchanged
        unless `apply_interaction` is called by the application.
        """
        self.online_updater = updater

    def apply_interaction(self, user_id, item_title, rating=None, sentiment=None, timestamp=None):
        """Best-effort incremental update of internal signals for a single interaction.

        - Delegates to attached `online_updater.ingest(...)` when present; otherwise
          performs lightweight local updates to review counts, popularity,
          rating and sentiment aggregates, and appends to `collab_model.df` if available.
        - Returns True on success, False on error.
        """
        # Delegate to external updater if provided
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
                # fallback to local best-effort updates
                pass

        try:
            prev = int(self._review_count_map.get(item_title, 0))
            new_count = prev + 1
            self._review_count_map[item_title] = new_count

            # popularity update relative to tracked max
            try:
                max_reviews = max(self._review_count_map.values()) if self._review_count_map else new_count
            except Exception:
                max_reviews = new_count
            self._popularity_map[item_title] = (new_count / max_reviews) if max_reviews > 0 else 0.0

            if rating is not None:
                try:
                    prev_rating = float(self._rating_map.get(item_title, 0.0))
                    prev_n = prev if prev > 0 else 0
                    raw_avg = (prev_rating * prev_n + float(rating)) / (prev_n + 1) if (prev_n + 1) > 0 else float(rating)
                    try:
                        global_avg = float(np.mean(list(self._rating_map.values()))) if self._rating_map else 3.0
                    except Exception:
                        global_avg = 3.0
                    self._rating_map[item_title] = bayesian_rating(raw_avg, new_count, global_avg=global_avg)
                except Exception:
                    pass

            if sentiment is not None:
                try:
                    prev_sent = self._sentiment_map.get(item_title)
                    if prev_sent is None:
                        self._sentiment_map[item_title] = float(sentiment)
                    else:
                        self._sentiment_map[item_title] = (float(prev_sent) * prev + float(sentiment)) / (prev + 1)
                except Exception:
                    pass

            # append to collab_model.df if available
            try:
                if self.collab_model is not None and hasattr(self.collab_model, 'df'):
                    import pandas as pd
                    row = {'user_id': user_id, 'title': item_title}
                    if rating is not None:
                        row['rating'] = float(rating)
                    if timestamp is not None:
                        row['timestamp'] = timestamp
                    self.collab_model.df = pd.concat([self.collab_model.df, pd.DataFrame([row])], ignore_index=True)
            except Exception:
                pass

            return True
        except Exception:
            return False

    def _cold_start_fallback(self, title, top_n, target_catalog=None):
        """
        Fallback when no model data exists for the title.
        Returns popular items from the same category or global popularity.
        """
        if self.item_df is None:
            return []

        df = self.item_df
        if target_catalog and 'catalog' in df.columns:
            df = df[df['catalog'].str.lower() == target_catalog.lower()]

        target_cat = self._category_map.get(title, '')
        if target_cat:
            cat_items = df[df['category'] == target_cat]
            if len(cat_items) >= top_n:
                df = cat_items

        return self.get_popular_fallback_items(
            top_n=top_n,
            source_df=df,
            exclude_title=title,
        )

    def get_popular_fallback_items(self, top_n=5, source_df=None, exclude_title=None):
        """
        Return globally popular items when personalization produces no candidates.
        """
        if self.item_df is None and source_df is None:
            return []

        df = source_df if source_df is not None else self.item_df
        if df is None or len(df) == 0:
            return []

        df = df.copy()
        if exclude_title is not None and 'title' in df.columns:
            df = df[df['title'] != exclude_title]
            global_avg = 3.0
        # Sort by Bayesian rating
        if 'rating' in df.columns and 'review_count' in df.columns:
            df['_bayesian'] = df.apply(lambda r: bayesian_rating(r['rating'], r.get('review_count', 0), global_avg), axis=1)
            df['_bayesian'] = df.apply(
                lambda r: bayesian_rating(r['rating'], r.get('review_count', 0), global_avg), axis=1
            )
            df = df.sort_values(
                ['_bayesian', 'review_count'],
                ascending=[False, False],
            )
        elif 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False)
        elif 'review_count' in df.columns:
            df = df.sort_values('review_count', ascending=False)

        results = []
        for _, row in df.head(top_n).iterrows():
            results.append({
                'title': row['title'],
                'content_score': 0.0,
                'collab_score': 0.0,
                'sentiment_score': (row.get('avg_sentiment', 0) + 1) / 2,
                'hybrid_score': round(self._rating_map.get(row['title'], 0) / 5, 4),
                'rating': round(float(row.get('rating', 0)), 2),
                'category': row.get('category', ''),
                'description': str(row.get('description', ''))[:200],
                'top_reviews': [],
            })
        return results