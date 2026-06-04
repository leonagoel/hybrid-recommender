"""
Hybrid Recommender
Combines content-based, collaborative, and NLP sentiment scores
using a weighted scoring system with normalization and re-ranking.

Improvements:
- Bayesian average rating to prevent rating bias
- Popularity-based cold start fallback
- Category warm-start for new users
- Better weight redistribution
- Optional causal debiasing via Inverse Propensity Scoring (IPS)
"""
import logging
import math
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

from src.model.causal_config import CausalConfig  # noqa: E402
from src.model.causal_model import CausalDebiaser  # noqa: E402

VALID_CANDIDATE_SOURCES = {"hybrid", "neural", "mixed"}


def bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10):
    """
    Bayesian average: smooths ratings toward the global mean.
    Items with few votes get pulled toward the average.
    """
    v = review_count
    m = min_votes
    C = global_avg
    return (v / (v + m)) * rating + (m / (v + m)) * C


class HybridRecommender:
    def __init__(self, content_model, collab_model=None, item_df=None,
                 alpha=0.4, beta=0.35, gamma=0.25,
                 normalization='minmax', weight_matrix=None,
                 use_causal_debiasing=False, causal_lambda=0.5, causal_clip=5.0,
                 causal_config=None, model_kwargs=None,
                 retrieval_model=None, neural_candidate_k=100,
                 kg_model=None, delta=0.0):
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
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fairness_enabled = False
        self.fairness_key = "category"
        self.fairness_max_share = 1.0
        self.kg_model = kg_model
        self.delta = delta
        self.retrieval_model = retrieval_model
        self.neural_candidate_k = max(1, int(neural_candidate_k or 100))
        self.last_candidate_context = {
            "requested": "hybrid",
            "effective": "hybrid",
            "fallback": False,
            "neural_candidate_count": 0,
        }
        self.online_updater = None

        # Expose model kwargs explicitly as structural configuration dictionaries
        self.model_kwargs = model_kwargs or {}

        # Apply exposed parameters if dynamic updates are supplied on runtime triggers
        if self.collab_model and self.model_kwargs:
            n_factors = self.model_kwargs.get("n_factors")
            use_implicit = self.model_kwargs.get("use_implicit")

            # Re-initialize or pass hyperparameters down safely if explicitly specified
            if n_factors is not None and hasattr(self.collab_model, 'n_factors'):
                self.collab_model.n_factors = n_factors
            if use_implicit is not None and hasattr(self.collab_model, 'use_implicit'):
                self.collab_model.use_implicit = use_implicit

        # # normalization: 'minmax' or 'zscore'
        self.normalization = normalization
        # dynamic weighting matrix (dict of context -> (alpha,beta,gamma))
        self.weight_matrix = weight_matrix or {}

        # Fairness defaults
        self.fairness_enabled = False
        self.fairness_key = 'category'
        self.fairness_max_share = 1.0

        # Causal debiasing — prefer CausalConfig when provided; fall back to raw params.
        # This keeps the old float-based API fully working while adding structured config.
        if causal_config is not None:
            # CausalConfig path: validate once, then build debiaser if enabled
            causal_config.validate()
            self.use_causal_debiasing = causal_config.enabled
            self._debiaser: CausalDebiaser | None = (
                CausalDebiaser.from_config(item_df, causal_config)
                if causal_config.enabled and item_df is not None
                else None
            )
            # Store config for introspection (e.g. API response, Streamlit UI)
            self._causal_config: CausalConfig | None = causal_config
        else:
            # Legacy raw-param path — unchanged behaviour
            self.use_causal_debiasing = use_causal_debiasing
            self._debiaser = (
                CausalDebiaser(item_df, blend_lambda=causal_lambda, clip_max=causal_clip)
                if use_causal_debiasing and item_df is not None
                else None
            )
            self._causal_config = None

        # Initialize fairness parameters
        self.fairness_enabled = False
        self.fairness_key = 'category'
        self.fairness_max_share = 1.0

        # Build sentiment + rating lookups
        self._sentiment_map = {}
        self._rating_map = {}
        self._review_count_map = {}
        self._category_map = {}
        self._popularity_map = {}
        self._catalog_map = {}
        self._item_id_to_title = {}
        self._title_set = set()

        if item_df is not None:
            global_avg = item_df['rating'].mean() if 'rating' in item_df.columns else 3.0

            for _, row in item_df.iterrows():
                title = row['title']
                self._title_set.add(title)
                if 'avg_sentiment' in item_df.columns:
                    self._sentiment_map[title] = row['avg_sentiment']

                raw_rating = float(row.get('rating', 0))
                review_count = row.get('review_count', 0)

                if np.isnan(review_count):
                    review_count = 0

                review_count = int(review_count)
                self._review_count_map[title] = review_count
                self._rating_map[title] = bayesian_rating(
                    raw_rating, review_count, global_avg
                )
                self._category_map[title] = row.get('category', '')
                self._catalog_map[title] = row.get('catalog', '')
                item_id = row.get('item_id', row.get('id', None))
                if item_id is not None and not (isinstance(item_id, float) and np.isnan(item_id)):
                    self._item_id_to_title[item_id] = title
                    self._item_id_to_title[str(item_id)] = title

            # Popularity rank (0-1 scale, higher = more popular)
            if 'review_count' in item_df.columns:
                max_reviews = item_df['review_count'].max()
                if max_reviews > 0:
                    for _, row in item_df.iterrows():
                        self._popularity_map[row['title']] = (
                            row['review_count'] / max_reviews
                        )

    def set_weights(self, alpha, beta, gamma):
        """Update the scoring weights. Normalized to sum to 1."""
        if any(math.isnan(w) for w in [alpha, beta, gamma]):
            raise ValueError("Weights must be finite numbers")
        if any(w < 0 for w in [alpha, beta, gamma]):
            raise ValueError("Weights must be non-negative")
        total = alpha + beta + gamma
        if total == 0:
            total = 1
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total

    def get_weights(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}

    def set_fairness(self, enabled=None, key=None, max_share=None):
        if enabled is not None:
            self.fairness_enabled = bool(enabled)
        if key is not None:
            self.fairness_key = key or 'category'
        if max_share is not None:
            try:
                self.fairness_max_share = float(max_share)
            except Exception:
                self.fairness_max_share = 1.0

    def get_fairness(self):
        return {
            'enabled': self.fairness_enabled,
            'key': self.fairness_key,
            'max_share': self.fairness_max_share,
        }

    def _fair_rerank(self, results, top_n, key, max_share):
        """
        Lightweight fairness-aware re-ranking to reduce over-exposure of a single group.

        Keeps hybrid_score ordering as much as possible while enforcing a max-per-group
        cap in the final top_n list.
        """
        if not results or top_n <= 1:
            return results[:top_n]

        try:
            max_share = float(max_share)
        except Exception:
            max_share = 1.0

        if not (0 < max_share <= 1):
            max_share = 1.0

        max_per_group = max(1, int(math.ceil(max_share * top_n)))
        key = key or 'category'

        group_counts = {}
        selected = []
        overflow = []

        for item in results:
            group = str(item.get(key, '') or '').strip().casefold() or 'unknown'
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

    def _normalize(self, scores):
        """Backward-compatible alias for the configured normalizer."""
        return self._normalize_scores(scores)

    def _normalize_scores(self, scores):
        """Normalize a list of numeric scores to [0,1].

        Supports 'minmax' and 'zscore'. When std is zero, falls back to
        constant 0.5 for all values.
        """
        if not scores:
            return scores
        arr = np.array(scores, dtype=float)
        if self.normalization == 'zscore':
            mu = float(np.nanmean(arr))
            sigma = float(np.nanstd(arr))
            if sigma == 0 or math.isnan(sigma):
                return [0.5] * len(arr)
            # map z-score to standard normal CDF to get values in (0,1)
            z = (arr - mu) / sigma
            cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
            return [float(v) for v in cdf]
        # default: min-max
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx - mn == 0 or math.isnan(mn) or math.isnan(mx):
            return [0.5] * len(arr)
        return [float((v - mn) / (mx - mn)) for v in arr]

    def _get_active_weights(self, base_a, base_b, base_g, user_id=None, candidate_titles=None):
        """Resolve active weights using configured weight_matrix and runtime signals.

        The matrix keys can include: 'default', 'cold_user', 'warm_user', 'no_collab',
        'no_sentiment', or 'category:<Name>' to override base weights for specific
        contexts. The returned weights are normalized to sum to 1.
        """
        a, b, g = base_a, base_b, base_g

        # Apply matrix by priority: default -> category -> user signals -> feature absence
        if 'default' in self.weight_matrix:
            da, db, dg = self.weight_matrix['default']
            a, b, g = da, db, dg

        # category overrides (if candidate_titles provided, pick most common category)
        try:
            if candidate_titles and self.item_df is not None:
                cats = self.item_df[self.item_df['title'].isin(candidate_titles)]['category'].dropna().tolist()
                if cats:
                    # use the modal category
                    top_cat = Counter(cats).most_common(1)[0][0]
                    key = f'category:{top_cat}'
                    if key in self.weight_matrix:
                        a, b, g = self.weight_matrix[key]
        except Exception:
            logger.warning("Failed to apply weight_matrix category override", exc_info=True)

        # user signals
        if user_id and self.collab_model and hasattr(self.collab_model, 'df'):
            try:
                user_interacts = int(len(self.collab_model.df[self.collab_model.df['user_id'] == user_id]))
                if 'warm_user' in self.weight_matrix and user_interacts > 10:
                    a, b, g = self.weight_matrix['warm_user']
                if 'cold_user' in self.weight_matrix and user_interacts < 3:
                    a, b, g = self.weight_matrix['cold_user']
            except Exception:
                logger.warning("Failed to check user interaction count for weight matrix", exc_info=True)

        # feature absence overrides
        if self.collab_model is None and 'no_collab' in self.weight_matrix:
            a, b, g = self.weight_matrix['no_collab']
        if not self._sentiment_map and 'no_sentiment' in self.weight_matrix:
            a, b, g = self.weight_matrix['no_sentiment']

        # Fallback: if matrix entries are partial tuples, keep bases
        try:
            a = float(a)
            b = float(b)
            g = float(g)
        except Exception:
            a, b, g = base_a, base_b, base_g

        total = a + b + g
        if total <= 0:
            return base_a, base_b, base_g
        return a / total, b / total, g / total

    def _normalize_candidate_source(self, candidate_source):
        source = (candidate_source or "hybrid").strip().lower()
        if source not in VALID_CANDIDATE_SOURCES:
            raise ValueError(
                f"candidate_source must be one of {sorted(VALID_CANDIDATE_SOURCES)}"
            )
        return source

    def _normalize_candidate_titles(self, titles):
        if not titles:
            return []
        normalized = []
        seen = set()
        for title in titles:
            if not title:
                continue
            title = str(title)
            if title in seen:
                continue
            seen.add(title)
            normalized.append(title)
        return normalized

    def _resolve_item_id_to_title(self, item_id):
        if item_id in self._item_id_to_title:
            return self._item_id_to_title[item_id]
        item_id_text = str(item_id)
        if item_id_text in self._item_id_to_title:
            return self._item_id_to_title[item_id_text]
        if item_id_text in self._title_set:
            return item_id_text
        return None

    def _resolve_neural_candidate_titles(self, user_id, top_k=None):
        if not user_id or self.retrieval_model is None:
            return []

        top_k = top_k or self.neural_candidate_k
        try:
            if hasattr(self.retrieval_model, "retrieve_candidates_for_user"):
                item_ids = self.retrieval_model.retrieve_candidates_for_user(user_id, top_k=top_k)
            else:
                user_map = getattr(self.retrieval_model, "user_id_map", {})
                user_idx = user_map.get(user_id)
                if user_idx is None:
                    return []
                item_ids = self.retrieval_model.retrieve_candidates(user_idx, top_k=top_k)
        except Exception:
            logger.warning("Neural candidate retrieval failed; falling back to hybrid.", exc_info=True)
            return []

        titles = []
        for item_id in item_ids or []:
            title = self._resolve_item_id_to_title(item_id)
            if title:
                titles.append(title)
        return self._normalize_candidate_titles(titles)

    def _candidate_allowed(self, title, target_catalog=None):
        if self._title_set and title not in self._title_set:
            return False
        if target_catalog and self._catalog_map:
            item_catalog = self._catalog_map.get(title, "")
            if str(item_catalog).lower() != str(target_catalog).lower():
                return False
        return True

    def _content_candidate_scores(self, source_title, candidate_titles):
        if not candidate_titles:
            return {}
        try:
            title_to_idx = getattr(self.content_model, "_title_to_idx", {})
            source_idx = title_to_idx.get(str(source_title).lower())
            if source_idx is None:
                return {}

            indices = []
            index_to_title = []
            for title in candidate_titles:
                idx = title_to_idx.get(str(title).lower())
                if idx is not None:
                    indices.append(idx)
                    index_to_title.append(title)

            if not indices:
                return {}

            from sklearn.metrics.pairwise import cosine_similarity

            matrix = self.content_model.matrix
            query_vec = matrix[source_idx].reshape(1, -1)
            scores = cosine_similarity(query_vec, matrix[indices]).flatten()
            return {
                title: float(score)
                for title, score in zip(index_to_title, scores, strict=False)
            }
        except Exception:
            logger.warning("Failed to score explicit content candidates.", exc_info=True)
            return {}

    def _collab_candidate_scores(self, source_title, candidate_titles):
        if not candidate_titles or not self.collab_model:
            return {}
        try:
            title_to_idx = getattr(self.collab_model, "_title_to_idx", {})
            source_idx = title_to_idx.get(source_title)
            if source_idx is None:
                return {}

            candidate_indices = []
            index_to_title = []
            for title in candidate_titles:
                idx = title_to_idx.get(title)
                if idx is not None:
                    candidate_indices.append(idx)
                    index_to_title.append(title)

            if not candidate_indices:
                return {}

            from sklearn.metrics.pairwise import cosine_similarity

            item_factors = self.collab_model.item_factors
            query_vec = item_factors[:, source_idx].reshape(1, -1)
            candidate_matrix = item_factors[:, candidate_indices].T
            scores = cosine_similarity(query_vec, candidate_matrix).flatten()
            return {
                title: float(score)
                for title, score in zip(index_to_title, scores, strict=False)
            }
        except Exception:
            logger.warning("Failed to score explicit collaborative candidates.", exc_info=True)
            return {}

    def _set_candidate_context(
        self,
        requested,
        effective,
        fallback=False,
        neural_candidate_count=0,
    ):
        self.last_candidate_context = {
            "requested": requested,
            "effective": effective,
            "fallback": bool(fallback),
            "neural_candidate_count": int(neural_candidate_count or 0),
        }

    def recommend(
        self,
        title,
        user_id=None,
        top_n=10,
        explain=False,
        target_catalog=None,
        weights=None,
        fairness=None,
        fairness_key=None,
        fairness_max_share=None,
        diversity=0.0,
        serendipity=0.0,
        candidate_source="hybrid",
        candidate_titles=None,
        candidate_k=None,
    ):
        """
        Get hybrid recommendations for a given item title.
        Returns list of dicts sorted by hybrid_score.
        """
        source_title = title
        requested_source = self._normalize_candidate_source(candidate_source)
        effective_source = requested_source
        candidate_limit = max(top_n * 3, int(candidate_k or self.neural_candidate_k))
        explicit_titles = self._normalize_candidate_titles(candidate_titles)
        neural_titles = []
        filter_titles = set(explicit_titles) if explicit_titles else None
        supplement_titles = list(explicit_titles)
        fallback_to_hybrid = False

        if requested_source in {"neural", "mixed"}:
            neural_titles = self._resolve_neural_candidate_titles(
                user_id=user_id,
                top_k=candidate_limit,
            )
            if neural_titles:
                if requested_source == "neural":
                    filter_titles = set(neural_titles)
                else:
                    supplement_titles.extend(neural_titles)
            elif requested_source == "neural" and not explicit_titles:
                effective_source = "hybrid"
                filter_titles = None
                fallback_to_hybrid = True

        # 1. Content-based scores
        content_recs = self.content_model.recommend(
            source_title,
            top_n=candidate_limit,
            target_catalog=target_catalog,
        )
        content_map = {}

        for r in content_recs:
            if not isinstance(r, dict):
                continue

            rec_title = r.get("title")
            if rec_title:
                content_map[rec_title] = r.get("content_score", 0.0)

        # 2. Collaborative scores
        collab_map = {}
        if self.collab_model:
            collab_recs = self.collab_model.recommend(
                source_title,
                top_n=candidate_limit,
                target_catalog=target_catalog,
            )
            for r in collab_recs:
                if not isinstance(r, dict):
                    continue

                rec_title = r.get("title")
                if not rec_title:
                    continue

                collab_map[rec_title] = r.get("collab_score", 0.0)

        # 3. Build unified candidates
        candidate_pool_titles = self._normalize_candidate_titles(
            list(filter_titles or []) + supplement_titles
        )
        if candidate_pool_titles:
            content_map.update(
                self._content_candidate_scores(source_title, candidate_pool_titles)
            )
            collab_map.update(
                self._collab_candidate_scores(source_title, candidate_pool_titles)
            )

        candidates = {}
        for rec_title, score in content_map.items():
            if rec_title == source_title:
                continue
            if not self._candidate_allowed(rec_title, target_catalog=target_catalog):
                continue
            candidates[rec_title] = {
                'title': rec_title,
                'raw_content': score,
                'raw_collab': collab_map.get(rec_title, 0.0),
                'raw_sentiment': self._sentiment_map.get(rec_title, 0.0),
            }

        for rec_title, score in collab_map.items():
            if rec_title == source_title:
                continue
            if not self._candidate_allowed(rec_title, target_catalog=target_catalog):
                continue
            if rec_title not in candidates:
                candidates[rec_title] = {
                    'title': rec_title,
                    'raw_content': 0.0,
                    'raw_collab': score,
                    'raw_sentiment': self._sentiment_map.get(rec_title, 0.0),
                }

        for rec_title in candidate_pool_titles:
            if rec_title == source_title:
                continue
            if not self._candidate_allowed(rec_title, target_catalog=target_catalog):
                continue
            candidates.setdefault(
                rec_title,
                {
                    'title': rec_title,
                    'raw_content': content_map.get(rec_title, 0.0),
                    'raw_collab': collab_map.get(rec_title, 0.0),
                    'raw_sentiment': self._sentiment_map.get(rec_title, 0.0),
                },
            )

        if filter_titles is not None:
            candidates = {
                rec_title: data
                for rec_title, data in candidates.items()
                if rec_title in filter_titles
            }

        if not candidates:
            if requested_source == "neural":
                recs = self.recommend(
                    source_title,
                    user_id=user_id,
                    top_n=top_n,
                    explain=explain,
                    target_catalog=target_catalog,
                    weights=weights,
                    fairness=fairness,
                    fairness_key=fairness_key,
                    fairness_max_share=fairness_max_share,
                    diversity=diversity,
                    serendipity=serendipity,
                    candidate_source="hybrid",
                )
                self._set_candidate_context(
                    requested_source,
                    "hybrid",
                    fallback=True,
                    neural_candidate_count=len(neural_titles),
                )
                return recs
            self._set_candidate_context(
                requested_source,
                effective_source,
                fallback=fallback_to_hybrid,
                neural_candidate_count=len(neural_titles),
            )
            return self._cold_start_fallback(source_title, top_n, target_catalog=target_catalog)

        items = list(candidates.values())
        a, b, g = self._get_active_weights(
            self.alpha,
            self.beta,
            self.gamma,
            user_id=user_id,
            candidate_titles=list(candidates.keys()),
        )
        if weights:
            try:
                if isinstance(weights, dict):
                    a = float(weights.get("alpha", a))
                    b = float(weights.get("beta", b))
                    g = float(weights.get("gamma", g))
                else:
                    a, b, g = [float(w) for w in weights]
                total = a + b + g
                if total > 0:
                    a, b, g = a / total, b / total, g / total
            except Exception:
                logger.warning("Invalid runtime weights supplied; using active defaults.")

        # 4. Normalize each component using configured normalizer
        content_raws = [it['raw_content'] for it in items]
        collab_raws = [it['raw_collab'] for it in items]
        sentiment_raws = [it['raw_sentiment'] for it in items]

        content_scores = self._normalize_scores(content_raws)
        collab_scores = self._normalize_scores(collab_raws)
        sentiment_scores = self._normalize_scores(sentiment_raws)

        kg_scores = [0.0] * len(items)
        if self.kg_model:
            try:
                kg_recs = self.kg_model.recommend(source_title, top_n=candidate_limit)
                kg_map = {
                    item['title']: item['kg_score']
                    for item in kg_recs
                    if isinstance(item, dict) and item.get('title')
                }
                kg_scores = self._normalize_scores(
                    [kg_map.get(item['title'], 0.0) for item in items]
                )
            except Exception:
                logger.warning("Knowledge-graph scoring failed; ignoring KG scores.", exc_info=True)

        # 6. Compute hybrid score with capped popularity boost to protect [0, 1] constraint
        results = []
        for i, item in enumerate(items):
            hybrid_base = (
                a * content_scores[i]
                + b * collab_scores[i]
                + g * sentiment_scores[i]
            )
            if self.kg_model and self.delta > 0:
                hybrid_base = (
                    (1 - self.delta) * hybrid_base
                    + self.delta * kg_scores[i]
                )

            # Light popularity boost (max 5% bonus) scaled to not leak over 1.0 boundary contract
            popularity = self._popularity_map.get(item['title'], 0.5)
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
                    source_title,
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
            return self.get_popular_fallback_items(top_n=top_n, exclude_title=source_title)

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

        self._set_candidate_context(
            requested_source,
            effective_source,
            fallback=fallback_to_hybrid,
            neural_candidate_count=len(neural_titles),
        )

        apply_fairness = self.fairness_enabled if fairness is None else bool(fairness)
        if apply_fairness:
            key = fairness_key or self.fairness_key
            max_share = self.fairness_max_share if fairness_max_share is None else fairness_max_share
            return self._fair_rerank(results, top_n, key, max_share)

        return results[:top_n]

    def recommend_for_user(
        self,
        user_id,
        top_n=10,
        explain=False,
        candidate_source="hybrid",
        candidate_k=None,
    ):
        """
        Get recommendations for a specific user.
        If the user is new (or no collab model exists), fallback to popular items.
        """
        requested_source = self._normalize_candidate_source(candidate_source)
        if requested_source in {"neural", "mixed"}:
            seed_title = None
            try:
                if self.collab_model is not None and hasattr(self.collab_model, "df"):
                    user_rows = self.collab_model.df[self.collab_model.df["user_id"] == user_id]
                    if not user_rows.empty:
                        seed_title = user_rows.iloc[-1]["title"]
            except Exception:
                seed_title = None

            if seed_title:
                recs = self.recommend(
                    seed_title,
                    user_id=user_id,
                    top_n=top_n,
                    explain=explain,
                    candidate_source=requested_source,
                    candidate_k=candidate_k,
                )
                if recs:
                    return recs

            if requested_source == "neural":
                self._set_candidate_context(
                    requested_source,
                    "hybrid",
                    fallback=True,
                    neural_candidate_count=0,
                )

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
            'popularity_bonus': round(0.05 * popularity, 4),
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
