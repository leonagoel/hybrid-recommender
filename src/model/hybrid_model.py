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
import math

import numpy as np

from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.knowledge_graph import KnowledgeGraphRecommender
from src.model.validation import validate_recommendations
from src.model.recommendation_history import history_tracker
from src.model.causal_config import CausalConfig
from src.model.causal_model import CausalDebiaser
from src.model.recommendation_history import history_tracker
from src.model.multi_objective import MultiObjectiveRanker
from src.model.validation import validate_recommendations


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
                 kg_model=None, delta=0.0,
                 normalization='minmax', weight_matrix=None,
                 use_causal_debiasing=False, causal_lambda=0.5, causal_clip=5.0,
                 causal_config=None, model_kwargs=None):
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
        self.kg_model = None
        self.delta = 0.0
        self.multi_objective_ranker = MultiObjectiveRanker()
        # Multi-Armed Bandit
        self.bandit_arms = [
            (0.6, 0.3, 0.1),
            (0.5, 0.4, 0.1),
            (0.4, 0.4, 0.2),
            (0.3, 0.5, 0.2),
        ]

        self.arm_rewards = {i: 0.0 for i in range(len(self.bandit_arms))}
        self.arm_counts = {i: 0 for i in range(len(self.bandit_arms))}
        self.epsilon = 0.1
        self.kg_model = kg_model
        self.delta = delta

        # Preserve caller-provided model_kwargs; fall back to empty dict if None
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

        if item_df is not None:
            global_avg = item_df['rating'].mean() if 'rating' in item_df.columns else 3.0

            for _, row in item_df.iterrows():
                title = row['title']
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

            # Popularity rank (0-1 scale, higher = more popular)
            if 'review_count' in item_df.columns:
                max_reviews = item_df['review_count'].max()
                if max_reviews > 0:
                    for _, row in item_df.iterrows():
                        self._popularity_map[row['title']] = (
                            row['review_count'] / max_reviews
                        )

            # Optional runtime hook for online updates (attachable)
            self.online_updater = None

    def set_weights(self, alpha, beta, gamma, delta=0.0):
        """Update the scoring weights. Normalized to sum to 1."""
        import math
        if any(math.isnan(w) for w in [alpha, beta, gamma, delta]):
            raise ValueError("Weights must be finite numbers")
        if any(w < 0 for w in [alpha, beta, gamma, delta]):
            raise ValueError("Weights must be non-negative")
        total = alpha + beta + gamma + delta
        if total == 0:
            total = 1
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.delta = delta / total

    def get_weights(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
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
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma, 'delta': self.delta}

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

    def _get_active_weights(self, base_a, base_b, base_g, base_d, user_id=None, candidate_titles=None):
        """Resolve active weights using configured weight_matrix and runtime signals.

        The matrix keys can include: 'default', 'cold_user', 'warm_user', 'no_collab',
        'no_sentiment', 'no_kg', or 'category:<Name>' to override base weights for specific
        contexts. The returned weights are normalized to sum to 1.
        """
        a, b, g, d = base_a, base_b, base_g, base_d

        def unpack_weights(val, default_d=0.0):
            if len(val) == 4:
                return val[0], val[1], val[2], val[3]
            elif len(val) == 3:
                return val[0], val[1], val[2], default_d
            return val

        # Apply matrix by priority: default -> category -> user signals -> feature absence
        if 'default' in self.weight_matrix:
            a, b, g, d = unpack_weights(self.weight_matrix['default'], d)

        # category overrides (if candidate_titles provided, pick most common category)
        try:
            if candidate_titles and self.item_df is not None:
                cats = self.item_df[self.item_df['title'].isin(candidate_titles)]['category'].dropna().tolist()
                if cats:
                    # use the modal category
                    from collections import Counter
                    top_cat = Counter(cats).most_common(1)[0][0]
                    key = f'category:{top_cat}'
                    if key in self.weight_matrix:
                        a, b, g, d = unpack_weights(self.weight_matrix[key], d)
        except Exception:
            pass

        # user signals
        if user_id and self.collab_model and hasattr(self.collab_model, 'df'):
            try:
                user_interacts = int(len(self.collab_model.df[self.collab_model.df['user_id'] == user_id]))
                if 'warm_user' in self.weight_matrix and user_interacts > 10:
                    a, b, g, d = unpack_weights(self.weight_matrix['warm_user'], d)
                if 'cold_user' in self.weight_matrix and user_interacts < 3:
                    a, b, g, d = unpack_weights(self.weight_matrix['cold_user'], d)
            except Exception:
                pass

        # feature absence overrides
        if self.collab_model is None and 'no_collab' in self.weight_matrix:
            a, b, g, d = unpack_weights(self.weight_matrix['no_collab'], d)
        if not self._sentiment_map and 'no_sentiment' in self.weight_matrix:
            a, b, g, d = unpack_weights(self.weight_matrix['no_sentiment'], d)
        if self.kg_model is None and 'no_kg' in self.weight_matrix:
            a, b, g, d = unpack_weights(self.weight_matrix['no_kg'], d)

        # Fallback: if matrix entries are partial tuples, keep bases
        try:
            a = float(a)
            b = float(b)
            g = float(g)
            d = float(d)
        except Exception:
            a, b, g, d = base_a, base_b, base_g, base_d

        total = a + b + g + d
        if total <= 0:
            return base_a, base_b, base_g, base_d
        return a / total, b / total, g / total, d / total

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
    ):
        """
        Get hybrid recommendations for a given item title.
        Returns list of dicts sorted by hybrid_score.
        """
        # 1. Content-based scores
        content_recs = self.content_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)
        all_titles = {r['title'] for r in content_recs}

        # 2. Collaborative scores
        collab_map = {}
        if self.collab_model:
            collab_recs = self.collab_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)
            for r in collab_recs:
                collab_map[r['title']] = r.get('collab_score', 0.0)
                all_titles.add(r['title'])

        # 2.5 Knowledge Graph scores
        kg_map = {}
        if self.kg_model:
            kg_recs = self.kg_model.recommend(title, top_n=top_n * 3, target_catalog=target_catalog)
            for r in kg_recs:
                if not isinstance(r, dict):
                    continue
                candidate_title = r.get("title")
                if not candidate_title:
                    continue
                kg_map[candidate_title] = r.get("kg_score", 0.0)
                all_titles.add(candidate_title)

        # 3. Build unified candidates
        candidates = {}
        for r in content_recs:
            candidates[r['title']] = {
                'title': r['title'],
                'raw_content': r['content_score'],
                'raw_collab': collab_map.get(r['title'], 0.0),
                'raw_sentiment': self._sentiment_map.get(r['title'], 0.0),
                'raw_gnn': 0.0,
                'raw_kg': kg_map.get(r['title'], 0.0),
            }

        for t in collab_map:
            if t not in candidates:
                candidates[t] = {
                    'title': t,
                    'raw_content': 0.0,
                    'raw_collab': collab_map[t],
                    'raw_sentiment': self._sentiment_map.get(t, 0.0),
                    'raw_gnn': 0.0,
                    'raw_kg': kg_map.get(t, 0.0),
                }

        for t in kg_map:
            if t not in candidates:
                candidates[t] = {
                    'title': t,
                    'raw_content': 0.0,
                    'raw_collab': collab_map.get(t, 0.0),
                    'raw_sentiment': self._sentiment_map.get(t, 0.0),
                    'raw_gnn': 0.0,
                    'raw_kg': kg_map[t],
                }

        # (Removed early fallback return here so validate_recommendations can log)
        if not candidates:
            return self._cold_start_fallback(title, top_n, target_catalog=target_catalog)

        items = list(candidates.values())

        # 4. Normalize each component using configured normalizer
        content_raws = [it['raw_content'] for it in items]
        collab_raws = [it['raw_collab'] for it in items]
        sentiment_raws = [it['raw_sentiment'] for it in items]
        kg_raws = [it.get('raw_kg', 0.0) for it in items]

        content_scores = self._normalize_scores(content_raws)
        collab_scores = self._normalize_scores(collab_raws)
        sentiment_scores = self._normalize_scores(sentiment_raws)
        kg_scores = self._normalize_scores(kg_raws)

        # 5. Resolve active weights (applies weight_matrix overrides and context signals).
        if weights is not None:
            a = weights.get("alpha", self.alpha)
            b = weights.get("beta", self.beta)
            g = weights.get("gamma", self.gamma)
            d = weights.get("delta", self.delta)
            # normalize provided weights
            tot = a + b + g + d
            if tot > 0:
                a, b, g, d = a / tot, b / tot, g / tot, d / tot
        else:
            kg_scores = [0.0] * len(items)

        # 5. Resolve active weights
        if weights:
            a, b, g = float(weights.get('alpha', self.alpha)), float(weights.get('beta', self.beta)), float(weights.get('gamma', self.gamma))
            total = a + b + g
            if total > 0:
                a, b, g = a / total, b / total, g / total
            else:
                a, b, g = self.alpha, self.beta, self.gamma
        else:
            candidate_titles = [it['title'] for it in items]
            arm_id = self.select_bandit_arm()
a, b, g = self.bandit_arms[arm_id]
            a, b, g, d = self._get_active_weights(
                self.alpha, self.beta, self.gamma, self.delta,
                user_id=user_id,
                candidate_titles=list(candidates.keys()),
            )

        # 6. Compute hybrid score with capped popularity boost to protect [0, 1] constraint
        results = []
        for i, item in enumerate(items):
            hybrid_base = (
                a * content_scores[i] +
                b * collab_scores[i] +
                g * sentiment_scores[i] +
                d * kg_scores[i]
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
                'kg_score': round(kg_scores[i], 4),
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
                    kg_scores[i],
                    popularity,
                    a,
                    b,
                    g,
                    d,
                    item,
                )
            results.append(result)

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        if not explain:
            results = self.multi_objective_ranker.rerank(results)
        if not results:
            return self.get_popular_fallback_items(...)

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

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self.get_popular_fallback_items(top_n=top_n, exclude_title=title),
            top_n=top_n,
            default_fallback_items=[t for t in self.item_df["title"].tolist() if t.lower() != title.lower()] if (self.item_df is not None and "title" in self.item_df.columns) else None,
            context="hybrid"
        )
    
    def recommend_for_user(self, user_id, top_n=10, explain=False):
        """
        Get recommendations for a specific user.
        If the user is new (or no collab model exists), fallback to popular items.
        """
        mapped_user_id = user_id
        if self.collab_model is not None:
            if user_id not in self.collab_model._user_to_idx:
                for key in self.collab_model._user_to_idx.keys():
                    if str(key) == str(user_id):
                        mapped_user_id = key
                        break

        if self.collab_model is None or mapped_user_id not in self.collab_model._user_to_idx:
            # Cold start fallback for new user
            recs = self._cold_start_fallback(title=None, top_n=top_n)
            return validate_recommendations(
                recs,
                fallback_fn=lambda top_n: self.get_popular_fallback_items(top_n=top_n),
                top_n=top_n,
                default_fallback_items=self.item_df["title"].tolist() if (self.item_df is not None and "title" in self.item_df.columns) else None,
                context="hybrid"
            )

        collab_recs = self.collab_model.predict_for_user(mapped_user_id, top_n=top_n * 3)
        _tracker = getattr(self, 'history_tracker', None)
        recent_titles = _tracker.get_recent_titles(mapped_user_id) if _tracker is not None else set()

        collab_recs = [ 
            rec for rec in collab_recs
            if rec["title"] not in recent_titles
        ]
        
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
        kg_score,
        popularity,
        alpha,
        beta,
        gamma,
        delta,
        raw_item,
    ):
        content_terms = []
        if hasattr(self.content_model, 'explain_similarity'):
            content_terms = self.content_model.explain_similarity(source_title, candidate_title)

        weighted_components = {
            'content': round(alpha * content_score, 4),
            'collaborative': round(beta * collab_score, 4),
            'sentiment': round(gamma * sentiment_score, 4),
            'knowledge_graph': round(delta * kg_score, 4),
            'popularity_bonus': round(0.05 * popularity, 4),
        }
        strongest = max(weighted_components, key=weighted_components.get)
        if strongest == "content":
            explanation_text = (
                f"Recommended due to strong content similarity "
                f"with '{source_title}'."
                )

        elif strongest == "collaborative":
            explanation_text = (
                "Recommended because users with similar preferences "
                "also interacted with this item."
                )
        elif strongest == "sentiment":
            explanation_text = (
                "Recommended because it has highly positive reviews."
                )
        elif strongest == "knowledge_graph":
            explanation_text = (
                "Recommended because of strong semantic relationships in the knowledge graph."
                )

        else:
            explanation_text = (
                "Recommended because of its popularity and overall score."
                )

        return {
            'source_item': source_title,
            'candidate_item': candidate_title,
            'explanation_text': explanation_text,
            'active_weights': {
                'alpha': round(alpha, 4),
                'beta': round(beta, 4),
                'gamma': round(gamma, 4),
                'delta': round(delta, 4),
            },
            'component_scores': {
                'content': round(content_score, 4),
                'collaborative': round(collab_score, 4),
                'sentiment': round(sentiment_score, 4),
                'knowledge_graph': round(kg_score, 4),
                'raw_content': round(raw_item['raw_content'], 4),
                'raw_collaborative': round(raw_item['raw_collab'], 4),
                'raw_sentiment': round(raw_item['raw_sentiment'], 4),
                'raw_knowledge_graph': round(raw_item.get('raw_kg', 0.0), 4),
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

    def register_feedback(self, arm_id, clicked=True):
        reward = 1.0 if clicked else 0.0
        self.update_bandit_reward(arm_id, reward)
        

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
            df['_bayesian'] = df.apply(
                lambda r: bayesian_rating(
                    r['rating'],
                    r.get('review_count', 0),
                    global_avg 
                    ),
                    axis=1

                    )
            
            df = df.sort_values(
                ['_bayesian', 'review_count'],
                ascending=[False, False],
            )
        elif 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False)
        # 3. Trending items
        else:
            try:
                from src.model.trending_model import TrendingRecommender
                trending_model = TrendingRecommender(df=df)
                trending_results = trending_model.get_trending_products(top_n=top_n)
                if trending_results:
                    return [
                        {
                            'title': item['title'],
                            'content_score': 0.0,
                            'collab_score': 0.0,
                            'sentiment_score': 0.0,
                            'hybrid_score': 0.0,
                            'rating': float(item.get('avg_rating', 0.0)),
                            'category': '',
                            'description': '',
                            'top_reviews': [],
                            'fallback': True,
                        }
                        for item in trending_results
                    ]
            except Exception:
                pass
            # 4. Empty list
            return []

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
    def _diversity_rerank(self, results, top_n, diversity=0.0, serendipity=0.0):
            

            """
            Re-ranks results to reduce filter bubbles.

            Args:
                results: list of recommendation dicts sorted by hybrid_score
                top_n: number of final results to return
                diversity: 0.0 = no diversity, 1.0 = max category variety
                serendipity: 0.0 = no surprises, 1.0 = more random/unexpected items

            Returns:
                Re-ranked list of recommendations
            """
            if not results:
                return results

            if diversity == 0.0 and serendipity == 0.0:
                return results[:top_n]

            selected = []
            remaining = results.copy()
            seen_categories = []

            while len(selected) < top_n and remaining:
                best = None
                best_score = -1

                for item in remaining:
                    score = item['hybrid_score']
                    category = item.get('category', 'unknown')

                    times_seen = seen_categories.count(category)
                    diversity_penalty = diversity * times_seen * 0.2
                    score = score - diversity_penalty

                    surprise_bonus = serendipity * np.random.uniform(0, 0.3)
                    score = score + surprise_bonus

                    if score > best_score:
                        best_score = score
                        best = item

                selected.append(best)
                remaining.remove(best)
                seen_categories.append(best.get('category', 'unknown'))

            return selected
    