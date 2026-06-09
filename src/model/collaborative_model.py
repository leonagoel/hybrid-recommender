"""
Collaborative Recommender
Uses Truncated SVD (matrix factorization) on the user-item interaction
matrix to discover latent factors and predict ratings.

Public interface
----------------
  recommend(title, top_n=10, target_catalog=None)
      Item-item similarity recommendations.
      Returns: List[{'title': str, 'collab_score': float}]

  predict_for_user(user_id, top_n=10, target_catalog=None)
      Personalised user recommendations via latent-factor dot product.
      Returns: List[{'title': str, 'predicted_score': float}]
      Cold-start users receive popularity-based fallback with
      {'title': str, 'predicted_score': float, 'fallback': True}.

  predict_rating(user_id, title) -> Optional[float]
      Point-estimate of the rating user would give to item.

Score-key contract
------------------
  recommend()          -> 'collab_score'    (item-item CF)
  predict_for_user()   -> 'predicted_score' (user-item MF)
  _popularity_fallback() accepts a score_key parameter so it can produce
  either key consistently depending on the calling method.

Improvements:
- Implicit feedback support (views, purchases -> confidence weights)
- Adaptive n_factors for sparse matrices
- User-based personalized recommendations via SVD
- Type-safe user_id matching (str/int coercion)
"""
__all__ = ["CollaborativeRecommender"]

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

from src.model.validation import validate_recommendations
import gc

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    def __init__(
        self,
        interaction_df: pd.DataFrame,
        n_factors: int = 50,
        use_implicit: bool = True,
    ) -> None:
        """Initialise the collaborative recommender and fit SVD.

        Parameters
        ----------
        interaction_df:
            DataFrame with columns 'user_id', 'title', 'rating'.
            Optionally 'views' and 'purchases' for implicit feedback.
        n_factors:
            Number of latent factors for SVD decomposition.
        use_implicit:
            When True, blend in implicit feedback signals when present.
        """
        self.df = interaction_df.copy()

        self.users  = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')

        self._user_to_idx  = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list    = list(self.titles.cat.categories)

        row  = self.users.cat.codes.values
        col  = self.titles.cat.codes.values
        data = self.df['rating'].values.astype(float)

        if use_implicit:
            alpha_implicit = 0.5
            if 'purchases' in self.df.columns:
                data = data + alpha_implicit * self.df['purchases'].fillna(0).values
            if 'views' in self.df.columns:
                data = data + (alpha_implicit * 0.5) * self.df['views'].fillna(0).values

        n_users = len(self._user_to_idx)
        n_items = len(self._title_to_idx)
        self.user_item_sparse = coo_matrix(
            (data, (row, col)), shape=(n_users, n_items)
        ).tocsr()
        # Cleanup temporary arrays to free memory
        del row, col, data
        import gc
        gc.collect()

        # Adaptive rank: reduce factors dynamically for sparse matrices
        min_dim = min(self.user_item_sparse.shape)
        density = (self.user_item_sparse.nnz / (n_users * n_items)
                   if (n_users * n_items) > 0 else 0)

        # FIX FOR ISSUE #483: Prevent array out-of-bounds collapse on small matrices
        if min_dim <= 2:
            self.svd = None
            self.user_factors = np.ones((n_users, 1))
            self.item_factors = np.ones((1, n_items))
        else:
            if density < 0.001:
                n_components = min(20, min_dim - 1)
            elif density < 0.01:
                n_components = min(30, min_dim - 1)
            else:
                n_components = min(n_factors, min_dim - 1)

            n_components = min(n_components, n_users - 1, n_items - 1)
            n_components = max(1, n_components)

            try:
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.user_factors = self.svd.fit_transform(self.user_item_sparse)
                self.item_factors = self.svd.components_
                # Cleanup heavy SVD objects after use to reduce memory pressure
                del self.svd
                import gc
                gc.collect()
            except ValueError:
                self.svd = None
                self.user_factors = np.ones((n_users, 1))
                self.item_factors = np.ones((1, n_items))

        # Build catalog map if catalog column is present in interaction_df
        self._catalog_map: Dict[str, str] = {}
        if 'catalog' in self.df.columns:
            self._catalog_map = dict(zip(self.df['title'], self.df['catalog']))

        # Online SGD learning rate — used by online_update() (Issue #1596)
        self._lr = 0.01
        self._reg = 0.02

    def online_update(self, user_id: str, title: str, rating: float) -> None:
        """
        Issue #1596 — Incremental (online) SGD update for Bayesian collaborative filtering.

        Updates the latent user and item factor vectors based on a single new
        interaction without full retraining.  Uses a regularised gradient step:

            err = rating - user_vec · item_vec
            user_vec += lr * (err * item_vec - reg * user_vec)
            item_vec += lr * (err * user_vec - reg * item_vec)

        Args:
            user_id: ID of the user who interacted.
            title:   Title of the item they interacted with.
            rating:  Observed rating / implicit feedback signal.
        """
        if user_id not in self._user_to_idx or title not in self._title_to_idx:
            # Unknown user or item — cannot update; silently skip (cold-start).
            logger.debug(
                "online_update skipped: unknown user=%s or item=%s", user_id, title
            )
            return

        u_idx = self._user_to_idx[user_id]
        i_idx = self._title_to_idx[title]

        user_vec = self.user_factors[u_idx].copy()
        # item_factors shape is (n_components, n_items) — column i is item vector
        item_vec = self.item_factors[:, i_idx].copy()

        prediction = float(np.dot(user_vec, item_vec))
        err = rating - prediction

        # Gradient step with L2 regularisation
        self.user_factors[u_idx] += self._lr * (err * item_vec - self._reg * user_vec)
        self.item_factors[:, i_idx] += self._lr * (err * user_vec - self._reg * item_vec)

        logger.debug(
            "online_update applied: user=%s item=%s rating=%.3f err=%.4f",
            user_id,
            title,
            rating,
            err,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        target_catalog: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Item-item collaborative recommendations using SVD latent space.

        Uses cosine similarity in the item latent-factor space to find items
        most similar to *title*.  Use this when the query is an *item*, not a
        user.  For personalised user recommendations use predict_for_user().

        Parameters
        ----------
        title:
            Query item title.  Unknown titles return an empty list.
        top_n:
            Maximum results to return, capped at 100.
        target_catalog:
            Optional catalog filter.

        Returns
        -------
        List[{'title': str, 'collab_score': float}]
            Sorted by 'collab_score' descending.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        top_n = min(top_n, 100)

        if title not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title]
        try:
            query_vec = self.item_factors[:, idx].reshape(1, -1)
            scores = cosine_similarity(query_vec, self.item_factors.T).flatten()
            sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        except Exception as exc:
            logger.error("recommend() similarity computation failed: %s", exc)
            sim_scores = []

        results: List[Dict[str, Any]] = []
        seen: set = set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue
            if target_catalog and self._catalog_map:
                if str(self._catalog_map.get(t, '')).lower() != str(target_catalog).lower():
                    continue
            seen.add(t)
            results.append({'title': t, 'collab_score': float(score)})
            if len(results) >= top_n:
                break

        return validate_recommendations(
            results,
            fallback_fn=lambda n: self._popularity_fallback(n, score_key='collab_score'),
            top_n=top_n,
            context="CF",
            force_padding=False,
        )

    def predict_for_user(
        self,
        user_id: Any,
        top_n: int = 10,
        target_catalog: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Personalised recommendations for a specific user.

        Uses latent-factor dot product (SVD) to score every unseen item for
        user_id.  Cold-start users (unknown user_id) receive popularity-based
        fallback results marked with 'fallback': True.

        Parameters
        ----------
        user_id:
            User identifier.  String and integer representations are matched
            interchangeably ("1" matches integer 1 in the index).
        top_n:
            Maximum results to return, capped at 100.
        target_catalog:
            Optional catalog filter.

        Returns
        -------
        List[{'title': str, 'predicted_score': float}]
            Sorted by 'predicted_score' descending.
            Cold-start items also carry 'fallback': True.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        top_n = min(top_n, 100)

        # Type-safe existence check: exact match first, then str coercion
        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            for key in self._user_to_idx:
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx:
            logger.warning(
                "Cold-start: user %r not found. Returning popularity fallback.",
                user_id,
            )
            recs = self._popularity_fallback(top_n, score_key='predicted_score')
            return validate_recommendations(
                recs,
                fallback_fn=None,
                top_n=top_n,
                context="CF",
                force_padding=False,
            )

        try:
            u_idx = self._user_to_idx[mapped_user_id]
            user_vec = self.user_factors[u_idx]
            scores = np.dot(user_vec, self.item_factors)

            seen_items = set(
                self.df[self.df['user_id'] == mapped_user_id]['title'].tolist()
            )

            scored: List[tuple] = []
            for i, score in enumerate(scores):
                t = self.title_list[i]
                if t in seen_items:
                    continue
                if target_catalog and self._catalog_map:
                    if str(self._catalog_map.get(t, '')).lower() != str(target_catalog).lower():
                        continue
                scored.append((t, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            results: List[Dict[str, Any]] = [
                {'title': t, 'predicted_score': s} for t, s in scored[:top_n]
            ]
        except Exception as exc:
            logger.error("predict_for_user() prediction failed: %s", exc)
            results = []

        return validate_recommendations(
            results,
            fallback_fn=lambda n: self._popularity_fallback(n, score_key='predicted_score'),
            top_n=top_n,
            context="CF",
            force_padding=False,
        )

    def predict_rating(self, user_id: Any, title: str) -> Optional[float]:
        """Predict the rating a user would give to an item.

        Parameters
        ----------
        user_id:
            User identifier (matched with type coercion).
        title:
            Item title.

        Returns
        -------
        float or None
            Raw dot-product score, or None when either user_id or title is
            unknown.
        """
        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            for key in self._user_to_idx:
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        u_idx = self._user_to_idx[mapped_user_id]
        i_idx = self._title_to_idx[title]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _popularity_fallback(
        self,
        top_n: int = 10,
        score_key: str = 'predicted_score',
    ) -> List[Dict[str, Any]]:
        """Return top-N popular items as a cold-start fallback.

        Parameters
        ----------
        top_n:
            Number of items to return.
        score_key:
            Score field name in the returned dicts.  Pass 'collab_score' when
            this fallback backs recommend(); pass 'predicted_score' (default)
            when it backs predict_for_user().

        Returns
        -------
        List[{score_key: float, 'title': str, 'fallback': True}]
        """
        logger.info("Using popularity-based fallback (score_key=%r).", score_key)

        item_counts = (
            self.df.groupby('title')['rating']
            .agg(['mean', 'count'])
            .reset_index()
        )

        if not item_counts.empty and 'count' in item_counts.columns:
            top_items = item_counts.nlargest(top_n, 'count')
        elif not item_counts.empty and 'mean' in item_counts.columns:
            top_items = item_counts.nlargest(top_n, 'mean')
        else:
            try:
                from src.model.trending_model import TrendingRecommender
                trending_model = TrendingRecommender(df=self.df)
                trending_results = trending_model.get_trending_products(top_n=top_n)
                if trending_results:
                    return [
                        {
                            'title': row['title'],
                            score_key: float(row.get('avg_rating', 0.0)),
                            'fallback': True,
                        }
                        for row in trending_results
                    ]
            except Exception:
                pass
            return []

        return [
            {
                'title': row['title'],
                score_key: round(float(row.get('mean', 0.0)), 4),
                'fallback': True,
            }
            for _, row in top_items.iterrows()
        ]
