"""
Collaborative Recommender
Uses Truncated SVD (matrix factorization) on the user-item interaction
matrix to discover latent factors and predict ratings.

Improvements:
- Implicit feedback support (views, purchases → confidence weights)
- Adaptive n_factors for sparse matrices
- User-based personalized recommendations
"""
__all__ = ["CollaborativeRecommender"]

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from src.model.validation import validate_recommendations

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    def __init__(self, interaction_df, n_factors=50, use_implicit=True):
        """
        interaction_df: DataFrame with columns 'user_id', 'title', 'rating'.
                        Optionally 'views' and 'purchases' for implicit feedback.
        n_factors: number of latent factors for SVD decomposition.
        use_implicit: blend in implicit feedback signals if available.
        """
        self.df = interaction_df.copy()

        # Map users and items to integer indices
        self.users = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')

        self._user_to_idx = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list = list(self.titles.cat.categories)

        # Build sparse user-item matrix
        row = self.users.cat.codes.values
        col = self.titles.cat.codes.values

        # Combine explicit ratings with implicit signals
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

        # Adaptive rank: reduce factors dynamically for sparse matrices
        min_dim = min(self.user_item_sparse.shape)
        density = self.user_item_sparse.nnz / (n_users * n_items) if (n_users * n_items) > 0 else 0

        # FIX FOR ISSUE #483: Prevent array out-of-bounds collapse on small matrices
        if min_dim <= 2:
            self.svd = None
            # Matching shapes perfectly to prevent slice dimensionality failures inside recommend()
            self.user_factors = np.ones((n_users, 1))
            self.item_factors = np.ones((1, n_items))
        else:
            if density < 0.001:
                n_components = min(20, min_dim - 1)
            elif density < 0.01:
                n_components = min(30, min_dim - 1)
            else:
                n_components = min(n_factors, min_dim - 1)

            # Keep n_components safely below absolute matrix dimension boundaries
            n_components = min(n_components, n_users - 1, n_items - 1)
            n_components = max(1, n_components)

            try:
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.user_factors = self.svd.fit_transform(self.user_item_sparse)
                self.item_factors = self.svd.components_
            except ValueError:
                # Safe baseline fallback if SVD initialization constraints fail on edge-case data shapes
                self.svd = None
                self.user_factors = np.ones((n_users, 1))
                self.item_factors = np.ones((1, n_items))

        # Build catalog map if catalog column is present in interaction_df
        self._catalog_map = {}
        if 'catalog' in self.df.columns:
            self._catalog_map = self.df.groupby('title')['catalog'].first().to_dict()

    def recommend(self, title, top_n=10, target_catalog=None):
        """
        Item-item collaborative recommendations using SVD latent space.
        Returns list of dicts: [{ 'title', 'collab_score' }, ...]
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
            sim_scores = list(enumerate(scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Collaborative recommendation similarity computation failed: {e}")
            if "NaN" in str(e) or "nan" in str(e):
                return validate_recommendations(
                    [{"title": "NaN Placeholder", "collab_score": float("nan")}],
                    fallback_fn=lambda top_n: self._popularity_fallback(top_n),
                    top_n=top_n,
                    context="CF",
                    force_padding=False
                )
            sim_scores = []

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue

            # Catalog filtering
            if target_catalog and self._catalog_map:
                item_catalog = self._catalog_map.get(t, '')
                if str(item_catalog).lower() != str(target_catalog).lower():
                    continue

            seen.add(t)
            results.append({
                'title': t,
                'collab_score': float(score),
            })
            if len(results) >= top_n:
                break

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n),
            top_n=top_n,
            context="CF",
            force_padding=False
        )

    def predict_for_user(self, user_id, top_n=10, target_catalog=None):
        """
        Personalized recommendations for a specific user.
        Predicts scores for all unseen items and returns top N.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        top_n = min(top_n, 100)

        # Safe type-aware existence check
        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            # Try type conversion to see if it matches (e.g., str "1" to int 1)
            for key in self._user_to_idx.keys():
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Cold-start detected for user '%s': no interaction history found. Falling back to popularity-based recommendations.", user_id)
            recs = self._popularity_fallback(top_n)
            return validate_recommendations(
                recs,
                fallback_fn=None,
                top_n=top_n,
                context="CF",
                force_padding=False
            )
            

        try:
            u_idx = self._user_to_idx[mapped_user_id]
            user_vec = self.user_factors[u_idx]
            scores = np.dot(user_vec, self.item_factors)

            # Exclude already-interacted items
            seen_items = set(
                self.df[self.df['user_id'] == user_id]['title'].tolist()
            )

            scored = []
            for i, score in enumerate(scores):
                t = self.title_list[i]
                if t in seen_items:
                    continue

                # Catalog filtering
                if target_catalog and self._catalog_map:
                    item_catalog = self._catalog_map.get(t, '')
                    if str(item_catalog).lower() != str(target_catalog).lower():
                        continue

                scored.append((t, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [{'title': t, 'predicted_score': s} for t, s in scored[:top_n]]
        except Exception as e:
            logger.error(f"Collaborative recommendation prediction computation failed: {e}")
            if "NaN" in str(e) or "nan" in str(e):
                return validate_recommendations(
                    [{"title": "NaN Placeholder", "predicted_score": float("nan")}],
                    fallback_fn=lambda top_n: self._popularity_fallback(top_n),
                    top_n=top_n,
                    context="CF",
                    force_padding=False
                )
            results = []
        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n),
            top_n=top_n,
            context="CF",
            force_padding=False
        )

    def predict_rating(self, user_id, title):
        """Predict the rating a user would give to an item."""
        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            for key in self._user_to_idx.keys():
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        u_idx = self._user_to_idx[mapped_user_id]
        i_idx = self._title_to_idx[title]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))
    
    def _popularity_fallback(self, top_n=10):
        # Fallback for cold-start users — top-N by interaction count (popularity)
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Using popularity-based fallback for cold-start user.")
    
        item_counts = self.df.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
    
        # 1. Most popular items
        if 'count' in item_counts.columns and not item_counts.empty:
            top_items = item_counts.nlargest(top_n, 'count')
        # 2. Highest rated items
        elif 'mean' in item_counts.columns and not item_counts.empty:
            top_items = item_counts.nlargest(top_n, 'mean')
        # 3. Trending items
        else:
            try:
                from src.model.trending_model import TrendingRecommender
                trending_model = TrendingRecommender(df=self.df)
                trending_results = trending_model.get_trending_products(top_n=top_n)
                if trending_results:
                    return [
                        {
                            'title': row['title'],
                            'predicted_score': float(row.get('avg_rating', 0.0)),
                            'fallback': True
                        }
                        for row in trending_results
                    ]
            except Exception:
                pass
            # 4. Empty list
            return []
    
        return [
        {
            'title': row['title'],
            'predicted_score': round(float(row.get('mean', 0.0)), 4),
            'fallback': True
        }
        for _, row in top_items.iterrows()
        ]
