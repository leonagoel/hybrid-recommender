"""
Collaborative Recommender
Uses Truncated SVD (matrix factorization) on the user-item interaction
matrix to discover latent factors and predict ratings.

Improvements:
- Implicit feedback support (views, purchases → confidence weights)
- Adaptive n_factors for sparse matrices
- User-based personalized recommendations
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix


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

        # Adaptive rank: reduce factors for very sparse matrices
        min_dim = min(self.user_item_sparse.shape)
        density = self.user_item_sparse.nnz / (n_users * n_items) if (n_users * n_items) > 0 else 0

        if density < 0.001:
            n_components = min(20, min_dim - 1)
        elif density < 0.01:
            n_components = min(30, min_dim - 1)
        else:
            n_components = min(n_factors, min_dim - 1)

        n_components = max(1, n_components)

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_sparse)
        self.item_factors = self.svd.components_

    def recommend(self, title, top_n=10):
        """
        Item-item collaborative recommendations using SVD latent space.
        Returns list of dicts: [{ 'title', 'collab_score' }, ...]
        """
        if title not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title]
        query_vec = self.item_factors[:, idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self.item_factors.T).flatten()

        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue
            seen.add(t)
            results.append({
                'title': t,
                'collab_score': float(score),
            })
            if len(results) >= top_n:
                break

        return results

    def predict_for_user(self, user_id, top_n=10):
        """
        Personalized recommendations for a specific user.
        Predicts scores for all unseen items and returns top N.
        """
        if user_id not in self._user_to_idx:
            return []

        u_idx = self._user_to_idx[user_id]
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
            scored.append((t, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'title': t, 'predicted_score': s} for t, s in scored[:top_n]]

    def predict_rating(self, user_id, title):
        """Predict the rating a user would give to an item."""
        if user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        u_idx = self._user_to_idx[user_id]
        i_idx = self._title_to_idx[title]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))