# src/model/gnn_model.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class GNNRecommender:
    """
    Graph Neural Network based recommender.
    Builds a user-item-category graph and generates recommendations.
    """

    def __init__(self, item_df, interaction_df):
        self.item_df = item_df
        self.interaction_df = interaction_df

        self.user_nodes = {}
        self.item_nodes = {}
        self.category_nodes = {}

        self.edge_index = None
        self.item_embeddings = None
        self.user_embeddings = None
        self._trained = False

    def build_graph(self):
        """Build node mappings and adjacency edges from item_df and interaction_df."""
        # Index users
        if self.interaction_df is not None and 'user_id' in self.interaction_df.columns:
            users = self.interaction_df['user_id'].unique()
            self.user_nodes = {u: i for i, u in enumerate(users)}
        else:
            self.user_nodes = {}

        # Index items
        if 'title' in self.item_df.columns:
            items = self.item_df['title'].unique()
            self.item_nodes = {t: i for i, t in enumerate(items)}
        else:
            self.item_nodes = {}

        # Index categories
        if 'category' in self.item_df.columns:
            categories = self.item_df['category'].dropna().unique()
            self.category_nodes = {c: i for i, c in enumerate(categories)}
        else:
            self.category_nodes = {}

        # Build edges: user -> item (from interaction_df)
        edges = []
        if self.interaction_df is not None and 'user_id' in self.interaction_df.columns:
            for _, row in self.interaction_df.iterrows():
                uid = row.get('user_id')
                title = row.get('title')
                if uid in self.user_nodes and title in self.item_nodes:
                    edges.append((self.user_nodes[uid], self.item_nodes[title]))

        # Build edges: item -> category
        if 'category' in self.item_df.columns:
            for _, row in self.item_df.iterrows():
                title = row.get('title')
                cat = row.get('category')
                if pd.notna(cat) and title in self.item_nodes and cat in self.category_nodes:
                    edges.append((self.item_nodes[title], len(self.user_nodes) + self.category_nodes[cat]))

        self.edge_index = edges

    def train(self, n_epochs=10, lr=0.01, embedding_dim=32):
        """
        Train simple node embeddings using a random-walk inspired approach.
        For each user-item pair in edges, update item embeddings toward user preference.
        """
        if not self.item_nodes:
            self.build_graph()

        n_items = len(self.item_nodes)
        n_users = len(self.user_nodes)
        if n_items == 0:
            return

        # Initialize item embeddings
        np.random.seed(42)
        self.item_embeddings = np.random.normal(0, 0.1, size=(n_items, embedding_dim))
        self.user_embeddings = np.random.normal(0, 0.1, size=(n_users, embedding_dim))

        # Simple gradient descent: pull item embeddings toward users who interacted
        item_idx_list = list(self.item_nodes.values())
        user_idx_list = list(self.user_nodes.values())

        # Build user->items mapping
        user_to_items = {u: [] for u in user_idx_list}
        for u_idx, i_idx in self.edge_index:
            if u_idx < n_users:
                user_to_items[u_idx].append(i_idx)

        for epoch in range(n_epochs):
            for u_idx, item_list in user_to_items.items():
                if not item_list:
                    continue
                user_vec = self.user_embeddings[u_idx]
                for i_idx in item_list:
                    item_vec = self.item_embeddings[i_idx]
                    # Move item slightly toward user
                    diff = user_vec - item_vec
                    self.item_embeddings[i_idx] += lr * diff
                    # Move user slightly toward item
                    self.user_embeddings[u_idx] += lr * 0.5 * diff

        self._trained = True

    def recommend(self, user_id, top_n=10):
        """Return top-N recommended items for user_id based on learned embeddings."""
        if not self._trained:
            self.train()

        if self.item_embeddings is None or not self.item_nodes:
            return []

        if user_id not in self.user_nodes:
            # Cold start: return most popular items (highest norm embeddings)
            norms = np.linalg.norm(self.item_embeddings, axis=1)
            top_indices = np.argsort(norms)[::-1][:top_n]
            idx_to_title = {v: k for k, v in self.item_nodes.items()}
            return [{"title": idx_to_title[i], "gnn_score": float(norms[i])} for i in top_indices]

        u_idx = self.user_nodes[user_id]
        user_vec = self.user_embeddings[u_idx]

        # Cosine similarity between user and all items
        similarities = cosine_similarity([user_vec], self.item_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_n]
        idx_to_title = {v: k for k, v in self.item_nodes.items()}

        return [
            {"title": idx_to_title[i], "gnn_score": float(similarities[i])}
            for i in top_indices
        ]