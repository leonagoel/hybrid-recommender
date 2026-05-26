"""
Federated Learning Module for Collaborative Filtering.
Enables training recommendation models (Collaborative Filtering)
across decentralized client nodes without centralizing raw user interactions.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from src.model.collaborative_model import CollaborativeRecommender


class FederatedClient:
    """
    Represents a decentralized client node (user).
    Holds local private interaction data and performs local computations.
    """

    def __init__(self, user_id: str, private_ratings: dict):
        """
        user_id: Unique client identifier.
        private_ratings: Dict of {item_title: rating} representing user's history.
        """
        self.user_id = user_id
        self.private_ratings = private_ratings
        self.user_factor = None

    def compute_local_user_factor(
        self, global_item_factors: np.ndarray, title_to_idx: dict, n_factors: int, reg: float = 0.05
    ) -> np.ndarray:
        """
        Computes the client's local user vector (latent factor) using Ridge Regression
        over the global item factors and the client's private ratings.
        """
        rated_titles = [t for t in self.private_ratings if t in title_to_idx]
        if not rated_titles:
            self.user_factor = np.zeros(n_factors)
            return self.user_factor

        # Extract global vectors for the rated items
        indices = [title_to_idx[t] for t in rated_titles]
        V_u = global_item_factors[:, indices]  # Shape: [n_factors, len(rated_titles)]
        ratings_vec = np.array([self.private_ratings[t] for t in rated_titles], dtype=float)

        # Solve Ridge Regression: (V_u * V_u.T + reg * I) * user_factor = V_u * ratings
        A = np.dot(V_u, V_u.T) + reg * np.eye(n_factors)
        b = np.dot(V_u, ratings_vec)
        self.user_factor = np.linalg.solve(A, b)
        return self.user_factor

    def compute_local_item_updates(
        self, global_item_factors: np.ndarray, title_to_idx: dict, reg: float = 0.05
    ) -> dict:
        """
        Computes local updates (gradients) for the global item factors based on the
        reconstruction error of private ratings. Only updates items the user has rated.
        Returns:
            Dict of {item_title: update_vector}
        """
        updates = {}
        if self.user_factor is None:
            return updates

        for title, rating in self.private_ratings.items():
            if title not in title_to_idx:
                continue
            idx = title_to_idx[title]
            v_i = global_item_factors[:, idx]  # Shape: [n_factors]

            # Error = raw rating - predicted rating
            pred_rating = np.dot(self.user_factor, v_i)
            error = rating - pred_rating

            # Update vector = error * user_factor - regularization * item_factor
            updates[title] = error * self.user_factor - reg * v_i
        return updates


class FederatedServer:
    """
    Central coordinator that aggregates updates from client nodes
    and updates the global collaborative model parameters.
    """

    def __init__(self, item_list: list, n_factors: int = 20, learning_rate: float = 0.05, reg: float = 0.05):
        """
        item_list: List of all unique item titles.
        n_factors: Number of SVD latent dimensions.
        learning_rate: Global learning rate for updating item factors.
        reg: Regularization strength.
        """
        self.item_list = item_list
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg

        self.title_to_idx = {t: i for i, t in enumerate(self.item_list)}
        
        # Initialize global item factors randomly
        np.random.seed(42)
        self.global_item_factors = np.random.normal(
            0.0, 0.1, size=(self.n_factors, len(self.item_list))
        )

    def aggregate_updates(self, client_updates_list: list):
        """
        Aggregates local updates from multiple clients using Federated Averaging (FedAvg)
        and performs a global gradient descent update step.
        """
        aggregated_updates = {title: [] for title in self.item_list}

        # Gather updates for each item
        for client_updates in client_updates_list:
            for title, update in client_updates.items():
                if title in aggregated_updates:
                    aggregated_updates[title].append(update)

        # Update global item factors
        for title, updates in aggregated_updates.items():
            if not updates:
                continue
            idx = self.title_to_idx[title]
            # Average update across all contributing clients
            avg_update = np.mean(updates, axis=0)
            self.global_item_factors[:, idx] += self.lr * avg_update


def train_federated_collaborative_model(
    interaction_df: pd.DataFrame,
    n_factors: int = 20,
    epochs: int = 5,
    lr: float = 0.05,
    reg: float = 0.05
) -> CollaborativeRecommender:
    """
    Simulates federated training of a CollaborativeRecommender model.
    1. Splits interaction data into decentralized local private client datasets.
    2. Runs federated learning epochs with local updates and server-side aggregation.
    3. Returns a fully populated CollaborativeRecommender instance.
    """
    if interaction_df.empty:
        raise ValueError("Cannot train on empty DataFrame.")

    unique_users = interaction_df['user_id'].unique()
    unique_items = interaction_df['title'].unique()

    # 1. Initialize Clients (decentralize user data)
    clients = []
    for user_id in unique_users:
        user_data = interaction_df[interaction_df['user_id'] == user_id]
        private_ratings = dict(zip(user_data['title'], user_data['rating']))
        clients.append(FederatedClient(user_id, private_ratings))

    # 2. Initialize Server
    server = FederatedServer(
        item_list=list(unique_items), n_factors=n_factors, learning_rate=lr, reg=reg
    )

    # 3. Federated Training Loop
    for _ in range(epochs):
        client_updates = []
        
        # Phase 1: Local computations on each client
        for client in clients:
            # Client updates its local user vector based on global item factors
            client.compute_local_user_factor(
                server.global_item_factors, server.title_to_idx, n_factors, reg
            )
            # Client computes local updates for global item factors
            updates = client.compute_local_item_updates(
                server.global_item_factors, server.title_to_idx, reg
            )
            client_updates.append(updates)

        # Phase 2: Central server aggregates updates and updates global factors
        server.aggregate_updates(client_updates)

    # Final iteration to ensure local user factors are synced with final global factors
    user_factors_list = []
    for client in clients:
        client.compute_local_user_factor(
            server.global_item_factors, server.title_to_idx, n_factors, reg
        )
        user_factors_list.append(client.user_factor)

    # 4. Construct final CollaborativeRecommender
    # We create a mock/empty instance and populate its SVD matrix factors
    recommender = CollaborativeRecommender(interaction_df, n_factors=n_factors)
    
    # Overwrite matrix factors and lookups
    recommender.title_list = list(unique_items)
    recommender._user_to_idx = {u: i for i, u in enumerate(unique_users)}
    recommender._title_to_idx = server.title_to_idx
    recommender.item_factors = server.global_item_factors
    recommender.user_factors = np.array(user_factors_list)

    return recommender
