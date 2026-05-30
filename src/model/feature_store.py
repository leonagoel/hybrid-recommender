
import numpy as np
import pickle
import os

class FeatureStore:
    """
    Centralized store for user and item embeddings.
    Used by HybridRecommender to cache and retrieve embeddings
    instead of recomputing them on every request.
    """

    def __init__(self, store_path="src/model/data"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self._user_embeddings = {}
        self._item_embeddings = {}

    def save_user_embedding(self, user_id, embedding):
        self._user_embeddings[user_id] = embedding
        self._persist("user_embeddings.pkl", self._user_embeddings)

    def get_user_embedding(self, user_id):
        self._load("user_embeddings.pkl", self._user_embeddings)
        return self._user_embeddings.get(user_id, None)

    def save_item_embedding(self, item_id, embedding):
        self._item_embeddings[item_id] = embedding
        self._persist("item_embeddings.pkl", self._item_embeddings)

    def get_item_embedding(self, item_id):
        self._load("item_embeddings.pkl", self._item_embeddings)
        return self._item_embeddings.get(item_id, None)

    def _persist(self, filename, data):
        path = os.path.join(self.store_path, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load(self, filename, target):
        path = os.path.join(self.store_path, filename)
        if os.path.exists(path) and not target:
            with open(path, "rb") as f:
                target.update(pickle.load(f))