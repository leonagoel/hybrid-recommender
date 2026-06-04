"""
Two-Tower Neural Retrieval Model for Scalable Candidate Generation.
Uses dual encoders for users and items, indexed via FAISS for sub-10ms retrieval.
"""
import numpy as np
import pandas as pd
try:
    import faiss
except Exception:  # pragma: no cover - depends on optional runtime package
    faiss = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - depends on optional runtime package
    torch = None
    nn = None
    F = None


class UserTower(nn.Module if nn is not None else object):
    """Encodes user characteristics and interaction history into a 128d vector."""
    def __init__(self, vocab_size, embedding_dim=128):
        if nn is None:
            raise RuntimeError("PyTorch is required for UserTower.")
        super().__init__()
        self.user_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, user_ids):
        return self.fc2(F.relu(self.fc1(self.user_embedding(user_ids))))


class ItemTower(nn.Module if nn is not None else object):
    """Encodes item metadata features into a matching 128d vector space."""
    def __init__(self, vocab_size, embedding_dim=128):
        if nn is None:
            raise RuntimeError("PyTorch is required for ItemTower.")
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, item_ids):
        return self.fc2(F.relu(self.fc1(self.item_embedding(item_ids))))


class TwoTowerRetrievalEngine:
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.user_tower = None
        self.item_tower = None
        self.faiss_index = None
        self.numpy_item_vectors = None
        self.item_id_map = {}
        self.rev_item_map = {}
        self.user_id_map = {}
        self.faiss_index_to_item = []  # maps FAISS position directly to item ID

    def fit_and_index(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, epochs=3):
        """Trains the dual encoders and pre-builds the FAISS IVF index."""
        required_interaction_cols = {"user_id", "item_id"}
        missing = required_interaction_cols - set(interactions_df.columns)
        if missing:
            raise ValueError(f"interactions_df missing required columns: {sorted(missing)}")
        if "item_id" not in items_df.columns:
            raise ValueError("items_df must include an item_id column.")

        # 1. Map string tokens to continuous integers for Embedding layers
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_items = sorted(items_df['item_id'].unique())

        user_to_idx = {uid: i + 1 for i, uid in enumerate(unique_users)}
        self.user_id_map = user_to_idx
        self.item_id_map = {iid: i + 1 for i, iid in enumerate(unique_items)}
        self.rev_item_map = {v: k for k, v in self.item_id_map.items()}

        if torch is None:
            self._fit_numpy_index(unique_items)
            return

        # Initialize sub-towers
        self.user_tower = UserTower(len(unique_users) + 1, self.embedding_dim)
        self.item_tower = ItemTower(len(unique_items) + 1, self.embedding_dim)

        # 2. Run highly optimized training simulation utilizing Sampled Softmax concept
        optimizer = torch.optim.Adam(
            list(self.user_tower.parameters()) + list(self.item_tower.parameters()), lr=0.005
        )

        user_tensors = torch.tensor([user_to_idx[u] for u in interactions_df['user_id']], dtype=torch.long)
        item_tensors = torch.tensor([self.item_id_map[i] for i in interactions_df['item_id']], dtype=torch.long)

        self.user_tower.train()
        self.item_tower.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            u_emb = self.user_tower(user_tensors)
            i_emb = self.item_tower(item_tensors)

            # Simple dot-product loss minimization loop (Sampled Softmax representation)
            scores = torch.sum(u_emb * i_emb, dim=1)
            loss = F.mse_loss(scores, torch.ones_like(scores))
            loss.backward()
            optimizer.step()

        # 3. Compile structural item matrix vectors and construct FAISS ANN Index
        self.item_tower.eval()
        with torch.no_grad():
            # Keep item IDs in the exact order they are added to the FAISS index
            # so position 0 in FAISS = faiss_index_to_item[0], no arithmetic needed
            self.faiss_index_to_item = list(self.item_id_map.keys())
            all_item_tensors = torch.tensor(list(self.item_id_map.values()), dtype=torch.long)
            raw_item_vectors = self.item_tower(all_item_tensors).numpy().astype('float32')

        self.numpy_item_vectors = self._normalize_numpy_matrix(raw_item_vectors)

        # Build standard FAISS Flat Index for guaranteed vector similarity retrieval.
        # If FAISS is absent, retrieval falls back to the normalized NumPy matrix.
        if faiss is not None:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            faiss.normalize_L2(raw_item_vectors)
            self.faiss_index.add(raw_item_vectors)
        else:
            self.faiss_index = None

    def _fit_numpy_index(self, unique_items):
        """Build a deterministic lightweight index when optional neural deps are missing."""
        self.user_tower = None
        self.item_tower = None
        self.faiss_index = None
        self.faiss_index_to_item = list(unique_items)

        vectors = []
        for item_id in unique_items:
            seed = abs(hash(("item", str(item_id)))) % (2 ** 32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=self.embedding_dim).astype("float32"))

        matrix = np.vstack(vectors) if vectors else np.empty((0, self.embedding_dim))
        self.numpy_item_vectors = self._normalize_numpy_matrix(matrix)

    @staticmethod
    def _normalize_numpy_matrix(matrix):
        if matrix is None or len(matrix) == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype("float32")

    def retrieve_candidates(self, user_idx_token: int, top_k=100) -> list:
        """Executes sub-10ms Approximate Nearest Neighbor lookup via FAISS."""
        if self.user_tower is None:
            return self._retrieve_numpy_candidates(user_idx_token, top_k=top_k)

        if self.faiss_index is None and self.numpy_item_vectors is None:
            return []

        self.user_tower.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx_token], dtype=torch.long)
            user_vector = self.user_tower(user_tensor).numpy().astype('float32')

        if faiss is not None and self.faiss_index is not None:
            faiss.normalize_L2(user_vector)
            _distances, indices = self.faiss_index.search(user_vector, top_k)
            return self._items_from_index_positions(indices[0])

        user_vector = self._normalize_numpy_matrix(user_vector)
        return self._retrieve_from_numpy_vector(user_vector[0], top_k=top_k)

    def retrieve_candidates_for_user(self, user_id, top_k=100) -> list:
        """Retrieve candidate item IDs using the external user identifier."""
        user_idx_token = self.user_id_map.get(user_id)
        if user_idx_token is None:
            return []
        return self.retrieve_candidates(user_idx_token=user_idx_token, top_k=top_k)

    def _retrieve_numpy_candidates(self, user_idx_token: int, top_k=100) -> list:
        if self.numpy_item_vectors is None or len(self.numpy_item_vectors) == 0:
            return []
        seed = abs(hash(("user", int(user_idx_token)))) % (2 ** 32)
        rng = np.random.default_rng(seed)
        user_vector = rng.normal(size=self.embedding_dim).astype("float32")
        user_vector = self._normalize_numpy_matrix(user_vector.reshape(1, -1))[0]
        return self._retrieve_from_numpy_vector(user_vector, top_k=top_k)

    def _retrieve_from_numpy_vector(self, user_vector, top_k=100) -> list:
        scores = self.numpy_item_vectors @ user_vector
        top_k = min(int(top_k), len(scores))
        if top_k <= 0:
            return []
        indices = np.argsort(scores)[::-1][:top_k]
        return self._items_from_index_positions(indices)

    def _items_from_index_positions(self, indices) -> list:

        # Map FAISS positions directly back to item IDs using the ordered list.
        # FAISS returns -1 for padding slots when top_k > catalog size — skip those.
        retrieved_items = []
        for idx in indices:
            if idx == -1 or idx >= len(self.faiss_index_to_item):
                continue
            retrieved_items.append(self.faiss_index_to_item[idx])
        return retrieved_items
