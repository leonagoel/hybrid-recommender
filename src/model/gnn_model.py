"""
src/model/gnn_model.py — PyTorch Geometric RGCN Knowledge-Graph GNN scaffold.

Issue #1576: Integrates a GNN layer into the hybrid recommender pipeline.
This module provides a placeholder RGCN-based graph neural network model
that can be trained on a knowledge graph stored at data/gnn_dataset/.

Usage:
    from src.model.gnn_model import GNNRecommender
    model = GNNRecommender(num_nodes=1000, num_relations=10, embedding_dim=64)
    embeddings = model.get_embeddings(node_ids)

To train the GNN:
    python scripts/train_gnn_kg.py --data_dir data/gnn_dataset/ --epochs 20
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default placeholder path for the GNN knowledge-graph dataset (Issue #1576)
GNN_DATASET_PATH = Path(
    os.environ.get("GNN_DATASET_PATH", "data/gnn_dataset/")
)


class GNNRecommender:
    """
    Issue #1576 — RGCN-based Graph Neural Network recommender scaffold.

    This class provides a stable, importable interface for GNN-based
    recommendations. When PyTorch Geometric is not installed it falls back
    to random embeddings so the rest of the pipeline is not broken.

    Attributes:
        num_nodes      (int): Number of nodes in the knowledge graph.
        num_relations  (int): Number of edge-relation types.
        embedding_dim  (int): Dimensionality of learned node embeddings.
        embeddings     (np.ndarray | None): Trained embeddings matrix (N × D).
    """

    def __init__(
        self,
        num_nodes: int = 1000,
        num_relations: int = 10,
        embedding_dim: int = 64,
        data_dir: Path | str | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.data_dir = Path(data_dir) if data_dir else GNN_DATASET_PATH
        self.embeddings: Optional[np.ndarray] = None
        self._model = None

        self._try_init_torch()

    def _try_init_torch(self) -> None:
        """
        Attempt to import PyTorch Geometric and initialise the RGCN model.
        Falls back to a numpy random-embedding placeholder if torch is absent.
        """
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
            logger.info(
                "GNNRecommender: PyTorch detected. "
                "Use scripts/train_gnn_kg.py to train on %s",
                self.data_dir,
            )
            # Placeholder: real RGCN layers would be constructed here using
            # torch_geometric.nn.RGCNConv once torch_geometric is installed.
            self.embeddings = np.zeros(
                (self.num_nodes, self.embedding_dim), dtype=np.float32
            )
        except ImportError:
            logger.warning(
                "PyTorch / PyTorch Geometric not installed. "
                "GNNRecommender will use random embeddings as a placeholder. "
                "Install with: pip install torch torch-geometric"
            )
            rng = np.random.default_rng(42)
            self.embeddings = rng.standard_normal(
                (self.num_nodes, self.embedding_dim)
            ).astype(np.float32)

    def get_embeddings(self, node_ids: List[int]) -> np.ndarray:
        """
        Return embedding vectors for the given node IDs.

        Args:
            node_ids: List of integer node indices.

        Returns:
            np.ndarray of shape (len(node_ids), embedding_dim).
        """
        if self.embeddings is None:
            raise RuntimeError("GNN embeddings not initialised.")
        ids = np.asarray(node_ids, dtype=int)
        ids = np.clip(ids, 0, self.num_nodes - 1)
        return self.embeddings[ids]

    def score(self, user_node_id: int, item_node_ids: List[int]) -> np.ndarray:
        """
        Compute cosine similarity scores between a user node and item nodes.

        Args:
            user_node_id:   Node index of the user.
            item_node_ids:  Node indices of candidate items.

        Returns:
            np.ndarray of shape (len(item_node_ids),) with similarity scores.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        user_emb = self.get_embeddings([user_node_id])          # (1, D)
        item_embs = self.get_embeddings(item_node_ids)           # (N, D)
        return cosine_similarity(user_emb, item_embs).flatten()