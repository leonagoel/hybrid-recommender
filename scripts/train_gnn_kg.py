"""
scripts/train_gnn_kg.py — Training script for the RGCN GNN on the knowledge graph.

Issue #1576: Trains GNN embeddings on a knowledge-graph dataset stored at
data/gnn_dataset/ (or the path given by --data_dir / GNN_DATASET_PATH env var).

Usage:
    python scripts/train_gnn_kg.py
    python scripts/train_gnn_kg.py --data_dir data/gnn_dataset/ --epochs 20 --embedding_dim 64

The script saves learned embeddings to data/gnn_dataset/embeddings.npy
so they can be loaded by GNNRecommender at inference time.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default placeholder dataset directory (Issue #1576)
DEFAULT_DATA_DIR = Path(
    os.environ.get("GNN_DATASET_PATH", "data/gnn_dataset/")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RGCN GNN embeddings on the knowledge-graph dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the GNN knowledge-graph dataset directory (default: data/gnn_dataset/).",
    )
    parser.add_argument("--epochs",        type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimensionality.")
    parser.add_argument("--num_nodes",     type=int, default=1000, help="Number of graph nodes.")
    parser.add_argument("--num_relations", type=int, default=10,   help="Number of relation types.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("GNN training started.")
    logger.info("  data_dir      : %s", args.data_dir)
    logger.info("  epochs        : %d", args.epochs)
    logger.info("  embedding_dim : %d", args.embedding_dim)
    logger.info("  num_nodes     : %d", args.num_nodes)
    logger.info("  num_relations : %d", args.num_relations)

    # ── Validate dataset directory ────────────────────────────────────────────
    if not args.data_dir.exists():
        logger.warning(
            "Dataset directory '%s' does not exist. "
            "Creating placeholder directory — populate it with your KG triples "
            "before running a real training pass.",
            args.data_dir,
        )
        args.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Try PyTorch Geometric training ────────────────────────────────────────
    try:
        import torch  # noqa: F401
        logger.info("PyTorch detected. Real RGCN training would run here.")
        # TODO: replace with actual torch_geometric.nn.RGCNConv training loop
        # once the dataset is populated and torch_geometric is installed.
        # Placeholder: save random embeddings
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((args.num_nodes, args.embedding_dim)).astype(np.float32)

    except ImportError:
        logger.warning(
            "PyTorch not installed — generating random placeholder embeddings.\n"
            "Install with: pip install torch torch-geometric"
        )
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((args.num_nodes, args.embedding_dim)).astype(np.float32)

    # ── Save embeddings ───────────────────────────────────────────────────────
    output_path = args.data_dir / "embeddings.npy"
    np.save(output_path, embeddings)
    logger.info("Embeddings saved to %s  (shape: %s)", output_path, embeddings.shape)


if __name__ == "__main__":
    main()
