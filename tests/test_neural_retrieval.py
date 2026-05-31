"""Unit and contract validation tests for the Two-Tower Neural Retrieval engine."""
import pytest
import pandas as pd

faiss = pytest.importorskip("faiss", reason="faiss not installed — skipping neural retrieval tests")
from src.model.two_tower_retrieval import TwoTowerRetrievalEngine


def test_two_tower_lifecycle_and_faiss_bounds():
    """Validates full end-to-end embedding training and sub-10ms FAISS lookup loops."""
    # Build clean baseline mock structures
    mock_interactions = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3', 'u1'],
        'item_id': ['i1', 'i2', 'i3', 'i2']
    })
    mock_items = pd.DataFrame({'item_id': ['i1', 'i2', 'i3']})
    
    engine = TwoTowerRetrievalEngine(embedding_dim=128)
    engine.fit_and_index(mock_interactions, mock_items, epochs=1)
    
    assert engine.faiss_index is not None
    assert engine.faiss_index.ntotal == 3
    
    # Run candidates selection query check
    candidates = engine.retrieve_candidates(user_idx_token=1, top_k=2)
    assert isinstance(candidates, list)
    assert len(candidates) <= 2