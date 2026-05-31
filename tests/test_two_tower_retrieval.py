import pytest
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.two_tower_retrieval import UserTower, ItemTower, TwoTowerRetrievalEngine


class TestUserTower:
    def test_init(self):
        tower = UserTower(vocab_size=100, embedding_dim=128)
        assert tower.embedding_dim == 128
        assert tower.user_embedding.num_embeddings == 100

    def test_forward(self):
        tower = UserTower(vocab_size=100, embedding_dim=128)
        user_ids = torch.tensor([1, 2, 3])
        output = tower(user_ids)
        assert output.shape == (3, 128)


class TestItemTower:
    def test_init(self):
        tower = ItemTower(vocab_size=50, embedding_dim=128)
        assert tower.embedding_dim == 128
        assert tower.item_embedding.num_embeddings == 50

    def test_forward(self):
        tower = ItemTower(vocab_size=50, embedding_dim=128)
        item_ids = torch.tensor([1, 2])
        output = tower(item_ids)
        assert output.shape == (2, 128)


class TestTwoTowerRetrievalEngine:
    def test_init(self):
        engine = TwoTowerRetrievalEngine(embedding_dim=128)
        assert engine.embedding_dim == 128
        assert engine.user_tower is None
        assert engine.item_tower is None
        assert engine.faiss_index is None

    def test_fit_and_index_requires_dataframes(self):
        engine = TwoTowerRetrievalEngine(embedding_dim=64)
        interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i3']
        })
        items_df = pd.DataFrame({
            'item_id': ['i1', 'i2', 'i3'],
            'title': ['Item 1', 'Item 2', 'Item 3']
        })
        engine.fit_and_index(interactions_df, items_df, epochs=1)
        assert engine.user_tower is not None
        assert engine.item_tower is not None
        assert engine.faiss_index is not None

    def test_retrieve_empty_query(self):
        engine = TwoTowerRetrievalEngine(embedding_dim=64)
        interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'item_id': ['i1', 'i2']
        })
        items_df = pd.DataFrame({
            'item_id': ['i1', 'i2'],
            'title': ['Item 1', 'Item 2']
        })
        engine.fit_and_index(interactions_df, items_df, epochs=1)
        results = engine.retrieve('unknown_user', top_n=5)
        assert isinstance(results, list)

    def test_faiss_index_mapping(self):
        engine = TwoTowerRetrievalEngine(embedding_dim=64)
        interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i3']
        })
        items_df = pd.DataFrame({
            'item_id': ['i1', 'i2', 'i3'],
            'title': ['Item 1', 'Item 2', 'Item 3']
        })
        engine.fit_and_index(interactions_df, items_df, epochs=1)
        assert len(engine.faiss_index_to_item) == 3
