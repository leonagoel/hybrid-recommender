import pytest
import numpy as np
import os
import tempfile
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.feature_store import FeatureStore


class TestFeatureStore:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = FeatureStore(store_path=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directory(self):
        assert os.path.exists(self.temp_dir)

    def test_save_and_get_user_embedding(self):
        embedding = np.random.randn(128).astype(np.float32)
        self.store.save_user_embedding("user1", embedding)
        retrieved = self.store.get_user_embedding("user1")
        assert retrieved is not None
        assert np.allclose(retrieved, embedding)

    def test_get_nonexistent_user_embedding(self):
        retrieved = self.store.get_user_embedding("nonexistent")
        assert retrieved is None

    def test_save_and_get_item_embedding(self):
        embedding = np.random.randn(64).astype(np.float32)
        self.store.save_item_embedding("item1", embedding)
        retrieved = self.store.get_item_embedding("item1")
        assert retrieved is not None
        assert np.allclose(retrieved, embedding)

    def test_get_nonexistent_item_embedding(self):
        retrieved = self.store.get_item_embedding("nonexistent")
        assert retrieved is None

    def test_multiple_user_embeddings(self):
        emb1 = np.random.randn(128)
        emb2 = np.random.randn(128)
        self.store.save_user_embedding("user1", emb1)
        self.store.save_user_embedding("user2", emb2)
        assert np.allclose(self.store.get_user_embedding("user1"), emb1)
        assert np.allclose(self.store.get_user_embedding("user2"), emb2)

    def test_multiple_item_embeddings(self):
        emb1 = np.random.randn(64)
        emb2 = np.random.randn(64)
        self.store.save_item_embedding("item1", emb1)
        self.store.save_item_embedding("item2", emb2)
        assert np.allclose(self.store.get_item_embedding("item1"), emb1)
        assert np.allclose(self.store.get_item_embedding("item2"), emb2)

    def test_persistence_across_instances(self):
        embedding = np.random.randn(128)
        self.store.save_user_embedding("user1", embedding)
        new_store = FeatureStore(store_path=self.temp_dir)
        retrieved = new_store.get_user_embedding("user1")
        assert np.allclose(retrieved, embedding)
