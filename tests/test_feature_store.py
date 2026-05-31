import sys

sys.path.insert(0, ".")


import numpy as np

# Import directly, bypassing __init__.py
from src.model.feature_store import FeatureStore


def test_save_and_get_user_embedding(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    vec = np.array([0.1, 0.2, 0.3])
    store.save_user_embedding("user_1", vec)
    assert np.allclose(store.get_user_embedding("user_1"), vec)


def test_save_and_get_item_embedding(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    vec = np.array([0.4, 0.5, 0.6])
    store.save_item_embedding("item_1", vec)
    assert np.allclose(store.get_item_embedding("item_1"), vec)


def test_missing_user_returns_none(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    assert store.get_user_embedding("unknown_user") is None


def test_missing_item_returns_none(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    assert store.get_item_embedding("unknown_item") is None


def test_clear_user_embeddings(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    vec = np.array([0.1, 0.2, 0.3])
    store.save_user_embedding("user_1", vec)
    store.clear_user_embeddings()
    assert store.get_user_embedding("user_1") is None


def test_clear_item_embeddings(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    vec = np.array([0.4, 0.5, 0.6])
    store.save_item_embedding("item_1", vec)
    store.clear_item_embeddings()
    assert store.get_item_embedding("item_1") is None


def test_clear_all(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    store.save_user_embedding("user_1", np.array([0.1, 0.2]))
    store.save_item_embedding("item_1", np.array([0.4, 0.5]))
    store.clear_all()
    assert store.get_user_embedding("user_1") is None
    assert store.get_item_embedding("item_1") is None


def test_clear_user_keeps_item(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    store.save_user_embedding("user_1", np.array([0.1, 0.2]))
    store.save_item_embedding("item_1", np.array([0.4, 0.5]))
    store.clear_user_embeddings()
    assert store.get_user_embedding("user_1") is None
    assert store.get_item_embedding("item_1") is not None
