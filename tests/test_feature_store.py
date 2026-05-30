import sys
sys.path.insert(0, '.')

import numpy as np
import pickle
import os

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

def test_custom_store_path_creation(tmp_path):
    custom_dir = tmp_path / "custom_subdir"
    assert not custom_dir.exists()
    store = FeatureStore(store_path=str(custom_dir))
    assert custom_dir.exists()

def test_empty_keys_returns_none(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    assert store.get_user_embedding("") is None
    assert store.get_item_embedding("") is None