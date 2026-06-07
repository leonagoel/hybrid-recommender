import pytest
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

def test_compute_hash_returns_hex_string(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("hello world")
    hash_val = store._compute_hash(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA-256 hex

def test_compute_hash_deterministic(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("hello world")
    h1 = store._compute_hash(test_file)
    h2 = store._compute_hash(test_file)
    assert h1 == h2

def test_compute_hash_different_files_different_hashes(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    f1 = str(tmp_path / "a.txt")
    f2 = str(tmp_path / "b.txt")
    with open(f1, "w") as f:
        f.write("content a")
    with open(f2, "w") as f:
        f.write("content b")
    h1 = store._compute_hash(f1)
    h2 = store._compute_hash(f2)
    assert h1 != h2

def test_save_hash_writes_manifest(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("data")
    store._save_hash(test_file)
    manifest_path = store._manifest_path
    assert os.path.exists(manifest_path)
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert test_file in manifest

def test_verify_hash_passes_for_saved_file(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("data")
    store._save_hash(test_file)
    # Should not raise
    store._verify_hash(test_file)

def test_verify_hash_raises_for_missing_manifest(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "orphan.txt")
    with open(test_file, "w") as f:
        f.write("data")
    # No manifest exists
    with pytest.raises(RuntimeError, match="Manifest not found"):
        store._verify_hash(test_file)

def test_verify_hash_raises_for_tampered_file(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("original")
    store._save_hash(test_file)
    # Tamper with the file
    with open(test_file, "w") as f:
        f.write("tampered")
    with pytest.raises(RuntimeError, match="Hash mismatch"):
        store._verify_hash(test_file)

def test_validate_magic_bytes_accepts_joblib(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    import joblib
    test_file = str(tmp_path / "test.joblib")
    joblib.dump({"key": "value"}, test_file)
    # Should not raise
    store._validate_magic_bytes(test_file)

def test_validate_magic_bytes_rejects_non_joblib(tmp_path):
    store = FeatureStore(store_path=str(tmp_path))
    test_file = str(tmp_path / "test.txt")
    with open(test_file, "w") as f:
        f.write("not a joblib file")
    with pytest.raises(RuntimeError, match="Invalid file format"):
        store._validate_magic_bytes(test_file)