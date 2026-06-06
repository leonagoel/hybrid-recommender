import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation import evaluation


def _dataset():
    return pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "category": ["x", "x", "y"],
            "rating": [4.0, 3.5, 5.0],
        }
    )


def test_tfidf_cache_rejects_pickle(monkeypatch, tmp_path):
    cache_path = tmp_path / "tfidf_matrix.pkl"
    cache_path.write_bytes(pickle.dumps({"unexpected": "payload"}))
    monkeypatch.setenv("TFIDF_CACHE", str(cache_path))

    with pytest.raises(RuntimeError, match="Refusing to load unsafe pickle"):
        evaluation._load_or_build_tfidf(_dataset())


def test_svd_cache_rejects_pickle(monkeypatch, tmp_path):
    cache_path = tmp_path / "svd_matrix.pkl"
    cache_path.write_bytes(pickle.dumps({"unexpected": "payload"}))
    monkeypatch.setenv("SVD_CACHE", str(cache_path))

    with pytest.raises(RuntimeError, match="Refusing to load unsafe pickle"):
        evaluation._load_or_build_svd(_dataset())


def test_tfidf_cache_loads_safe_sparse_npz(monkeypatch, tmp_path):
    cache_path = tmp_path / "tfidf_matrix.npz"
    expected = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    sparse.save_npz(cache_path, expected)
    monkeypatch.setenv("TFIDF_CACHE", str(cache_path))

    loaded = evaluation._load_or_build_tfidf(_dataset())

    assert (loaded != expected).nnz == 0


def test_svd_cache_loads_safe_npy_without_pickle(monkeypatch, tmp_path):
    cache_path = tmp_path / "svd_matrix.npy"
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.save(cache_path, expected, allow_pickle=False)
    monkeypatch.setenv("SVD_CACHE", str(cache_path))

    loaded = evaluation._load_or_build_svd(_dataset())

    np.testing.assert_array_equal(loaded, expected)
