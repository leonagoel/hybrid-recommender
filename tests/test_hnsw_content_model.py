import sys
import importlib
from pathlib import Path
import types

import numpy as np
import pandas as pd


def _make_fake_hnswlib():
    mod = types.ModuleType("hnswlib")

    class Index:
        def __init__(self, space, dim):
            self.space = space
            self.dim = dim
            self._data = None
            self._ids = None

        def init_index(self, max_elements, ef_construction=200, M=16):
            self.max_elements = max_elements

        def add_items(self, vectors, ids):
            self._data = np.asarray(vectors, dtype="float32")
            self._ids = np.asarray(ids, dtype=int)

        def set_ef(self, ef):
            self.ef = ef

        @property
        def element_count(self):
            return 0 if self._data is None else self._data.shape[0]

        def knn_query(self, query_vec, k=1):
            # query_vec: shape (1, dim)
            q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
            data = self._data
            if data is None or data.shape[0] == 0:
                return np.empty((1, 0), dtype=int), np.empty((1, 0), dtype=float)

            # cosine similarity -> distances = 1 - cosine_sim
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            d_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-12)
            sims = np.dot(q_norm, d_norm.T).flatten()
            # distances as hnswlib would return for 'cosine' space: 1 - sim
            dists = 1.0 - sims
            order = np.argsort(dists)[:k]
            labels = self._ids[order].astype(int)
            return np.expand_dims(labels, 0), np.expand_dims(dists[order], 0)

    mod.Index = Index
    return mod


def _make_fake_sentence_transformers():
    mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, show_progress_bar=False):
            # deterministic lightweight encoding: map each text to a small vector
            vecs = []
            for t in texts:
                s = str(t or "")
                a = len(s)
                b = len(set(s))
                c = sum(ord(ch) for ch in s) % 7
                v = np.array([a, b, c, 1.0], dtype=float)
                # normalize to unit length for cosine
                v = v / (np.linalg.norm(v) + 1e-12)
                vecs.append(v)
            return np.vstack(vecs)

    mod.SentenceTransformer = SentenceTransformer
    return mod


def test_hnsw_integration(tmp_path, monkeypatch):
    # Ensure project root is on sys.path
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Inject fake hnswlib and sentence_transformers before import
    fake_hnsw = _make_fake_hnswlib()
    fake_st = _make_fake_sentence_transformers()

    monkeypatch.setitem(sys.modules, "hnswlib", fake_hnsw)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    # (Re)import the content_model module so it picks up our fakes
    import importlib

    content_mod = importlib.import_module("src.model.content_model")
    importlib.reload(content_mod)

    # Build a small synthetic dataframe
    df = pd.DataFrame(
        {
            "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "combined": ["alpha text", "beta text", "gamma text", "delta text", "epsilon text"],
        }
    )

    rec = content_mod.ContentRecommender(df, model_name="dummy", batch_size=2)

    # HNSW index should be built using our fake
    assert rec._hnsw_index is not None
    assert rec._hnsw_index.element_count == len(df)

    # Recommend for 'Alpha'
    results = rec.recommend("Alpha", top_n=2)
    assert isinstance(results, list)
    # Should return 2 neighbors (excluding the query itself)
    assert len(results) == 2
    for r in results:
        assert "title" in r and "content_score" in r
        assert isinstance(r["content_score"], float)

    # Search by free text
    search_res = rec.search("alpha text", top_n=3)
    assert isinstance(search_res, list)
    assert len(search_res) >= 1
    for r in search_res[:3]:
        assert "title" in r and "score" in r
        # allow small floating point rounding beyond 1.0
        score = float(r["score"])
        assert -1e-6 <= score <= 1.0 + 1e-6
