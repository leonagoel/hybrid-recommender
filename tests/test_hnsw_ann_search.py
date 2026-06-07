import importlib
import sys
import types
import builtins
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Dummy SentenceTransformer to avoid network/model downloads
class DummySentenceTransformer:
    def __init__(self, model_name=None):
        # simple mapping for predictable embeddings
        self._map = {
            'a': np.array([1.0, 0.0]),
            'b': np.array([0.0, 1.0]),
            'c': np.array([0.9, 0.1]),
            'd': np.array([0.0, 0.0]),
        }

    def encode(self, texts, show_progress_bar=False):
        out = []
        for t in texts:
            # texts here are the 'combined' values we set in the test df
            out.append(self._map.get(t, np.array([0.0, 0.0])))
        return np.vstack(out)


def _make_df():
    return pd.DataFrame({
        'title': ['A', 'B', 'C', 'D'],
        'combined': ['a', 'b', 'c', 'd'],
        'item_id': [1, 2, 3, 4]
    })


def reload_content_module():
    if 'src.model.content_model' in sys.modules:
        return importlib.reload(sys.modules['src.model.content_model'])
    return importlib.import_module('src.model.content_model')


def test_fallback_when_hnswlib_missing(monkeypatch):
    # Simulate ImportError for hnswlib by monkeypatching __import__
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'hnswlib' or name.startswith('hnswlib.'):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    # Reload module so the import behavior is evaluated
    content_model = reload_content_module()
    # Replace SentenceTransformer with dummy
    monkeypatch.setattr(content_model, 'SentenceTransformer', DummySentenceTransformer)

    df = _make_df()
    recomm = content_model.ContentRecommender(df)

    # ANN should be disabled when hnswlib is not importable
    assert getattr(recomm, '_ann_enabled', False) is False

    results = recomm.recommend('A', top_n=2)
    # Top recommendation (excluding the source itself) should be 'C'
    assert len(results) >= 1
    assert results[0]['title'] == 'C'


def test_ann_index_creation(monkeypatch):
    # Create fake hnswlib with a simple Index implementation
    class FakeIndex:
        def __init__(self, space, dim):
            self.space = space
            self.dim = dim
            self.data = None
            self.ids = None

        def init_index(self, max_elements, ef_construction, M):
            pass

        def add_items(self, data, ids):
            self.data = np.asarray(data)
            self.ids = np.asarray(ids)

        def set_ef(self, ef):
            pass

        def knn_query(self, vector, k):
            v = np.asarray(vector)
            if v.ndim == 2:
                v = v[0]
            sims = cosine_similarity(v.reshape(1, -1), self.data).flatten()
            idxs = np.argsort(sims)[::-1][:k]
            labels = self.ids[idxs].reshape(1, -1)
            dists = (1.0 - sims[idxs]).reshape(1, -1)
            return labels, dists

    fake_module = types.SimpleNamespace(Index=FakeIndex)
    monkeypatch.setitem(sys.modules, 'hnswlib', fake_module)

    content_model = reload_content_module()
    monkeypatch.setattr(content_model, 'SentenceTransformer', DummySentenceTransformer)

    df = _make_df()
    recomm = content_model.ContentRecommender(df)

    assert getattr(recomm, '_ann_enabled', False) is True
    assert getattr(recomm, '_ann_index', None) is not None


def test_ann_recommendation_parity(monkeypatch):
    # Fake hnswlib as above
    class FakeIndex:
        def __init__(self, space, dim):
            self.space = space
            self.dim = dim
            self.data = None
            self.ids = None

        def init_index(self, max_elements, ef_construction, M):
            pass

        def add_items(self, data, ids):
            self.data = np.asarray(data)
            self.ids = np.asarray(ids)

        def set_ef(self, ef):
            pass

        def knn_query(self, vector, k):
            v = np.asarray(vector)
            if v.ndim == 2:
                v = v[0]
            sims = cosine_similarity(v.reshape(1, -1), self.data).flatten()
            idxs = np.argsort(sims)[::-1][:k]
            labels = self.ids[idxs].reshape(1, -1)
            dists = (1.0 - sims[idxs]).reshape(1, -1)
            return labels, dists

    fake_module = types.SimpleNamespace(Index=FakeIndex)
    monkeypatch.setitem(sys.modules, 'hnswlib', fake_module)

    content_model = reload_content_module()
    monkeypatch.setattr(content_model, 'SentenceTransformer', DummySentenceTransformer)
    df = _make_df()
    recomm_ann = content_model.ContentRecommender(df)
    res_ann = [r['title'] for r in recomm_ann.recommend('A', top_n=3)]

    # Now simulate missing hnswlib
    monkeypatch.delitem(sys.modules, 'hnswlib', raising=False)
    content_model_noann = reload_content_module()
    monkeypatch.setattr(content_model_noann, 'SentenceTransformer', DummySentenceTransformer)
    recomm_noann = content_model_noann.ContentRecommender(df)
    res_noann = [r['title'] for r in recomm_noann.recommend('A', top_n=3)]

    # Top-1 should be identical (final ranking uses exact cosine on candidates)
    assert res_ann[0] == res_noann[0]


def test_ann_failure_fallback(monkeypatch):
    # Fake hnswlib where init_index raises
    class BrokenIndex:
        def __init__(self, space, dim):
            pass

        def init_index(self, max_elements, ef_construction, M):
            raise RuntimeError('init failed')

    fake_module = types.SimpleNamespace(Index=BrokenIndex)
    monkeypatch.setitem(sys.modules, 'hnswlib', fake_module)

    content_model = reload_content_module()
    monkeypatch.setattr(content_model, 'SentenceTransformer', DummySentenceTransformer)
    df = _make_df()
    recomm = content_model.ContentRecommender(df)

    # Should have silently disabled ANN
    assert getattr(recomm, '_ann_enabled', False) is False

    results = recomm.recommend('A', top_n=2)
    assert len(results) >= 1
    assert results[0]['title'] == 'C'
