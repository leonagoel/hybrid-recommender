"""
Tests for the admin dashboard endpoint (issue #71).
Run with: pytest tests/test_dashboard.py -v
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

from backend import main as backend_main


# ─── Fake Supabase client ────────────────────────────────────────────

class _FakeQuery:
    """
    Records the chain of calls and returns a result based on which
    table + selection it represents. The full chain is collapsed at
    .execute() into a SimpleNamespace(data=..., count=...).
    """

    def __init__(self, table_name, dataset):
        self._table = table_name
        self._dataset = dataset
        self._selected = None
        self._count_mode = None
        self._filters = []

    def select(self, columns, count=None):
        self._selected = columns
        self._count_mode = count
        return self

    def limit(self, _n):
        return self

    def range(self, _a, _b):
        return self

    def offset(self, _n):
        return self

    def order(self, _col, **_kwargs):
        return self

    def in_(self, col, values):
        self._filters.append(('in', col, set(values)))
        return self

    def eq(self, col, value):
        self._filters.append(('eq', col, value))
        return self

    def execute(self):
        rows = list(self._dataset.get(self._table, []))
        for kind, col, value in self._filters:
            if kind == 'in':
                rows = [r for r in rows if r.get(col) in value]
            elif kind == 'eq':
                rows = [r for r in rows if r.get(col) == value]

        if self._count_mode == 'exact':
            return SimpleNamespace(data=[], count=len(self._dataset.get(self._table, [])))
        return SimpleNamespace(data=rows, count=None)


class _FakeSupabase:
    def __init__(self, dataset):
        self._dataset = dataset

    def table(self, name):
        return _FakeQuery(name, self._dataset)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(backend_main.app)


@pytest.fixture(autouse=True)
def reset_models_state():
    """Snapshot and restore the in-memory models dict around each test."""
    saved = {
        "ready": backend_main.models["ready"],
        "last_trained_at": backend_main.models["last_trained_at"],
    }
    yield
    backend_main.models["ready"] = saved["ready"]
    backend_main.models["last_trained_at"] = saved["last_trained_at"]


def _patch_supabase(monkeypatch, dataset):
    fake = _FakeSupabase(dataset)
    monkeypatch.setattr(backend_main, 'get_supabase', lambda: fake)


# ─── Tests ───────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "total_products",
    "total_users",
    "total_interactions",
    "avg_recommendation_score",
    "avg_sentiment_score",
    "top_5_recommended_products",
    "model_last_trained",
}


def test_dashboard_schema_with_data(client, monkeypatch):
    """Populated database: response has the exact required schema."""
    dataset = {
        'products': [
            {'id': 1, 'title': 'A', 'category': 'Electronics', 'rating': 4.5, 'avg_sentiment': 0.6, 'review_count': 100},
            {'id': 2, 'title': 'B', 'category': 'Books',       'rating': 3.0, 'avg_sentiment': 0.1, 'review_count': 20},
            {'id': 3, 'title': 'C', 'category': 'Electronics', 'rating': 5.0, 'avg_sentiment': 0.9, 'review_count': 50},
        ],
        'purchases': [
            {'id': 10, 'user_id': 'u1', 'product_id': 1},
            {'id': 11, 'user_id': 'u1', 'product_id': 2},
            {'id': 12, 'user_id': 'u2', 'product_id': 1},
        ],
    }
    _patch_supabase(monkeypatch, dataset)
    backend_main.models["last_trained_at"] = None

    res = client.get('/api/dashboard')
    assert res.status_code == 200
    body = res.json()

    assert set(body.keys()) == REQUIRED_KEYS
    assert body['total_products'] == 3
    assert body['total_users'] == 2
    assert body['total_interactions'] == 3
    assert isinstance(body['avg_recommendation_score'], float)
    assert isinstance(body['avg_sentiment_score'], float)
    assert body['avg_recommendation_score'] == pytest.approx((4.5 + 3.0 + 5.0) / 3, rel=1e-3)
    assert isinstance(body['top_5_recommended_products'], list)
    assert len(body['top_5_recommended_products']) <= 5
    # Product 1 has the most purchases — it should rank first.
    assert body['top_5_recommended_products'][0]['id'] == 1
    assert body['top_5_recommended_products'][0]['interactions'] == 2
    assert body['model_last_trained'] is None


def test_dashboard_empty_dataset(client, monkeypatch):
    """Empty database: zeros, empty list, null timestamp — no error."""
    _patch_supabase(monkeypatch, {'products': [], 'purchases': []})
    backend_main.models["last_trained_at"] = None

    res = client.get('/api/dashboard')
    assert res.status_code == 200
    body = res.json()

    assert set(body.keys()) == REQUIRED_KEYS
    assert body['total_products'] == 0
    assert body['total_users'] == 0
    assert body['total_interactions'] == 0
    assert body['avg_recommendation_score'] == 0.0
    assert body['avg_sentiment_score'] == 0.0
    assert body['top_5_recommended_products'] == []
    assert body['model_last_trained'] is None


def test_dashboard_reports_last_trained_timestamp(client, monkeypatch):
    """model_last_trained should reflect the in-memory training timestamp."""
    _patch_supabase(monkeypatch, {'products': [], 'purchases': []})
    fixed_ts = "2026-05-19T12:00:00+00:00"
    backend_main.models["last_trained_at"] = fixed_ts

    res = client.get('/api/dashboard')
    assert res.status_code == 200
    assert res.json()['model_last_trained'] == fixed_ts


def test_dashboard_top_products_fallback_when_no_purchases(client, monkeypatch):
    """With products but no purchases, top_5 falls back to top-rated products."""
    dataset = {
        'products': [
            {'id': 1, 'title': 'Low',  'category': 'X', 'rating': 2.0, 'avg_sentiment': 0.1, 'review_count': 5},
            {'id': 2, 'title': 'High', 'category': 'X', 'rating': 4.9, 'avg_sentiment': 0.8, 'review_count': 80},
        ],
        'purchases': [],
    }
    _patch_supabase(monkeypatch, dataset)

    res = client.get('/api/dashboard')
    body = res.json()
    assert res.status_code == 200
    assert len(body['top_5_recommended_products']) == 2
    # Fallback returns the product list as-is from the fake (no ordering applied
    # in the fake), but interactions should be 0 for every fallback entry.
    for p in body['top_5_recommended_products']:
        assert p['interactions'] == 0
