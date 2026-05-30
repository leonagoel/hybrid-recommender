"""
Tests for the /api/categories endpoint.

Covers:
- Preferred RPC (get_distinct_categories) returns sorted, deduplicated results.
- Legacy RPC (get_categories) used when preferred RPC fails.
- Direct table fallback used when both RPCs fail — no 5 000-row limit applied.
- Response is served from cache on the second request (no extra DB call).
- Cache is invalidated after data upload clears the response cache.
- Empty category values are filtered out.
- Endpoint returns 200 with empty list when all paths fail.
- Results are always sorted alphabetically.
- Results contain no duplicates.
"""

import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main
from backend.main import _fetch_categories_from_db

client = TestClient(main.app)


# ── Fake Supabase plumbing ────────────────────────────────────────────

class _RPCResult:
    def __init__(self, data, fail=False):
        self._data = data
        self._fail = fail

    def execute(self):
        if self._fail:
            raise RuntimeError("RPC unavailable")
        return SimpleNamespace(data=self._data)


class _TableQuery:
    """Chainable fake for sb.table(...).select(...).execute()."""

    def __init__(self, rows, fail=False):
        self._rows = rows
        self._fail = fail
        self.select_called = False
        self.limit_applied = None

    def select(self, *args, **kwargs):
        self.select_called = True
        return self

    def limit(self, n, *args, **kwargs):
        self.limit_applied = n
        return self

    def execute(self):
        if self._fail:
            raise RuntimeError("Table query failed")
        return SimpleNamespace(data=self._rows)


class FakeSupabase:
    """Configurable fake Supabase client.

    rpc_map: dict mapping rpc_name → list | None | Exception subclass.
    table_rows: rows returned by the table() fallback.
    table_fail: if True the table query raises.
    """

    def __init__(self, rpc_map=None, table_rows=None, table_fail=False):
        self._rpc_map = rpc_map or {}
        self._table_rows = table_rows or []
        self._table_fail = table_fail
        self.table_query = _TableQuery(self._table_rows, fail=table_fail)

    def rpc(self, name, _params):
        entry = self._rpc_map.get(name)
        if entry is None:
            return _RPCResult(None, fail=True)
        if entry is False:
            return _RPCResult(None, fail=True)
        return _RPCResult(entry, fail=False)

    def table(self, _name):
        return self.table_query


# ── Helpers ───────────────────────────────────────────────────────────

def _clear():
    main._clear_response_cache()


def _distinct_rpc_rows(*cats):
    return [{"category": c} for c in cats]


# ── Preferred RPC path ────────────────────────────────────────────────

def test_preferred_rpc_returns_sorted_categories(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("Tools", "Books", "Electronics"),
        }),
    )
    response = client.get("/api/categories")
    assert response.status_code == 200
    cats = response.json()["categories"]
    assert cats == sorted(cats)
    assert "Books" in cats
    assert "Tools" in cats
    assert "Electronics" in cats


def test_preferred_rpc_deduplicates_results(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("Books", "Books", "Tools"),
        }),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert len(cats) == len(set(cats)), "Duplicate categories must be removed"


def test_preferred_rpc_filters_empty_strings(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("", "Books", ""),
        }),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert "" not in cats
    assert cats == ["Books"]


# ── Legacy RPC fallback ───────────────────────────────────────────────

def test_falls_back_to_legacy_rpc_when_preferred_fails(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": False,  # fail
            "get_categories": ["Apparel", "Garden"],
        }),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert "Apparel" in cats
    assert "Garden" in cats


def test_legacy_rpc_result_is_sorted(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": False,
            "get_categories": ["Zebra", "Apple", "Mango"],
        }),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert cats == sorted(cats)


# ── Direct table fallback ─────────────────────────────────────────────

def test_falls_back_to_table_when_both_rpcs_fail(monkeypatch):
    _clear()
    table_rows = [
        {"category": "Home"},
        {"category": "Garden"},
        {"category": "Garden"},  # duplicate — must be deduped
    ]
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={}, table_rows=table_rows),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert sorted(cats) == ["Garden", "Home"]


def test_table_fallback_does_not_apply_row_limit(monkeypatch):
    """The table fallback must NOT call .limit() — the 5 000-row cap is removed."""
    _clear()
    table_rows = [{"category": "A"}, {"category": "B"}]
    fake_sb = FakeSupabase(rpc_map={}, table_rows=table_rows)
    monkeypatch.setattr(main, "get_supabase", lambda: fake_sb)

    client.get("/api/categories")

    assert fake_sb.table_query.limit_applied is None, (
        "Table fallback must not call .limit() — removing the 5 000-row cap is the fix"
    )


def test_table_fallback_filters_empty_category_values(monkeypatch):
    _clear()
    table_rows = [
        {"category": ""},
        {"category": "Books"},
        {"category": None},
        {"category": "Tools"},
    ]
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={}, table_rows=table_rows),
    )
    cats = client.get("/api/categories").json()["categories"]
    assert "" not in cats
    assert None not in cats
    assert sorted(cats) == ["Books", "Tools"]


# ── All paths fail ────────────────────────────────────────────────────

def test_returns_empty_list_when_all_paths_fail(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={}, table_rows=[], table_fail=True),
    )
    response = client.get("/api/categories")
    assert response.status_code == 200
    assert response.json() == {"categories": []}


# ── Caching behaviour ─────────────────────────────────────────────────

def test_second_request_served_from_cache(monkeypatch):
    _clear()
    call_count = {"n": 0}

    def make_supabase():
        call_count["n"] += 1
        return FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("Books", "Tools"),
        })

    monkeypatch.setattr(main, "get_supabase", make_supabase)

    client.get("/api/categories")
    client.get("/api/categories")

    assert call_count["n"] == 1, "Second call must be served from cache, not DB"


def test_cache_invalidated_after_clear(monkeypatch):
    _clear()
    call_count = {"n": 0}

    def make_supabase():
        call_count["n"] += 1
        return FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("Books"),
        })

    monkeypatch.setattr(main, "get_supabase", make_supabase)

    client.get("/api/categories")
    main._clear_response_cache()  # simulates a data upload clearing all caches
    client.get("/api/categories")

    assert call_count["n"] == 2, "After cache clear, a fresh DB call must be made"


def test_cached_response_matches_original(monkeypatch):
    _clear()
    monkeypatch.setattr(
        main, "get_supabase",
        lambda: FakeSupabase(rpc_map={
            "get_distinct_categories": _distinct_rpc_rows("Alpha", "Beta"),
        }),
    )
    first = client.get("/api/categories").json()
    second = client.get("/api/categories").json()
    assert first == second


# ── _fetch_categories_from_db unit tests ─────────────────────────────

def test_fetch_categories_preferred_rpc_dict_rows():
    sb = FakeSupabase(rpc_map={
        "get_distinct_categories": [{"category": "C"}, {"category": "A"}, {"category": "B"}],
    })
    result = _fetch_categories_from_db(sb)
    assert result == ["A", "B", "C"]


def test_fetch_categories_preferred_rpc_returns_empty_falls_to_legacy():
    sb = FakeSupabase(rpc_map={
        "get_distinct_categories": [],  # empty list — truthy None check bypasses, empty list also
        "get_categories": ["X", "Y"],
    })
    result = _fetch_categories_from_db(sb)
    # Empty list from preferred RPC triggers fallback to legacy.
    assert result == ["X", "Y"]


def test_fetch_categories_all_rpcs_fail_table_used():
    table_rows = [{"category": "Widgets"}, {"category": "Gadgets"}]
    sb = FakeSupabase(rpc_map={}, table_rows=table_rows)
    result = _fetch_categories_from_db(sb)
    assert sorted(result) == ["Gadgets", "Widgets"]


def test_fetch_categories_all_fail_returns_empty():
    sb = FakeSupabase(rpc_map={}, table_rows=[], table_fail=True)
    result = _fetch_categories_from_db(sb)
    assert result == []


def test_fetch_categories_result_always_sorted():
    sb = FakeSupabase(rpc_map={
        "get_distinct_categories": [{"category": "Z"}, {"category": "A"}, {"category": "M"}],
    })
    result = _fetch_categories_from_db(sb)
    assert result == sorted(result)
