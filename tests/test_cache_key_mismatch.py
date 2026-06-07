"""
Tests for Issue #925: precomputed recommendation cache entries never served
due to cache-key mismatch between the build-time precomputation path and
the request-serving path.

Root cause (before fix)
------------------------
_precompute_recommendation_cache() built keys with five parts:

    _cache_key("recommend", title, top_n, explain, "")
    → "recommend:<title>:10:false:"          (4 colons)

get_recommendations() built keys with eight parts:

    _cache_key("recommend", query_title, top_n, explain,
               user_id or "", target_catalog or "",
               model_version or "", strategy or "")
    → "recommend:<title>:10:false::::"       (7 colons)

Because _cache_key() is a simple colon-join, these strings are never equal.
Every request was a cache miss regardless of what was precomputed.

Fix
----
_recommendation_cache_key() is the single authoritative key builder used by
both paths.  It always encodes all eight fields so that a default request
(no user_id, no catalog, no model version, no strategy) generates the same
key as the corresponding precomputed entry.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os
os.environ.setdefault("SUPABASE_URL",   "http://fake")
os.environ.setdefault("SUPABASE_ANON_KEY", "fake")
os.environ.setdefault("ADMIN_API_TOKEN",   "fake-token")

from backend import main  # noqa: E402  (env vars must be set first)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache():
    main._clear_response_cache()


class _FakeHybrid:
    """Minimal HybridRecommender stand-in for cache tests."""

    def __init__(self, recs=None):
        self.calls = 0
        self._recs = recs or [{"title": "MatchItem", "hybrid_score": 0.9}]

    def get_weights(self):
        return {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}

    def recommend(self, title, top_n=10, explain=False, target_catalog=None, **_):
        self.calls += 1
        return self._recs[:top_n]


# ---------------------------------------------------------------------------
# 1. _recommendation_cache_key — unit tests
# ---------------------------------------------------------------------------

class TestRecommendationCacheKeyFunction(unittest.TestCase):
    """Unit tests for _recommendation_cache_key()."""

    def test_function_exists(self):
        self.assertTrue(
            hasattr(main, "_recommendation_cache_key"),
            "_recommendation_cache_key must be exported from backend.main",
        )

    def test_default_call_matches_default_api_call(self):
        """
        Regression: precompute call (no optional params) must produce the
        same key as an API request with all optional params absent / empty.
        """
        precompute_key = main._recommendation_cache_key("Item A", 10, False)
        api_key = main._recommendation_cache_key(
            "Item A", 10, False, "", "", "", ""
        )
        self.assertEqual(
            precompute_key,
            api_key,
            "Key built with defaults must equal key built with explicit empty strings",
        )

    def test_old_precompute_key_does_not_match_api_key(self):
        """
        Regression proof: demonstrate that the old 5-part key (used before
        the fix) never matched the 8-part API key.
        """
        old_precompute = main._cache_key("recommend", "Item A", 10, False, "")
        api_key = main._recommendation_cache_key("Item A", 10, False)
        self.assertNotEqual(
            old_precompute,
            api_key,
            "The old buggy precompute key must differ from the canonical key "
            "to confirm the regression existed",
        )

    def test_key_is_lowercase(self):
        """Keys must be lower-cased (inherited from _cache_key)."""
        key = main._recommendation_cache_key("BIG PRODUCT TITLE", 10, False)
        self.assertEqual(key, key.lower())

    def test_key_strips_whitespace_from_title(self):
        """Titles with leading/trailing whitespace must produce the same key."""
        k1 = main._recommendation_cache_key("  widget  ", 10, False)
        k2 = main._recommendation_cache_key("widget", 10, False)
        self.assertEqual(k1, k2)

    def test_different_titles_produce_different_keys(self):
        k1 = main._recommendation_cache_key("Alpha", 10, False)
        k2 = main._recommendation_cache_key("Beta", 10, False)
        self.assertNotEqual(k1, k2)

    def test_different_top_n_produces_different_keys(self):
        k1 = main._recommendation_cache_key("Alpha", 5, False)
        k2 = main._recommendation_cache_key("Alpha", 10, False)
        self.assertNotEqual(k1, k2)

    def test_explain_true_vs_false_produces_different_keys(self):
        k1 = main._recommendation_cache_key("Alpha", 10, True)
        k2 = main._recommendation_cache_key("Alpha", 10, False)
        self.assertNotEqual(k1, k2)

    def test_user_id_encoded_in_key(self):
        k_anon = main._recommendation_cache_key("Alpha", 10, False, user_id="")
        k_user = main._recommendation_cache_key("Alpha", 10, False, user_id="user123")
        self.assertNotEqual(k_anon, k_user)

    def test_target_catalog_encoded_in_key(self):
        k_no_cat = main._recommendation_cache_key("Alpha", 10, False, target_catalog="")
        k_cat    = main._recommendation_cache_key("Alpha", 10, False, target_catalog="electronics")
        self.assertNotEqual(k_no_cat, k_cat)

    def test_model_version_encoded_in_key(self):
        k1 = main._recommendation_cache_key("Alpha", 10, False, model_version="")
        k2 = main._recommendation_cache_key("Alpha", 10, False, model_version="v1.2")
        self.assertNotEqual(k1, k2)

    def test_strategy_encoded_in_key(self):
        k1 = main._recommendation_cache_key("Alpha", 10, False, strategy="")
        k2 = main._recommendation_cache_key("Alpha", 10, False, strategy="cold")
        self.assertNotEqual(k1, k2)

    def test_none_optional_params_treated_same_as_empty_string(self):
        """None values (from Optional query params) must equal empty-string defaults."""
        k_none  = main._recommendation_cache_key(
            "Alpha", 10, False, None, None, None, None
        )
        k_empty = main._recommendation_cache_key(
            "Alpha", 10, False, "", "", "", ""
        )
        self.assertEqual(k_none, k_empty)

    def test_key_is_deterministic(self):
        """Same arguments always produce the same key."""
        k1 = main._recommendation_cache_key("Alpha", 10, False, "u1", "cat", "v1", "pop")
        k2 = main._recommendation_cache_key("Alpha", 10, False, "u1", "cat", "v1", "pop")
        self.assertEqual(k1, k2)


# ---------------------------------------------------------------------------
# 2. Precompute → API cache-hit integration
# ---------------------------------------------------------------------------

class TestPrecomputeThenCacheHit(unittest.TestCase):
    """
    End-to-end: after _precompute_recommendation_cache() writes a cache
    entry, a GET /api/recommend request for the same title must return
    X-Cache: HIT without invoking the recommendation model again.
    """

    def setUp(self):
        _clear_cache()
        self.fake_hybrid = _FakeHybrid()
        main.models.update(
            {
                "ready":    True,
                "hybrid":   self.fake_hybrid,
                "item_df":  __import__("pandas").DataFrame(
                    {"title": ["Product X", "Product Y"]}
                ),
                "collab":   None,
            }
        )

    def tearDown(self):
        _clear_cache()
        main.models.update(
            {"ready": False, "hybrid": None, "item_df": None, "collab": None}
        )

    def test_precomputed_entry_is_served_as_cache_hit(self):
        """
        Regression test: precomputed cache entry for 'Product X' must be
        retrieved by a GET /api/recommend/Product%20X request.
        """
        from fastapi.testclient import TestClient

        # --- Step 1: precompute ---
        count = main._precompute_recommendation_cache(top_n=10, explain=False)
        self.assertGreater(count, 0, "Precomputation must write at least one entry")
        calls_after_precompute = self.fake_hybrid.calls

        # --- Step 2: serve via API ---
        client = TestClient(main.app)
        response = client.get("/api/recommend/Product%20X?top_n=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("x-cache"), "HIT",
            "A precomputed entry must be served as a cache HIT; "
            f"x-cache header was '{response.headers.get('x-cache')}'. "
            "This was the bug: the precompute key and API key did not match.",
        )
        # The model must NOT have been called again after precomputation
        self.assertEqual(
            self.fake_hybrid.calls,
            calls_after_precompute,
            "Model must not be invoked for a cache-hit request",
        )

    def test_precomputed_entry_returns_correct_payload(self):
        """Cached payload must contain the same recommendations as the precomputed run."""
        from fastapi.testclient import TestClient

        main._precompute_recommendation_cache(top_n=10, explain=False)
        client = TestClient(main.app)
        response = client.get("/api/recommend/Product%20X?top_n=10")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("recommendations", body)
        self.assertGreater(len(body["recommendations"]), 0)
        self.assertEqual(body["query_item"], "Product X")

    def test_request_without_precompute_is_a_cache_miss(self):
        """Without precomputation the same request must be a cache MISS."""
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        response = client.get("/api/recommend/Product%20X?top_n=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("x-cache"), "MISS",
            "Without precomputation the first request must be a MISS",
        )


# ---------------------------------------------------------------------------
# 3. Precompute write path — key format
# ---------------------------------------------------------------------------

class TestPrecomputeWritesCorrectKey(unittest.TestCase):
    """Verify _precompute_recommendation_cache writes under the canonical key."""

    def setUp(self):
        _clear_cache()
        self.fake_hybrid = _FakeHybrid()
        main.models.update(
            {
                "ready":   True,
                "hybrid":  self.fake_hybrid,
                "item_df": __import__("pandas").DataFrame(
                    {"title": ["Widget Alpha"]}
                ),
                "collab":  None,
            }
        )

    def tearDown(self):
        _clear_cache()
        main.models.update(
            {"ready": False, "hybrid": None, "item_df": None, "collab": None}
        )

    def test_written_key_matches_canonical_key(self):
        """The key stored by precompute must equal _recommendation_cache_key output."""
        main._precompute_recommendation_cache(top_n=10, explain=False)

        expected_key = main._recommendation_cache_key("Widget Alpha", 10, False)

        # The cache entry must exist under the canonical key
        cached = main._get_cached_response(expected_key)
        self.assertIsNotNone(
            cached,
            f"No cache entry found at canonical key '{expected_key}'. "
            "Precompute must write using _recommendation_cache_key().",
        )

    def test_old_buggy_key_not_populated(self):
        """The old 5-part key must NOT have an entry after the fix."""
        main._precompute_recommendation_cache(top_n=10, explain=False)

        old_key = main._cache_key("recommend", "Widget Alpha", 10, False, "")
        cached = main._get_cached_response(old_key)
        self.assertIsNone(
            cached,
            f"Old buggy key '{old_key}' must not have a cache entry after the fix.",
        )

    def test_cache_precomputed_flag_present(self):
        """Precomputed entries must carry the cache_precomputed=True flag."""
        main._precompute_recommendation_cache(top_n=10, explain=False)

        key = main._recommendation_cache_key("Widget Alpha", 10, False)
        cached = main._get_cached_response(key)

        self.assertIsNotNone(cached)
        self.assertTrue(
            cached.get("cache_precomputed"),
            "Precomputed entries must have cache_precomputed=True in the payload",
        )


# ---------------------------------------------------------------------------
# 4. Regression: key format stability
# ---------------------------------------------------------------------------

class TestCacheKeyStability(unittest.TestCase):
    """Keys must not change shape across software versions (backward compat)."""

    def test_canonical_key_has_eight_colon_separated_parts(self):
        """The key must always consist of exactly 8 colon-joined parts."""
        key = main._recommendation_cache_key("My Title", 5, True, "u1", "cat", "v1", "pop")
        parts = key.split(":")
        self.assertEqual(
            len(parts), 8,
            f"Key must have 8 parts separated by 7 colons; got {len(parts)}: {key!r}",
        )

    def test_default_key_has_eight_parts(self):
        key = main._recommendation_cache_key("My Title", 10, False)
        parts = key.split(":")
        self.assertEqual(len(parts), 8)

    def test_key_parts_order(self):
        """Parts must appear in the documented order."""
        key = main._recommendation_cache_key(
            "My Product", 15, True, "alice", "electronics", "v2", "cold"
        )
        parts = key.split(":")
        self.assertEqual(parts[0], "recommend")
        self.assertIn("my product", parts[1])  # title, lower-cased
        self.assertEqual(parts[2], "15")       # top_n
        self.assertEqual(parts[3], "true")     # explain
        self.assertEqual(parts[4], "alice")    # user_id
        self.assertEqual(parts[5], "electronics")  # target_catalog
        self.assertEqual(parts[6], "v2")       # model_version
        self.assertEqual(parts[7], "cold")     # strategy
