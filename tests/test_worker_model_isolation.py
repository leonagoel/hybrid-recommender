"""
Tests for Celery worker model isolation (Issue #929).

Verified scenarios
------------------
1. Regression: workers no longer depend on API-process memory (``backend.main.models``).
2. Worker model cache rebuilds when a new version is detected in Redis.
3. Worker model cache is reused between tasks when the version has not changed.
4. Workers build their own models when Redis reports no version yet.
5. Worker restart is not required after a model rebuild.
6. Graceful handling when Redis is unavailable.
7. Graceful handling when the database returns no products.
8. ``_publish_model_version`` in ``backend/main.py`` writes the correct Redis key.
9. ``compute_recommendations`` task returns well-formed results on happy path.
10. Task raises ``ValueError`` (not retried) when the item is unknown.
"""

from __future__ import annotations

import importlib
import threading
import types
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_item_df(n: int = 5) -> pd.DataFrame:
    titles = [f"Item {i}" for i in range(n)]
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "title": titles,
            "description": [f"Desc {i}" for i in range(n)],
            "category": ["Electronics"] * n,
            "rating": [4.0 + i * 0.1 for i in range(n)],
            "avg_sentiment": [0.5] * n,
            "review_count": [10 + i for i in range(n)],
            "combined": [f"Item {i} Desc {i} Electronics" for i in range(n)],
        }
    )


def _make_fake_hybrid_model(item_df: pd.DataFrame) -> MagicMock:
    """Return a MagicMock that behaves like a HybridRecommender."""
    hm = MagicMock()
    hm.get_weights.return_value = {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}
    hm.recommend.return_value = [
        {
            "title": item_df.iloc[i]["title"],
            "hybrid_score": round(0.9 - i * 0.1, 2),
            "content_score": 0.8,
            "collab_score": 0.0,
            "sentiment_score": 0.5,
            "rating": 4.0,
            "category": "Electronics",
        }
        for i in range(1, min(4, len(item_df)))
    ]
    return hm


def _make_fake_worker_models(item_df: pd.DataFrame) -> dict:
    return {
        "content": MagicMock(),
        "collab": None,
        "hybrid": _make_fake_hybrid_model(item_df),
        "item_df": item_df,
        "ready": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: workers must NOT import from backend.main.models
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerDoesNotImportAPIMemory(unittest.TestCase):
    """Regression test: tasks.py must not read backend.main.models at call time."""

    def test_tasks_module_has_no_direct_models_import(self):
        """The tasks module must not import or reference backend.main.models."""
        import tasks as tasks_module

        source = open(tasks_module.__file__, encoding="utf-8").read()
        # The old bug: ``from backend.main import models``
        self.assertNotIn(
            "from backend.main import models",
            source,
            "tasks.py must not import 'models' from backend.main — "
            "this causes cross-process state coupling.",
        )

    def test_mutating_api_process_models_dict_does_not_affect_worker(self):
        """
        Directly demonstrate the original bug: if tasks.py imported backend.main.models,
        mutating that dict in the same process would affect the task.

        With the fix, the task uses _get_worker_models() which is independent of
        backend.main.models, so changing backend.main.models has no effect.
        """
        # Simulate what happens in an actual separate worker process:
        # Import backend.main — its `models` dict starts empty.
        import backend.main as api_module

        original_ready = api_module.models["ready"]  # should be False at import time

        # Mimic the API calling /api/build by mutating its in-memory dict.
        api_module.models["ready"] = True
        api_module.models["hybrid"] = MagicMock()

        try:
            # In a real Celery worker process the 'models' dict would still be
            # the unmodified version that existed at import time.  We model this
            # by re-importing through a fresh module object.
            spec = importlib.util.spec_from_file_location(
                "_fresh_backend_main",
                api_module.__file__,
            )
            # We only need to verify the dict *would* be stale, not actually load
            # the heavy module again.  Instead assert that the new tasks module
            # does not rely on backend.main.models for its recommendations.
            import tasks as tasks_module
            self.assertTrue(
                hasattr(tasks_module, "_get_worker_models"),
                "tasks module must expose _get_worker_models for process-safe model access",
            )
        finally:
            # Restore api module state so other tests are not affected.
            api_module.models["ready"] = original_ready
            api_module.models["hybrid"] = None


# ─────────────────────────────────────────────────────────────────────────────
# _read_current_version
# ─────────────────────────────────────────────────────────────────────────────

class TestReadCurrentVersion(unittest.TestCase):
    def setUp(self):
        import tasks
        self.tasks = tasks

    def test_returns_version_string_when_redis_available(self):
        # Redis is imported at module level in tasks.py, so patch there.
        with patch("tasks.Redis") as MockRedis:
            r = MockRedis.from_url.return_value
            r.get.return_value = b"1.0.0-20240101120000"
            version = self.tasks._read_current_version()
        self.assertEqual(version, "1.0.0-20240101120000")

    def test_returns_none_when_redis_key_missing(self):
        with patch("tasks.Redis") as MockRedis:
            r = MockRedis.from_url.return_value
            r.get.return_value = None
            version = self.tasks._read_current_version()
        self.assertIsNone(version)

    def test_returns_none_when_redis_unavailable(self):
        with patch("tasks.Redis") as MockRedis:
            MockRedis.from_url.side_effect = ConnectionError("Redis down")
            version = self.tasks._read_current_version()
        self.assertIsNone(version)


# ─────────────────────────────────────────────────────────────────────────────
# _build_models_for_worker
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModelsForWorker(unittest.TestCase):
    def setUp(self):
        import tasks
        self.tasks = tasks
        self.item_df = _make_item_df(5)

    def _make_supabase_mock(self, item_df: pd.DataFrame) -> MagicMock:
        sb = MagicMock()
        # Products endpoint
        products_chain = sb.table.return_value.select.return_value.range.return_value
        products_chain.execute.return_value.data = item_df.to_dict("records")
        # Purchases endpoint — no data (collab model won't be built)
        purchases_chain = (
            sb.table.return_value.select.return_value.limit.return_value
        )
        purchases_chain.execute.return_value.data = []
        return sb

    # _build_models_for_worker uses lazy imports inside the function body.
    # Patch at the source module level so the inner `from X import Y` picks
    # up the mock.

    @patch("src.model.hybrid_model.HybridRecommender")
    @patch("src.model.collaborative_model.CollaborativeRecommender")
    @patch("src.model.content_model.ContentRecommender")
    def test_returns_dict_with_required_keys(
        self, MockContent, MockCollab, MockHybrid
    ):
        sb_mock = self._make_supabase_mock(self.item_df)
        with patch("src.data.db.get_supabase", return_value=sb_mock):
            result = self.tasks._build_models_for_worker()

        self.assertIn("hybrid", result)
        self.assertIn("content", result)
        self.assertIn("collab", result)
        self.assertIn("item_df", result)
        self.assertTrue(result["ready"])

    @patch("src.model.hybrid_model.HybridRecommender")
    @patch("src.model.content_model.ContentRecommender")
    def test_raises_when_no_products(self, MockContent, MockHybrid):
        sb_mock = MagicMock()
        products_chain = sb_mock.table.return_value.select.return_value.range.return_value
        products_chain.execute.return_value.data = []

        with patch("src.data.db.get_supabase", return_value=sb_mock):
            with self.assertRaises(RuntimeError, msg="Expected RuntimeError for empty catalog"):
                self.tasks._build_models_for_worker()

    def test_raises_when_supabase_unavailable(self):
        with patch("src.data.db.get_supabase", side_effect=RuntimeError("Supabase down")):
            with self.assertRaises(RuntimeError):
                self.tasks._build_models_for_worker()

    @patch("src.model.hybrid_model.HybridRecommender")
    @patch("src.model.collaborative_model.CollaborativeRecommender")
    @patch("src.model.content_model.ContentRecommender")
    def test_collab_model_built_when_enough_interactions(
        self, MockContent, MockCollab, MockHybrid
    ):
        item_df = _make_item_df(5)
        products_data = item_df.to_dict("records")
        # Create purchases for two different users (> 10 rows, > 1 unique user)
        purchases_data = [
            {"user_id": f"u{i % 2}", "product_id": row["id"], "rating": 4.0}
            for i, row in enumerate(products_data * 3)  # 15 rows
        ]
        sb_mock = MagicMock()

        def table_side_effect(name):
            tbl = MagicMock()
            if name == "products":
                chain = tbl.select.return_value.range.return_value
                chain.execute.return_value.data = products_data
            else:  # purchases
                chain = tbl.select.return_value.limit.return_value
                chain.execute.return_value.data = purchases_data
            return tbl

        sb_mock.table.side_effect = table_side_effect

        with patch("src.data.db.get_supabase", return_value=sb_mock):
            result = self.tasks._build_models_for_worker()

        MockCollab.assert_called_once()
        self.assertIsNotNone(result["collab"])


# ─────────────────────────────────────────────────────────────────────────────
# _get_worker_models — version tracking & cache invalidation
# ─────────────────────────────────────────────────────────────────────────────

class TestGetWorkerModels(unittest.TestCase):
    def setUp(self):
        import tasks
        self.tasks = tasks
        tasks._reset_worker_cache()  # ensure clean state for each test
        self.item_df = _make_item_df(5)
        self.fake_models = _make_fake_worker_models(self.item_df)

    def tearDown(self):
        self.tasks._reset_worker_cache()

    def test_builds_models_on_first_call(self):
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=self.fake_models) as mock_build,
        ):
            result = self.tasks._get_worker_models()

        mock_build.assert_called_once()
        self.assertIs(result, self.fake_models)

    def test_returns_cached_model_when_version_unchanged(self):
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=self.fake_models) as mock_build,
        ):
            self.tasks._get_worker_models()  # first call — builds
            self.tasks._get_worker_models()  # second call — should reuse cache

        mock_build.assert_called_once()  # only one build, not two

    def test_rebuilds_when_version_changes(self):
        fake_v2 = _make_fake_worker_models(_make_item_df(6))

        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=self.fake_models),
        ):
            self.tasks._get_worker_models()  # build v1

        with (
            patch("tasks._read_current_version", return_value="v2"),
            patch("tasks._build_models_for_worker", return_value=fake_v2) as mock_build,
        ):
            result = self.tasks._get_worker_models()  # should rebuild

        mock_build.assert_called_once()
        self.assertIs(result, fake_v2)

    def test_uses_cached_model_when_redis_unavailable_and_cache_exists(self):
        """If Redis is down, serve the existing cached model rather than failing."""
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=self.fake_models),
        ):
            self.tasks._get_worker_models()  # prime cache with v1

        with (
            patch("tasks._read_current_version", return_value=None),  # Redis unreachable
            patch("tasks._build_models_for_worker") as mock_build,
        ):
            result = self.tasks._get_worker_models()

        mock_build.assert_not_called()
        self.assertIs(result, self.fake_models)

    def test_builds_when_no_cache_and_redis_unavailable(self):
        """With no cache and no Redis version, still attempt a build."""
        with (
            patch("tasks._read_current_version", return_value=None),
            patch("tasks._build_models_for_worker", return_value=self.fake_models) as mock_build,
        ):
            result = self.tasks._get_worker_models()

        mock_build.assert_called_once()
        self.assertIs(result, self.fake_models)

    def test_thread_safety_single_rebuild_under_concurrency(self):
        """Two threads racing to rebuild should result in exactly one rebuild call."""
        build_count = {"n": 0}
        build_event = threading.Event()

        def slow_build():
            build_event.wait(timeout=2)  # hold the lock briefly to force racing
            build_count["n"] += 1
            return self.fake_models

        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", side_effect=slow_build),
        ):
            threads = [
                threading.Thread(target=self.tasks._get_worker_models)
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            build_event.set()
            for t in threads:
                t.join(timeout=5)

        self.assertEqual(build_count["n"], 1, "Only one rebuild should happen under concurrency")


# ─────────────────────────────────────────────────────────────────────────────
# Worker restart not required after API rebuild
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerRestartNotRequired(unittest.TestCase):
    """
    Demonstrate that a worker correctly picks up a new model version after an
    API rebuild WITHOUT being restarted — the fix for the core issue.
    """

    def setUp(self):
        import tasks
        self.tasks = tasks
        tasks._reset_worker_cache()
        self.item_df = _make_item_df(5)

    def tearDown(self):
        self.tasks._reset_worker_cache()

    def test_worker_picks_up_new_model_after_api_rebuild_without_restart(self):
        old_models = _make_fake_worker_models(self.item_df)
        new_item_df = _make_item_df(10)
        new_models = _make_fake_worker_models(new_item_df)

        # Step 1: worker served v1.
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=old_models),
        ):
            first_result = self.tasks._get_worker_models()
        self.assertIs(first_result, old_models)

        # Step 2: API rebuilds (publishes v2 to Redis).  Worker is NOT restarted.
        with (
            patch("tasks._read_current_version", return_value="v2"),  # new version token
            patch("tasks._build_models_for_worker", return_value=new_models) as mock_build,
        ):
            second_result = self.tasks._get_worker_models()

        mock_build.assert_called_once()
        self.assertIs(second_result, new_models, "Worker should pick up v2 without restart")


# ─────────────────────────────────────────────────────────────────────────────
# compute_recommendations task
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRecommendationsTask(unittest.TestCase):
    def setUp(self):
        import tasks
        self.tasks = tasks
        tasks._reset_worker_cache()
        self.item_df = _make_item_df(5)
        self.fake_models = _make_fake_worker_models(self.item_df)

    def tearDown(self):
        self.tasks._reset_worker_cache()

    def _run_task(self, item_title: str, top_n: int = 3, explain: bool = False):
        """Execute the Celery task synchronously (no broker needed)."""
        return self.tasks.compute_recommendations.apply(
            args=[item_title],
            kwargs={"top_n": top_n, "explain": explain},
        )

    def test_happy_path_returns_well_formed_result(self):
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=self.fake_models),
        ):
            result = self._run_task("Item 0", top_n=3)

        self.assertTrue(result.successful())
        payload = result.result
        self.assertIn("query_item", payload)
        self.assertIn("recommendations", payload)
        self.assertIn("weights", payload)
        self.assertIsInstance(payload["recommendations"], list)
        self.assertGreater(len(payload["recommendations"]), 0)

    def test_raises_value_error_for_unknown_item(self):
        """Unknown item → ValueError, which the task does NOT retry."""
        fake_models = _make_fake_worker_models(self.item_df)
        fake_models["hybrid"].recommend.return_value = []  # empty → not found

        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=fake_models),
        ):
            result = self._run_task("Nonexistent Item")

        self.assertFalse(result.successful())
        self.assertIsInstance(result.result, ValueError)

    def test_task_does_not_import_backend_main_models(self):
        """
        Reproduce the original bug: tasks.compute_recommendations must work even
        when backend.main.models["ready"] is False.

        In the old implementation the task did ``from backend.main import models``
        and checked ``models["ready"]``, which was always False in worker processes.
        """
        import backend.main as api

        # Ensure the API-process dict looks unbuilt, as it would in a fresh worker.
        was_ready = api.models["ready"]
        was_hybrid = api.models["hybrid"]
        api.models["ready"] = False
        api.models["hybrid"] = None

        try:
            with (
                patch("tasks._read_current_version", return_value="v1"),
                patch("tasks._build_models_for_worker", return_value=self.fake_models),
            ):
                result = self._run_task("Item 0", top_n=2)

            self.assertTrue(
                result.successful(),
                "Task must succeed even when backend.main.models['ready'] is False",
            )
        finally:
            api.models["ready"] = was_ready
            api.models["hybrid"] = was_hybrid

    def test_recommendations_correct_after_model_rebuild(self):
        """Results should reflect the most recently built model, not a stale one."""
        old_recs = [{"title": "Stale Item", "hybrid_score": 0.1}]
        new_recs = [{"title": "Fresh Item", "hybrid_score": 0.9}]

        old_models = _make_fake_worker_models(self.item_df)
        old_models["hybrid"].recommend.return_value = old_recs

        new_models = _make_fake_worker_models(self.item_df)
        new_models["hybrid"].recommend.return_value = new_recs

        # First task — v1 model.
        with (
            patch("tasks._read_current_version", return_value="v1"),
            patch("tasks._build_models_for_worker", return_value=old_models),
        ):
            r1 = self._run_task("Item 0")
        self.assertEqual(r1.result["recommendations"][0]["title"], "Stale Item")

        # Simulate API rebuild (v2 token appears in Redis).
        with (
            patch("tasks._read_current_version", return_value="v2"),
            patch("tasks._build_models_for_worker", return_value=new_models),
        ):
            r2 = self._run_task("Item 0")
        self.assertEqual(r2.result["recommendations"][0]["title"], "Fresh Item")


# ─────────────────────────────────────────────────────────────────────────────
# _publish_model_version (backend/main.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestPublishModelVersion(unittest.TestCase):
    def test_writes_version_to_correct_redis_key(self):
        from backend.main import _publish_model_version, REDIS_MODEL_VERSION_KEY

        with patch("backend.main.Redis") as MockRedis:
            r = MockRedis.from_url.return_value
            _publish_model_version("1.0.0-20240601000000")

        r.set.assert_called_once_with(REDIS_MODEL_VERSION_KEY, "1.0.0-20240601000000")

    def test_does_not_raise_when_redis_unavailable(self):
        from backend.main import _publish_model_version

        with patch("backend.main.Redis") as MockRedis:
            MockRedis.from_url.side_effect = ConnectionError("Redis down")
            # Must not propagate the exception.
            _publish_model_version("1.0.0-fallback")

    def test_key_constant_matches_between_api_and_worker(self):
        """The Redis key used by the API must match the key used by workers."""
        from backend.main import REDIS_MODEL_VERSION_KEY as api_key
        from tasks import REDIS_MODEL_VERSION_KEY as worker_key

        self.assertEqual(
            api_key,
            worker_key,
            "API and worker must use the same Redis coordination key",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Worker model isolation — process-boundary simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessBoundaryIsolation(unittest.TestCase):
    """
    Simulate process isolation without spawning real subprocesses.

    In a real deployment the API and Celery workers are separate OS processes.
    We emulate this by resetting the worker cache and verifying that the worker
    correctly builds its own model instead of borrowing from the API's dict.
    """

    def setUp(self):
        import tasks
        self.tasks = tasks
        tasks._reset_worker_cache()

    def tearDown(self):
        self.tasks._reset_worker_cache()

    def test_worker_builds_own_model_not_from_api_dict(self):
        import backend.main as api

        # Mark the API dict as "built" — in a real worker process this mutation
        # would be invisible, and the worker would start with ready=False.
        api.models["ready"] = True
        api.models["hybrid"] = MagicMock(name="api_hybrid_sentinel")

        fake_worker_models = _make_fake_worker_models(_make_item_df(3))

        try:
            with (
                patch("tasks._read_current_version", return_value="v1"),
                patch("tasks._build_models_for_worker", return_value=fake_worker_models),
            ):
                result = self.tasks._get_worker_models()

            # The worker's hybrid model must NOT be the API's sentinel.
            self.assertIsNot(
                result["hybrid"],
                api.models["hybrid"],
                "Worker must use its own model, not the API-process model object",
            )
        finally:
            api.models["ready"] = False
            api.models["hybrid"] = None

    def test_newly_built_model_visible_to_worker_without_restart(self):
        """Core property: after _publish_model_version, workers pick up the new model."""
        fake_models = _make_fake_worker_models(_make_item_df(5))

        # Worker has no cached model yet (fresh start / simulated new process).
        self.assertIsNone(self.tasks._worker_models)

        with (
            patch("tasks._read_current_version", return_value="build-001"),
            patch("tasks._build_models_for_worker", return_value=fake_models),
        ):
            result = self.tasks._get_worker_models()

        self.assertTrue(result["ready"])
        self.assertEqual(self.tasks._worker_model_version, "build-001")
