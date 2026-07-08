"""
Regression tests for ANN (HNSW) support in ContentRecommender.

Previously:
  1. hnswlib was not in requirements.txt, so the module-level import silently
     fell back to hnswlib=None and _ann_enabled was never set True.
  2. _ann_index and _ann_enabled were never initialized in __init__,
     so getattr(self, '_ann_enabled', False) was always False even if
     hnswlib had been installed.

The fix adds hnswlib>=0.7.0 to requirements.txt and builds the HNSW index
in __init__ when hnswlib is importable and the catalog is small enough to
densify without OOM.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_hnswlib_in_requirements():
    reqs = Path("requirements.txt").read_text()
    assert "hnswlib" in reqs, "hnswlib must be listed in requirements.txt"


def test_ann_index_initialized_in_init():
    """__init__ must set _ann_index and _ann_enabled so ANN path is reachable."""
    source = Path("src/model/content_model.py").read_text()
    assert "self._ann_index = None" in source, (
        "_ann_index must be explicitly initialized in __init__"
    )
    assert "self._ann_enabled = False" in source, (
        "_ann_enabled must be explicitly initialized in __init__"
    )


def test_ann_build_guarded_by_hnswlib_check():
    """HNSW index build must be guarded by hnswlib is not None."""
    source = Path("src/model/content_model.py").read_text()
    init_end = source.find("def recommend(")
    init_section = source[:init_end]
    assert "hnswlib is not None" in init_section, (
        "HNSW index construction must be guarded by 'if hnswlib is not None'"
    )
    assert "self._ann_enabled = True" in init_section, (
        "_ann_enabled must be set True on successful HNSW index build"
    )


def test_ann_build_has_size_guard():
    """Dense conversion must be skipped for catalogs too large to fit in memory."""
    source = Path("src/model/content_model.py").read_text()
    init_end = source.find("def recommend(")
    init_section = source[:init_end]
    assert "n_items * dim" in init_section, (
        "A size guard on n_items * dim must exist to avoid OOM on large catalogs"
    )


def test_result_uses_score_loop_variable_not_scores_array():
    """recommend() must use the loop variable 'score', not 'scores[i]'."""
    source = Path("src/model/content_model.py").read_text()
    recommend_start = source.find("def recommend(")
    recommend_section = source[recommend_start:]
    assert "float(scores[i])" not in recommend_section, (
        "scores[i] in the result loop causes NameError in the ANN success path "
        "where 'scores' is never assigned — use the loop variable 'score' instead"
    )
    assert '"content_score": float(score)' in recommend_section, (
        "result dict must use float(score) — the loop variable already holds the similarity value"
    )
