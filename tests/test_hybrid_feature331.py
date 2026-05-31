import pandas as pd
from src.model.hybrid_model import HybridRecommender


def test_minmax_normalization():
    h = HybridRecommender(None, None, item_df=None, normalization='minmax')
    res = h._normalize_scores([0.0, 50.0, 100.0])
    assert res[0] == 0.0
    assert res[1] == 0.5
    assert res[2] == 1.0


def test_zscore_constant_fallback():
    h = HybridRecommender(None, None, item_df=None, normalization='zscore')
    res = h._normalize_scores([0.0, 0.0, 0.0])
    assert all(abs(v - 0.5) < 1e-6 for v in res)


def test_zscore_monotonic_range():
    h = HybridRecommender(None, None, item_df=None, normalization='zscore')
    vals = [1.0, 2.0, 3.0]
    res = h._normalize_scores(vals)
    assert res == sorted(res)
    assert all(0.0 < v < 1.0 for v in res)


def test_weight_matrix_category_override():
    item_df = pd.DataFrame({'title': ['A', 'B', 'C'], 'category': ['Audio', 'Audio', 'Books']})
    wm = {'default': (0.4, 0.4, 0.2), 'category:Audio': (0.6, 0.3, 0.1)}
    h = HybridRecommender(None, None, item_df=item_df, weight_matrix=wm)
    a, b, g = h._get_active_weights(0.4, 0.4, 0.2, user_id=None, candidate_titles=['A', 'B'])
    tot = 0.6 + 0.3 + 0.1
    assert abs(a - (0.6 / tot)) < 1e-6


def test_weight_matrix_warm_user_override():
    item_df = pd.DataFrame({'title': ['A', 'B', 'C'], 'category': ['Audio', 'Audio', 'Books']})
    collab = type('C', (), {})()
    collab.df = pd.DataFrame({'user_id': [42] * 12})
    wm = {'default': (0.5, 0.4, 0.1), 'warm_user': (0.2, 0.7, 0.1)}
    h = HybridRecommender(None, collab, item_df=item_df, weight_matrix=wm)
    a, b, g = h._get_active_weights(0.5, 0.4, 0.1, user_id=42, candidate_titles=None)
    tot = 0.2 + 0.7 + 0.1
    assert abs(a - (0.2 / tot)) < 1e-6


def test_diversity_rerank_empty_results():
    h = HybridRecommender(None, None, item_df=None)
    assert h._diversity_rerank([], top_n=5) == []


def test_diversity_rerank_zeros():
    h = HybridRecommender(None, None, item_df=None)
    results = [{"title": "A", "hybrid_score": 0.9, "category": "X"}]
    assert h._diversity_rerank(results, top_n=5, diversity=0.0, serendipity=0.0) == results


def test_diversity_rerank_applies_diversity_penalty():
    h = HybridRecommender(None, None, item_df=None)
    results = [
        {"title": "A", "hybrid_score": 0.9, "category": "X"},
        {"title": "B", "hybrid_score": 0.8, "category": "X"},
        {"title": "C", "hybrid_score": 0.7, "category": "Y"}
    ]
    reranked = h._diversity_rerank(results, top_n=2, diversity=1.0, serendipity=0.0)
    assert len(reranked) == 2
    assert reranked[0]["title"] == "A"
    assert reranked[1]["title"] == "C"
