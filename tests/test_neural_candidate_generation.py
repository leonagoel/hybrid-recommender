import pandas as pd

from src.model.hybrid_model import HybridRecommender


class FakeContentModel:
    def __init__(self, item_df):
        self.df = item_df

    def recommend(self, title, top_n=10, target_catalog=None):
        return [
            {"title": "Product B", "content_score": 0.8},
            {"title": "Product C", "content_score": 0.6},
        ][:top_n]


class FakeCollabModel:
    _user_to_idx = {"u1": 0}

    def recommend(self, title, top_n=10, target_catalog=None):
        return [
            {"title": "Product C", "collab_score": 0.9},
            {"title": "Product B", "collab_score": 0.2},
        ][:top_n]

    def predict_for_user(self, user_id, top_n=10):
        return [{"title": "Product C", "predicted_score": 0.9}]


class FakeRetrievalModel:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve_candidates_for_user(self, user_id, top_k=100):
        return self.candidates.get(user_id, [])[:top_k]


def _item_df():
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104],
            "title": ["Product A", "Product B", "Product C", "Product D"],
            "description": ["A", "B", "C", "D"],
            "category": ["Electronics", "Electronics", "Books", "Books"],
            "rating": [4.8, 4.1, 4.4, 4.9],
            "review_count": [100, 80, 50, 500],
            "avg_sentiment": [0.1, 0.3, 0.4, 0.9],
            "combined": ["Product A", "Product B", "Product C", "Product D"],
        }
    )


def _recommender(candidates=None):
    item_df = _item_df()
    return HybridRecommender(
        FakeContentModel(item_df),
        FakeCollabModel(),
        item_df,
        retrieval_model=FakeRetrievalModel(candidates or {}),
    )


def test_neural_candidate_source_reranks_retrieved_candidates_only():
    model = _recommender({"u1": [102, 104]})

    recs = model.recommend(
        "Product A",
        user_id="u1",
        top_n=5,
        candidate_source="neural",
    )

    assert {rec["title"] for rec in recs}.issubset({"Product B", "Product D"})
    assert model.last_candidate_context == {
        "requested": "neural",
        "effective": "neural",
        "fallback": False,
        "neural_candidate_count": 2,
    }


def test_neural_candidate_source_falls_back_to_hybrid_when_unavailable():
    model = _recommender({"other": [104]})

    recs = model.recommend(
        "Product A",
        user_id="u1",
        top_n=2,
        candidate_source="neural",
    )

    assert [rec["title"] for rec in recs]
    assert model.last_candidate_context["requested"] == "neural"
    assert model.last_candidate_context["effective"] == "hybrid"
    assert model.last_candidate_context["fallback"] is True


def test_mixed_candidate_source_supplements_hybrid_pool_with_neural_candidates():
    model = _recommender({"u1": [104]})

    recs = model.recommend(
        "Product A",
        user_id="u1",
        top_n=3,
        candidate_source="mixed",
    )

    assert "Product D" in [rec["title"] for rec in recs]
    assert model.last_candidate_context["effective"] == "mixed"
    assert model.last_candidate_context["neural_candidate_count"] == 1
