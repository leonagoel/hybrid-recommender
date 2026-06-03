import pandas as pd

from src.model.hybrid_model import HybridRecommender


class DummyContentModel:
    def __init__(self, recommend_return, df=None):
        self._recommend_return = recommend_return
        self.df = (
            df
            if df is not None
            else pd.DataFrame(
                columns=["title", "description", "top_reviews"]
            )
        )

    def recommend(self, title, top_n=None, target_catalog=None):
        return self._recommend_return


def test_empty_recommendations_do_not_crash():
    content = DummyContentModel([])

    hr = HybridRecommender(
        content_model=content,
        collab_model=None,
        item_df=None,
    )

    result = hr.recommend("unknown", top_n=5)

    assert isinstance(result, list)
    assert result == []


def test_recommendation_missing_title_is_ignored():
    content = DummyContentModel(
        [
            {
                "content_score": 0.9,
            }
        ]
    )

    hr = HybridRecommender(
        content_model=content,
        collab_model=None,
        item_df=None,
    )

    result = hr.recommend("unknown", top_n=5)

    assert isinstance(result, list)
    assert result == []


def test_non_dict_entries_are_ignored():
    content = DummyContentModel(
        [
            None,
            "bad",
            123,
            [],
        ]
    )

    hr = HybridRecommender(
        content_model=content,
        collab_model=None,
        item_df=None,
    )

    result = hr.recommend("unknown", top_n=5)

    assert isinstance(result, list)
    assert result == []


def test_valid_recommendation_is_preserved():
    df = pd.DataFrame(
        [
            {
                "title": "Valid Item",
                "description": "Sample description",
                "top_reviews": [],
            }
        ]
    )

    content = DummyContentModel(
        [
            {
                "title": "Valid Item",
                "content_score": 0.8,
            }
        ],
        df=df,
    )

    hr = HybridRecommender(
        content_model=content,
        collab_model=None,
        item_df=None,
    )

    result = hr.recommend("source", top_n=5)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["title"] == "Valid Item"


def test_mixed_valid_and_malformed_entries():
    df = pd.DataFrame(
        [
            {
                "title": "Valid Item",
                "description": "Sample description",
                "top_reviews": [],
            }
        ]
    )

    content = DummyContentModel(
        [
            None,
            "bad",
            {},
            {"content_score": 0.5},
            {
                "title": "Valid Item",
                "content_score": 0.8,
            },
        ],
        df=df,
    )

    hr = HybridRecommender(
        content_model=content,
        collab_model=None,
        item_df=None,
    )

    result = hr.recommend("source", top_n=5)

    assert len(result) == 1
    assert result[0]["title"] == "Valid Item"