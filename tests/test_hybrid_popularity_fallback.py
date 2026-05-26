import pandas as pd

from src.model.hybrid_model import HybridRecommender


class EmptyContentModel:
    def __init__(self, df):
        self.df = df

    def recommend(self, title, top_n=10):
        return []


def make_recommender():
    item_df = pd.DataFrame({
        "title": ["Product A", "Product B", "Product C"],
        "description": ["A", "B", "C"],
        "category": ["Electronics", "Electronics", "Home"],
        "rating": [4.5, 3.8, 4.9],
        "review_count": [120, 45, 200],
        "avg_sentiment": [0.6, 0.2, 0.8],
    })
    return HybridRecommender(EmptyContentModel(item_df), item_df=item_df)


def test_unknown_item_uses_global_popularity_fallback():
    recs = make_recommender().recommend("Unknown Product", top_n=2)

    assert [rec["title"] for rec in recs] == ["Product C", "Product A"]


def test_popularity_fallback_excludes_source_title():
    recs = make_recommender().get_popular_fallback_items(
        top_n=3,
        exclude_title="Product C",
    )

    titles = [rec["title"] for rec in recs]
    assert "Product C" not in titles
    assert titles[0] == "Product A"
