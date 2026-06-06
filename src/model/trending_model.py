import pandas as pd


class TrendingRecommender:
    def __init__(self, df=None, data_path="datasets/sample_products.csv"):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(data_path)

    def get_trending_products(self, top_n=10):
        product_stats = (
            self.df.groupby(["item_id", "title"])
            .agg(
                avg_rating=("rating", "mean"),
                total_views=("views", "sum"),
                total_purchases=("purchases", "sum"),
            )
            .reset_index()
        )

        product_stats["trending_score"] = (
            0.5 * product_stats["total_purchases"]
            + 0.3 * product_stats["total_views"]
            + 0.2 * product_stats["avg_rating"]
        )

        trending_products = product_stats.sort_values(
            by="trending_score",
            ascending=False
        )

        return trending_products.head(top_n).to_dict(orient="records")