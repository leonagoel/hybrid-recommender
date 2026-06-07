# src/model/gnn_model.py

class GNNRecommender:
    """
    Graph Neural Network based recommender.
    Builds a user-item-category graph and generates recommendations.
    """

    def __init__(self, item_df, interaction_df):
        self.item_df = item_df
        self.interaction_df = interaction_df

        self.user_nodes = {}
        self.item_nodes = {}
        self.category_nodes = {}

        self.edge_index = None

    def build_graph(self):
        pass

    def train(self):
        pass

    def recommend(self, user_id, top_n=10):
        return []