import pandas as pd

from src.model.knowledge_graph_model import KnowledgeGraphRecommender


if __name__ == '__main__':
    df = pd.read_csv('datasets/books.csv')

    model = KnowledgeGraphRecommender(df)

    recs = model.recommend('Harry Potter', top_n=5)

    print(recs)
