import pandas as pd

from src.model.knowledge_graph_model import KnowledgeGraphRecommender

if __name__ == '__main__':
    from pathlib import Path
    _BASE_DIR = Path(__file__).resolve().parent.parent  # project root

    df = pd.read_csv(_BASE_DIR / 'datasets' / 'books.csv')

    model = KnowledgeGraphRecommender(df)

    recs = model.recommend('Harry Potter', top_n=5)

    print(recs)
