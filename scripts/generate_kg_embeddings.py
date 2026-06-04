import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.knowledge_graph_model import KnowledgeGraphRecommender


if __name__ == '__main__':
    df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'booksdata.csv')

    model = KnowledgeGraphRecommender(df)

    recs = model.recommend('Harry Potter', top_n=5)

    print(recs)
