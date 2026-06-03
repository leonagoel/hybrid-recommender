import pandas as pd

from src.model.knowledge_graph_model import KnowledgeGraphRecommender


def test_kg_recommendations():
    df = pd.DataFrame({
        'title': [
            'Book A',
            'Book B',
            'Book C'
        ],
        'category': [
            'Fantasy',
            'Fantasy',
            'Science'
        ]
    })

    model = KnowledgeGraphRecommender(df)

    recs = model.recommend('Book A', top_n=2)

    assert len(recs) > 0
    assert 'title' in recs[0]
    assert 'kg_score' in recs[0]
