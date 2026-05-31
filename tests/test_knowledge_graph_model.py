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


def test_kg_recommendations_invalid_title():
    df = pd.DataFrame({
        'title': ['Book A', 'Book B'],
        'category': ['Fantasy', 'Fantasy']
    })
    model = KnowledgeGraphRecommender(df)
    recs = model.recommend('Unknown Book', top_n=2)
    assert recs == []


def test_kg_recommendations_respects_top_n():
    df = pd.DataFrame({
        'title': ['Book A', 'Book B', 'Book C', 'Book D'],
        'category': ['Fantasy', 'Fantasy', 'Fantasy', 'Fantasy']
    })
    model = KnowledgeGraphRecommender(df)
    recs = model.recommend('Book A', top_n=2)
    assert len(recs) == 2


def test_kg_recommendations_missing_optional_columns():
    df = pd.DataFrame({
        'title': ['Book A', 'Book B']
    })
    model = KnowledgeGraphRecommender(df)
    recs = model.recommend('Book A', top_n=1)
    assert len(recs) == 1
