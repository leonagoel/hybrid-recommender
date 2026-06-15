import numpy as np
import pandas as pd
import pytest
from src.model.knowledge_graph_model import KnowledgeGraphRecommender


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'title':    ['Book A', 'Book B', 'Book C', 'Book D'],
        'category': ['Fantasy', 'Fantasy', 'Science', 'Science'],
        'author':   ['Author X', 'Author X', 'Author Y', 'Author Z'],
        'genre':    ['Epic', 'Epic', 'Hard SF', 'Soft SF'],
        'keywords': ['magic,dragon', 'magic,elves', 'space,physics', 'space,emotion'],
    })


# --- TransE / DistMult / ComplEx all return valid recs ---
def test_transe(sample_df):
    recs = KnowledgeGraphRecommender(sample_df, model_type="TransE").recommend("Book A", top_n=2)
    assert len(recs) > 0 and 'kg_score' in recs[0]

def test_distmult(sample_df):
    recs = KnowledgeGraphRecommender(sample_df, model_type="DistMult").recommend("Book A", top_n=2)
    assert len(recs) > 0 and 'kg_score' in recs[0]

def test_complex(sample_df):
    recs = KnowledgeGraphRecommender(sample_df, model_type="ComplEx").recommend("Book A", top_n=2)
    assert len(recs) > 0 and 'kg_score' in recs[0]

def test_unknown_title_returns_empty(sample_df):
    recs = KnowledgeGraphRecommender(sample_df).recommend("Ghost Title")
    assert recs == []

def test_scores_in_valid_range(sample_df):
    recs = KnowledgeGraphRecommender(sample_df).recommend("Book A", top_n=3)
    for rec in recs:
        assert -1.0 <= rec['kg_score'] <= 1.0

# --- get_embedding ---
def test_get_embedding_shape(sample_df):
    emb = KnowledgeGraphRecommender(sample_df, embedding_dim=32).get_embedding("Book A")
    assert emb.shape == (32,)

def test_get_embedding_complex_double_dim(sample_df):
    emb = KnowledgeGraphRecommender(sample_df, embedding_dim=32, model_type="ComplEx").get_embedding("Book A")
    assert emb.shape == (64,)

def test_get_embedding_unknown_returns_none(sample_df):
    assert KnowledgeGraphRecommender(sample_df).get_embedding("Ghost") is None

# --- Keyword triples ---
def test_keyword_triples_generated(sample_df):
    model = KnowledgeGraphRecommender(sample_df)
    kw_rel = model.relation_to_idx['similar_keywords']
    kw_triples = [t for t in model.triples if t[1] == kw_rel]
    assert len(kw_triples) >= 2  # magic overlap + space overlap

# --- Invalid model type ---
def test_invalid_model_type_raises(sample_df):
    with pytest.raises(ValueError, match="Unsupported model_type"):
        KnowledgeGraphRecommender(sample_df, model_type="BadModel")

# --- Edge cases ---
def test_minimal_df_no_metadata():
    df = pd.DataFrame({'title': ['Item 1', 'Item 2', 'Item 3']})
    model = KnowledgeGraphRecommender(df)
    assert model.triples == []

def test_single_item_df():
    df = pd.DataFrame({'title': ['Lonely Book'], 'category': ['Fantasy']})
    recs = KnowledgeGraphRecommender(df).recommend("Lonely Book", top_n=5)
    assert recs == []