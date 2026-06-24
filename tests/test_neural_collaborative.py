import pytest
import pandas as pd
from src.model.neural_collaborative_model import NeuralCollaborativeRecommender

@pytest.fixture
def sample_interaction_df():
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3'],
        'title': ['itemA', 'itemB', 'itemB', 'itemC', 'itemA', 'itemC'],
        'rating': [5.0, 4.0, 4.5, 3.0, 2.0, 5.0],
        'purchases': [1, 0, 1, 0, 0, 1],
        'views': [5, 2, 8, 1, 3, 10]
    }
    return pd.DataFrame(data)

def test_neural_collab_initialization(sample_interaction_df):
    model = NeuralCollaborativeRecommender(sample_interaction_df, embedding_dim=8, epochs=1)
    assert model.model is not None

def test_neural_collab_recommend(sample_interaction_df):
    model = NeuralCollaborativeRecommender(sample_interaction_df, embedding_dim=8, epochs=1)
    recs = model.recommend('itemA', top_n=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
    if len(recs) > 0:
        assert 'title' in recs[0]
        assert 'collab_score' in recs[0]

def test_neural_collab_predict_for_user(sample_interaction_df):
    model = NeuralCollaborativeRecommender(sample_interaction_df, embedding_dim=8, epochs=1)
    recs = model.predict_for_user('u1', top_n=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
    if len(recs) > 0:
        assert 'title' in recs[0]
        assert 'predicted_score' in recs[0]

def test_neural_collab_predict_rating(sample_interaction_df):
    model = NeuralCollaborativeRecommender(sample_interaction_df, embedding_dim=8, epochs=1)
    score = model.predict_rating('u1', 'itemC')
    assert isinstance(score, float)
