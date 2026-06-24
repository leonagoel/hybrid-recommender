import pandas as pd
from src.model.hybrid_model import HybridRecommender


class _ContentStub:
    def __init__(self):
        self.df = pd.DataFrame([
            {'title': 'Product A', 'description': 'desc A'},
            {'title': 'Product B', 'description': 'desc B'},
        ])


class _CollabStub:
    def __init__(self):
        self._user_to_idx = {'u1': 0}

    def predict_for_user(self, user_id, top_n=10):
        return [
            {'title': 'Product A', 'predicted_score': 4.5},
            {'title': 'Product B', 'predicted_score': 3.8},
        ]


def test_recommend_for_user_returns_list_when_debiasing_disabled():
    content = _ContentStub()
    collab = _CollabStub()
    model = HybridRecommender(content, collab, item_df=None)
    model.use_causal_debiasing = False
    
    recs = model.recommend_for_user('u1', top_n=2)
    assert recs is not None
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert recs[0]['title'] == 'Product A'
    assert recs[1]['title'] == 'Product B'
