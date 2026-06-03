import pandas as pd

from src.model.hybrid_model import HybridRecommender
from src.model.online_updater import OnlineUpdater


class DummyContent:
    def __init__(self, df):
        self.df = df


def test_online_updater_applies_incremental_updates():
    item_df = pd.DataFrame([
        {'title': 'A', 'rating': 4.0, 'review_count': 2, 'avg_sentiment': 0.5, 'category': 'X'}
    ])

    content_df = pd.DataFrame([
        {'title': 'A', 'description': 'Test', 'top_reviews': [], 'category': 'X'}
    ])

    content_model = DummyContent(content_df)

    hr = HybridRecommender(content_model, collab_model=None, item_df=item_df)

    # initial conditions
    assert hr._review_count_map.get('A', 0) == 2
    assert 'A' in hr._rating_map
    assert abs(hr._sentiment_map.get('A', 0.0) - 0.5) < 1e-6

    updater = OnlineUpdater()
    hr.set_online_updater(updater)

    ok = hr.apply_interaction(user_id='u1', item_title='A', rating=5.0, sentiment=0.8)
    assert ok is True

    # review count increased
    assert hr._review_count_map['A'] == 3

    # sentiment moved toward new value
    assert abs(hr._sentiment_map['A'] - 0.6) < 1e-6

    # rating should have increased (best-effort)
    assert hr._rating_map['A'] > 4.0


def test_apply_interaction_without_updater_is_ok():
    item_df = pd.DataFrame([
        {'title': 'B', 'rating': 3.0, 'review_count': 0, 'avg_sentiment': 0.0, 'category': 'Y'}
    ])
    content_df = pd.DataFrame([{'title': 'B', 'description': '', 'top_reviews': [], 'category': 'Y'}])
    hr = HybridRecommender(DummyContent(content_df), collab_model=None, item_df=item_df)

    # no updater attached
    ok = hr.apply_interaction(user_id='u2', item_title='B', rating=4.0, sentiment=0.2)
    assert ok is True
    assert hr._review_count_map['B'] == 1
