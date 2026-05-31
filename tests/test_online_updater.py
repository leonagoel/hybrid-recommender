import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.online_updater import OnlineUpdater
from src.model.hybrid_model import HybridRecommender


class _MockContent:
    def __init__(self):
        self.df = pd.DataFrame([{"title": "Item A", "combined": "Desc A", "description": "D", "top_reviews": []}])
        self.matrix = None
        class M:
            def encode(self, texts, show_progress_bar=False):
                return [[0.1 * len(t)] for t in texts]
        self.model = M()


class _MockCollab:
    def __init__(self):
        self.df = pd.DataFrame(columns=["user_id", "title", "rating"]) 
    def partial_fit(self, *args, **kwargs):
        return True


def test_online_updater_smoke():
    item_df = pd.DataFrame([{"title": "Item A", "rating": 4.0, "review_count": 1, "views": 5}])
    content = _MockContent()
    collab = _MockCollab()
    updater = OnlineUpdater(content_model=content, collab_model=collab, item_df=item_df)

    res = updater.ingest_interaction(user_id='u1', item_title='Item A', rating=5.0, review_text='Great')
    assert res['popularity'] is True
    assert res['rating'] is True
    # views incremented
    assert int(updater.item_df.loc[updater.item_df['title']=='Item A','views'].iloc[0]) >= 6


def test_hybrid_integration_apply_interaction():
    item_df = pd.DataFrame([{"title": "Item A", "rating": 4.0, "review_count": 1, "views": 10}])
    content = _MockContent()
    collab = _MockCollab()
    hybrid = HybridRecommender(content_model=content, collab_model=collab, item_df=item_df)

    updater = OnlineUpdater(content_model=content, collab_model=collab, item_df=item_df)
    hybrid.set_online_updater(updater)

    # call via hybrid API
    out = hybrid.apply_interaction(user_id='u2', item_title='Item A', rating=3.0, review_text='OK')
    assert isinstance(out, dict)
    # hybrid maps updated
    assert hybrid._review_count_map.get('Item A', 0) >= 2
