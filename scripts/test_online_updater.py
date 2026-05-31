"""Smoke test for OnlineUpdater.

Run from repository root:
    python scripts/test_online_updater.py

This test is lightweight and dependency-free.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.online_updater import OnlineUpdater
import pandas as pd


def _make_mock_content():
    class MockContent:
        def __init__(self):
            self.df = pd.DataFrame([{"title": "Item A", "combined": "A nice widget"}])
            # simple dense matrix
            self.matrix = None
            # model with encode method is optional
            class M:
                def encode(self, texts, show_progress_bar=False):
                    return [[0.1 * len(t)] for t in texts]
            self.model = M()

    return MockContent()


def _make_mock_collab():
    class MockCollab:
        def __init__(self):
            self.df = pd.DataFrame(columns=["user_id", "title", "rating"])

        def partial_fit(self, *args, **kwargs):
            # accept many signatures
            return True

    return MockCollab()


def run_smoke():
    item_df = pd.DataFrame([{"title": "Item A", "rating": 4.0, "review_count": 1, "views": 10}])
    content = _make_mock_content()
    collab = _make_mock_collab()

    updater = OnlineUpdater(content_model=content, collab_model=collab, hybrid=None, item_df=item_df)

    res = updater.ingest_interaction(user_id="user1", item_title="Item A", rating=5.0, review_text="Great product!")
    print("ingest result:", res)

    # Basic assertions
    assert res.get("popularity") is True
    assert res.get("rating") is True
    # sentiment may be False if analyzer missing

    # New views should be incremented
    v = int(updater.item_df.loc[updater.item_df["title"] == "Item A", "views"].iloc[0])
    assert v >= 11

    # Collab df appended or partial_fit accepted
    print("updated item row:")
    print(updater.item_df.to_dict(orient="records"))

    print("Smoke test passed")


if __name__ == "__main__":
    try:
        run_smoke()
    except AssertionError as e:
        print("Smoke test failed:", e)
        raise


