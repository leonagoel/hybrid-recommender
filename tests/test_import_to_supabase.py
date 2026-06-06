from types import SimpleNamespace

import pandas as pd

from scripts import import_to_supabase


class FakeProductsTable:
    def __init__(self):
        self.rows = []
        self.calls = []

    def upsert(self, rows, **kwargs):
        self.rows.extend(rows)
        self.calls.append(("upsert", rows, kwargs))
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


class FakeSupabase:
    def __init__(self):
        self.products = FakeProductsTable()

    def table(self, name):
        assert name == "products"
        return self.products


def test_build_product_row_initializes_review_count():
    row = pd.Series({
        "title": "Widget",
        "description": "Useful widget",
        "category": "Tools",
        "rating": "4.5",
        "sentiment": "0.25",
    })

    product = import_to_supabase.build_product_row(row)

    assert product == {
        "title": "Widget",
        "description": "Useful widget",
        "category": "Tools",
        "rating": 4.5,
        "avg_sentiment": 0.25,
        "review_count": 0,
        "metadata": {},
    }


def test_import_dataset_sends_review_count_to_supabase(tmp_path, monkeypatch):
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "title,description,category,rating\n"
        "Widget,Useful widget,Tools,4.5\n"
        "Book,Readable book,Books,3.5\n",
        encoding="utf-8",
    )
    fake_supabase = FakeSupabase()
    monkeypatch.setattr(import_to_supabase, "get_supabase_admin", lambda: fake_supabase)

    imported = import_to_supabase.import_dataset(str(csv_path), batch_size=2)

    assert imported == 2
    assert len(fake_supabase.products.rows) == 2
    assert all(row["review_count"] == 0 for row in fake_supabase.products.rows)
    assert all("review_count" in row for row in fake_supabase.products.rows)
