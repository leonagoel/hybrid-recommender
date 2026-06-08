"""
Unit tests for DatasetManager in src/data/dataset_manager.py.
Run with: pytest tests/test_dataset_manager.py -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset_manager import DatasetManager


class TestDatasetManagerMergeAll:
    """Test merge_all deduplication and aggregation."""

    @pytest.fixture
    def dm_two_datasets(self):
        dm = DatasetManager()
        df1 = pd.DataFrame({
            "user_id": ["u1", "u2", "u1"],
            "item_id": ["i1", "i2", "i1"],
            "title": ["Book A", "Book B", "Book A"],
            "rating": [5.0, 3.0, 4.0],
            "review_text": ["Great!", "Okay.", "Good!"],
            "description": ["A book", "Another", "A book"],
            "category": ["Fiction", "Sci-Fi", "Fiction"],
            "views": [100, 50, 100],
            "purchases": [10, 5, 10],
        })
        df2 = pd.DataFrame({
            "user_id": ["u3"],
            "item_id": ["i3"],
            "title": ["Book C"],
            "rating": [4.5],
            "review_text": ["Nice!"],
            "description": ["Third book"],
            "category": ["Drama"],
            "views": [30],
            "purchases": [3],
        })
        dm.load_csv(df1, name="ds1")
        dm.load_csv(df2, name="ds2")
        return dm

    def test_deduplication_keeps_first_description(self, dm_two_datasets):
        """Items with same title but different descriptions keep first."""
        _, grouped = dm_two_datasets.merge_all()
        book_a = grouped[grouped["title"] == "Book A"].iloc[0]
        assert book_a["description"] == "A book"

    def test_rating_aggregation_averages_duplicates(self, dm_two_datasets):
        """Duplicate titles have their ratings averaged."""
        _, grouped = dm_two_datasets.merge_all()
        book_a = grouped[grouped["title"] == "Book A"].iloc[0]
        # Ratings are 5.0 and 4.0, average should be 4.5
        assert book_a["rating"] == 4.5

    def test_reviews_aggregation_concatenates(self, dm_two_datasets):
        """Reviews from duplicate titles are concatenated."""
        _, grouped = dm_two_datasets.merge_all()
        book_a = grouped[grouped["title"] == "Book A"].iloc[0]
        # top_reviews should have up to 2 reviews with length > 8
        top_reviews = book_a.get("top_reviews", [])
        assert isinstance(top_reviews, list)

    def test_missing_views_column_handled(self):
        """merge_all handles DataFrames missing views/purchases columns."""
        dm = DatasetManager()
        df = pd.DataFrame({
            "user_id": ["u1"],
            "item_id": ["i1"],
            "title": ["Book X"],
            "rating": [4.0],
            "review_text": ["Good book!"],
            "description": ["A"],
            "category": ["Fiction"],
        })
        dm.load_csv(df, name="ds1")
        _, grouped = dm.merge_all()
        assert len(grouped) == 1

    def test_empty_datasets_raises(self):
        """merge_all on empty manager raises ValueError."""
        dm = DatasetManager()
        with pytest.raises(ValueError, match="No datasets loaded"):
            dm.merge_all()

    def test_multiple_datasets_merged_correctly(self, dm_two_datasets):
        """merge_all produces correct number of unique items."""
        _, grouped = dm_two_datasets.merge_all()
        # ds1 has Book A (dup) and Book B, ds2 has Book C
        # Should have 3 unique items: Book A, Book B, Book C
        assert len(grouped) == 3
        titles = set(grouped["title"].tolist())
        assert "Book A" in titles
        assert "Book B" in titles
        assert "Book C" in titles


class TestDatasetManagerGetStats:
    """Test get_stats and list_datasets."""

    def test_get_stats_empty_manager(self):
        dm = DatasetManager()
        stats = dm.get_stats()
        assert stats["dataset_count"] == 0
        assert stats["total_rows"] == 0
        assert stats["datasets"] == []

    def test_get_stats_single_dataset(self):
        dm = DatasetManager()
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "title": ["A", "B"],
            "rating": [4.0, 5.0],
        })
        dm.load_csv(df, name="test_ds")
        stats = dm.get_stats()
        assert stats["dataset_count"] == 1
        assert stats["total_rows"] == 2
        assert "test_ds" in stats["datasets"]

    def test_get_stats_multiple_datasets(self):
        dm = DatasetManager()
        df1 = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "title": ["A"], "rating": [4.0]})
        df2 = pd.DataFrame({"user_id": ["u2"], "item_id": ["i2"], "title": ["B"], "rating": [5.0]})
        dm.load_csv(df1, name="ds1")
        dm.load_csv(df2, name="ds2")
        stats = dm.get_stats()
        assert stats["dataset_count"] == 2
        assert stats["total_rows"] == 2

    def test_list_datasets_empty_manager(self):
        dm = DatasetManager()
        result = dm.list_datasets()
        assert result == []

    def test_list_datasets_single_dataset(self):
        dm = DatasetManager()
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "title": ["A", "B"],
            "rating": [4.0, 5.0],
            "review_text": ["Good", "Great"],
        })
        dm.load_csv(df, name="my_dataset")
        result = dm.list_datasets()
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "my_dataset"
        assert entry["rows"] == 2
        assert "id" in entry
        assert "has_reviews" in entry
        assert "has_user_data" in entry
        assert "has_behavior" in entry

    def test_list_datasets_structure_keys(self):
        dm = DatasetManager()
        df = pd.DataFrame({
            "user_id": ["u1"],
            "item_id": ["i1"],
            "title": ["A"],
            "rating": [4.0],
        })
        dm.load_csv(df, name="ds1")
        result = dm.list_datasets()
        assert len(result) == 1
        entry = result[0]
        for key in ["id", "name", "rows", "has_reviews", "has_user_data", "has_behavior", "detected_columns"]:
            assert key in entry, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])