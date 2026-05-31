"""
Unit tests for DatasetManager.
"""
import io
import os
import pytest
import pandas as pd
from src.data.dataset_manager import DatasetManager


def test_dataset_manager_init_empty():
    manager = DatasetManager()
    assert len(manager._datasets) == 0


def test_dataset_manager_load_csv_file_not_found():
    manager = DatasetManager()
    with pytest.raises(FileNotFoundError):
        manager.load_csv("nonexistent_path_xyz_123.csv")


def test_dataset_manager_load_and_remove():
    manager = DatasetManager()
    csv_data = "title,description,category\nBook A,Desc A,Fantasy\nBook B,Desc B,SciFi"
    buffer = io.StringIO(csv_data)
    
    ds_id = manager.load_csv(buffer, name="test_data.csv")
    assert ds_id in manager._datasets
    assert len(manager.list_datasets()) == 1

    removed = manager.remove_dataset(ds_id)
    assert removed is True
    assert ds_id not in manager._datasets
    assert len(manager.list_datasets()) == 0

    assert manager.remove_dataset("invalid_id") is False


def test_dataset_manager_get_stats():
    manager = DatasetManager()
    csv_data = "title,description,category\nBook A,Desc A,Fantasy"
    buffer = io.StringIO(csv_data)
    manager.load_csv(buffer, name="test_data.csv")

    stats = manager.get_stats()
    assert stats["dataset_count"] == 1
    assert stats["total_rows"] == 1
    assert "test_data.csv" in stats["datasets"]


def test_dataset_manager_merge_all():
    manager = DatasetManager()
    csv_data1 = "title,description,category,rating\nBook A,Desc A,Fantasy,4.0"
    csv_data2 = "title,description,category,rating\nBook A,Desc A,Fantasy,5.0\nBook B,Desc B,SciFi,3.0"
    
    manager.load_csv(io.StringIO(csv_data1), name="ds1.csv")
    manager.load_csv(io.StringIO(csv_data2), name="ds2.csv")

    merged, grouped = manager.merge_all()
    # merged should contain all interaction rows
    assert len(merged) == 3
    # grouped should be deduplicated by title and rating averaged (Book A: (4.0+5.0)/2 = 4.5)
    assert len(grouped) == 2
    book_a_rating = grouped[grouped["title"] == "Book A"]["rating"].iloc[0]
    assert book_a_rating == 4.5


def test_dataset_manager_merge_empty_value_error():
    manager = DatasetManager()
    with pytest.raises(ValueError, match="No datasets loaded."):
        manager.merge_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
