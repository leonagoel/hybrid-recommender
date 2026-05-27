import pytest
import pandas as pd

from src.data.data_adapter import adapt_data


def test_adapt_data_accepts_valid_interaction_dataset():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "item_id": ["i1", "i2"],
        "rating": [4.0, 5.0],
        "title": ["Item One", "Item Two"],
    })

    adapted, meta = adapt_data(df)

    assert meta["has_user_data"] is True
    assert list(adapted["user_id"]) == ["u1", "u2"]


def test_adapt_data_rejects_missing_interaction_columns():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "rating": [4.0, 5.0],
    })

    with pytest.raises(ValueError, match="item_id or title"):
        adapt_data(df)


def test_adapt_data_rejects_blank_core_identifiers():
    df = pd.DataFrame({
        "user_id": ["u1", " "],
        "item_id": ["i1", "i2"],
        "rating": [4.0, 5.0],
    })

    with pytest.raises(ValueError, match="user_id"):
        adapt_data(df)


def test_adapt_data_rejects_non_numeric_ratings():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "item_id": ["i1", "i2"],
        "rating": [4.0, "bad"],
    })

    with pytest.raises(ValueError, match="rating must be numeric"):
        adapt_data(df)
