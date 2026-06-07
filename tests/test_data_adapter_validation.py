import pytest
import pandas as pd
import numpy as np

from src.data.data_adapter import (
    adapt_data,
    validate_dataframe,
    validate_recommender_inputs,
    _has_blank_values,
    detect_column,
    read_file,
)


class TestDetectColumn:
    def test_detect_column_exact_match(self):
        cols = ["user_id", "product_id", "rating"]
        assert detect_column(cols, ["user_id"]) == "user_id"

    def test_detect_column_case_insensitive(self):
        cols = ["User_ID", "Product_ID", "Rating"]
        assert detect_column(cols, ["user_id"]) == "User_ID"

    def test_detect_column_substring_match(self):
        cols = ["user_id", "review_text", "rating"]
        assert detect_column(cols, ["review"]) == "review_text"

    def test_detect_column_empty_keywords_returns_none(self):
        cols = ["user_id", "product_id", "rating"]
        assert detect_column(cols, []) is None

    def test_detect_column_no_match_returns_none(self):
        cols = ["user_id", "product_id", "rating"]
        assert detect_column(cols, ["nonexistent"]) is None

    def test_detect_column_avoids_false_positive_customer_rating(self):
        cols = ["customer_rating", "score"]
        result = detect_column(cols, ["customer"])
        assert result is None or "customer_rating" not in str(result)


class TestHasBlankValues:
    def test_has_blank_values_nulls(self):
        series = pd.Series(["a", None, "b"])
        assert _has_blank_values(series)

    def test_has_blank_values_empty_strings(self):
        series = pd.Series(["a", "", "b"])
        assert _has_blank_values(series)

    def test_has_blank_values_whitespace_only(self):
        series = pd.Series(["a", "  ", "b"])
        assert _has_blank_values(series)

    def test_has_blank_values_clean_data(self):
        series = pd.Series(["a", "b", "c"])
        assert not _has_blank_values(series)

    def test_has_blank_values_mixed_null_and_valid(self):
        series = pd.Series([None, "valid", "also valid"])
        assert _has_blank_values(series)


class TestValidateDataFrame:
    def test_validate_dataframe_empty_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)

    def test_validate_dataframe_single_column_raises(self):
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 columns"):
            validate_dataframe(df)

    def test_validate_dataframe_valid_two_columns(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        assert validate_dataframe(df) is True


class TestValidateRecommenderInputs:
    def test_validate_recommender_inputs_missing_user_col(self):
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "rating": [4.0, 5.0],
        })
        with pytest.raises(ValueError, match="user_id"):
            validate_recommender_inputs(df)

    def test_validate_recommender_inputs_missing_item_and_title(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "rating": [4.0, 5.0],
        })
        with pytest.raises(ValueError, match="item_id or title"):
            validate_recommender_inputs(df)

    def test_validate_recommender_inputs_missing_rating(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
        })
        with pytest.raises(ValueError, match="rating"):
            validate_recommender_inputs(df)

    def test_validate_recommender_inputs_all_valid(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [4.0, 5.0],
        })
        assert validate_recommender_inputs(
            df, user_col="user_id", item_id_col="item_id", rating_col="rating"
        ) is True

    def test_validate_recommender_inputs_blank_user_id(self):
        df = pd.DataFrame({
            "user_id": ["u1", ""],
            "item_id": ["i1", "i2"],
            "rating": [4.0, 5.0],
        })
        with pytest.raises(ValueError, match="user_id"):
            validate_recommender_inputs(
                df, user_col="user_id", item_id_col="item_id", rating_col="rating"
            )

    def test_validate_recommender_inputs_blank_rating(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [4.0, None],
        })
        with pytest.raises(ValueError, match="rating"):
            validate_recommender_inputs(
                df, user_col="user_id", item_id_col="item_id", rating_col="rating"
            )

    def test_validate_recommender_inputs_non_numeric_rating(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [4.0, "bad"],
        })
        with pytest.raises(ValueError, match="rating must be numeric"):
            validate_recommender_inputs(
                df, user_col="user_id", item_id_col="item_id", rating_col="rating"
            )


class TestAdaptDataInteractionValidation:
    def test_adapt_data_accepts_valid_interaction_dataset(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [4.0, 5.0],
            "title": ["Item One", "Item Two"],
        })

        adapted, meta = adapt_data(df)

        assert meta["has_user_data"] is True
        assert list(adapted["user_id"]) == ["u1", "u2"]

    def test_adapt_data_rejects_missing_interaction_columns(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "rating": [4.0, 5.0],
        })

        with pytest.raises(ValueError, match="item_id or title"):
            adapt_data(df)

    def test_adapt_data_rejects_blank_core_identifiers(self):
        df = pd.DataFrame({
            "user_id": ["u1", " "],
            "item_id": ["i1", "i2"],
            "rating": [4.0, 5.0],
        })

        with pytest.raises(ValueError, match="user_id"):
            adapt_data(df)

    def test_adapt_data_rejects_non_numeric_ratings(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [4.0, "bad"],
        })

        with pytest.raises(ValueError, match="rating must be numeric"):
            adapt_data(df)
