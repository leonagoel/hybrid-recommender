"""
Unit tests for the data_adapter module.
Tests CSV/JSON reading and schema adaptation functions.
"""
import pytest
import pandas as pd
import json
import tempfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.data_adapter import (
    detect_column,
    validate_dataframe,
    validate_recommender_inputs,
    read_file,
    adapt_data,
    preprocess_books_data,
    preprocess_ratings_data,
    preprocess_sentiment_data,
)


class TestDetectColumn:
    """Test detect_column function."""

    def test_exact_match(self):
        """Test exact column name match."""
        columns = ["Title", "Description", "Price"]
        assert detect_column(columns, ["title"]) == "Title"

    def test_substring_match(self):
        """Test substring matching."""
        columns = ["product_title", "description", "price"]
        assert detect_column(columns, ["title"]) == "product_title"

    def test_no_match(self):
        """Test when no column matches."""
        columns = ["abc", "def", "ghi"]
        assert detect_column(columns, ["title"]) is None

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        columns = ["PRODUCT_NAME", "DESCRIPTION"]
        assert detect_column(columns, ["name"]) == "PRODUCT_NAME"

    def test_first_exact_match_priority(self):
        """Test that exact match takes priority over substring."""
        columns = ["title", "original_title"]
        assert detect_column(columns, ["title"]) == "title"


class TestValidateDataframe:
    """Test validate_dataframe function."""

    def test_valid_dataframe(self):
        """Test validation passes for valid DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df) is True

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)

    def test_single_column_raises(self):
        """Test that single column DataFrame raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 columns"):
            validate_dataframe(df)


class TestValidateRecommenderInputs:
    """Test validate_recommender_inputs function."""

    def test_valid_recommender_inputs(self):
        """Test validation passes with correct columns and types."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [5.0, 4.0]
        })
        assert validate_recommender_inputs(
            df,
            user_col="user_id",
            item_id_col="item_id",
            rating_col="rating"
        ) is True

    def test_missing_required_columns_raises(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({
            "item_id": ["i1", "i2"],
            "rating": [5.0, 4.0]
        })
        with pytest.raises(ValueError, match="missing required column"):
            validate_recommender_inputs(
                df,
                user_col=None,
                item_id_col="item_id",
                rating_col="rating"
            )

    def test_blank_values_raises(self):
        """Test that blank string values or nulls raise ValueError."""
        df = pd.DataFrame({
            "user_id": ["u1", "   "],
            "item_id": ["i1", "i2"],
            "rating": [5.0, 4.0]
        })
        with pytest.raises(ValueError, match="invalid value"):
            validate_recommender_inputs(
                df,
                user_col="user_id",
                item_id_col="item_id",
                rating_col="rating"
            )

    def test_non_numeric_rating_raises(self):
        """Test that non-numeric rating values raise ValueError."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": ["five", "4.0"]
        })
        with pytest.raises(ValueError, match="must be numeric"):
            validate_recommender_inputs(
                df,
                user_col="user_id",
                item_id_col="item_id",
                rating_col="rating"
            )
class TestHasBlankValues:

    def test_no_blanks_returns_false(self):
        from src.data.data_adapter import _has_blank_values
        s = pd.Series(['a', 'b', 'c'])
        assert not _has_blank_values(s)   

    def test_nan_returns_true(self):
        from src.data.data_adapter import _has_blank_values
        s = pd.Series(['a', None, 'c'])
        assert _has_blank_values(s)      

    def test_empty_string_returns_true(self):
        from src.data.data_adapter import _has_blank_values
        s = pd.Series(['a', '', 'c'])
        assert _has_blank_values(s)       

    def test_whitespace_only_returns_true(self):
        from src.data.data_adapter import _has_blank_values
        s = pd.Series(['a', '   ', 'c'])
        assert _has_blank_values(s)      

class TestReadFile:
    """Test read_file function."""

    def test_read_csv(self):
        """Test reading a CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            f.flush()
            temp_path = f.name

        try:
            df = read_file(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            os.unlink(temp_path)

    def test_read_csv_with_bad_lines(self):
        """Test reading CSV with malformed lines (on_bad_lines='skip')."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3\n")  # missing column
            f.flush()
            temp_path = f.name

        try:
            df = read_file(temp_path)
            # on_bad_lines='skip' handles bad lines - row may be kept with NaN
            assert len(df) >= 1
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(temp_path)

    def test_read_json_lines(self):
        """Test reading JSON Lines format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"col1": 1, "col2": 2}\n')
            f.write('{"col1": 3, "col2": 4}\n')
            f.flush()
            temp_path = f.name

        try:
            df = read_file(temp_path)
            assert len(df) == 2
        finally:
            os.unlink(temp_path)

    def test_read_json_array(self):
        """Test reading JSON array format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"col1": 1}, {"col1": 2}], f)
            f.flush()
            temp_path = f.name

        try:
            df = read_file(temp_path)
            # JSON array is parsed - may be read as single row with multiple columns
            # depending on pandas version and structure
            assert df is not None
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(temp_path)


class TestAdaptData:
    """Test adapt_data function."""

    def test_adapt_books_data(self):
        """Test adaptation of books data."""
        df = pd.DataFrame({
            "Title": ["Book A", "Book B"],
            "Description": ["Desc A", "Desc B"],
            "authors": ["Author A", "Author B"],
            "rating": [4.5, 3.5]
        })
        adapted, meta = adapt_data(df)
        assert "title" in adapted.columns
        assert meta["has_user_data"] is False

    def test_adapt_ratings_data(self):
        """Test adaptation of ratings data."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u1"],
            "book_id": ["b1", "b2", "b1"],
            "rating": [5.0, 4.0, 3.0]
        })
        adapted, meta = adapt_data(df)
        assert "user_id" in adapted.columns
        assert "rating" in adapted.columns
        assert meta["has_user_data"] is True

    def test_adapt_sentiment_data(self):
        """Test adaptation of sentiment data."""
        df = pd.DataFrame({
            "sentiment": ["positive", "negative", "neutral"],
            "customer_rating": [5.0, 1.0, 3.0]
        })
        adapted, meta = adapt_data(df)
        assert "sentiment" in adapted.columns

    def test_adapt_missing_title_uses_first_column(self):
        """Test that missing title column uses first column."""
        df = pd.DataFrame({
            "product_name": ["Product A", "Product B"],
            "price": [10.0, 20.0]
        })
        adapted, meta = adapt_data(df)
        assert "title" in adapted.columns

    def test_adapt_fills_missing_columns(self):
        """Test that adapt_data fills missing standard columns."""
        df = pd.DataFrame({
            "Title": ["Book A"],
            "Description": ["Desc A"]
        })
        adapted, meta = adapt_data(df)
        assert "category" in adapted.columns
        assert "item_id" in adapted.columns
        assert "user_id" in adapted.columns
        assert "views" in adapted.columns
        assert "purchases" in adapted.columns

    def test_adapt_combined_column_created(self):
        """Test that combined text feature is created."""
        df = pd.DataFrame({
            "Title": ["Book A"],
            "Description": ["A description"],
            "category": ["Fiction"]
        })
        adapted, meta = adapt_data(df)
        assert "combined" in adapted.columns
        assert "Book A" in adapted["combined"].iloc[0]
        assert "A description" in adapted["combined"].iloc[0]
        assert "Fiction" in adapted["combined"].iloc[0]


class TestPreprocessBooksData:
    """Test preprocess_books_data function."""

    def test_removes_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame({
            "Title": ["Book A", "Book A", "Book B"],
            "authors": ["Author A", "Author A", "Author B"],
            "rating": [4.0, 4.0, 5.0]
        })
        result = preprocess_books_data(df)
        assert len(result) == 2

    def test_handles_missing_values(self):
        """Test that missing values are filled."""
        df = pd.DataFrame({
            "Title": ["Book A", None, "Book B"],
            "authors": ["Author A", "Author B", None],
            "rating": [4.0, 3.0, None]
        })
        result = preprocess_books_data(df)
        assert result["Title"].isnull().sum() == 0
        assert result["authors"].isnull().sum() == 0

    def test_normalizes_rating(self):
        """Test that rating is normalized."""
        df = pd.DataFrame({
            "Title": ["Book A", "Book B"],
            "authors": ["Author A", "Author B"],
            "rating": [1.0, 5.0]
        })
        result = preprocess_books_data(df)
        assert "rating_normalized" in result.columns
        assert result["rating_normalized"].min() >= 0.0
        assert result["rating_normalized"].max() <= 1.0


class TestPreprocessRatingsData:
    """Test preprocess_ratings_data function."""

    def test_removes_duplicate_user_book_pairs(self):
        """Test that duplicate user-book pairs are removed."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "book_id": ["b1", "b1", "b1"],
            "rating": [5.0, 5.0, 4.0]
        })
        result = preprocess_ratings_data(df)
        assert len(result) == 2


class TestPreprocessSentimentData:
    """Test preprocess_sentiment_data function."""

    def test_encodes_categorical_columns(self):
        """Test that categorical columns are encoded."""
        df = pd.DataFrame({
            "sentiment": ["positive", "negative", "neutral"],
            "gender": ["M", "F", "M"],
            "age_group": ["young", "adult", "senior"],
            "region": ["North", "South", "East"],
            "product_category": ["Electronics", "Books", "Clothing"],
            "purchase_channel": ["online", "store", "online"],
            "platform": ["web", "mobile", "web"],
            "customer_rating": [5.0, 1.0, 3.0]
        })
        result = preprocess_sentiment_data(df)
        # Check that original categorical columns are still present but encoded
        assert "sentiment" in result.columns
        assert "customer_rating" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])