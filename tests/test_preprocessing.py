"""
Unit tests for data_preprocessing module.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.data_preprocessing import (
    handle_missing_values,
    remove_duplicates,
    normalize_ratings,
    encode_categorical,
    preprocess,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame for testing preprocessing steps."""
    return pd.DataFrame({
        'user_id': ['u1', 'u2', 'u2', 'u3'],
        'book_id': ['b1', 'b2', 'b2', 'b3'],
        'rating': [1.0, 3.0, 3.0, 5.0],
        'title': ['Book A', 'Book B', 'Book B', None],
        'authors': ['Author X', 'Author Y', 'Author Y', 'Author Z'],
    })


# ─── handle_missing_values ───────────────────────────────────────────────────

class TestHandleMissingValues:
    def test_fills_text_columns(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result['title'].isnull().sum() == 0

    def test_no_fully_empty_rows(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result.dropna(how='all').shape == result.shape


# ─── remove_duplicates ───────────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_removes_duplicate_user_item_pairs(self, sample_df):
        result = remove_duplicates(sample_df)
        assert result.duplicated(subset=['user_id', 'book_id']).sum() == 0

    def test_keeps_unique_rows(self, sample_df):
        result = remove_duplicates(sample_df)
        assert len(result) == 3


# ─── normalize_ratings ───────────────────────────────────────────────────────

class TestNormalizeRatings:
    def test_adds_normalized_column(self, sample_df):
        result = normalize_ratings(sample_df)
        assert 'rating_normalized' in result.columns

    def test_normalized_range_is_0_to_1(self, sample_df):
        result = normalize_ratings(sample_df)
        assert result['rating_normalized'].min() >= 0.0
        assert result['rating_normalized'].max() <= 1.0


# ─── encode_categorical ──────────────────────────────────────────────────────

class TestEncodeCategorical:
    def test_adds_encoded_columns(self, sample_df):
        result = encode_categorical(sample_df)
        assert 'authors_encoded' in result.columns

    def test_encoded_column_is_numeric(self, sample_df):
        result = encode_categorical(sample_df)
        assert result['authors_encoded'].dtype in ['int32', 'int64']


# ─── preprocess ──────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_returns_dataframe(self, sample_df):
        result = preprocess(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_values(self, sample_df):
        result = preprocess(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_no_duplicates(self, sample_df):
        result = preprocess(sample_df)
        assert result.duplicated().sum() == 0

    def test_rating_normalized_exists(self, sample_df):
        result = preprocess(sample_df)
        assert 'rating_normalized' in result.columns


# ─── Edge Cases ────────────────────────────────────────────────────────────────

class TestHandleMissingValuesEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        result = handle_missing_values(df)
        assert len(result) == 0

    def test_all_null_columns(self):
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [None, None, None]
        })
        result = handle_missing_values(df)
        assert len(result) == 3

    def test_mixed_type_columns(self):
        df = pd.DataFrame({
            'text_col': ['a', None, 'c'],
            'num_col': [1.0, None, 3.0]
        })
        result = handle_missing_values(df)
        assert result['text_col'].isnull().sum() == 0
        assert result['num_col'].isnull().sum() == 0

    def test_numeric_uses_median(self):
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0, None, 5.0]
        })
        result = handle_missing_values(df)
        assert result['num_col'].notna().all()
        assert result['num_col'].dtype in ['float64', 'int64']


class TestRemoveDuplicatesEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['a', 'b'])
        result = remove_duplicates(df)
        assert len(result) == 0

    def test_no_duplicates(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestNormalizeRatingsEdgeCases:
    def test_no_rating_column(self):
        df = pd.DataFrame({'title': ['a', 'b', 'c']})
        result = normalize_ratings(df)
        assert 'rating_normalized' not in result.columns

    def test_all_same_values(self):
        df = pd.DataFrame({'rating': [3.0, 3.0, 3.0]})
        result = normalize_ratings(df)
        assert result['rating_normalized'].min() >= 0.0
        assert result['rating_normalized'].max() <= 1.0


class TestEncodeCategoricalEdgeCases:
    def test_empty_list_of_columns(self):
        df = pd.DataFrame({'authors': ['a', 'b', 'c']})
        result = encode_categorical(df, columns=[])
        assert 'authors_encoded' not in result.columns

    def test_missing_columns_in_dataframe(self):
        df = pd.DataFrame({'title': ['a', 'b', 'c']})
        result = encode_categorical(df, columns=['nonexistent'])
        assert len(result) == 3

    def test_normalize_ratings_single_row(self):
        df = pd.DataFrame({'rating': [4.0]})
        result = normalize_ratings(df)
        assert len(result) == 1
        assert 'rating_normalized' in result.columns

    def test_encode_categorical_nan_values(self):
        df = pd.DataFrame({'authors': ['a', None, 'b']})
        result = encode_categorical(df, columns=None)
        assert 'authors_encoded' in result.columns

    def test_preprocess_all_nan(self):
        df = pd.DataFrame({
            'user_id': [None, None],
            'book_id': [None, None],
            'rating': pd.Series([None, None], dtype='float64')
        })
        result = preprocess(df)
        assert len(result) > 0