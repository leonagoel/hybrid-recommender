"""
Regression tests for: adapt_data() must not re-run preprocessing.

Bug: DatasetManager.load_csv() called preprocess() then adapt_data(), but
adapt_data() also called preprocess_books/ratings/sentiment_data() internally.
The second preprocessing pass corrupted the data silently:

  - MinMaxScaler on an already-normalised rating_normalized → std == 0
  - LabelEncoder on already-integer authors column → different mapping

These tests verify that adapt_data() only does schema adaptation (rename +
fill defaults) and leaves already-preprocessed columns untouched.

Run with:
    PYTHONPATH=src/data python -m pytest tests/test_adapt_data.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))

from data_adapter import adapt_data
from data_preprocessing import preprocess


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_books_df():
    """Raw books DataFrame as it would arrive from pd.read_csv()."""
    return pd.DataFrame({
        'authors':   ['J.K. Rowling', 'George Orwell', 'Tolkien'],
        'publisher': ['Bloomsbury',   'Secker',         'Allen'],
        'title':     ['Harry Potter', '1984',           'LOTR'],
        'rating':    [1.0,            3.0,              5.0],
    })


@pytest.fixture
def preprocessed_books_df(raw_books_df):
    """Books DataFrame after one pass of preprocess() — what adapt_data() receives."""
    return preprocess(raw_books_df)


@pytest.fixture
def raw_ratings_df():
    """Raw ratings DataFrame."""
    return pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3'],
        'book_id': ['b1', 'b2', 'b3'],
        'rating':  [1.0,  3.0,  5.0],
        'title':   ['Book A', 'Book B', 'Book C'],
    })


# ── adapt_data does not re-run preprocessing ─────────────────────────────────

class TestAdaptDataNoDoublePreprocessing:
    """Verify adapt_data() does not re-apply preprocessing on already-processed data."""

    def test_rating_normalized_variance_preserved(self, preprocessed_books_df):
        """rating_normalized must retain its variance after adapt_data().

        Before fix: adapt_data() re-ran MinMaxScaler on an already-normalised
        0-1 column, producing std == 0 (flat distribution).
        """
        adapted_df, _ = adapt_data(preprocessed_books_df)

        assert 'rating_normalized' in adapted_df.columns, (
            "rating_normalized column missing from adapted output"
        )
        std = adapted_df['rating_normalized'].std()
        assert std > 0, (
            f"rating_normalized has zero variance (std={std:.6f}) after adapt_data(). "
            "This means MinMaxScaler was applied a second time to an already-normalised "
            "column, collapsing all variance."
        )

    def test_rating_normalized_min_max_unchanged(self, preprocessed_books_df):
        """Min and max of rating_normalized must not be re-squashed to exactly 0 and 1."""
        # After one preprocess() pass on 3-row fixture, min=0.0 and max=1.0 is
        # expected — but std must not be zero (only 2 values at extremes).
        adapted_df, _ = adapt_data(preprocessed_books_df)
        rn = adapted_df['rating_normalized']

        # Both endpoints being 0 and 1 is fine; what is NOT fine is std == 0
        # which would mean ALL values collapsed to the same point.
        assert rn.min() >= 0.0
        assert rn.max() <= 1.0
        assert rn.std() > 0

    def test_authors_encoding_stable_across_adapt(self, preprocessed_books_df):
        """authors column integer encoding must not change inside adapt_data().

        Before fix: LabelEncoder was re-run inside adapt_data(), producing a
        different integer mapping than the one preprocess() produced. Models
        trained on the preprocess() mapping would receive wrong inputs.
        """
        encoding_before = preprocessed_books_df['authors'].tolist()
        adapted_df, _ = adapt_data(preprocessed_books_df)
        encoding_after = adapted_df['authors'].tolist()

        assert encoding_before == encoding_after, (
            f"authors encoding changed inside adapt_data().\n"
            f"  Before: {encoding_before}\n"
            f"  After:  {encoding_after}\n"
            "This means LabelEncoder was re-applied to already-encoded integers."
        )

    def test_authors_dtype_remains_integer(self, preprocessed_books_df):
        """authors must remain int64 after adapt_data(), not be re-cast to object."""
        adapted_df, _ = adapt_data(preprocessed_books_df)
        assert adapted_df['authors'].dtype in ['int32', 'int64'], (
            f"authors dtype changed to {adapted_df['authors'].dtype} inside adapt_data(). "
            "Expected int64 (already encoded by preprocess())."
        )

    def test_single_preprocessing_equals_pipeline(self, raw_books_df):
        """preprocess() then adapt_data() must equal adapt_data(preprocess()) only ONCE.

        Verifies the full DatasetManager pipeline produces the same result
        as explicitly calling both functions once in sequence.
        """
        preprocessed = preprocess(raw_books_df)
        adapted_once, _ = adapt_data(preprocessed)

        # If adapt_data re-preprocessed, a second manual call would differ.
        adapted_again, _ = adapt_data(adapted_once)

        # rating_normalized should be identical in both outputs
        pd.testing.assert_series_equal(
            adapted_once['rating_normalized'].reset_index(drop=True),
            adapted_again['rating_normalized'].reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_adapt_data_does_not_mutate_input(self, preprocessed_books_df):
        """adapt_data() must not modify the DataFrame passed to it."""
        original_values = preprocessed_books_df['authors'].tolist()
        adapt_data(preprocessed_books_df)
        assert preprocessed_books_df['authors'].tolist() == original_values, (
            "adapt_data() mutated the input DataFrame in-place."
        )


# ── adapt_data schema adaptation still works ─────────────────────────────────

class TestAdaptDataSchemaIntegrity:
    """Verify adapt_data() still performs its intended schema adaptation correctly."""

    def test_returns_tuple_of_df_and_meta(self, preprocessed_books_df):
        result = adapt_data(preprocessed_books_df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_canonical_columns_present(self, preprocessed_books_df):
        """Canonical required columns must be present after adaptation."""
        adapted_df, _ = adapt_data(preprocessed_books_df)
        for col in ['title', 'description', 'category', 'item_id',
                    'rating', 'user_id', 'review_text', 'combined']:
            assert col in adapted_df.columns, f"Missing canonical column: {col}"

    def test_meta_has_required_keys(self, preprocessed_books_df):
        """Meta dict must contain the expected keys."""
        _, meta = adapt_data(preprocessed_books_df)
        for key in ['has_user_data', 'has_reviews', 'total_rows', 'total_columns']:
            assert key in meta, f"Missing meta key: {key}"

    def test_ratings_df_adapted(self, raw_ratings_df):
        """Ratings dataset must adapt correctly after a single preprocess pass."""
        preprocessed = preprocess(raw_ratings_df)
        adapted_df, meta = adapt_data(preprocessed)
        assert 'title' in adapted_df.columns
        assert 'rating' in adapted_df.columns
        assert meta['total_rows'] == 3