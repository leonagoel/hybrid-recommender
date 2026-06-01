"""
Unit tests for the NLP sentiment engine module.
Tests NLTK VADER sentiment analysis functions.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.nlp_engine import (
    analyze_sentiment,
    sentiment_label,
    batch_analyze,
    aggregate_sentiment_by_item,
)


class TestAnalyzeSentiment:
    """Test analyze_sentiment function."""

    def test_positive_text(self):
        """Test that positive text returns positive score."""
        score = analyze_sentiment("This is amazing! I love it!")
        assert score > 0.05

    def test_negative_text(self):
        """Test that negative text returns negative score."""
        score = analyze_sentiment("This is terrible! I hate it!")
        assert score < -0.05

    def test_neutral_text(self):
        """Test that neutral text returns score near zero."""
        score = analyze_sentiment("The product is a thing.")
        assert -0.05 <= score <= 0.05

    def test_empty_string(self):
        """Test that empty string returns 0.0."""
        score = analyze_sentiment("")
        assert score == 0.0

    def test_none_input(self):
        """Test that None input returns 0.0."""
        score = analyze_sentiment(None)
        assert score == 0.0

    def test_whitespace_only(self):
        """Test that whitespace-only text returns 0.0."""
        score = analyze_sentiment("   \n\t  ")
        assert score == 0.0

    def test_non_string_int_input(self):
        """Test that integer input returns 0.0."""
        score = analyze_sentiment(42)
        assert score == 0.0

    def test_non_string_list_input(self):
        """Test that list input returns 0.0."""
        score = analyze_sentiment(["text", "more"])
        assert score == 0.0

    def test_non_string_dict_input(self):
        """Test that dict input returns 0.0."""
        score = analyze_sentiment({"key": "value"})
        assert score == 0.0

    def test_very_long_text(self):
        """Test sentiment analysis on very long text."""
        long_text = "This is great! " * 100
        score = analyze_sentiment(long_text)
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        """Compound score must always be in [-1.0, 1.0]."""
        for text in ["I love this!", "I hate this!", "It is okay.", ""]:
            score = analyze_sentiment(text)
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestSentimentLabel
# ---------------------------------------------------------------------------

class TestSentimentLabel:
    """Test sentiment_label boundary conditions."""

    def test_positive_label_above_threshold(self):
        """Score >= 0.05 should return 'positive'."""
        assert sentiment_label(0.05) == "positive"
        assert sentiment_label(0.5) == "positive"
        assert sentiment_label(1.0) == "positive"

    def test_negative_label_below_threshold(self):
        """Score <= -0.05 should return 'negative'."""
        assert sentiment_label(-0.05) == "negative"
        assert sentiment_label(-0.5) == "negative"
        assert sentiment_label(-1.0) == "negative"

    def test_neutral_label_between_thresholds(self):
        """Score strictly between -0.05 and 0.05 should return 'neutral'."""
        assert sentiment_label(0.0) == "neutral"
        assert sentiment_label(0.04) == "neutral"
        assert sentiment_label(-0.04) == "neutral"

    def test_label_is_string(self):
        """Return type must always be str."""
        for score in [-1.0, -0.1, 0.0, 0.1, 1.0]:
            assert isinstance(sentiment_label(score), str)

    def test_label_values_are_known(self):
        """Labels must be one of the three known values."""
        valid = {"positive", "negative", "neutral"}
        for score in [-1.0, -0.05, 0.0, 0.05, 1.0]:
            assert sentiment_label(score) in valid


# ---------------------------------------------------------------------------
# TestBatchAnalyze
# ---------------------------------------------------------------------------

class TestBatchAnalyze:
    """Test batch_analyze function."""

    @pytest.fixture()
    def review_df(self):
        """Small DataFrame with a review_text column."""
        return pd.DataFrame(
            {
                "title": ["Item A", "Item B", "Item C"],
                "review_text": [
                    "This is absolutely fantastic!",
                    "Terrible experience, very disappointed.",
                    "It works fine.",
                ],
            }
        )

    def test_adds_sentiment_score_column(self, review_df):
        """batch_analyze must add a sentiment_score column."""
        result = batch_analyze(review_df)
        assert "sentiment_score" in result.columns

    def test_adds_sentiment_label_column(self, review_df):
        """batch_analyze must add a sentiment_label column."""
        result = batch_analyze(review_df)
        assert "sentiment_label" in result.columns

    def test_does_not_mutate_original(self, review_df):
        """Input DataFrame must not be modified in-place."""
        original_cols = list(review_df.columns)
        batch_analyze(review_df)
        assert list(review_df.columns) == original_cols

    def test_score_range(self, review_df):
        """Every sentiment_score must be in [-1.0, 1.0]."""
        result = batch_analyze(review_df)
        assert result["sentiment_score"].between(-1.0, 1.0).all()

    def test_label_values_are_valid(self, review_df):
        """Every sentiment_label must be one of the three known values."""
        result = batch_analyze(review_df)
        valid = {"positive", "negative", "neutral"}
        assert set(result["sentiment_label"].unique()).issubset(valid)

    def test_row_count_unchanged(self, review_df):
        """Output must have the same number of rows as input."""
        result = batch_analyze(review_df)
        assert len(result) == len(review_df)

    def test_missing_text_column_fallback(self):
        """When the specified text column is absent, scores default to 0.0 / neutral."""
        df = pd.DataFrame({"title": ["A", "B"], "description": ["good", "bad"]})
        result = batch_analyze(df, text_col="review_text")
        assert "sentiment_score" in result.columns
        assert (result["sentiment_score"] == 0.0).all()
        assert (result["sentiment_label"] == "neutral").all()

    def test_custom_text_column(self):
        """batch_analyze should respect a custom text_col argument."""
        df = pd.DataFrame({"comment": ["I love it!", "I hate it!"]})
        result = batch_analyze(df, text_col="comment")
        assert "sentiment_score" in result.columns
        assert len(result) == 2

    def test_empty_reviews_return_zero(self):
        """Empty / blank review strings should produce score 0.0."""
        df = pd.DataFrame({"review_text": ["", "   "]})
        result = batch_analyze(df)
        assert (result["sentiment_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# TestAggregateSentimentByItem
# ---------------------------------------------------------------------------

class TestAggregateSentimentByItem:
    """Test aggregate_sentiment_by_item function."""

    @pytest.fixture()
    def multi_review_df(self):
        """DataFrame with multiple reviews per item."""
        return pd.DataFrame(
            {
                "title": ["Book A", "Book A", "Book B", "Book B", "Book B"],
                "review_text": [
                    "Amazing read, loved it!",
                    "Quite good overall.",
                    "Terrible, waste of money.",
                    "Not great, not terrible.",
                    "Absolutely awful.",
                ],
            }
        )

    def test_returns_dataframe(self, multi_review_df):
        """aggregate_sentiment_by_item must return a DataFrame."""
        result = aggregate_sentiment_by_item(multi_review_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_avg_sentiment_column(self, multi_review_df):
        """Output must contain avg_sentiment column."""
        result = aggregate_sentiment_by_item(multi_review_df)
        assert "avg_sentiment" in result.columns

    def test_has_review_count_column(self, multi_review_df):
        """Output must contain review_count column."""
        result = aggregate_sentiment_by_item(multi_review_df)
        assert "review_count" in result.columns

    def test_one_row_per_item(self, multi_review_df):
        """Output must have exactly one row per unique item."""
        result = aggregate_sentiment_by_item(multi_review_df)
        assert len(result) == multi_review_df["title"].nunique()

    def test_review_counts_are_correct(self, multi_review_df):
        """review_count must match the number of reviews per item."""
        result = aggregate_sentiment_by_item(multi_review_df)
        result = result.set_index("title")
        assert result.loc["Book A", "review_count"] == 2
        assert result.loc["Book B", "review_count"] == 3

    def test_avg_sentiment_in_valid_range(self, multi_review_df):
        """avg_sentiment must be within [-1.0, 1.0]."""
        result = aggregate_sentiment_by_item(multi_review_df)
        assert result["avg_sentiment"].between(-1.0, 1.0).all()

    def test_works_when_sentiment_score_already_present(self):
        """If sentiment_score column already exists it must not re-run batch_analyze."""
        df = pd.DataFrame(
            {
                "title": ["X", "X", "Y"],
                "sentiment_score": [0.8, 0.6, -0.5],
            }
        )
        result = aggregate_sentiment_by_item(df)
        assert len(result) == 2
        x_row = result[result["title"] == "X"].iloc[0]
        assert abs(x_row["avg_sentiment"] - 0.7) < 0.01

    def test_custom_item_column(self):
        """Should work with a non-default item column name."""
        df = pd.DataFrame(
            {
                "product": ["P1", "P1", "P2"],
                "review_text": ["great!", "good.", "bad."],
            }
        )
        result = aggregate_sentiment_by_item(df, item_col="product")
        assert "product" in result.columns
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])