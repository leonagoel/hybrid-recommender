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

    def test_very_long_text(self):
        """Test sentiment analysis on very long text."""
        long_text = "This is great! " * 100
        score = analyze_sentiment(long_text)
        assert isinstance(score, float)


class TestSentimentLabel:
    """Test sentiment_label function."""

    def test_positive_threshold(self):
        """Test positive threshold at boundary (0.05)."""
        assert sentiment_label(0.06) == "positive"
        assert sentiment_label(0.5) == "positive"

    def test_negative_threshold(self):
        """Test negative threshold at boundary (-0.05)."""
        assert sentiment_label(-0.06) == "negative"
        assert sentiment_label(-0.5) == "negative"

    def test_boundary_positive_is_positive(self):
        """Test that 0.05 threshold is considered positive (>= 0.05 is positive)."""
        assert sentiment_label(0.05) == "positive"
        assert sentiment_label(0.051) == "positive"

    def test_boundary_negative_is_negative(self):
        """Test that -0.05 threshold is considered negative (<= -0.05 is negative)."""
        assert sentiment_label(-0.05) == "negative"
        assert sentiment_label(-0.051) == "negative"

    def test_just_below_positive_threshold(self):
        """Test that just below 0.05 is neutral."""
        assert sentiment_label(0.049) == "neutral"

    def test_neutral_middle(self):
        """Test neutral in the middle range."""
        assert sentiment_label(0.0) == "neutral"


class TestBatchAnalyze:
    """Test batch_analyze function."""

    def test_with_text_column(self):
        """Test batch analyze with valid text column."""
        df = pd.DataFrame({
            "review_text": [
                "This is amazing!",
                "This is terrible!",
                "This is okay."
            ]
        })
        result = batch_analyze(df, text_col="review_text")
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert len(result) == 3

    def test_missing_text_column(self):
        """Test batch analyze when text column does not exist."""
        df = pd.DataFrame({
            "other_col": ["a", "b", "c"]
        })
        result = batch_analyze(df, text_col="review_text")
        assert result["sentiment_score"].iloc[0] == 0.0
        assert result["sentiment_label"].iloc[0] == "neutral"

    def test_empty_dataframe(self):
        """Test batch analyze with empty DataFrame."""
        df = pd.DataFrame(columns=["review_text"])
        result = batch_analyze(df, text_col="review_text")
        assert len(result) == 0

    def test_preserves_other_columns(self):
        """Test that batch_analyze preserves other DataFrame columns."""
        df = pd.DataFrame({
            "review_text": ["Great!", "Terrible!"],
            "product_id": ["P1", "P2"]
        })
        result = batch_analyze(df, text_col="review_text")
        assert "product_id" in result.columns
        assert result["product_id"].tolist() == ["P1", "P2"]


class TestAggregateSentimentByItem:
    """Test aggregate_sentiment_by_item function."""

    def test_aggregation_with_multiple_reviews(self):
        """Test aggregation with multiple reviews per item."""
        df = pd.DataFrame({
            "title": ["Item A", "Item A", "Item B"],
            "review_text": [
                "This is great!",
                "I love it!",
                "Not bad."
            ]
        })
        df = batch_analyze(df, text_col="review_text")
        result = aggregate_sentiment_by_item(df, item_col="title")
        assert "avg_sentiment" in result.columns
        assert "review_count" in result.columns
        assert len(result) == 2

    def test_single_review_per_item(self):
        """Test aggregation with single review per item."""
        df = pd.DataFrame({
            "title": ["Item A", "Item B"],
            "review_text": [
                "This is great!",
                "This is terrible!"
            ]
        })
        df = batch_analyze(df, text_col="review_text")
        result = aggregate_sentiment_by_item(df, item_col="title")
        assert len(result) == 2
        assert result["review_count"].sum() == 2

    def test_empty_groups(self):
        """Test aggregation when some items have no reviews."""
        df = pd.DataFrame({
            "title": ["Item A", "Item B", "Item C"],
            "review_text": ["Great!", "Terrible!", ""]
        })
        df = batch_analyze(df, text_col="review_text")
        result = aggregate_sentiment_by_item(df, item_col="title")
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])