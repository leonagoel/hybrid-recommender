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


class TestSentimentLabel:
    """Test sentiment_label function."""

    def test_positive_score(self):
        """Test that score >= 0.05 returns 'positive'."""
        assert sentiment_label(0.05) == "positive"
        assert sentiment_label(0.5) == "positive"
        assert sentiment_label(1.0) == "positive"

    def test_negative_score(self):
        """Test that score <= -0.05 returns 'negative'."""
        assert sentiment_label(-0.05) == "negative"
        assert sentiment_label(-0.5) == "negative"
        assert sentiment_label(-1.0) == "negative"

    def test_neutral_score(self):
        """Test that score between -0.05 and 0.05 returns 'neutral'."""
        assert sentiment_label(0.04) == "neutral"
        assert sentiment_label(0.0) == "neutral"
        assert sentiment_label(-0.04) == "neutral"

    def test_boundary_positive(self):
        """Test boundary case at 0.05."""
        assert sentiment_label(0.05) == "positive"

    def test_boundary_negative(self):
        """Test boundary case at -0.05."""
        assert sentiment_label(-0.05) == "negative"


class TestBatchAnalyze:
    """Test batch_analyze function."""

    def test_batch_analyze_adds_columns(self):
        """Test that batch_analyze adds sentiment_score and sentiment_label."""
        df = pd.DataFrame({
            "review_text": ["I love this!", "This is terrible.", "It is okay."]
        })
        result = batch_analyze(df)
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns

    def test_batch_analyze_positive_label(self):
        """Test that positive text gets 'positive' label."""
        df = pd.DataFrame({"review_text": ["I love this product!"]})
        result = batch_analyze(df)
        assert result["sentiment_label"].iloc[0] == "positive"

    def test_batch_analyze_negative_label(self):
        """Test that negative text gets 'negative' label."""
        df = pd.DataFrame({"review_text": ["This is horrible!"]})
        result = batch_analyze(df)
        assert result["sentiment_label"].iloc[0] == "negative"

    def test_batch_analyze_missing_column(self):
        """Test that missing text column returns neutral defaults."""
        df = pd.DataFrame({"other_col": ["a", "b"]})
        result = batch_analyze(df, text_col="missing_col")
        assert result["sentiment_score"].iloc[0] == 0.0
        assert result["sentiment_label"].iloc[0] == "neutral"

    def test_batch_analyze_preserves_original(self):
        """Test that batch_analyze does not modify original DataFrame."""
        df = pd.DataFrame({"review_text": ["Great!", "Terrible."]})
        original_cols = list(df.columns)
        _ = batch_analyze(df)
        assert list(df.columns) == original_cols

    def test_batch_analyze_custom_column_name(self):
        """Test batch_analyze with custom text column name."""
        df = pd.DataFrame({"custom_text": ["I am happy!", "I am sad!"]})
        result = batch_analyze(df, text_col="custom_text")
        assert "sentiment_score" in result.columns


class TestAggregateSentimentByItem:
    """Test aggregate_sentiment_by_item function."""

    def test_aggregate_creates_summary(self):
        """Test that aggregate creates summary DataFrame."""
        df = pd.DataFrame({
            "title": ["Item A", "Item A", "Item B"],
            "review_text": ["Great!", "Good!", "Terrible!"]
        })
        df = batch_analyze(df)
        result = aggregate_sentiment_by_item(df)
        assert "title" in result.columns
        assert "avg_sentiment" in result.columns
        assert "review_count" in result.columns

    def test_aggregate_correct_counts(self):
        """Test that review counts are correct."""
        df = pd.DataFrame({
            "title": ["Item A", "Item A", "Item B"],
            "review_text": ["Great!", "Good!", "Terrible!"]
        })
        df = batch_analyze(df)
        result = aggregate_sentiment_by_item(df)
        item_a_count = result[result["title"] == "Item A"]["review_count"].iloc[0]
        assert item_a_count == 2

    def test_aggregate_auto_analyzes(self):
        """Test that aggregate auto-runs batch_analyze if not done."""
        df = pd.DataFrame({
            "title": ["Item A"],
            "review_text": ["Great!"]
        })
        result = aggregate_sentiment_by_item(df)
        assert "sentiment_score" not in df.columns
        assert "avg_sentiment" in result.columns

    def test_aggregate_respects_item_column(self):
        """Test that aggregate uses specified item column."""
        df = pd.DataFrame({
            "product_name": ["Item X", "Item X", "Item Y"],
            "review_text": ["Great!", "Good!", "Terrible!"]
        })
        df = batch_analyze(df)
        result = aggregate_sentiment_by_item(df, item_col="product_name")
        assert "product_name" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])