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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])