import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.nlp_engine import (
    analyze_sentiment,
    sentiment_label,
    batch_analyze,
    aggregate_sentiment_by_item,
    compute_product_sentiment
)


class TestAnalyzeSentiment:
    def test_positive_text(self):
        score = analyze_sentiment("I love this product! It's amazing!")
        assert score > 0.05

    def test_negative_text(self):
        score = analyze_sentiment("This is terrible. I hate it. Worst ever!")
        assert score < -0.05

    def test_empty_text(self):
        score = analyze_sentiment("")
        assert score == 0.0

    def test_none_text(self):
        score = analyze_sentiment(None)
        assert score == 0.0

    def test_whitespace_text(self):
        score = analyze_sentiment("   ")
        assert score == 0.0


class TestSentimentLabel:
    def test_positive_label(self):
        assert sentiment_label(0.5) == 'positive'
        assert sentiment_label(0.05) == 'positive'

    def test_negative_label(self):
        assert sentiment_label(-0.5) == 'negative'
        assert sentiment_label(-0.05) == 'negative'

    def test_neutral_label(self):
        assert sentiment_label(0.0) == 'neutral'


class TestBatchAnalyze:
    def test_batch_analyze_basic(self):
        df = pd.DataFrame({'review_text': ['Great!', 'Terrible!', 'Okay']})
        result = batch_analyze(df)
        assert 'sentiment_score' in result.columns
        assert 'sentiment_label' in result.columns
        assert len(result) == 3

    def test_batch_analyze_missing_column(self):
        df = pd.DataFrame({'other_col': ['data']})
        result = batch_analyze(df)
        assert 'sentiment_score' in result.columns
        assert result['sentiment_score'].iloc[0] == 0.0


class TestAggregateSentimentByItem:
    def test_aggregate_basic(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item A', 'Item B'],
            'review_text': ['Great', 'Good', 'Bad']
        })
        result = aggregate_sentiment_by_item(df)
        assert 'avg_sentiment' in result.columns
        assert 'review_count' in result.columns
        assert len(result) == 2


class TestComputeProductSentiment:
    def test_empty_reviews(self):
        result = compute_product_sentiment([])
        assert result is None

    def test_none_reviews(self):
        result = compute_product_sentiment(None)
        assert result is None

    def test_invalid_reviews(self):
        result = compute_product_sentiment(["", " ", None])
        assert result is None

    def test_valid_reviews(self):
        reviews = ["I love this product", "It's amazing", "Best purchase ever"]
        result = compute_product_sentiment(reviews)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0.0
