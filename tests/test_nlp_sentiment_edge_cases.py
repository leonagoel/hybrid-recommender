from src.model.nlp_engine import analyze_sentiment, batch_analyze, sentiment_label


class TestAnalyzeSentimentEdgeCases:

    def test_empty_string(self):
        result = analyze_sentiment("")
        assert result == 0.0

    def test_whitespace_only(self):
        result = analyze_sentiment("   \t\n  ")
        assert result == 0.0

    def test_none_input(self):
        result = analyze_sentiment(None)
        assert result == 0.0

    def test_non_string_int_returns_zero(self):
        result = analyze_sentiment(42)
        assert result == 0.0

    def test_non_string_float_returns_zero(self):
        result = analyze_sentiment(3.14)
        assert result == 0.0

    def test_non_string_list_returns_zero(self):
        result = analyze_sentiment(["hello", "world"])
        assert result == 0.0

    def test_non_string_dict_returns_zero(self):
        result = analyze_sentiment({"key": "value"})
        assert result == 0.0

    def test_non_string_boolean_returns_zero(self):
        result = analyze_sentiment(True)
        assert result == 0.0

    def test_very_long_text(self):
        long_text = "a" * 20000
        result = analyze_sentiment(long_text)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_unicode_text(self):
        result = analyze_sentiment("This is a great day!")
        assert isinstance(result, float)

    def test_negation_in_text(self):
        result = analyze_sentiment("This is not good.")
        assert isinstance(result, float)
        assert result < 0.05


class TestSentimentLabel:

    def test_positive_threshold(self):
        assert sentiment_label(0.05) == "positive"
        assert sentiment_label(0.5) == "positive"
        assert sentiment_label(1.0) == "positive"

    def test_negative_threshold(self):
        assert sentiment_label(-0.05) == "negative"
        assert sentiment_label(-0.5) == "negative"
        assert sentiment_label(-1.0) == "negative"

    def test_neutral_boundary(self):
        assert sentiment_label(0.04) == "neutral"
        assert sentiment_label(-0.04) == "neutral"
        assert sentiment_label(0.0) == "neutral"


class TestBatchAnalyze:

    def test_batch_with_missing_column(self):
        import pandas as pd

        df = pd.DataFrame({"title": ["Book A", "Book B"]})
        result = batch_analyze(df, text_col="nonexistent_column")
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert result["sentiment_score"].iloc[0] == 0.0
        assert result["sentiment_label"].iloc[0] == "neutral"

    def test_batch_with_valid_reviews(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "review_text": [
                    "This is absolutely fantastic!",
                    "This is terrible and I hate it.",
                    "Okay, nothing special.",
                    "",
                ]
            }
        )
        result = batch_analyze(df, text_col="review_text")
        assert len(result) == 4
        assert result["sentiment_label"].iloc[0] == "positive"
        assert result["sentiment_label"].iloc[1] == "negative"
