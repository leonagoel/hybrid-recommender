"""
NLP Sentiment Engine
Uses NLTK VADER for lightweight sentiment analysis on user review text.
"""
import nltk
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# Download VADER lexicon (only on first run)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try: 
    nltk.data.find('corpora/stopwords') 
except LookupError: 
    nltk.download('stopwords', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of a single text string.
    Returns the VADER compound score in range [-1.0, 1.0].
      > 0.05  → positive
      < -0.05 → negative
      else    → neutral
    """
    if not text or not isinstance(text, str) or text.strip() == '':
        return 0.0
    scores = _analyzer.polarity_scores(text)
    return scores['compound']


def sentiment_label(score: float) -> str:
    """Convert a compound score to a human-readable label."""
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def batch_analyze(df: pd.DataFrame, text_col: str = 'review_text') -> pd.DataFrame:
    """
    Add sentiment_score and sentiment_label columns to the DataFrame.
    Operates on the specified text column.
    """
    df = df.copy()
    if text_col not in df.columns:
        df['sentiment_score'] = 0.0
        df['sentiment_label'] = 'neutral'
        return df

    df['sentiment_score'] = df[text_col].apply(analyze_sentiment)
    df['sentiment_label'] = df['sentiment_score'].apply(sentiment_label)
    return df


def aggregate_sentiment_by_item(df: pd.DataFrame, item_col: str = 'title') -> pd.DataFrame:
    """
    Compute average sentiment score per unique item.
    Returns a DataFrame with columns: [item_col, 'avg_sentiment', 'review_count'].
    """
    if 'sentiment_score' not in df.columns:
        df = batch_analyze(df)

    agg = df.groupby(item_col).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        review_count=('sentiment_score', 'count')
    ).reset_index()

    return agg
# Common English stopwords
_custom_stopwords = {
    'top',
    'toprated',
    'top-rated',
    'users',
    'user',
    'value',
    'quality',
    'modern',
    'designed',
    'item',
    'items',
    'convenience',
    'premium',
    'pro',
    'plus',
    'lite'
}

_stop_words = list(set(stopwords.words('english')).union(_custom_stopwords))

def clean_text(text: str) -> str:
    """
    Clean product description text for keyword extraction.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_keywords(text: str, top_n: int = 5) -> list:
    """
    Extract top TF-IDF keywords from product descriptions.

    Returns:
        List[str] → top keywords/tags
    """

    if not text or not isinstance(text, str):
        return []

    cleaned = clean_text(text)

    if not cleaned:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words=_stop_words,
            max_features=top_n,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform([cleaned])

        keywords = vectorizer.get_feature_names_out()

        # Convert spaces to hyphens for cleaner tags
        return [k.replace(' ', '-') for k in keywords]

    except Exception:
        return []
    

