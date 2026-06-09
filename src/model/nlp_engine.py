"""
NLP Sentiment Engine
Uses NLTK VADER for lightweight sentiment analysis on user review text.
"""
import nltk
import numpy as np
import pandas as pd
from typing import List, Optional

# Download VADER lexicon (only on first run)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> float:
    """Analyze the sentiment of a single text string using VADER polarity scores.

    Args:
        text (str): The raw text string or user review to analyze.

    Returns:
        float: The calculated VADER compound sentiment score bounded between 
            -1.0 (highly negative) and 1.0 (highly positive). Returns 0.0 
            if the string is empty or invalid.
    """
    def analyze_sentiment(text: str) -> float:
        if text is None:
            return 0.0
        

    if not isinstance(text, str):
        return 0.0

    text = text.strip()
    if not text:
        return 0.0

    scores = _analyzer.polarity_scores(text)
    return scores['compound']


def sentiment_label(score: float) -> str:
    """Convert a numerical VADER compound sentiment score to a string category.

    Args:
        score (float): The compound sentiment score to classify.

    Returns:
        str: Human-readable category label mapping to 'positive' (score >= 0.05),
            'negative' (score <= -0.05), or 'neutral' (all other values).
    """
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def batch_analyze(df: pd.DataFrame, text_col: str = 'review_text') -> pd.DataFrame:
    """Process an entire text column in a DataFrame to attach sentiment metrics.

    Applies sentence-level extraction rules sequentially across rows, tracking 
    both raw scores and categorical string descriptors.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text reviews.
        text_col (str, optional): Target column header containing string entries 
            to analyze. Defaults to 'review_text'.

    Returns:
        pd.DataFrame: A shallow copy of the modified DataFrame containing two 
            new fields: 'sentiment_score' and 'sentiment_label'.
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
    """Compute structural average sentiment parameters grouped by item tracking identities.

    Triggers batch evaluation metrics if raw calculations are missing, grouping 
    records to calculate mean intensity metrics alongside frequency properties.

    Args:
        df (pd.DataFrame): Data matrix tracking interaction history rows.
        item_col (str, optional): Unique key field grouping baseline products. 
            Defaults to 'title'.

    Returns:
        pd.DataFrame: A grouped DataFrame containing columns for the item key, 
            'avg_sentiment' (float mean score), and 'review_count' (integer total).
    """
    if 'sentiment_score' not in df.columns:
        df = batch_analyze(df, text_col='review_text')

    agg = df.groupby(item_col).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        review_count=('sentiment_score', 'count')
    ).reset_index()

    return agg


def compute_product_sentiment(reviews: List[str]) -> Optional[float]:
    """Dynamically compute average sentiment for unindexed items during pipeline fallbacks.

    Cleans empty structures out of runtime text arrays, providing an isolated fallback 
    calculation metric to protect live endpoints from tracking blank baselines.

    Args:
        reviews (list): Array tracking string content pieces or comments.

    Returns:
        float | None: A 4-decimal rounded float mean score value if valid data 
            is parsed; otherwise None.
    """
    if not reviews:
        return None

    valid_reviews = [
        review for review in reviews
        if isinstance(review, str) and review.strip()
    ]

    if not valid_reviews:
        return None

    scores = np.array(
    [analyze_sentiment(review) for review in valid_reviews]
)
    
    return round(scores.mean(), 4)
