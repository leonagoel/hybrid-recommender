"""
Data Adapter — Unified schema adapter for CSV and JSON datasets.
Detects columns automatically and normalizes to a standard schema
used by all recommender models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocess_books_data(df):
    """
    Preprocess books dataset.
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    # Handle missing numeric values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    categorical_cols = ['authors', 'publisher']

    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize ratings
    if 'rating' in df.columns:
        scaler = MinMaxScaler()

        df['rating_normalized'] = scaler.fit_transform(
            df[['rating']]
        )

    return df


def preprocess_ratings_data(df):
    """
    Preprocess ratings dataset.
    """

    df = df.copy()

    # Remove duplicates
    if 'user_id' in df.columns and 'book_id' in df.columns:
        df = df.drop_duplicates(subset=['user_id', 'book_id'])
    else:
        df = df.drop_duplicates()

    # Handle missing categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    # Handle missing numeric values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Normalize ratings
    if 'rating' in df.columns:
        scaler = MinMaxScaler()

        df['rating_normalized'] = scaler.fit_transform(
            df[['rating']]
        )

    return df


def preprocess_sentiment_data(df):
    """
    Preprocess customer sentiment dataset.
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    # Handle missing numeric values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    categorical_cols = [
        'gender',
        'age_group',
        'region',
        'product_category',
        'purchase_channel',
        'platform',
        'sentiment'
    ]

    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize customer rating
    if 'customer_rating' in df.columns:
        scaler = MinMaxScaler()

        df['rating_normalized'] = scaler.fit_transform(
            df[['customer_rating']]
        )

    return df


def detect_column(columns, keywords):
    """Detect a column by matching against a list of keywords (case-insensitive)."""
    # First pass: exact matches
    for key in keywords:
        for col in columns:
            if col.lower() == key:
                return col
                
    # Second pass: substring matches
    for key in keywords:
        for col in columns:
            if key in col.lower():
                return col
                
    return None


def validate_dataframe(df):
    """
    Validate DataFrame.
    """

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if len(df.columns) < 2:
        raise ValueError(
            "DataFrame must have at least 2 columns."
        )

    return True


def read_file(path_or_buffer, file_format=None):
    """
    Read CSV or JSON into DataFrame.
    """

    if file_format is None and isinstance(path_or_buffer, str):

        if path_or_buffer.lower().endswith('.json'):
            file_format = 'json'

        else:
            file_format = 'csv'

    if file_format == 'json':

        try:
            df = pd.read_json(path_or_buffer, lines=True)

        except ValueError:

            if hasattr(path_or_buffer, 'seek'):
                path_or_buffer.seek(0)

            df = pd.read_json(path_or_buffer)

    else:

        for encoding in ['utf-8', 'latin-1', 'cp1252']:

            try:

                if hasattr(path_or_buffer, 'seek'):
                    path_or_buffer.seek(0)

                df = pd.read_csv(
                    path_or_buffer,
                    on_bad_lines='skip',
                    low_memory=False,
                    encoding=encoding,
                )

                break

            except (UnicodeDecodeError, UnicodeError):
                continue

        else:

            if hasattr(path_or_buffer, 'seek'):
                path_or_buffer.seek(0)

            df = pd.read_csv(
                path_or_buffer,
                on_bad_lines='skip',
                low_memory=False,
                encoding='utf-8',
                encoding_errors='replace',
            )

    return df


def adapt_data(df):
    """
    Adapt any DataFrame into unified schema.
    """

    validate_dataframe(df)

    # Apply preprocessing automatically

    if 'authors' in df.columns or 'publisher' in df.columns:

        df = preprocess_books_data(df)

    elif 'user_id' in df.columns and 'rating' in df.columns:

        df = preprocess_ratings_data(df)

    elif 'sentiment' in df.columns:

        df = preprocess_sentiment_data(df)

    columns = df.columns

    title_col = detect_column(
        columns,
        ['title', 'name', 'product_name', 'item_name']
    )

    desc_col = detect_column(
        columns,
        ['desc', 'summary', 'overview', 'about']
    )

    user_col = detect_column(
        columns,
        ['user_id', 'user', 'reviewer', 'customer']
    )

    rating_col = detect_column(
        columns,
        ['rating', 'score', 'stars']
    )

    review_col = detect_column(
        columns,
        ['review', 'text', 'comment', 'feedback', 'review_text']
    )

    category_col = detect_column(
        columns,
        ['category', 'genre', 'tags', 'type', 'department']
    )

    item_id_col = detect_column(
        columns,
        ['item_id', 'product_id', 'asin',
         'isbn', 'book_id', 'movie_id']
    )

    views_col = detect_column(
        columns,
        ['views', 'clicks', 'impressions']
    )

    purchase_col = detect_column(
        columns,
        ['purchases', 'orders', 'bought', 'transactions']
    )

    df = df.copy()

    rename_map = {}

    if title_col:
        rename_map[title_col] = 'title'

    if desc_col:
        rename_map[desc_col] = 'description'

    if user_col:
        rename_map[user_col] = 'user_id'

    if rating_col:
        rename_map[rating_col] = 'rating'

    if review_col:
        rename_map[review_col] = 'review_text'

    if category_col:
        rename_map[category_col] = 'category'

    if item_id_col:
        rename_map[item_id_col] = 'item_id'

    if views_col:
        rename_map[views_col] = 'views'

    if purchase_col:
        rename_map[purchase_col] = 'purchases'

    df = df.rename(columns=rename_map)
    # Safety net: drop duplicate columns that may arise when both an exact match
    # (e.g. 'title') and a partial match (e.g. 'original_title') exist in the dataset.
    df = df.loc[:, ~df.columns.duplicated()]

    # Fill missing safely

    if 'title' in df.columns:
        df['title'] = df['title'].fillna('Unknown')

    else:
        df['title'] = df.iloc[:, 0].astype(str)

    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')

    else:
        df['description'] = ''

    if 'category' not in df.columns:
        df['category'] = ''

    else:
        df['category'] = df['category'].fillna('')

    if 'item_id' not in df.columns:
        df['item_id'] = range(len(df))

    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].fillna('')

    else:
        df['review_text'] = ''

    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(
            df['rating'],
            errors='coerce'
        ).fillna(0)

    else:
        df['rating'] = 0.0

    if 'user_id' not in df.columns:
        df['user_id'] = 'anonymous'

    if 'views' not in df.columns:
        df['views'] = 0

    if 'purchases' not in df.columns:
        df['purchases'] = 0

    # Combined text feature

    df['combined'] = (
        df['title'].astype(str) + ' ' +
        df['description'].astype(str) + ' ' +
        df['category'].astype(str)
    )

    meta = {
        'title_col': title_col,
        'desc_col': desc_col,
        'user_col': user_col,
        'rating_col': rating_col,
        'review_col': review_col,
        'category_col': category_col,
        'item_id_col': item_id_col,
        'views_col': views_col,
        'purchase_col': purchase_col,
        'has_user_data': (
            user_col is not None and rating_col is not None
        ),
        'has_reviews': review_col is not None,
        'has_behavior': (
            views_col is not None or purchase_col is not None
        ),
        'total_rows': len(df),
        'total_columns': len(df.columns),
    }

    return df, meta