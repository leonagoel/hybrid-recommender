import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os


def preprocess_books_data(filepath="datasets/booksdata.csv"):
    """
    Preprocess the books dataset.

    Operations:
    - Remove duplicate entries
    - Handle missing values
    - Encode categorical columns
    - Normalize ratings from 1–5 to 0–1 scale

    Returns:
        Cleaned pandas DataFrame
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()

    print(f"After removing duplicates: {df.shape}")

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

    # Normalize ratings if present
    if 'rating' in df.columns:
        scaler = MinMaxScaler()

        df['rating_normalized'] = scaler.fit_transform(
            df[['rating']]
        )

    print(f"Final shape: {df.shape}")

    return df


def preprocess_ratings_data(filepath="datasets/ratings.csv"):
    """
    Preprocess the ratings dataset.

    Operations:
    - Remove duplicate user-book pairs
    - Handle missing values
    - Normalize ratings from 1–5 to 0–1 scale

    Returns:
        Cleaned pandas DataFrame
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")

    # Remove duplicate user-book pairs
    if 'user_id' in df.columns and 'book_id' in df.columns:
        df = df.drop_duplicates(subset=['user_id', 'book_id'])
    else:
        df = df.drop_duplicates()

    print(f"After removing duplicates: {df.shape}")

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

    print(f"Final shape: {df.shape}")

    return df


def preprocess_sentiment_data(
    filepath="datasets/customer_sentiment.csv"
):
    """
    Preprocess the customer sentiment dataset.

    Operations:
    - Remove duplicates
    - Handle missing values
    - Encode categorical columns
    - Normalize customer ratings

    Returns:
        Cleaned pandas DataFrame
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()

    print(f"After removing duplicates: {df.shape}")

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

    # Normalize customer ratings
    if 'customer_rating' in df.columns:
        scaler = MinMaxScaler()

        df['rating_normalized'] = scaler.fit_transform(
            df[['customer_rating']]
        )

    print(f"Final shape: {df.shape}")

    return df


def handle_missing_values(df):
    """Handle missing values by filling text columns and numeric columns."""
    df = df.copy()
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    """Remove duplicate entries or duplicate user-item pairs."""
    df = df.copy()
    if 'user_id' in df.columns and 'book_id' in df.columns:
        df = df.drop_duplicates(subset=['user_id', 'book_id'])
    else:
        df = df.drop_duplicates()
    return df


def normalize_ratings(df):
    """Normalize ratings column using MinMaxScaler."""
    df = df.copy()
    if 'rating' in df.columns:
        scaler = MinMaxScaler()
        df['rating_normalized'] = scaler.fit_transform(df[['rating']])
    return df


def encode_categorical(df):
    """Encode categorical columns using LabelEncoder."""
    df = df.copy()
    le = LabelEncoder()
    if 'authors' in df.columns:
        df['authors_encoded'] = le.fit_transform(df['authors'].astype(str))
    return df


def preprocess(df):
    """Run full preprocessing pipeline on a DataFrame."""
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = normalize_ratings(df)
    df = encode_categorical(df)
    return df


if __name__ == "__main__":

    print("=== Preprocessing Books Data ===")

    books_df = preprocess_books_data()

    print("\n=== Preprocessing Ratings Data ===")

    ratings_df = preprocess_ratings_data()

    print("\n=== Preprocessing Sentiment Data ===")

    sentiment_df = preprocess_sentiment_data()

    print("\n✅ All datasets preprocessed successfully!")

    print(f"Books Dataset Shape: {books_df.shape}")
    print(f"Ratings Dataset Shape: {ratings_df.shape}")
    print(f"Sentiment Dataset Shape: {sentiment_df.shape}")
