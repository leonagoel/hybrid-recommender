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
