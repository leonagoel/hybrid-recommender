import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
def handle_missing_values(df):
    return df.fillna("")

def encode_categorical(df):
    """
    Encode categorical (object) columns using label encoding.
    Creates new *_encoded columns without modifying original data.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))

    return df

def remove_duplicates(df: pd.DataFrame, subset=None):
    """
    Remove duplicate rows from dataset.
    """
    if subset:
        return df.drop_duplicates(subset=subset)
    return df.drop_duplicates()
def normalize_ratings(df, column='rating'):
    if column not in df.columns:
        return df

    min_val = df[column].min()
    max_val = df[column].max()

    if max_val == min_val:
        df['rating_normalized'] = 0.0
    else:
        df['rating_normalized'] = (
            (df[column] - min_val) / (max_val - min_val)
        )

    return df
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

    # le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
         
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))

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
    # Normalize ratings if present
    if 'rating' in df.columns:
        min_val = df['rating'].min()
        max_val = df['rating'].max()

        if max_val == min_val:
            df['rating_normalized'] = 0.0
        else:
            df['rating_normalized'] = (
                (df['rating'] - min_val) / (max_val - min_val)
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

def preprocess(df):
    """
    Master preprocessing function for a single dataframe.
    Expected by tests.
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Normalize rating if exists
    if 'rating' in df.columns:
        min_val = df['rating'].min()
        max_val = df['rating'].max()

        if max_val == min_val:
            df['rating_normalized'] = 0.0
        else:
            df['rating_normalized'] = (
                (df['rating'] - min_val) / (max_val - min_val)
            )

    return df