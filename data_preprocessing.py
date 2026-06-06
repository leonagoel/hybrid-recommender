import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Union
from pathlib import Path
import os

def preprocess_books_data(data: Union[str, Path, pd.DataFrame] = "datasets/booksdata.csv"):
    """
    Preprocess the books dataset.
    - Removes duplicate entries
    - Handles missing values
    - Normalizes ratings from 1-5 to 0-1 scale
    Args:
        data: File path (str or Path) to a CSV file, or a pandas DataFrame.
    Returns: cleaned pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        raise TypeError(f"Expected str, Path, or pd.DataFrame — got {type(data)}")

    print(f"Original shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    df = df.dropna(subset=['title', 'authors'])
    df['description'] = df['description'].fillna('No description available')

    scaler = MinMaxScaler()
    df['rating_normalized'] = scaler.fit_transform(df[['rating']])

    print(f"Final shape: {df.shape}")
    return df

def preprocess_ratings_data(data: Union[str, Path, pd.DataFrame] = "datasets/ratings.csv"):
    """
    Preprocess the ratings dataset.
    - Removes duplicate user-book pairs
    - Handles missing values
    - Normalizes ratings from 1-5 to 0-1 scale
    Args:
        data: File path (str or Path) to a CSV file, or a pandas DataFrame.
    Returns: cleaned pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        raise TypeError(f"Expected str, Path, or pd.DataFrame — got {type(data)}")

    print(f"Original shape: {df.shape}")

    df = df.drop_duplicates(subset=['user_id', 'book_id'])
    print(f"After removing duplicates: {df.shape}")

    df = df.dropna()

    scaler = MinMaxScaler()
    df['rating_normalized'] = scaler.fit_transform(df[['rating']])

    print(f"Final shape: {df.shape}")
    return df

def preprocess_sentiment_data(data: Union[str, Path, pd.DataFrame] = "datasets/Customer_Sentiment.csv"):
    """
    Preprocess the customer sentiment dataset.
    - Removes duplicates
    - Handles missing values
    - Encodes categorical columns (gender, region, sentiment etc)
    - Normalizes customer_rating to 0-1 scale
    Args:
        data: File path (str or Path) to a CSV file, or a pandas DataFrame.
    Returns: cleaned pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        raise TypeError(f"Expected str, Path, or pd.DataFrame — got {type(data)}")

    print(f"Original shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    df = df.dropna()

    categorical_cols = ['gender', 'age_group', 'region', 
                       'product_category', 'purchase_channel', 
                       'platform', 'sentiment']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    scaler = MinMaxScaler()
    df['rating_normalized'] = scaler.fit_transform(
        df[['customer_rating']])

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
    print(f"Books: {books_df.shape}")
    print(f"Ratings: {ratings_df.shape}")
    print(f"Sentiment: {sentiment_df.shape}")