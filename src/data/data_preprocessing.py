import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values. Text columns get 'Unknown', numeric get median."""
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].fillna('Unknown')
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and reset index."""
    return df.drop_duplicates().reset_index(drop=True)


def normalize_ratings(
    df: pd.DataFrame,
    column: str = 'rating'
) -> pd.DataFrame:
    """Normalize rating column to 0-1 scale."""
    df = df.copy()
    if column in df.columns:
        scaler = MinMaxScaler()
        df[f'{column}_normalized'] = scaler.fit_transform(df[[column]])
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: list = None
) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    if columns is None:
        columns = ['authors']
        add_suffix = True
    else:
        add_suffix = False
    for col in columns:
        if col in df.columns:
            encoded_val = LabelEncoder().fit_transform(df[col].astype(str))
            if add_suffix:
                df[f'{col}_encoded'] = encoded_val
            else:
                df[col] = encoded_val
    return df


def preprocess_books_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the books dataset.

    Operations:
    - Remove duplicate entries
    - Handle missing values
    - Encode categorical columns
    - Normalize ratings from 1-5 to 0-1 scale

    Returns:
        Cleaned pandas DataFrame
    """
    print(f'Original shape: {df.shape}')
    df = remove_duplicates(df)
    print(f'After removing duplicates: {df.shape}')
    df = handle_missing_values(df)
    df = encode_categorical(df, ['authors', 'publisher'])
    if 'rating' in df.columns:
        df = normalize_ratings(df, 'rating')
    print(f'Final shape: {df.shape}')
    return df


def preprocess_ratings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ratings dataset.

    Operations:
    - Remove duplicate user-book pairs
    - Handle missing values
    - Normalize ratings from 1-5 to 0-1 scale

    Returns:
        Cleaned pandas DataFrame
    """
    print(f'Original shape: {df.shape}')
    if 'user_id' in df.columns and 'book_id' in df.columns:
        df = df.drop_duplicates(subset=['user_id', 'book_id'])
    else:
        df = df.drop_duplicates()
    print(f'After removing duplicates: {df.shape}')
    df = handle_missing_values(df)
    if 'rating' in df.columns:
        df = normalize_ratings(df, 'rating')
    print(f'Final shape: {df.shape}')
    return df


def preprocess_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
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
    print(f'Original shape: {df.shape}')
    df = remove_duplicates(df)
    print(f'After removing duplicates: {df.shape}')
    df = handle_missing_values(df)
    df = encode_categorical(df, [
        'gender', 'age_group', 'region',
        'product_category', 'purchase_channel',
        'platform', 'sentiment'
    ])
    if 'customer_rating' in df.columns:
        df = normalize_ratings(df, 'customer_rating')
    print(f'Final shape: {df.shape}')
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Detects dataset type and applies
    appropriate preprocessing. Returns clean DataFrame ready for
    model input.
    """
    columns = df.columns.str.lower()
    if 'authors' in columns or 'publisher' in columns:
        return preprocess_books_data(df)
    elif 'user_id' in columns or 'book_id' in columns:
        return preprocess_ratings_data(df)
    elif 'sentiment' in columns:
        return preprocess_sentiment_data(df)
    return handle_missing_values(df)


if __name__ == '__main__':
    print('=== Preprocessing Books Data ===')
    books_df = pd.read_csv('datasets/booksdata.csv')
    books_df = preprocess_books_data(books_df)

    print('\n=== Preprocessing Ratings Data ===')
    ratings_df = pd.read_csv('datasets/ratings.csv')
    ratings_df = preprocess_ratings_data(ratings_df)

    print('\n=== Preprocessing Sentiment Data ===')
    sentiment_df = pd.read_csv('datasets/customer_sentiment.csv')
    sentiment_df = preprocess_sentiment_data(sentiment_df)

    print('\nAll datasets preprocessed successfully!')
    print(f'Books Dataset Shape: {books_df.shape}')
    print(f'Ratings Dataset Shape: {ratings_df.shape}')
    print(f'Sentiment Dataset Shape: {sentiment_df.shape}')
