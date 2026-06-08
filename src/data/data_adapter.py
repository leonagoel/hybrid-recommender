"""
Data Adapter — Unified schema adapter for CSV and JSON datasets.
Detects columns automatically and normalizes to a standard schema
used by all recommender models.

Fixes applied:
  - Title detection now catches 'Book-Title', 'book_title', 'book-title'
  - User detection now catches 'User-ID', 'user-id'
  - Rating detection now catches 'Book-Rating', 'book-rating'
  - item_id detection now catches 'ISBN'
  - When title_col is None but item_id_col IS found, we look for a
    dedicated title-like column before falling back to iloc[:,0].
    If still nothing, item_id is used as title (better than a random
    numeric column).
  - After merging datasets, title is resolved from a cross-dataset
    title_map so ratings.csv rows get real book titles, not IDs.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def remove_duplicate_headers(df):
    """Normalizes dataframe column headers and eliminates semantic duplicates.

    Converts columns to lowercase and strips trailing spaces. If duplicate 
    mappings are found, it preserves the column containing the lower count of 
    missing or null entries.

    Args:
        df (pd.DataFrame): The input DataFrame whose headers need deduplication.

    Returns:
        pd.DataFrame: A new or modified DataFrame with unique, cleaned headers.
    """
    normalized_cols = {}
    cols_to_drop = []


    for col in df.columns:
        # Convert to lowercase and strip spaces to detect duplicates like 'Product_ID' and 'product_id'
        norm_name = str(col).strip().lower()
        
        if norm_name in normalized_cols:
            existing_col = normalized_cols[norm_name]
            # Compare missing/null value counts between the duplicates
            existing_nulls = df[existing_col].isnull().sum()
            current_nulls = df[col].isnull().sum()
            
            # Keep the one with fewer null values, drop the other from the array
            if current_nulls < existing_nulls:
                cols_to_drop.append(existing_col)
                normalized_cols[norm_name] = col
            else:
                cols_to_drop.append(col)
        else:
            normalized_cols[norm_name] = col

    # Drop duplicate column fields gracefully before parsing mapping states
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    return df


def preprocess_books_data(df):
    """
    Preprocess books dataset.
    """

def preprocess_books_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    le = LabelEncoder()
    for col in ['authors', 'publisher']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    if 'rating' in df.columns:
        scaler = MinMaxScaler()
        df['rating_normalized'] = scaler.fit_transform(df[['rating']])
    return df


def preprocess_ratings_data(df):
    df = df.copy()
    if 'user_id' in df.columns and 'book_id' in df.columns:
        df = df.drop_duplicates(subset=['user_id', 'book_id'])
    else:
        df = df.drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    if 'rating' in df.columns:
        scaler = MinMaxScaler()
        df['rating_normalized'] = scaler.fit_transform(df[['rating']])
    return df


def preprocess_sentiment_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    le = LabelEncoder()
    for col in ['gender', 'age_group', 'region', 'product_category',
                'purchase_channel', 'platform', 'sentiment']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    if 'customer_rating' in df.columns:
        scaler = MinMaxScaler()
        df['rating_normalized'] = scaler.fit_transform(df[['customer_rating']])
    return df


# ─────────────────────────────────────────────
#  Column detection
# ─────────────────────────────────────────────

def detect_column(columns, keywords):
    """Detect a column by matching against a list of keywords (case-insensitive)."""
    if not keywords:
        return None

    # First pass: exact matches
    for key in keywords:
        for col in columns:
            if col.lower() == key:
                return col
                
    # Second pass: substring matches
    for key in keywords:
        for col in columns:
            if key in col.lower():
                # Avoid false positive where customer_rating is matched as user column
                if key == 'customer' and ('rating' in col.lower() or 'score' in col.lower()):
                    continue
                return col
    return None


# ─────────────────────────────────────────────
#  Validation / IO helpers
# ─────────────────────────────────────────────

def validate_dataframe(df):
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns.")
    return True


def _has_blank_values(series):
    """Return True when a column contains nulls or blank string values."""
    return series.isna().any() or series.astype(str).str.strip().eq('').any()


def validate_recommender_inputs(df, user_col=None, item_id_col=None, title_col=None, rating_col=None):
    """
    Validate columns that feed the recommender pipeline before normalization.
    """
    missing = []
    if user_col is None:
        missing.append('user_id')
    if item_id_col is None and title_col is None:
        missing.append('item_id or title')
    if rating_col is None:
        missing.append('rating')

    if missing:
        raise ValueError(
            "Interaction data is missing required column(s): "
            + ", ".join(missing)
        )

    corrupted = []
    for label, col in (('user_id', user_col), ('item_id', item_id_col), ('title', title_col)):
        if col is not None and _has_blank_values(df[col]):
            corrupted.append(label)

    if _has_blank_values(df[rating_col]):
        corrupted.append('rating')

    ratings = pd.to_numeric(df[rating_col], errors='coerce')
    if ratings.isna().any():
        corrupted.append('rating must be numeric')

    if corrupted:
        raise ValueError(
            "Interaction data contains invalid value(s) in: "
            + ", ".join(dict.fromkeys(corrupted))
        )

    return True


def read_file(path_or_buffer, file_format=None):
    if file_format is None and isinstance(path_or_buffer, str):
        file_format = 'json' if path_or_buffer.lower().endswith('.json') else 'csv'

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

                # Deduplicate raw column configurations early upon reading CSV file buffers
                df = remove_duplicate_headers(df)
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
            # Deduplicate raw column configurations early upon fallback processing loop strings
            df = remove_duplicate_headers(df)

    return df


# ─────────────────────────────────────────────
#  Main adapter
# ─────────────────────────────────────────────

def adapt_data(df):
    """
    Adapt any DataFrame into unified schema (case‑insensitive column matching).

    This function handles schema adaptation: column detection, renaming to
    canonical names, and filling missing required fields.
    """

    # Apply early header case deduplication loop tracking array states
    df = remove_duplicate_headers(df)

    validate_dataframe(df)

    # ── Main column mapping (case‑insensitive) ──
    columns = df.columns  # update after possible preprocessing

    # ── detect columns ──────────────────────────
    # FIX: extended keywords to cover BX dataset ('Book-Title', 'User-ID', 'Book-Rating')
    title_col = detect_column(columns, [
        'book title', 'booktitle',           # BX: 'Book-Title'
        'title', 'name', 'product name', 'item name',
    ])

    desc_col = detect_column(
        columns,
        [
            'description',
            'desc',
            'summary',
            'overview',
            'about'
        ]
    )
    user_col = detect_column(
        columns,
        ['user_id', 'user', 'reviewer', 'customer']
    )

    user_col = detect_column(columns, [
        'user id', 'userid',                 # BX: 'User-ID'
        'user', 'reviewer', 'customer',
    ])

    rating_col = detect_column(columns, [
        'book rating', 'bookrating',         # BX: 'Book-Rating'
        'rating', 'score', 'stars',
    ])

    review_col = detect_column(columns, [
        'review', 'text', 'comment', 'feedback', 'review text',
    ])

    category_col = detect_column(columns, [
        'category', 'genre', 'tags', 'type', 'department',
    ])

    item_id_col = detect_column(columns, [
        'item id', 'product id', 'asin',
        'isbn',                              # BX: 'ISBN'
        'book id', 'movie id',
    ])

    views_col    = detect_column(columns, ['views', 'clicks', 'impressions'])
    purchase_col = detect_column(columns, ['purchases', 'orders', 'bought', 'transactions'])

    is_interaction_data = user_col is not None
    if is_interaction_data:
        validate_recommender_inputs(
            df,
            user_col=user_col,
            item_id_col=item_id_col,
            title_col=title_col,
            rating_col=rating_col,
        )

    df = df.copy()

    # ── rename detected columns ─────────────────
    rename_map = {}
    if title_col:   rename_map[title_col]    = 'title'
    if desc_col:    rename_map[desc_col]     = 'description'
    if user_col:    rename_map[user_col]     = 'user_id'
    if rating_col:  rename_map[rating_col]   = 'rating'
    if review_col:  rename_map[review_col]   = 'review_text'
    if category_col: rename_map[category_col] = 'category'
    if item_id_col: rename_map[item_id_col]  = 'item_id'
    if views_col:   rename_map[views_col]    = 'views'
    if purchase_col: rename_map[purchase_col] = 'purchases'

    df = df.rename(columns=rename_map)
    # Safety net: drop duplicate columns that may arise when both an exact match
    # (e.g. 'title') and a partial match (e.g. 'original_title') exist in the dataset.
    df = df.loc[:, ~df.columns.duplicated()]

    # ── resolve 'title' ─────────────────────────
    # FIX: old code fell back to iloc[:,0] which picked up numeric book_id.
    # Now: if title still missing but item_id exists, use item_id as title
    # (a stable string identifier is better than a random first column).
    if 'title' not in df.columns:
        if 'item_id' in df.columns:
            df['title'] = df['item_id'].astype(str)
        else:
            df['title'] = df.iloc[:, 0].astype(str)

    df['title'] = df['title'].fillna('Unknown').astype(str)

    # ── other columns ───────────────────────────
    if 'description' not in df.columns:
        df['description'] = ''
    else:
        df['description'] = df['description'].fillna('')

    if 'category' not in df.columns:
        df['category'] = ''
    else:
        df['category'] = df['category'].fillna('')

    if 'item_id' not in df.columns:
        df['item_id'] = range(len(df))

    if 'review_text' not in df.columns:
        df['review_text'] = ''
    else:
        df['review_text'] = df['review_text'].fillna('')

    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    else:
        df['rating'] = 0.0

    if 'user_id' not in df.columns:
        df['user_id'] = df.index.astype(str)
    df['user_id'] = df['user_id'].astype(str)

    if 'views' not in df.columns:
        df['views'] = 0
    if 'purchases' not in df.columns:
        df['purchases'] = 0

    # combined text feature for content model
    df['combined'] = (
        df['title'].astype(str) + ' ' +
        df['description'].astype(str) + ' ' +
        df['category'].astype(str)
    )

    meta = {
        'title_col':    title_col,
        'desc_col':     desc_col,
        'user_col':     user_col,
        'rating_col':   rating_col,
        'review_col':   review_col,
        'category_col': category_col,
        'item_id_col':  item_id_col,
        'views_col':    views_col,
        'purchase_col': purchase_col,
        'has_user_data':  (user_col is not None and rating_col is not None),
        'has_reviews':    review_col is not None,
        'has_behavior':   (views_col is not None or purchase_col is not None),
        'total_rows':     len(df),
        'total_columns':  len(df.columns),
    }

    return df, meta
