from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
_DATASETS = _BASE_DIR / 'datasets'

# Load datasets
books = pd.read_csv(_DATASETS / 'books.csv')
ratings = pd.read_csv(_DATASETS / 'ratings.csv')

# Merge
df = pd.merge(ratings, books, on="book_id")

# Keep useful columns
df = df[['user_id', 'book_id', 'rating', 'title', 'authors']]

# Create description
df['description'] = df['title'] + " " + df['authors']

# Reduce size (for speed)
df = df.head(5000)

# Save with YOUR name
df.to_csv(_DATASETS / 'booksdata.csv', index=False)

print("Dataset prepared successfully!")
