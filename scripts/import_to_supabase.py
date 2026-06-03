"""
Import products from CSV/JSON datasets into Supabase PostgreSQL.
Processes data in batches to handle large files (250k+ rows).

Usage:
    python scripts/import_to_supabase.py
    python scripts/import_to_supabase.py --file datasets/Books.csv --batch-size 2000
"""
import os
import sys
import argparse
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from tqdm import tqdm
from src.data.data_adapter import adapt_data
from src.data.db import get_supabase_admin


def chunked(df: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    """Yield successive chunks of a DataFrame."""
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


def import_dataset(file_path: str, batch_size: int = 1000, run_sentiment: bool = False) -> int:
    """Import a single dataset file into the products table."""
    print(f"\n{'='*60}")
    print(f"  Importing: {os.path.basename(file_path)}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Read file
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.json':
        raw_df = pd.read_json(file_path, lines=True)
    elif ext == '.csv':
        raw_df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    else:
        print(f"Unsupported format: {ext}")
        return 0

    print(f"  Raw rows: {len(raw_df):,}")

    # Adapt columns
    adapted_df, meta = adapt_data(raw_df)
    print(f"  Adapted rows: {len(adapted_df):,}")
    print(f"  Detected columns: {', '.join(k for k, v in meta.items() if k.endswith('_col') and v)}")

    # Deduplicate by title
    adapted_df = adapted_df.drop_duplicates(subset='title', keep='first')
    print(f"  Unique titles: {len(adapted_df):,}")

    # Sentiment analysis (optional — slow on large datasets)
    if run_sentiment and 'review_text' in adapted_df.columns:
        from src.model.nlp_engine import analyze_sentiment

        print("  Running sentiment analysis...")
        adapted_df['sentiment'] = adapted_df['review_text'].apply(
            lambda x: analyze_sentiment(str(x)) if pd.notna(x) and str(x).strip() else 0.0
        )
    else:
        adapted_df['sentiment'] = 0.0

    # Prepare for insert
    sb = get_supabase_admin()
    inserted = 0
    errors = 0

    for chunk in tqdm(list(chunked(adapted_df, batch_size)), desc="  Uploading"):
        rows = []
        for _, row in chunk.iterrows():
            rows.append(build_product_row(row))

        try:
            result = sb.table('products').upsert(
                rows,
                on_conflict='title',
                ignore_duplicates=True
            ).execute()
            inserted += len(rows)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  ⚠ Batch error: {str(e)[:200]}")

    print(f"\n  ✅ Imported {inserted:,} products ({errors} batch errors)")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description='Import datasets into Supabase')
    parser.add_argument('--file', type=str, help='Specific file to import')
    parser.add_argument('--batch-size', type=int, default=1000, help='Rows per batch')
    parser.add_argument('--sentiment', action='store_true', help='Run sentiment analysis')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')

    if args.file:
        files = [args.file]
    else:
        # Default: import all CSV/JSON files in datasets/
        files = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(('.csv', '.json')):
                files.append(os.path.join(data_dir, f))

    if not files:
        print("No dataset files found. Place CSV/JSON files in datasets/")
        return

    print(f"\nFound {len(files)} dataset file(s)")
    total = 0
    for f in files:
        path = f if os.path.isabs(f) else os.path.join(data_dir, f)
        if os.path.exists(path):
            total += import_dataset(path, args.batch_size, args.sentiment)
        else:
            print(f"  ✗ File not found: {path}")

    print(f"\n{'='*60}")
    print(f"  Total products imported: {total:,}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
