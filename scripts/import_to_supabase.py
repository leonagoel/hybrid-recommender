"""
Import products from CSV/JSON datasets into Supabase PostgreSQL.
Processes data in batches to handle large files (250k+ rows).

Usage:
    python scripts/import_to_supabase.py
    python scripts/import_to_supabase.py --file datasets/Books.csv --batch-size 2000

Optimized via Issue #490: Implements strict pathlib absolute context mappings 
to prevent relative lookup path anomalies across multi-tier runtime environments.
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def chunked(df, size):
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def _safe_int(value, default=0):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, default)


def build_product_row(row):
    """Normalize one adapted product row for the Supabase products table."""
    return {
        'title': str(row.get('title', 'Unknown'))[:500],
        'description': str(row.get('description', ''))[:2000],
        'category': str(row.get('category', ''))[:200],
        'rating': _safe_float(row.get('rating', 0)),
        'avg_sentiment': _safe_float(row.get('sentiment', 0)),
        'review_count': _safe_int(row.get('review_count', 0)),
        'metadata': {},
    }


def import_dataset(file_path, batch_size=1000, run_sentiment=False):
    """Import a single dataset file into the products table."""
    file_path_obj = Path(file_path)
    if not file_path_obj.is_absolute():
        file_path_obj = (PROJECT_ROOT / file_path_obj).resolve()

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data.data_adapter import adapt_data
    from src.data.db import get_supabase_admin
    print(f"\n{'='*60}")
    print(f"  Importing: {file_path_obj.name}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Read file format using verified pathlib extensions
    ext = file_path_obj.suffix.lower()
    if ext == '.json':
        raw_df = pd.read_json(str(file_path_obj), lines=True)
    elif ext == '.csv':
        raw_df = pd.read_csv(str(file_path_obj), on_bad_lines='skip', low_memory=False)
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

    print(f"\n  Code execution block finished. Imported {inserted:,} products ({errors} batch errors)")
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Import datasets into Supabase')
    parser.add_argument('--file', type=str, help='Specific file to import')
    parser.add_argument('--batch-size', type=int, default=1000, help='Rows per batch')
    parser.add_argument('--sentiment', action='store_true', help='Run sentiment analysis')
    args = parser.parse_args()

    # FIX FOR ISSUE #490: Safely anchor data directory to root layout path
    data_dir = PROJECT_ROOT / "datasets"

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (PROJECT_ROOT / file_path).resolve()
        files = [file_path]
    else:
        # Default: import all CSV/JSON files in datasets/
        files = []
        if data_dir.exists():
            for f in sorted(os.listdir(data_dir)):
                if f.endswith(('.csv', '.json')):
                    files.append(data_dir / f)

    if not files:
        print(f"No dataset files found. Place CSV/JSON files in absolute target path: {data_dir}")
        return

    print(f"\nFound {len(files)} dataset file(s)")
    total = 0
    for f in files:
        # Evaluate clean path context references regardless of execution boundaries
        path_check = f if f.is_absolute() else data_dir / f.name
        
        if path_check.exists():
            total += import_dataset(str(path_check), args.batch_size, args.sentiment)
        else:
            print(f"  ✗ File not found at target: {path_check}")

    print(f"\n{'='*60}")
    print(f"  Total products imported: {total:,}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
