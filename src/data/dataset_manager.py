import pandas as pd
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from data_adapter import adapt_data
from data_preprocessing import preprocess


class DatasetManager:
    """
    Manages multiple loaded datasets.
    Each dataset is adapted into a unified schema on load.
    All datasets can be merged into a single DataFrame for the recommender.
    """

    def __init__(self):
        self._datasets = {}  # id → { 'name': str, 'raw': df, 'adapted': df, 'meta': dict }

    # ------------------------------------------------------------------
    def load_csv(self, file_path_or_buffer: Any, name: Optional[str] = None, catalog: Optional[str] = None) -> str:
        """
        Load a CSV file (path string or file-like object) into the manager.
        Returns the dataset ID.
        """
        if isinstance(file_path_or_buffer, str):
            if not os.path.exists(file_path_or_buffer):
                raise FileNotFoundError(f"File not found: {file_path_or_buffer}")
            raw_df = pd.read_csv(file_path_or_buffer, on_bad_lines='skip', low_memory=False)
            if name is None:
                name = os.path.basename(file_path_or_buffer)
        else:
            raw_df = pd.read_csv(file_path_or_buffer, on_bad_lines='skip', low_memory=False)
            if name is None:
                name = 'uploaded_dataset'

        if catalog is None:
            catalog = os.path.splitext(name)[0]

        raw_df = preprocess(raw_df)
        adapted_df, meta = adapt_data(raw_df)
        adapted_df['catalog'] = catalog
        
        ds_id = str(uuid.uuid4())[:8]

        self._datasets[ds_id] = {
            'name': name,
            'catalog': catalog,
            'raw': raw_df,
            'adapted': adapted_df,
            'meta': meta,
        }
        return ds_id

    # ------------------------------------------------------------------
    def remove_dataset(self, ds_id):
        """Remove a loaded dataset by ID."""
        if ds_id in self._datasets:
            del self._datasets[ds_id]
            return True
        return False

    # ------------------------------------------------------------------
    def list_datasets(self):
        """Return a summary of all loaded datasets."""
        result = []
        for ds_id, ds in self._datasets.items():
            result.append({
                'id': ds_id,
                'name': ds['name'],
                'rows': ds['meta']['total_rows'],
                'has_reviews': ds['meta']['has_reviews'],
                'has_user_data': ds['meta']['has_user_data'],
                'has_behavior': ds['meta']['has_behavior'],
                'detected_columns': {
                    k: v for k, v in ds['meta'].items()
                    if k.endswith('_col') and v is not None
                },
            })
        return result

    # ------------------------------------------------------------------
    def get_stats(self):
        """Aggregate statistics across all loaded datasets."""
        total_rows = sum(ds['meta']['total_rows'] for ds in self._datasets.values())
        return {
            'dataset_count': len(self._datasets),
            'total_rows': total_rows,
            'datasets': [d['name'] for d in self._datasets.values()],
        }

    # ------------------------------------------------------------------
    def merge_all(self):
        """
        Merge all adapted datasets into a single DataFrame.
        Deduplicates by (title) — keeps the first occurrence,
        but aggregates ratings & reviews from duplicates.
        """
        if not self._datasets:
            raise ValueError("No datasets loaded.")

        frames = [ds['adapted'] for ds in self._datasets.values()]
        merged = pd.concat(frames, ignore_index=True)

        # Remove duplicate columns if any exist (can happen with raw data)
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # Deduplicate items — aggregate per unique title
        # Keep first description/category, average rating, concat reviews
        
        # Ensure only columns that actually exist in merged are in the agg dict
        agg_dict = {
            'item_id':      'first',
            'description':  'first',
            'category':     'first',
            'combined':     'first',
            'user_id':      'first',
            'rating':       'mean',
            'review_text':  lambda x: ' '.join(x.astype(str)),
            'views':        'sum',
            'purchases':    'sum',
            'catalog':      'first',
        }
        
        # Filter agg_dict to only include columns present in merged
        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in merged.columns}

        grouped = merged.groupby('title', as_index=False).agg(valid_agg_dict)

        # Add top 2 reviews list
        if 'review_text' in merged.columns:
            def get_reviews(series):
                valid = [s for s in series.astype(str) if s and str(s).strip() and len(str(s)) > 8]
                return valid[:2]
            
            reviews_df = merged.groupby('title')['review_text'].agg(get_reviews).reset_index()
            reviews_df.rename(columns={'review_text': 'top_reviews'}, inplace=True)
            grouped = grouped.merge(reviews_df, on='title', how='left')

        # But we also keep the full interaction-level data for collaborative model
        return merged, grouped

    # ------------------------------------------------------------------
    def get_interaction_df(self):
        """Return the full interaction-level DataFrame (user × item ratings)."""
        if not self._datasets:
            raise ValueError("No datasets loaded.")
        frames = [ds['adapted'] for ds in self._datasets.values()]
        return pd.concat(frames, ignore_index=True)
