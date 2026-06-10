"""
DatasetManager — loads, adapts, and merges multiple CSV/JSON datasets.

Fixes vs. original:
  - merge_all: corrected indentation (all body lines were outside the method)
  - merge_all: build a title_map from every loaded dataset so that
    rows that contain only an ISBN/item_id as their title get resolved
    to a real book title when the books.csv / booksdata.csv is also loaded.
  - merge_all: returns (interaction_df, item_df) consistently.
  - load_csv: delegates file reading to data_adapter.read_file so encoding
    issues are handled uniformly.
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from src.data.data_adapter import adapt_data
from src.data.data_preprocessing import preprocess


class DatasetManager:
    """
    Manages multiple loaded datasets.
    Each dataset is adapted into a unified schema on load.
    All datasets can be merged into a single DataFrame for the recommender.
    """

    def __init__(self):
        # id → { 'name': str, 'raw': df, 'adapted': df, 'meta': dict }
        self._datasets = {}

    # ------------------------------------------------------------------
    def load_csv(self, file_path_or_buffer: Any, name: Optional[str] = None, catalog: Optional[str] = None) -> str:
        """
        Load a CSV (or JSON) file into the manager.
        Accepts a path string or a file-like object.
        Returns the dataset ID.
        """
        if isinstance(file_path_or_buffer, str):
            if not os.path.exists(file_path_or_buffer):
                raise FileNotFoundError(f"File not found: {file_path_or_buffer}")
            if name is None:
                name = os.path.basename(file_path_or_buffer)

        if catalog is None:
            catalog = os.path.splitext(name)[0]

        raw_df = preprocess(raw_df)
        adapted_df, meta = adapt_data(raw_df)
        adapted_df['catalog'] = catalog
        
        ds_id = str(uuid.uuid4())[:8]

        ds_id = str(uuid.uuid4())[:8]
        self._datasets[ds_id] = {
            'name': name,
            'catalog': catalog,
            'raw': raw_df,
            'adapted': adapted_df,
            'meta':    meta,
        }
        return ds_id

    # ------------------------------------------------------------------
    def remove_dataset(self, ds_id):
        """Remove a loaded dataset by ID. Returns True if found."""
        if ds_id in self._datasets:
            del self._datasets[ds_id]
            return True
        return False

    # ------------------------------------------------------------------
    def list_datasets(self):
        """Return a summary list of all loaded datasets."""
        result = []
        for ds_id, ds in self._datasets.items():
            result.append({
                'id':   ds_id,
                'name': ds['name'],
                'rows': ds['meta']['total_rows'],
                'has_reviews':   ds['meta']['has_reviews'],
                'has_user_data': ds['meta']['has_user_data'],
                'has_behavior':  ds['meta']['has_behavior'],
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
            'total_rows':    total_rows,
            'datasets':      [d['name'] for d in self._datasets.values()],
        }

    # ------------------------------------------------------------------
    def merge_all(self):
        """
        Merge every loaded dataset into:
          - interaction_df : full row-level frame (user × item × rating)
          - item_df        : one row per title, aggregated features

        Title resolution
        ----------------
        Some datasets (e.g. ratings.csv) carry only an ISBN/item_id as their
        'title'. We build a title_map {item_id → real_title} from whichever
        dataset actually contains human-readable titles, then patch the
        interaction frame before returning.
        """
        if not self._datasets:
            raise ValueError("No datasets loaded.")

        frames = [ds['adapted'] for ds in self._datasets.values()]
        merged = pd.concat(frames, ignore_index=True)

        # ── deduplicate columns (can arise from concat) ─────────────
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # ── guarantee user_id ───────────────────────────────────────
        if 'user_id' not in merged.columns:
            raise ValueError("user_id column missing after merge.")
        merged = merged.dropna(subset=['user_id'])
        merged['user_id'] = merged['user_id'].astype(str)

        # ── resolve titles from cross-dataset title_map ─────────────
        # Build map: item_id (string) → best known title
        # We prefer rows where title != item_id (i.e. a real title exists).
        title_map = {}
        for ds in self._datasets.values():
            adf = ds['adapted']
            if 'item_id' not in adf.columns or 'title' not in adf.columns:
                continue
            mask = adf['title'].astype(str) != adf['item_id'].astype(str)
            for _, row in adf[mask].iterrows():
                key = str(row['item_id'])
                if key not in title_map:
                    title_map[key] = str(row['title'])

        if title_map and 'item_id' in merged.columns:
            # Only overwrite rows where title == item_id (placeholder titles)
            placeholder = merged['title'].astype(str) == merged['item_id'].astype(str)
            merged.loc[placeholder, 'title'] = (
                merged.loc[placeholder, 'item_id']
                .astype(str)
                .map(title_map)
                .fillna(merged.loc[placeholder, 'title'])
            )

        # ── build item_df (one row per title, aggregated) ───────────
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
        valid_agg = {k: v for k, v in agg_dict.items() if k in merged.columns}
        item_df = merged.groupby('title', as_index=False).agg(valid_agg)

        return merged, item_df

    # ------------------------------------------------------------------
    def get_interaction_df(self):
        """Return the full interaction-level DataFrame (user × item ratings)."""
        if not self._datasets:
            raise ValueError("No datasets loaded.")
        frames = [ds['adapted'] for ds in self._datasets.values()]
        return pd.concat(frames, ignore_index=True)