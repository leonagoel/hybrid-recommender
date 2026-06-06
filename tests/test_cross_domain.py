import pytest
import pandas as pd
import numpy as np
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset_manager import DatasetManager
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender


@pytest.fixture
def multi_catalog_dataset_manager():
    """Create a DatasetManager with loaded Books and Movies catalogs."""
    dm = DatasetManager()
    
    # 1. Books Catalog
    books_df = pd.DataFrame({
        'title': ['Harry Potter', 'The Hobbit', 'Dune'],
        'description': ['Wizard boy fantasy', 'Hobbit ring fantasy', 'Sci-fi desert planet'],
        'category': ['Fantasy', 'Fantasy', 'Sci-Fi'],
        'rating': [4.8, 4.5, 4.6],
        'user_id': ['user_1', 'user_2', 'user_1'],
    })
    books_csv = books_df.to_csv(index=False)
    dm.load_csv(io.StringIO(books_csv), name='books.csv', catalog='Books')
    
    # 2. Movies Catalog
    movies_df = pd.DataFrame({
        'title': ['Star Wars', 'The Matrix', 'Avatar'],
        'description': ['Space opera sci-fi', 'Simulation sci-fi matrix', 'Blue aliens sci-fi'],
        'category': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi'],
        'rating': [4.7, 4.9, 4.3],
        'user_id': ['user_1', 'user_3', 'user_2'],
    })
    movies_csv = movies_df.to_csv(index=False)
    dm.load_csv(io.StringIO(movies_csv), name='movies.csv', catalog='Movies')
    
    return dm


def test_dataset_manager_catalog_tagging(multi_catalog_dataset_manager):
    """Verify that datasets are correctly tagged and stored with catalog metadata."""
    dm = multi_catalog_dataset_manager
    datasets = dm.list_datasets()
    
    assert len(datasets) == 2
    ds_values = list(dm._datasets.values())
    assert ds_values[0]['catalog'] == 'Books'
    assert ds_values[1]['catalog'] == 'Movies'


def test_dataset_manager_merge_aggregates_catalog(multi_catalog_dataset_manager):
    """Verify merged dataset aggregates the catalog column correctly."""
    dm = multi_catalog_dataset_manager
    merged, grouped = dm.merge_all()
    
    assert 'catalog' in merged.columns
    assert 'catalog' in grouped.columns
    
    # Ensure items belong to correct catalog
    assert grouped[grouped['title'] == 'Harry Potter']['catalog'].iloc[0] == 'Books'
    assert grouped[grouped['title'] == 'Star Wars']['catalog'].iloc[0] == 'Movies'


def test_content_recommender_catalog_filtering(multi_catalog_dataset_manager):
    """Verify that ContentRecommender filters results by target_catalog."""
    dm = multi_catalog_dataset_manager
    _, grouped = dm.merge_all()
    
    recommender = ContentRecommender(grouped)
    
    # Recommend across all catalogs (cross-domain)
    all_recs = recommender.recommend('Dune', top_n=5)
    all_titles = [r['title'] for r in all_recs]
    assert len(all_recs) > 0
    
    # Recommends cross-domain items (should find Star Wars / The Matrix / Avatar)
    has_movies = any(t in ['Star Wars', 'The Matrix', 'Avatar'] for t in all_titles)
    assert has_movies
    
    # Recommend filtered to Books domain only
    books_only = recommender.recommend('Dune', top_n=5, target_catalog='Books')
    for r in books_only:
        assert r['title'] in ['Harry Potter', 'The Hobbit']
        
    # Recommend filtered to Movies domain only
    movies_only = recommender.recommend('Dune', top_n=5, target_catalog='Movies')
    for r in movies_only:
        assert r['title'] in ['Star Wars', 'The Matrix', 'Avatar']


def test_collaborative_recommender_catalog_filtering(multi_catalog_dataset_manager):
    """Verify that CollaborativeRecommender filters recommendations by catalog."""
    dm = multi_catalog_dataset_manager
    merged, _ = dm.merge_all()
    
    recommender = CollaborativeRecommender(merged)
    
    # Personalised recommendations filtered to Books catalog
    books_only = recommender.predict_for_user('user_1', top_n=5, target_catalog='Books')
    for r in books_only:
        # User 1 already interacted with Harry Potter and Dune. The Hobbit is the remaining book.
        assert r['title'] == 'The Hobbit'
        
    # Personalised recommendations filtered to Movies catalog
    movies_only = recommender.predict_for_user('user_1', top_n=5, target_catalog='Movies')
    for r in movies_only:
        assert r['title'] in ['The Matrix', 'Avatar']


def test_hybrid_recommender_cross_domain_fallback(multi_catalog_dataset_manager):
    """Verify that HybridRecommender falls back safely to cold-start filtered by catalog."""
    dm = multi_catalog_dataset_manager
    merged, grouped = dm.merge_all()
    
    content = ContentRecommender(grouped)
    collab = CollaborativeRecommender(merged)
    hybrid = HybridRecommender(content, collab, grouped)
    
    # Recommend for a new/unseen title (triggers cold start) filtered by Books catalog
    recs = hybrid.recommend('Unknown Title', top_n=5, target_catalog='Books')
    assert len(recs) > 0
    for r in recs:
        assert r['title'] in ['Harry Potter', 'The Hobbit', 'Dune']
