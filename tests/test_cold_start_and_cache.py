"""
Tests for cold-start recommendation endpoint and cache hit/miss metrics.
"""
import pandas as pd
from fastapi.testclient import TestClient

import backend.main as backend

client = TestClient(backend.app)


def setup_module():
    # Build a tiny item_df used by the cold-start helper
    data = [
        {'id': 1, 'title': 'Red Bicycle', 'description': 'A red bicycle for kids', 'category': 'Toys', 'rating': 4.5, 'review_count': 10},
        {'id': 2, 'title': 'Mountain Bike', 'description': 'An all-terrain mountain bike', 'category': 'Sports', 'rating': 4.7, 'review_count': 50},
        {'id': 3, 'title': 'Blue Bicycle', 'description': 'A blue bicycle with basket', 'category': 'Toys', 'rating': 4.2, 'review_count': 5},
    ]
    df = pd.DataFrame(data)
    df['combined'] = df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['category'].astype(str)
    backend.models['item_df'] = df
    backend.models['ready'] = True


def test_cold_start_endpoint_returns_recommendations():
    resp = client.get('/api/recommend/cold_start', params={'title': 'Kids Bike', 'description': 'red bike', 'top_n': 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 'recommendations' in body
    assert len(body['recommendations']) >= 1


def test_cache_metrics_hit_miss():
    # Clear cache and metrics
    backend._clear_response_cache()

    # Initially misses should be incremented when reading a non-existent key
    _ = backend._get_cached_response('nonexistent-key')
    before = client.get('/api/cache_metrics').json()
    assert before['misses'] >= 1

    # Set a cached value and then read it
    backend._set_cached_response('k1', {'ok': True})
    _ = backend._get_cached_response('k1')
    after = client.get('/api/cache_metrics').json()
    assert after['hits'] >= 1
