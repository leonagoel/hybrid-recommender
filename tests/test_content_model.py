import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestContentRecommender:
    def test_init_without_model(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'combined': ['Description A', 'Description B']
        })
        with patch('src.model.content_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
            mock_st.return_value = mock_model
            from src.model.content_model import ContentRecommender
            recommender = ContentRecommender(df, batch_size=256)
            assert recommender.df is not None
            assert recommender.matrix.shape[0] == 2

    def test_recommend_item_not_found(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'combined': ['Description A', 'Description B']
        })
        with patch('src.model.content_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
            mock_st.return_value = mock_model
            from src.model.content_model import ContentRecommender
            recommender = ContentRecommender(df)
            results = recommender.recommend("NonExistent Item")
            assert results == []

    def test_recommend_basic(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B', 'Item C'],
            'combined': ['Description A', 'Description B', 'Description C']
        })
        with patch('src.model.content_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.eye(3, 384).astype(np.float32)
            mock_st.return_value = mock_model
            from src.model.content_model import ContentRecommender
            recommender = ContentRecommender(df)
            results = recommender.recommend("Item A", top_n=2)
            assert isinstance(results, list)

    def test_explain_similarity(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'combined': ['Description A', 'Description B']
        })
        with patch('src.model.content_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.eye(2, 384).astype(np.float32)
            mock_st.return_value = mock_model
            from src.model.content_model import ContentRecommender
            recommender = ContentRecommender(df)
            explanation = recommender.explain_similarity("Item A", "Item B")
            assert isinstance(explanation, list)

    def test_explain_similarity_not_found(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'combined': ['Description A', 'Description B']
        })
        with patch('src.model.content_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.eye(2, 384).astype(np.float32)
            mock_st.return_value = mock_model
            from src.model.content_model import ContentRecommender
            recommender = ContentRecommender(df)
            explanation = recommender.explain_similarity("Item A", "NonExistent")
            assert explanation == []
