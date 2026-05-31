import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.knowledge_graph_model import KnowledgeGraphRecommender


class TestKnowledgeGraphRecommender:
    def test_init_with_dataframe(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B', 'Item C'],
            'category': ['cat1', 'cat1', 'cat2'],
            'author': ['Author X', 'Author X', 'Author Y']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=32)
        assert kg.embedding_dim == 32
        assert len(kg.entity_to_idx) == 3
        assert len(kg.relation_to_idx) == 3

    def test_build_entities(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'category': ['cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=16)
        assert 'Item A' in kg.entity_to_idx
        assert 'Item B' in kg.entity_to_idx
        assert kg.entity_to_idx['Item A'] == 0
        assert kg.entity_to_idx['Item B'] == 1

    def test_build_relations(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'category': ['cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df)
        assert 'same_category' in kg.relation_to_idx
        assert 'same_author' in kg.relation_to_idx
        assert 'same_genre' in kg.relation_to_idx

    def test_recommend_existing_item(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B', 'Item C'],
            'category': ['cat1', 'cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=32)
        recs = kg.recommend('Item A', top_n=2)
        assert len(recs) <= 2
        assert all('title' in r for r in recs)
        assert all('kg_score' in r for r in recs)

    def test_recommend_nonexistent_item(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'category': ['cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df)
        recs = kg.recommend('NonExistent', top_n=5)
        assert recs == []

    def test_recommend_top_n_bounds(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E'],
            'category': ['cat1', 'cat1', 'cat1', 'cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=32)
        recs = kg.recommend('Item A', top_n=10)
        assert len(recs) <= 4

    def test_entity_embeddings_shape(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'category': ['cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=32)
        assert kg.entity_embeddings.shape == (2, 32)

    def test_relation_embeddings_shape(self):
        df = pd.DataFrame({
            'title': ['Item A', 'Item B'],
            'category': ['cat1', 'cat2']
        })
        kg = KnowledgeGraphRecommender(df, embedding_dim=16)
        assert kg.relation_embeddings.shape == (3, 16)
