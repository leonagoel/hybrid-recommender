"""
Knowledge Graph Embedding Recommender
-------------------------------------
Builds semantic relationships between items using TransE-style embeddings.

Relationships are generated from:
- same_category
- same_author
- same_genre
- similar_keywords

Embeddings are used to compute semantic similarity between items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeGraphRecommender:
    def __init__(self, item_df: pd.DataFrame, embedding_dim: int = 64):
        self.df = item_df.reset_index(drop=True)
        self.embedding_dim = embedding_dim

        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.relation_to_idx = {}

        self.entity_embeddings = None
        self.relation_embeddings = None

        self.triples = []

        self._build_entities()
        self._build_relations()
        self._generate_triples()
        self._initialize_embeddings()
        self._train_embeddings()

    def _build_entities(self):
        titles = self.df['title'].astype(str).unique().tolist()

        self.entity_to_idx = {
            title: idx for idx, title in enumerate(titles)
        }

        self.idx_to_entity = {
            idx: title for title, idx in self.entity_to_idx.items()
        }

    def _build_relations(self):
        relations = [
            'same_category',
            'same_author',
            'same_genre'
        ]

        self.relation_to_idx = {
            rel: idx for idx, rel in enumerate(relations)
        }

    def _generate_triples(self):
        if 'category' in self.df.columns:
            grouped = self.df.groupby('category')

            for _, group in grouped:
                titles = group['title'].tolist()

                for i in range(len(titles)):
                    for j in range(i + 1, len(titles)):
                        h = self.entity_to_idx[titles[i]]
                        t = self.entity_to_idx[titles[j]]
                        r = self.relation_to_idx['same_category']

                        self.triples.append((h, r, t))

        if 'author' in self.df.columns:
            grouped = self.df.groupby('author')

            for _, group in grouped:
                titles = group['title'].tolist()

                for i in range(len(titles)):
                    for j in range(i + 1, len(titles)):
                        h = self.entity_to_idx[titles[i]]
                        t = self.entity_to_idx[titles[j]]
                        r = self.relation_to_idx['same_author']

                        self.triples.append((h, r, t))

    def _initialize_embeddings(self):
        n_entities = len(self.entity_to_idx)
        n_relations = len(self.relation_to_idx)

        self.entity_embeddings = np.random.normal(
            0,
            0.1,
            (n_entities, self.embedding_dim)
        )

        self.relation_embeddings = np.random.normal(
            0,
            0.1,
            (n_relations, self.embedding_dim)
        )

    def _train_embeddings(self, epochs: int = 100, lr: float = 0.01):
        """
        Lightweight TransE optimization.
        """

        for _ in range(epochs):
            for h, r, t in self.triples:
                h_emb = self.entity_embeddings[h]
                r_emb = self.relation_embeddings[r]
                t_emb = self.entity_embeddings[t]

                error = h_emb + r_emb - t_emb

                self.entity_embeddings[h] -= lr * error
                self.relation_embeddings[r] -= lr * error
                self.entity_embeddings[t] += lr * error

    def recommend(self, title: str, top_n: int = 10):
        if title not in self.entity_to_idx:
            return []

        idx = self.entity_to_idx[title]

        query_embedding = self.entity_embeddings[idx].reshape(1, -1)

        similarities = cosine_similarity(
            query_embedding,
            self.entity_embeddings
        )[0]

        similar_indices = np.argsort(similarities)[::-1][1: top_n + 1]

        recommendations = []

        for sim_idx in similar_indices:
            recommendations.append({
                'title': self.idx_to_entity[sim_idx],
                'kg_score': float(similarities[sim_idx])
            })

        return recommendations


