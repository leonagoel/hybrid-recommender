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
    def __init__(
        self,
        item_df: pd.DataFrame,
        embedding_dim: int = 64,
        model_type: str = "TransE",   # NEW — accepts "TransE", "DistMult", "ComplEx"
        epochs: int = 100,             # NEW — was hardcoded before
        lr: float = 0.01,              # NEW — was hardcoded before
    ):
        if model_type not in ("TransE", "DistMult", "ComplEx"):
            raise ValueError(f"Unsupported model_type '{model_type}'.")
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
            'same_genre',
            'similar_keywords',
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

        if 'genre' in self.df.columns:
            grouped = self.df.groupby('genre')
            for _, group in grouped:
                titles = group['title'].astype(str).tolist()
                for i in range(len(titles)):
                    for j in range(i + 1, len(titles)):
                        h = self.entity_to_idx.get(titles[i])
                        t = self.entity_to_idx.get(titles[j])
                        if h is not None and t is not None:
                            r = self.relation_to_idx['same_genre']
                            self.triples.append((h, r, t))

        if 'keywords' in self.df.columns:
            kw_rel = self.relation_to_idx['similar_keywords']
            kw_series = self.df['keywords'].fillna('').astype(str)

            def _kw_set(s):
                return set(k.strip().lower() for k in s.split(',') if k.strip())

            kw_sets = [_kw_set(kw) for kw in kw_series]
            titles = self.df['title'].astype(str).tolist()

            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    if kw_sets[i] and kw_sets[j] and kw_sets[i] & kw_sets[j]:
                        h = self.entity_to_idx.get(titles[i])
                        t = self.entity_to_idx.get(titles[j])
                        if h is not None and t is not None:
                            self.triples.append((h, kw_rel, t))

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

        # ComplEx needs imaginary counterparts
        if self.model_type == "ComplEx":
            self.entity_embeddings_im = np.random.normal(0, 0.1, (n_entities, self.embedding_dim))
            self.relation_embeddings_im = np.random.normal(0, 0.1, (n_relations, self.embedding_dim))

    def get_embedding(self, title: str):
        """Return the learned embedding for an item. ComplEx returns real+imaginary concatenated."""
        if title not in self.entity_to_idx:
            return None
        idx = self.entity_to_idx[title]
        if self.model_type == "ComplEx":
            return np.concatenate([self.entity_embeddings[idx], self.entity_embeddings_im[idx]])
        return self.entity_embeddings[idx].copy()

    def recommend(self, title: str, top_n: int = 10):
        if title not in self.entity_to_idx:
            return []
        idx = self.entity_to_idx[title]

        if self.model_type == "ComplEx":
            all_embs = np.concatenate([self.entity_embeddings, self.entity_embeddings_im], axis=1)
            query_emb = all_embs[idx].reshape(1, -1)
            similarities = cosine_similarity(query_emb, all_embs)[0]
        else:
            query_emb = self.entity_embeddings[idx].reshape(1, -1)
            similarities = cosine_similarity(query_emb, self.entity_embeddings)[0]

        similar_indices = np.argsort(similarities)[::-1][1: top_n + 1]
        return [
            {'title': self.idx_to_entity[sim_idx], 'kg_score': float(similarities[sim_idx])}
            for sim_idx in similar_indices
        ]


