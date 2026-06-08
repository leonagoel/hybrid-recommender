"""
Knowledge Graph Embeddings for Semantic Item Relationships.
Extracts triples from product attributes and user purchases,
trains a TransE model using NumPy mini-batch gradient descent,
and calculates semantic item similarity scores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional


class KnowledgeGraphRecommender:
    """
    Simulates a knowledge graph where entities are items, categories, or attributes,
    and relations describe connections (e.g. belongs_to_category, co_purchased_with).
    Learns embeddings using the TransE (h + r ~ t) translation model in pure NumPy.
    """

    def __init__(
        self,
        item_df: pd.DataFrame,
        purchases_df: Optional[pd.DataFrame] = None,
        n_factors: int = 20,
        margin: float = 1.0,
        lr: float = 0.05,
        epochs: int = 5,
        batch_size: int = 64
    ):
        """
        item_df: DataFrame of catalog items with 'title', 'category', 'rating', 'avg_sentiment'
        purchases_df: Optional purchases data containing 'user_id', 'product_id'
        n_factors: Number of latent embedding dimensions
        margin: Ranking loss margin (gamma)
        lr: Learning rate for SGD updates
        epochs: Number of training epochs
        batch_size: Mini-batch size
        """
        self.item_df = item_df.copy()
        self.purchases_df = purchases_df.copy() if purchases_df is not None else None
        self.n_factors = n_factors
        self.margin = margin
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.triples: List[Tuple[str, str, str]] = []
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}

        self.entity_embeddings = None
        self.relation_embeddings = None

        self._build_graph()
        self._train()

    def _build_graph(self):
        triples = []

        # 1. Map categories, ratings, and sentiment relations
        global_avg = self.item_df['rating'].mean() if 'rating' in self.item_df.columns and len(self.item_df) > 0 else 3.0

        for _, row in self.item_df.iterrows():
            title = row['title']
            category = row.get('category')
            rating = row.get('rating', 0.0)
            avg_sentiment = row.get('avg_sentiment', 0.0)

            # Category relation: (Item, belongs_to_category, Category)
            if pd.notna(category) and str(category).strip():
                triples.append((title, "belongs_to_category", str(category).strip()))

            # Rating level: (Item, has_rating_level, level)
            if pd.notna(rating):
                if rating >= 4.0:
                    r_level = "high_rating"
                elif rating >= 3.0:
                    r_level = "mid_rating"
                else:
                    r_level = "low_rating"
                triples.append((title, "has_rating_level", r_level))

            # Sentiment level: (Item, has_sentiment_level, level)
            if pd.notna(avg_sentiment):
                if avg_sentiment >= 0.1:
                    s_level = "positive_sentiment"
                elif avg_sentiment <= -0.1:
                    s_level = "negative_sentiment"
                else:
                    s_level = "neutral_sentiment"
                triples.append((title, "has_sentiment_level", s_level))

        # 2. Co-purchase relations: (ItemA, co_purchased_with, ItemB)
        if self.purchases_df is not None and not self.purchases_df.empty:
            product_id_to_title = {}
            if 'id' in self.item_df.columns:
                product_id_to_title = dict(zip(self.item_df['id'], self.item_df['title']))

            # Group by user to find items co-purchased together
            user_groups = self.purchases_df.groupby('user_id')
            for _, group in user_groups:
                p_ids = group['product_id'].dropna().tolist()
                titles = [product_id_to_title[pid] for pid in p_ids if pid in product_id_to_title]
                # Cap list size to prevent quadratic scaling
                titles = titles[:15]
                for i in range(len(titles)):
                    for j in range(i + 1, len(titles)):
                        triples.append((titles[i], "co_purchased_with", titles[j]))
                        triples.append((titles[j], "co_purchased_with", titles[i]))

        self.triples = triples

        # Extract entities and relations
        entities = set()
        relations = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        self.entity_to_idx = {e: i for i, e in enumerate(sorted(entities))}
        self.idx_to_entity = {i: e for e, i in self.entity_to_idx.items()}
        self.relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}
        self.idx_to_relation = {i: r for r, i in self.relation_to_idx.items()}

    def _train(self):
        if not self.triples:
            # Fallback for empty graph
            self.entity_embeddings = np.zeros((0, self.n_factors))
            self.relation_embeddings = np.zeros((0, self.n_factors))
            return

        n_entities = len(self.entity_to_idx)
        n_relations = len(self.relation_to_idx)

        # Initialize embeddings randomly
        np.random.seed(42)
        self.entity_embeddings = np.random.uniform(
            -6.0 / np.sqrt(self.n_factors),
            6.0 / np.sqrt(self.n_factors),
            size=(n_entities, self.n_factors)
        )
        self.relation_embeddings = np.random.uniform(
            -6.0 / np.sqrt(self.n_factors),
            6.0 / np.sqrt(self.n_factors),
            size=(n_relations, self.n_factors)
        )

        # Normalize entity vectors to unit sphere
        self.entity_embeddings /= np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)

        triples_idx = [
            (self.entity_to_idx[h], self.relation_to_idx[r], self.entity_to_idx[t])
            for h, r, t in self.triples
        ]

        # Optimize using SGD
        for _ in range(self.epochs):
            np.random.shuffle(triples_idx)

            for start in range(0, len(triples_idx), self.batch_size):
                batch = triples_idx[start:start + self.batch_size]

                for h, r, t in batch:
                    # Corrupt head or tail to create a negative triple
                    h_prime, t_prime = h, t
                    if np.random.rand() < 0.5:
                        h_prime = np.random.randint(0, n_entities)
                        while h_prime == h and n_entities > 1:
                            h_prime = np.random.randint(0, n_entities)
                    else:
                        t_prime = np.random.randint(0, n_entities)
                        while t_prime == t and n_entities > 1:
                            t_prime = np.random.randint(0, n_entities)

                    # Compute TransE loss
                    emb_h = self.entity_embeddings[h]
                    emb_r = self.relation_embeddings[r]
                    emb_t = self.entity_embeddings[t]

                    emb_h_prime = self.entity_embeddings[h_prime]
                    emb_t_prime = self.entity_embeddings[t_prime]

                    d_pos = np.sum((emb_h + emb_r - emb_t) ** 2)
                    d_neg = np.sum((emb_h_prime + emb_r - emb_t_prime) ** 2)

                    loss = self.margin + d_pos - d_neg
                    if loss > 0:
                        # Gradient of positive triple
                        grad_pos = 2 * (emb_h + emb_r - emb_t)
                        # Gradient of negative triple
                        grad_neg = 2 * (emb_h_prime + emb_r - emb_t_prime)

                        # Update positive entities and relation
                        self.entity_embeddings[h] -= self.lr * grad_pos
                        self.entity_embeddings[t] += self.lr * grad_pos
                        self.relation_embeddings[r] -= self.lr * (grad_pos - grad_neg)

                        # Update corrupted entities
                        self.entity_embeddings[h_prime] += self.lr * grad_neg
                        self.entity_embeddings[t_prime] -= self.lr * grad_neg

                        # Renormalize updated entities to keep them on unit sphere
                        self.entity_embeddings[h] /= np.linalg.norm(self.entity_embeddings[h])
                        self.entity_embeddings[t] /= np.linalg.norm(self.entity_embeddings[t])
                        self.entity_embeddings[h_prime] /= np.linalg.norm(self.entity_embeddings[h_prime])
                        self.entity_embeddings[t_prime] /= np.linalg.norm(self.entity_embeddings[t_prime])

    def recommend(self, title: str, top_n: int = 10, target_catalog: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get semantic KG similarity recommendations for a given item title.
        Returns: list of dicts: [{'title': str, 'kg_score': float}]
        """
        if title not in self.entity_to_idx:
            return []

        h_idx = self.entity_to_idx[title]
        h_emb = self.entity_embeddings[h_idx]

        scores = []
        valid_items = self.item_df

        if target_catalog and 'catalog' in valid_items.columns:
            valid_items = valid_items[valid_items['catalog'].str.lower() == target_catalog.lower()]

        for _, row in valid_items.iterrows():
            item_title = row['title']
            if item_title == title or item_title not in self.entity_to_idx:
                continue

            t_idx = self.entity_to_idx[item_title]
            t_emb = self.entity_embeddings[t_idx]

            dot_product = np.dot(h_emb, t_emb)
            norm_h = np.linalg.norm(h_emb)
            norm_t = np.linalg.norm(t_emb)

            if norm_h > 0 and norm_t > 0:
                sim = dot_product / (norm_h * norm_t)
            else:
                sim = 0.0

            scores.append((item_title, float(sim)))

        # Sort by similarity score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [{'title': t, 'kg_score': score} for t, score in scores[:top_n]]
