"""
Content-Based Recommender
Uses SentenceTransformers to generate semantic embeddings of item metadata
and cosine similarity to find similar items.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, item_df, model_name='all-MiniLM-L6-v2'):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        """
        self.df = item_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        
        # Generate embeddings for all items
        texts = self.df['combined'].fillna('').tolist()
        self.matrix = self.model.encode(texts, show_progress_bar=False)
        
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'])
        }

    def recommend(self, title, top_n=10, target_catalog=None):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title.lower()]
        query_vec = self.matrix[idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        
        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.df.iloc[i]['title']
            if t.lower() == title.lower() or t in seen:
                continue
            
            # Catalog filtering
            if target_catalog and 'catalog' in self.df.columns:
                item_catalog = self.df.iloc[i].get('catalog', '')
                if str(item_catalog).lower() != str(target_catalog).lower():
                    continue

            seen.add(t)
            results.append({
                'title': t,
                'content_score': float(score),
            })
            if len(results) >= top_n:
                break

        return results

    def explain_similarity(self, source_title, candidate_title, top_n=5):
        """
        Return a placeholder or basic explanation since dense vectors 
        don't have interpretable individual features like TF-IDF terms.
        """
        if source_title.lower() not in self._title_to_idx or candidate_title.lower() not in self._title_to_idx:
            return []

        source_idx = self._title_to_idx[source_title.lower()]
        candidate_idx = self._title_to_idx[candidate_title.lower()]
        
        score = cosine_similarity(
            self.matrix[source_idx].reshape(1, -1), 
            self.matrix[candidate_idx].reshape(1, -1)
        )[0][0]
        
        return [{'term': 'semantic_similarity', 'score': round(float(score), 4)}]

    def search(self, query, top_n=20, target_catalog=None):
        """
        Search items by query text using semantic similarity.
        Returns list of matching item titles with scores.
        """
        query_vec = self.model.encode([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Determine candidate indices matching similarity threshold or top N
        top_indices = scores.argsort()[::-1]

        results = []
        seen = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            t = self.df.iloc[idx]['title']
            if t in seen:
                continue

            # Catalog filtering
            if target_catalog and 'catalog' in self.df.columns:
                item_catalog = self.df.iloc[idx].get('catalog', '')
                if str(item_catalog).lower() != str(target_catalog).lower():
                    continue

            seen.add(t)
            
            tp = self.df.iloc[idx].get('top_reviews', [])
            top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': t,
                'score': float(scores[idx]),
                'item_id': str(self.df.iloc[idx].get('item_id', idx)),
                'category': self.df.iloc[idx].get('category', ''),
                'description': str(self.df.iloc[idx].get('description', ''))[:200],
                'top_reviews': top_reviews,
            })
            if len(results) >= top_n:
                break
        return results
