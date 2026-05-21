"""
Content-Based Recommender
Uses TF-IDF vectorization on item metadata (title + description + category)
and cosine similarity to find similar items.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, item_df):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        """
        self.df = item_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(self.df['combined'].fillna(''))
        # Do not compute full similarity matrix here to avoid OOM
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'])
        }

    def recommend(self, title, top_n=10):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title.lower()]
        query_vec = self.matrix[idx]
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.df.iloc[i]['title']
            if t == title or t in seen:
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
        """Return top TF-IDF terms shared by the source and candidate item."""
        if source_title not in self._title_to_idx or candidate_title not in self._title_to_idx:
            return []

        source_idx = self._title_to_idx[source_title]
        candidate_idx = self._title_to_idx[candidate_title]
        contributions = self.matrix[source_idx].multiply(self.matrix[candidate_idx]).toarray().ravel()
        if not np.any(contributions):
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = contributions.argsort()[::-1]
        terms = []
        for idx in top_indices:
            score = float(contributions[idx])
            if score <= 0:
                break
            terms.append({
                'term': str(feature_names[idx]),
                'score': round(score, 4),
            })
            if len(terms) >= top_n:
                break
        return terms

    def search(self, query, top_n=20):
        """
        Search items by query text using TF-IDF similarity.
        Returns list of matching item titles with scores.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        seen = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            t = self.df.iloc[idx]['title']
            if t in seen:
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
        return results
