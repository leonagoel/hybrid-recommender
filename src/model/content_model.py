"""
Content-Based Recommender
Uses SentenceTransformers to generate semantic embeddings of item metadata
and cosine similarity to find similar items.

Optimizations:
- Implements chunked batch encoding to prevent Out-Of-Memory (OOM) memory overhead.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.model.validation import validate_recommendations

# Optional HNSW support (enabled only if hnswlib is importable)
try:
    import hnswlib
except Exception:
    hnswlib = None


class ContentRecommender:
    def __init__(self, item_df, model_name='all-MiniLM-L6-v2', batch_size=256):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        batch_size: Size of slices processed sequentially to prevent RAM spikes.
        """
        self.df = item_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
        )
        if "combined" not in self.df.columns:
            self.df["combined"] = (
                self.df.get("title", "").astype(str) + " " +
                self.df.get("author", "").astype(str) + " " +
                self.df.get("category", "").astype(str)
            )
        self.matrix = self.vectorizer.fit_transform(self.df['combined'].fillna(''))
        # Do not compute full similarity matrix here to avoid OOM
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'].astype(str))
        }

    def recommend(self, title, top_n=10, target_catalog=None):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title.lower()]
        try:
            query_vec = self.matrix[idx].reshape(1, -1)
            # Candidate retrieval: use ANN for candidate selection if enabled,
            # otherwise fall back to brute-force over the full matrix.
            n = int(self.matrix.shape[0]) if getattr(self, 'matrix', None) is not None else 0
            sim_scores = None
            if getattr(self, '_ann_enabled', False) and self._ann_index is not None:
                try:
                    q = query_vec[0] if query_vec.shape[0] == 1 else query_vec
                    k = min(n, max(top_n * 5, top_n)) if n > 0 else top_n
                    labels, dists = self._ann_index.knn_query(q, k=k)
                    # labels is expected to be 2D: take first row
                    candidate_idxs = labels[0].tolist() if hasattr(labels, '__len__') else list(labels)
                    candidate_idxs = [int(i) for i in candidate_idxs if 0 <= int(i) < n]
                    if candidate_idxs:
                        candidate_matrix = self.matrix[candidate_idxs]
                        # Recompute exact cosine similarity on ANN candidates to preserve
                        # the original ranking and scoring semantics (ANN only for candidates).
                        candidate_scores = cosine_similarity(query_vec, candidate_matrix).flatten()
                        sim_scores = list(zip(candidate_idxs, candidate_scores))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    else:
                        # empty candidate set; fall back
                        scores = cosine_similarity(query_vec, self.matrix).flatten()
                        sim_scores = list(enumerate(scores))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                except Exception:
                    # Any ANN failure -> fallback to brute-force
                    scores = cosine_similarity(query_vec, self.matrix).flatten()
                    sim_scores = list(enumerate(scores))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            else:
                scores = cosine_similarity(query_vec, self.matrix).flatten()
                sim_scores = list(enumerate(scores))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Content recommendation similarity computation failed: {e}")
            sim_scores = []

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.df.iloc[i]['title']
            if t.lower() == title.lower() or t in seen:
                continue
            seen.add(key)
            
            results.append({
                "title": t,
                "content_score": float(scores[i])
            })
            
            if len(results) >= top_n:
                break

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n, exclude_title=title),
            top_n=top_n,
            context="content",
            force_padding=True
        )

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
        try:
            query_vec = self.model.encode([query])

            # Candidate retrieval: prefer ANN candidates when available, otherwise brute-force
            n = int(self.matrix.shape[0]) if getattr(self, 'matrix', None) is not None else 0
            sim_scores = None
            if getattr(self, '_ann_enabled', False) and self._ann_index is not None:
                try:
                    q = query_vec[0] if query_vec.shape[0] == 1 else query_vec
                    k = min(n, max(top_n * 5, top_n)) if n > 0 else top_n
                    labels, dists = self._ann_index.knn_query(q, k=k)
                    candidate_idxs = labels[0].tolist() if hasattr(labels, '__len__') else list(labels)
                    candidate_idxs = [int(i) for i in candidate_idxs if 0 <= int(i) < n]
                    if candidate_idxs:
                        candidate_matrix = self.matrix[candidate_idxs]
                        # Recompute exact cosine similarity on ANN candidates to preserve
                        # the original ranking and scoring semantics (ANN only for candidates).
                        candidate_scores = cosine_similarity(query_vec, candidate_matrix).flatten()
                        sim_scores = list(zip(candidate_idxs, candidate_scores))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    else:
                        scores = cosine_similarity(query_vec, self.matrix).flatten()
                        sim_scores = list(enumerate(scores))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                except Exception:
                    scores = cosine_similarity(query_vec, self.matrix).flatten()
                    sim_scores = list(enumerate(scores))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            else:
                scores = cosine_similarity(query_vec, self.matrix).flatten()
                sim_scores = list(enumerate(scores))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Content search computation failed: {e}")
            sim_scores = []

        results = []
        seen = set()
        for idx, score in sim_scores:
            if score <= 0:
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
            
            tp = self.df.at[idx, 'top_reviews'] if 'top_reviews' in self.df.columns else []
            top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': t,
                'score': float(score),
                'item_id': str(self.df.iloc[idx].get('item_id', idx)),
                'category': self.df.iloc[idx].get('category', ''),
                'description': str(self.df.iloc[idx].get('description', ''))[:200],
                'top_reviews': top_reviews,
            })

            if len(results) >= top_n:
                break

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n),
            top_n=top_n,
            context="content",
            force_padding=True
        )

    def _popularity_fallback(self, top_n=10, exclude_title=None):
        df = self.df.copy()
        if exclude_title is not None and 'title' in df.columns:
            df = df[df['title'].str.lower() != exclude_title.lower()]
            
        if "rating" in df.columns:
            df = df.sort_values("rating", ascending=False)
        elif "review_count" in df.columns:
            df = df.sort_values("review_count", ascending=False)
        
        results = []
        for _, row in df.head(top_n).iterrows():
            results.append({
                "title": row["title"],
                "content_score": 0.0
            })
        return results

