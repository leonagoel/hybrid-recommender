"""
Content-Based Recommender
Uses SentenceTransformers to generate semantic embeddings of item metadata
and cosine similarity to find similar items.

Optimizations:
- Implements chunked batch encoding to prevent Out-Of-Memory (OOM) memory overhead.
- Implements HNSW approximate nearest neighbor index for sub-linear search (Issue #315).
  Falls back to brute-force cosine similarity when hnswlib is not installed.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False


class ContentRecommender:
    def __init__(self, item_df, model_name='all-MiniLM-L6-v2', batch_size=256):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        batch_size: Size of slices processed sequentially to prevent RAM spikes.
        """
        self.df = item_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)

        # Generate embeddings using optimized sequential batching
        texts = self.df['combined'].fillna('').tolist()

        # FIX FOR ISSUE #485: Process text slices sequentially to prevent massive host RAM peaks
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_encodings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings_list.append(batch_encodings)

        # Stack slices cleanly into a single final continuous array allocation
        self.matrix = np.vstack(embeddings_list) if embeddings_list else np.empty((0, 0))

        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'])
        }

        # Build HNSW index for approximate nearest neighbor search (Issue #315)
        self._hnsw_index = None
        if _HNSWLIB_AVAILABLE and self.matrix.ndim == 2 and self.matrix.shape[0] > 0:
            dim = self.matrix.shape[1]
            index = hnswlib.Index(space='cosine', dim=dim)
            index.init_index(
                max_elements=self.matrix.shape[0],
                ef_construction=200,
                M=16,
            )
            index.add_items(
                self.matrix.astype('float32'),
                list(range(self.matrix.shape[0])),
            )
            index.set_ef(50)
            self._hnsw_index = index

    def recommend(self, title, top_n=10, target_catalog=None):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title.lower()]
        query_vec = self.matrix[idx].reshape(1, -1)

        if self._hnsw_index is not None:
            k = min(top_n * 3 + 1, self._hnsw_index.element_count)
            labels, distances = self._hnsw_index.knn_query(query_vec.astype('float32'), k=k)
            sim_scores = [
                (int(labels[0][j]), float(1.0 - distances[0][j]))
                for j in range(len(labels[0]))
            ]
        else:
            scores = cosine_similarity(query_vec, self.matrix).flatten()
            sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

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

        if self._hnsw_index is not None:
            k = min(top_n * 3, self._hnsw_index.element_count)
            labels, distances = self._hnsw_index.knn_query(
                query_vec.astype('float32'), k=max(1, k)
            )
            hnsw_scores = {
                int(labels[0][j]): float(1.0 - distances[0][j])
                for j in range(len(labels[0]))
            }
            top_indices = [i for i, _ in sorted(hnsw_scores.items(), key=lambda x: x[1], reverse=True)]
        else:
            scores_arr = cosine_similarity(query_vec, self.matrix).flatten()
            hnsw_scores = None
            top_indices = scores_arr.argsort()[::-1]

        results = []
        seen = set()
        for idx in top_indices:
            score = hnsw_scores[idx] if hnsw_scores is not None else float(scores_arr[idx])
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

            tp = self.df.iloc[idx].get('top_reviews', [])
            top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': t,
                'score': score,
                'item_id': str(self.df.iloc[idx].get('item_id', idx)),
                'category': self.df.iloc[idx].get('category', ''),
                'description': str(self.df.iloc[idx].get('description', ''))[:200],
                'top_reviews': top_reviews,
            })
            if len(results) >= top_n:
                break

        return results

