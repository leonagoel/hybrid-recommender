"""
Hybrid Recommender
Combines content-based, collaborative, and NLP sentiment scores
using a weighted scoring system with normalization and re-ranking.

Improvements:
- Bayesian average rating to prevent rating bias
- Popularity-based cold start fallback
- Category warm-start for new users
- Better weight redistribution
"""
import numpy as np


def bayesian_rating(rating, review_count, global_avg=3.0, min_votes=10):
    """
    Bayesian average: smooths ratings toward the global mean.
    Items with few votes get pulled toward the average.
    """
    v = review_count
    m = min_votes
    C = global_avg
    return (v / (v + m)) * rating + (m / (v + m)) * C


class HybridRecommender:
    def __init__(self, content_model, collab_model=None, item_df=None,
                 alpha=0.4, beta=0.35, gamma=0.25):
        """
        content_model:  ContentRecommender instance
        collab_model:   CollaborativeRecommender instance (optional)
        item_df:        DataFrame with 'avg_sentiment', 'rating', 'review_count' columns
        alpha:          weight for content-based score
        beta:           weight for collaborative score
        gamma:          weight for sentiment score
        """
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Build sentiment + rating lookups
        self._sentiment_map = {}
        self._rating_map = {}
        self._review_count_map = {}
        self._category_map = {}
        self._popularity_map = {}

        if item_df is not None:
            global_avg = item_df['rating'].mean() if 'rating' in item_df.columns else 3.0

            for _, row in item_df.iterrows():
                title = row['title']
                if 'avg_sentiment' in item_df.columns:
                    self._sentiment_map[title] = row['avg_sentiment']

                raw_rating = float(row.get('rating', 0))
                review_count = row.get('review_count', 0)

                if np.isnan(review_count):
                    review_count = 0

                review_count = int(review_count)
                self._review_count_map[title] = review_count
                self._rating_map[title] = bayesian_rating(
                    raw_rating, review_count, global_avg
                )
                self._category_map[title] = row.get('category', '')

            # Popularity rank (0-1 scale, higher = more popular)
            if 'review_count' in item_df.columns:
                max_reviews = item_df['review_count'].max()
                if max_reviews > 0:
                    for _, row in item_df.iterrows():
                        self._popularity_map[row['title']] = (
                            row['review_count'] / max_reviews
                        )

    def set_weights(self, alpha, beta, gamma):
        """Update the scoring weights. Normalized to sum to 1."""
        total = alpha + beta + gamma
        if total == 0:
            total = 1
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total

    def get_weights(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}

    def _normalize(self, scores):
        """Min-max normalize a list of scores to [0, 1]."""
        if not scores:
            return scores
        mn, mx = min(scores), max(scores)
        if mx - mn == 0:
            return [0.5] * len(scores)
        return [(v - mn) / (mx - mn) for v in scores]

    def recommend(self, title, user_id=None, top_n=10, explain=False):
        """
        Get hybrid recommendations for a given item title.
        Returns list of dicts sorted by hybrid_score.
        """
        # 1. Content-based scores
        content_recs = self.content_model.recommend(title, top_n=top_n * 3)
        all_titles = {r['title'] for r in content_recs}

        # 2. Collaborative scores
        collab_map = {}
        if self.collab_model:
            collab_recs = self.collab_model.recommend(title, top_n=top_n * 3)
            for r in collab_recs:
                collab_map[r['title']] = r['collab_score']
                all_titles.add(r['title'])

        # 3. Build unified candidates
        candidates = {}
        for r in content_recs:
            candidates[r['title']] = {
                'title': r['title'],
                'raw_content': r['content_score'],
                'raw_collab': collab_map.get(r['title'], 0.0),
                'raw_sentiment': self._sentiment_map.get(r['title'], 0.0),
            }

        for t in collab_map:
            if t not in candidates:
                candidates[t] = {
                    'title': t,
                    'raw_content': 0.0,
                    'raw_collab': collab_map[t],
                    'raw_sentiment': self._sentiment_map.get(t, 0.0),
                }

        if not candidates:
            return self._cold_start_fallback(title, top_n)

        items = list(candidates.values())

        # 4. Normalize each component
        content_scores = self._normalize([it['raw_content'] for it in items])
        collab_scores = self._normalize([it['raw_collab'] for it in items])
        sentiment_scores = [(it['raw_sentiment'] + 1) / 2 for it in items]

        # 5. Determine active weights dynamically
        a, b, g = self.alpha, self.beta, self.gamma
        
        if user_id and self.collab_model and user_id in self.collab_model._user_to_idx:
            user_interacts = len(self.collab_model.df[self.collab_model.df['user_id'] == user_id])
            if user_interacts > 10:
                b += 0.2  # high confidence in collaborative
            elif user_interacts < 3:
                a += 0.2  # fallback to content for cold users
                
        if self.collab_model is None:
            b = 0
        if not self._sentiment_map:
            g = 0
            
        total = a + b + g
        if total > 0:
            a, b, g = a / total, b / total, g / total

        # 6. Compute hybrid score with popularity boost
        results = []
        for i, item in enumerate(items):
            hybrid = (
                a * content_scores[i] +
                b * collab_scores[i] +
                g * sentiment_scores[i]
            )

            # Light popularity boost (max 5% bonus)
            popularity = self._popularity_map.get(item['title'], 0.5)
            hybrid += 0.05 * popularity

            # Lookup info from content model's df
            row_data = self.content_model.df[
                self.content_model.df['title'] == item['title']
            ]
            avg_rating = self._rating_map.get(item['title'], 0.0)
            category = self._category_map.get(item['title'], '')
            description = ''
            top_reviews = []
            if len(row_data) > 0:
                description = str(row_data.iloc[0].get('description', ''))[:200]
                tp = row_data.iloc[0].get('top_reviews', [])
                top_reviews = tp if isinstance(tp, list) else []

            result = {
                'title': item['title'],
                'content_score': round(content_scores[i], 4),
                'collab_score': round(collab_scores[i], 4),
                'sentiment_score': round(sentiment_scores[i], 4),
                'hybrid_score': round(hybrid, 4),
                'rating': round(avg_rating, 2),
                'category': category,
                'description': description,
                'top_reviews': top_reviews,
            }
            if explain:
                result['explanation'] = self._build_explanation(
                    title,
                    item['title'],
                    content_scores[i],
                    collab_scores[i],
                    sentiment_scores[i],
                    popularity,
                    a,
                    b,
                    g,
                    item,
                )
            results.append(result)

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_n]

    def _build_explanation(
        self,
        source_title,
        candidate_title,
        content_score,
        collab_score,
        sentiment_score,
        popularity,
        alpha,
        beta,
        gamma,
        raw_item,
    ):
        content_terms = []
        if hasattr(self.content_model, 'explain_similarity'):
            content_terms = self.content_model.explain_similarity(source_title, candidate_title)

        weighted_components = {
            'content': round(alpha * content_score, 4),
            'collaborative': round(beta * collab_score, 4),
            'sentiment': round(gamma * sentiment_score, 4),
            'popularity_bonus': round(0.05 * popularity, 4),
        }
        strongest = max(weighted_components, key=weighted_components.get)

        return {
            'source_item': source_title,
            'candidate_item': candidate_title,
            'active_weights': {
                'alpha': round(alpha, 4),
                'beta': round(beta, 4),
                'gamma': round(gamma, 4),
            },
            'component_scores': {
                'content': round(content_score, 4),
                'collaborative': round(collab_score, 4),
                'sentiment': round(sentiment_score, 4),
                'raw_content': round(raw_item['raw_content'], 4),
                'raw_collaborative': round(raw_item['raw_collab'], 4),
                'raw_sentiment': round(raw_item['raw_sentiment'], 4),
            },
            'weighted_components': weighted_components,
            'top_content_terms': content_terms,
            'signals': {
                'strongest_component': strongest,
                'collaborative_match': raw_item['raw_collab'] > 0,
                'sentiment_polarity': self._sentiment_label(raw_item['raw_sentiment']),
                'popularity': round(popularity, 4),
            },
        }

    @staticmethod
    def _sentiment_label(score):
        if score > 0.2:
            return 'positive'
        if score < -0.2:
            return 'negative'
        return 'neutral'

    def _cold_start_fallback(self, title, top_n):
        """
        Fallback when no model data exists for the title.
        Returns popular items from the same category.
        """
        if self.item_df is None:
            return []

        target_cat = self._category_map.get(title, '')
        df = self.item_df

        if target_cat:
            cat_items = df[df['category'] == target_cat]
            if len(cat_items) >= top_n:
                df = cat_items

        # Sort by Bayesian rating
        if 'rating' in df.columns and 'review_count' in df.columns:
            df = df.copy()
            df['_bayesian'] = df.apply(
                lambda r: bayesian_rating(r['rating'], r.get('review_count', 0)), axis=1
            )
            df = df.sort_values('_bayesian', ascending=False)
        elif 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False)

        results = []
        for _, row in df.head(top_n).iterrows():
            results.append({
                'title': row['title'],
                'content_score': 0.0,
                'collab_score': 0.0,
                'sentiment_score': (row.get('avg_sentiment', 0) + 1) / 2,
                'hybrid_score': round(self._rating_map.get(row['title'], 0) / 5, 4),
                'rating': round(float(row.get('rating', 0)), 2),
                'category': row.get('category', ''),
                'description': str(row.get('description', ''))[:200],
                'top_reviews': [],
            })
        return results
