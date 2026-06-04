"""Lightweight OnlineUpdater for best-effort incremental updates.

This component is intentionally small and optional. It accepts incoming
interaction events and updates provided recommender structures where safe.

The design goals are:
- Best-effort updates (never raise on missing optional parts)
- No retraining or heavy computation
- Preserve backward compatibility
"""
from typing import Optional

import numpy as np

from src.model.hybrid_model import bayesian_rating


class OnlineUpdater:
    """A minimal online updater that can ingest interaction events.

    The `ingest` method accepts named parameters and, if a `recommender`
    instance is provided, will attempt to update its internal maps. The
    operation is best-effort and will not require or trigger retraining.
    """

    def __init__(self):
        # Placeholder for potential state (counters, buffers) later.
        self.buffer = []

    def ingest(self, user_id: Optional[str] = None, item_title: Optional[str] = None,
               rating: Optional[float] = None, sentiment: Optional[float] = None,
               timestamp: Optional[object] = None, recommender: Optional[object] = None):
        """Ingest a single interaction event.

        recommender: optional HybridRecommender instance; when provided, we
        update recommender internal maps in a best-effort manner.
        """
        # Record to an in-memory buffer for eventual flushing (not required)
        self.buffer.append({'user_id': user_id, 'title': item_title, 'rating': rating, 'sentiment': sentiment, 'timestamp': timestamp})

        if recommender is None or item_title is None:
            return True

        try:
            # 1) Review count
            prev = int(recommender._review_count_map.get(item_title, 0))
            new_count = prev + 1
            recommender._review_count_map[item_title] = new_count

            # 2) Popularity
            try:
                max_reviews = max(recommender._review_count_map.values()) if recommender._review_count_map else new_count
            except Exception:
                max_reviews = new_count
            recommender._popularity_map[item_title] = (new_count / max_reviews) if max_reviews > 0 else 0.0

            # 3) Rating
            if rating is not None:
                try:
                    prev_rating = float(recommender._rating_map.get(item_title, 0.0))
                    prev_n = prev if prev > 0 else 0
                    raw_avg = (prev_rating * prev_n + float(rating)) / (prev_n + 1) if (prev_n + 1) > 0 else float(rating)
                    try:
                        global_avg = float(np.mean(list(recommender._rating_map.values()))) if recommender._rating_map else 3.0
                    except Exception:
                        global_avg = 3.0
                    recommender._rating_map[item_title] = bayesian_rating(raw_avg, new_count, global_avg=global_avg)
                except Exception:
                    pass

            # 4) Sentiment
            if sentiment is not None:
                try:
                    prev_sent = recommender._sentiment_map.get(item_title)
                    if prev_sent is None:
                        recommender._sentiment_map[item_title] = float(sentiment)
                    else:
                        recommender._sentiment_map[item_title] = (float(prev_sent) * prev + float(sentiment)) / (prev + 1)
                except Exception:
                    pass

            # 5) Collaborative history best-effort
            try:
                if hasattr(recommender, 'collab_model') and recommender.collab_model is not None and hasattr(recommender.collab_model, 'df'):
                    try:
                        import pandas as pd
                        row = {'user_id': user_id, 'title': item_title}
                        if rating is not None:
                            row['rating'] = float(rating)
                        if timestamp is not None:
                            row['timestamp'] = timestamp
                        recommender.collab_model.df = pd.concat([recommender.collab_model.df, pd.DataFrame([row])], ignore_index=True)
                    except Exception:
                        pass
            except Exception:
                pass

            return True
        except Exception:
            return False
