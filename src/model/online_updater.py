"""Online updater for light-weight incremental model updates.

This module provides a small, dependency-tolerant helper `OnlineUpdater`
that can ingest single user-item interactions (ratings, reviews) and
update lightweight state used by the recommender without full retraining.

It performs safe operations:
- updates popularity counts and normalized popularity map
- updates rating and review_count aggregates (Bayesian smoothing)
- updates sentiment via `src.model.nlp_engine.batch_analyze` when available
- attempts to append new content embeddings to a `ContentRecommender`
- attempts to call `partial_fit` / `update` on collaborative models if present

This is a pragmatic, incremental pipeline for production setups that
want to reflect new feedback quickly while deferring heavy retraining.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

try:
    # optional sentiment helper
    from src.model.nlp_engine import analyze_sentiment
except Exception:
    analyze_sentiment = None  # type: ignore


def _bayesian_rating(rating: float, review_count: int, global_avg: float = 3.0, min_votes: int = 10) -> float:
    v = review_count
    m = min_votes
    C = global_avg
    return (v / (v + m)) * rating + (m / (v + m)) * C


class OnlineUpdater:
    """Incremental updater for content/collab/sentiment maps.

    Usage:
        updater = OnlineUpdater(content_model=content, collab_model=collab, hybrid=hybrid)
        updater.ingest_interaction(user_id, item_title, rating=4.0, review_text="Nice")
    """

    def __init__(
        self,
        content_model: Optional[object] = None,
        collab_model: Optional[object] = None,
        hybrid: Optional[object] = None,
        item_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.content_model = content_model
        self.collab_model = collab_model
        self.hybrid = hybrid
        self.item_df = item_df

        # lightweight internal counters if item_df not provided
        if item_df is None:
            self._counts = {}

    def ingest_interaction(
        self,
        user_id: str,
        item_title: str,
        rating: Optional[float] = None,
        review_text: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """Process a single interaction and update lightweight model state.

        Returns a dict describing which components were updated.
        """
        updated = {
            "popularity": False,
            "rating": False,
            "sentiment": False,
            "content_embedding": False,
            "collab": False,
        }

        # 1) Update item_df or local counters
        if self.item_df is not None and "title" in self.item_df.columns:
            # Find or create row
            mask = self.item_df["title"] == item_title
            if mask.any():
                idx = self.item_df[mask].index[0]
                row = self.item_df.loc[idx]
            else:
                # create minimal row
                idx = len(self.item_df)
                self.item_df.loc[idx] = {"title": item_title}
                row = self.item_df.loc[idx]

            # update review count
            rc = int(row.get("review_count", 0) or 0) + (1 if rating is not None or review_text else 0)
            self.item_df.at[idx, "review_count"] = rc

            # update rating aggregate if provided
            if rating is not None:
                prev_rating = float(row.get("rating", 0) or 0)
                # naive incremental average: assume rating is additional datapoint
                if prev_rating == 0:
                    new_avg = rating
                else:
                    new_avg = (prev_rating * (rc - 1) + rating) / rc
                self.item_df.at[idx, "rating"] = new_avg
                updated["rating"] = True

            # update popularity (views/purchases proxies)
            prev_views = int(row.get("views", 0) or 0)
            self.item_df.at[idx, "views"] = prev_views + 1
            updated["popularity"] = True

            # reflect into hybrid maps if present
            if self.hybrid is not None:
                try:
                    self.hybrid._review_count_map[item_title] = int(self.item_df.at[idx, "review_count"])
                    self.hybrid._rating_map[item_title] = _bayesian_rating(
                        float(self.item_df.at[idx, "rating"]), int(self.item_df.at[idx, "review_count"])
                    )
                    # update popularity_map normalized to max known views
                    if "views" in self.item_df.columns:
                        maxv = int(self.item_df["views"].max() or 1)
                        self.hybrid._popularity_map[item_title] = int(self.item_df.at[idx, "views"]) / maxv
                except Exception:
                    pass

        else:
            # fallback counter
            self._counts[item_title] = self._counts.get(item_title, 0) + 1
            updated["popularity"] = True

        # 2) Update sentiment if review_text available and analyzer present
        if review_text and analyze_sentiment is not None:
            try:
                s = analyze_sentiment(review_text)
                # analyzer returns dict with 'compound' or 'score'
                score = s.get("compound", s.get("score", 0.0))
                if self.hybrid is not None:
                    self.hybrid._sentiment_map[item_title] = (
                        (self.hybrid._sentiment_map.get(item_title, 0.0) + score) / 2.0
                    )
                updated["sentiment"] = True
            except Exception:
                pass

        # 3) Try to update content_model embeddings by appending new encoding
        if self.content_model is not None:
            try:
                # attempt to generate embedding using model.encode
                text = None
                if hasattr(self.content_model, "df") and "combined" in self.content_model.df.columns:
                    # find combined text for item if present
                    row = self.content_model.df[self.content_model.df["title"] == item_title]
                    if not row.empty:
                        text = row.iloc[0].get("combined")
                if text is None and review_text is not None:
                    text = review_text

                if text is not None and hasattr(self.content_model, "model") and hasattr(self.content_model.model, "encode"):
                    vec = np.asarray(self.content_model.model.encode([text], show_progress_bar=False))
                    # append to matrix
                    try:
                        self.content_model.matrix = np.vstack([self.content_model.matrix, vec])
                    except Exception:
                        # try to convert to dense then append
                        try:
                            cur = np.asarray(self.content_model.matrix)
                            self.content_model.matrix = np.vstack([cur, vec])
                        except Exception:
                            pass
                    # append row to df if missing
                    if hasattr(self.content_model, "df") and self.content_model.df[self.content_model.df["title"] == item_title].empty:
                        self.content_model.df = pd.concat([
                            self.content_model.df,
                            pd.DataFrame([{"title": item_title, "combined": text}])
                        ], ignore_index=True)
                    updated["content_embedding"] = True
            except Exception:
                pass

        # 4) Try to inform collaborative model of new interaction
        if self.collab_model is not None:
            try:
                # common incremental APIs
                if hasattr(self.collab_model, "partial_fit"):
                    # some implementations accept (user, item, rating)
                    try:
                        self.collab_model.partial_fit(user_id, item_title, rating)
                        updated["collab"] = True
                    except Exception:
                        # try with a single interaction dict
                        try:
                            self.collab_model.partial_fit([{"user_id": user_id, "title": item_title, "rating": rating}])
                            updated["collab"] = True
                        except Exception:
                            pass
                elif hasattr(self.collab_model, "update"):
                    try:
                        self.collab_model.update(user_id=user_id, item_title=item_title, rating=rating)
                        updated["collab"] = True
                    except Exception:
                        pass
                elif hasattr(self.collab_model, "df"):
                    # append to underlying interactions df if present
                    try:
                        self.collab_model.df = pd.concat([
                            self.collab_model.df,
                            pd.DataFrame([{"user_id": user_id, "title": item_title, "rating": rating}])
                        ], ignore_index=True)
                        updated["collab"] = True
                    except Exception:
                        pass
            except Exception:
                pass

        return updated

