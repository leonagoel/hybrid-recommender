"""knn_collaborative.py - KNN-based user collaborative filtering (Issue #51)"""
from __future__ import annotations
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class KNNCollaborativeRecommender:
    def __init__(self, interaction_df: pd.DataFrame, k: int = 10, min_common_items: int = 1):
        self.k = k
        self.min_common_items = min_common_items
        self._user_to_idx = {}
        self._idx_to_user = {}
        self._title_to_idx = {}
        self._idx_to_title = {}
        self._matrix = np.array([])
        self._fit(interaction_df)

    def _fit(self, df: pd.DataFrame) -> None:
        required = {"user_id", "title", "rating"}
        if not required.issubset(df.columns):
            raise ValueError(f"interaction_df missing columns: {required - set(df.columns)}")
        df = df.dropna(subset=["user_id", "title", "rating"]).copy()
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        users = sorted(df["user_id"].unique())
        titles = sorted(df["title"].unique())
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._idx_to_user = {i: u for u, i in self._user_to_idx.items()}
        self._title_to_idx = {t: i for i, t in enumerate(titles)}
        self._idx_to_title = {i: t for t, i in self._title_to_idx.items()}
        matrix = np.zeros((len(users), len(titles)), dtype=np.float32)
        for _, row in df.iterrows():
            matrix[self._user_to_idx[row["user_id"]], self._title_to_idx[row["title"]]] = float(row["rating"])
        self._matrix = matrix
        logger.info("KNN fitted: %d users, %d items", len(users), len(titles))

    def _user_similarity(self, user_vec: np.ndarray) -> np.ndarray:
        if self._matrix.shape[0] == 0:
            return np.array([])
        return cosine_similarity(user_vec.reshape(1, -1), self._matrix).flatten()

    def _top_k_similar_users(self, user_vec, exclude_indices):
        sims = self._user_similarity(user_vec)
        for idx in exclude_indices:
            if 0 <= idx < len(sims):
                sims[idx] = -1.0
        common = np.count_nonzero((self._matrix > 0) & (user_vec > 0), axis=1)
        sims[common < self.min_common_items] = -1.0
        top_k = np.argsort(sims)[::-1][:self.k]
        return [(int(i), float(sims[i])) for i in top_k if sims[i] > 0]

    def recommend(self, title: str, top_n: int = 10, user_id: Optional[str] = None, target_catalog: Optional[str] = None) -> list:
        if title not in self._title_to_idx and user_id not in self._user_to_idx:
            return []
        if user_id and user_id in self._user_to_idx:
            query_vec = self._matrix[self._user_to_idx[user_id]].copy()
            exclude = {self._user_to_idx[user_id]}
        elif title in self._title_to_idx:
            item_idx = self._title_to_idx[title]
            raters = np.where(self._matrix[:, item_idx] > 0)[0]
            if len(raters) == 0:
                return []
            query_vec = self._matrix[raters].mean(axis=0)
            exclude = set(raters.tolist())
        else:
            return []
        neighbours = self._top_k_similar_users(query_vec, exclude)
        if not neighbours:
            return self._popularity_fallback(top_n)
        item_scores = np.zeros(self._matrix.shape[1], dtype=np.float64)
        sim_sum = np.zeros(self._matrix.shape[1], dtype=np.float64)
        for n_idx, sim in neighbours:
            mask = self._matrix[n_idx] > 0
            item_scores[mask] += sim * self._matrix[n_idx][mask]
            sim_sum[mask] += sim
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted = np.where(sim_sum > 0, item_scores / sim_sum, 0.0)
        if title in self._title_to_idx:
            predicted[self._title_to_idx[title]] = 0.0
        if user_id and user_id in self._user_to_idx:
            predicted[np.where(self._matrix[self._user_to_idx[user_id]] > 0)[0]] = 0.0
        top_indices = np.argsort(predicted)[::-1][:top_n]
        max_score = float(predicted.max()) if predicted.max() > 0 else 1.0
        results = []
        for idx in top_indices:
            score = float(predicted[idx])
            if score <= 0:
                break
            results.append({"title": self._idx_to_title[idx], "collab_score": round(score / max_score, 4)})
        return results

    def _popularity_fallback(self, top_n: int) -> list:
        counts = np.count_nonzero(self._matrix, axis=0)
        top = np.argsort(counts)[::-1][:top_n]
        max_c = int(counts.max()) if counts.max() > 0 else 1
        return [{"title": self._idx_to_title[int(i)], "collab_score": round(counts[i] / max_c, 4)} for i in top if counts[i] > 0]
