"""
Cold-start recommendation handler.

Solves the cold-start problem for new users without rating history using:
- Onboarding survey for initial preferences
- Popular/trending items for discovery
- Demographic-based filtering
- Hybrid bootstrap approach
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """
    Handles recommendations for new users without rating history.
    Uses hybrid approach combining multiple signals.
    """

    def __init__(self, item_df: pd.DataFrame, interaction_df: pd.DataFrame):
        """
        Initialize cold-start handler with item and interaction data.

        Args:
            item_df: DataFrame with columns 'title', 'genres', 'popularity', 'avg_rating'
            interaction_df: DataFrame with columns 'user_id', 'title', 'rating'
        """
        self.item_df = item_df.copy()
        self.interaction_df = interaction_df.copy()

        self._compute_item_popularity()
        self._compute_demographic_clusters()

    def _compute_item_popularity(self) -> None:
        """Compute popularity score for trending items."""
        if 'popularity' not in self.item_df.columns:
            item_counts = self.interaction_df['title'].value_counts()
            self.item_df['popularity'] = self.item_df['title'].map(item_counts).fillna(0)
        else:
            self.item_df['popularity'] = self.item_df['popularity'].fillna(0)

        self.popular_items = (
            self.item_df.nlargest(20, 'popularity')[['title', 'popularity', 'avg_rating']]
            .to_dict('records')
        )

    def _compute_demographic_clusters(self) -> None:
        """Group users by preference patterns for demographic-based recommendations."""
        self.demographic_profiles = {}

        if 'genres' in self.interaction_df.columns:
            genre_groups = self.interaction_df.groupby('genres')['title'].apply(list).to_dict()
            for genre, items in genre_groups.items():
                self.demographic_profiles[genre] = items

    def get_onboarding_survey(self) -> Dict[str, Any]:
        """
        Generate onboarding survey for new users.
        Returns top genres and items for preference collection.
        """
        top_genres = []
        if 'genres' in self.item_df.columns:
            genre_counts = self.item_df['genres'].value_counts().head(10)
            top_genres = genre_counts.index.tolist()

        top_items = self.item_df.nlargest(10, 'avg_rating')[['title', 'genres', 'avg_rating']].to_dict('records')

        return {
            'survey_type': 'onboarding',
            'genres': top_genres,
            'sample_items': top_items,
            'instructions': 'Please rate these items or select your preferred genres to get personalized recommendations',
        }

    def get_popular_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending/popular items for new user discovery.

        Args:
            limit: Number of items to return

        Returns:
            List of popular items with scores
        """
        return self.popular_items[:limit]

    def get_demographic_recommendations(self, user_preferences: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recommendations based on demographic similarity.

        Args:
            user_preferences: User's stated preferences (genres, language, etc)
            limit: Number of items to return

        Returns:
            List of demographically similar items
        """
        recommendations = []

        preferred_genres = user_preferences.get('genres', [])
        if preferred_genres:
            for genre in preferred_genres:
                if genre in self.demographic_profiles:
                    recommendations.extend(self.demographic_profiles[genre])

        if not recommendations:
            recommendations = [item['title'] for item in self.popular_items]

        recommendations = list(set(recommendations))[:limit]

        return [
            {
                'title': title,
                'reason': 'Matches your preferred genres',
                'cold_start_score': 0.7,
            }
            for title in recommendations
        ]

    def bootstrap_user_profile(self, user_id: str, onboarding_ratings: Dict[str, float]) -> None:
        """
        Bootstrap user profile from onboarding survey ratings.
        Adds initial interactions to enable collaborative filtering.

        Args:
            user_id: New user identifier
            onboarding_ratings: Dict of {item_title: rating} from onboarding
        """
        if not onboarding_ratings:
            logger.warning(f'No onboarding ratings provided for user {user_id}')
            return

        new_rows = [
            {
                'user_id': user_id,
                'title': title,
                'rating': rating,
            }
            for title, rating in onboarding_ratings.items()
        ]

        self.interaction_df = pd.concat(
            [self.interaction_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )
        logger.info(f'Bootstrapped user {user_id} with {len(onboarding_ratings)} initial ratings')

    def recommend_for_new_user(self, user_preferences: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get hybrid recommendations for new user.
        Combines popular items, demographic-based, and random exploration.

        Args:
            user_preferences: User's stated preferences from onboarding
            limit: Number of items to return

        Returns:
            List of hybrid recommendations
        """
        recommendations = []

        popular = self.get_popular_items(limit=int(limit * 0.4))
        demographic = self.get_demographic_recommendations(user_preferences, limit=int(limit * 0.5))
        random_exploration = self._get_random_exploration(limit=int(limit * 0.1))

        for i, item in enumerate(popular):
            item['weight'] = 0.4
            item['reason'] = 'Popular with users like you'
            recommendations.append(item)

        for i, item in enumerate(demographic):
            item['weight'] = 0.5
            recommendations.append(item)

        recommendations.extend(random_exploration)

        return sorted(recommendations, key=lambda x: x.get('weight', 0), reverse=True)[:limit]

    def _get_random_exploration(self, limit: int = 2) -> List[Dict[str, Any]]:
        """Get random items for exploration and discovery."""
        sample_size = min(limit, len(self.item_df))
        random_items = self.item_df.sample(n=sample_size)

        return [
            {
                'title': row['title'],
                'reason': 'Recommended for you to explore',
                'weight': 0.1,
                'cold_start_score': 0.3,
            }
            for _, row in random_items.iterrows()
        ]
