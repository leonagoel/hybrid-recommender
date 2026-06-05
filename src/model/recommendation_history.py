from collections import defaultdict
from datetime import datetime


class RecommendationHistory:
    def __init__(self):
        self.history = defaultdict(list)

    def add_recommendation(self, user_id, item_title):
        self.history[user_id].append({
            "title": item_title,
            "timestamp": datetime.now()
        })

        # Keep only last 50 records
        self.history[user_id] = self.history[user_id][-50:]

    def get_history(self, user_id):
        return self.history.get(user_id, [])

    def get_recent_titles(self, user_id):
        return {
            item["title"]
            for item in self.history.get(user_id, [])
        }


history_tracker = RecommendationHistory()