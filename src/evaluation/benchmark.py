"""
Benchmarking Pipeline for Hybrid Recommender
Compares baseline models against the Semantic-Hybrid approach.
"""
import os
import sys
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.evaluation import ndcg_at_k, average_precision_at_k, _build_test_data

class RandomRecommender:
    def __init__(self, item_titles):
        self.items = list(item_titles)
        
    def recommend(self, title, user_id=None, top_n=10):
        recs = random.sample(self.items, min(top_n, len(self.items)))
        return [{'title': t} for t in recs]

class PopularityRecommender:
    def __init__(self, item_df):
        if 'rating' in item_df.columns:
            self.popular_items = item_df.sort_values('rating', ascending=False)['title'].tolist()
        else:
            self.popular_items = item_df['title'].tolist()
        
    def recommend(self, title, user_id=None, top_n=10):
        return [{'title': t} for t in self.popular_items[:top_n]]

def run_benchmark():
    print("Building test data and base models...")
    content_model, collab_model, item_df, test_pairs = _build_test_data()
    
    if not test_pairs:
        print("Not enough data to run benchmark.")
        return
        
    from src.model.hybrid_model import HybridRecommender
    hybrid_model = HybridRecommender(content_model, collab_model, item_df, alpha=0.4, beta=0.4, gamma=0.2)
    random_model = RandomRecommender(item_df['title'].unique())
    popular_model = PopularityRecommender(item_df)
    
    models = {
        "Random": random_model,
        "Popularity": popular_model,
        "Semantic-Content": content_model,
        "SVD-Collaborative": collab_model,
        "Semantic-Hybrid": hybrid_model
    }
    
    K = 10
    results = []
    
    print(f"\nRunning Benchmark on {len(test_pairs)} test users (Top-K={K})...")
    
    for name, model in models.items():
        if model is None:
            continue
            
        ndcgs = []
        maps = []
        
        for user_id, query_item, relevant_items in test_pairs:
            if hasattr(model, 'predict_for_user') and name == "SVD-Collaborative":
                recs_raw = model.predict_for_user(user_id, top_n=K)
            elif name == "Semantic-Hybrid":
                recs_raw = model.recommend(query_item, user_id=user_id, top_n=K)
            else:
                recs_raw = model.recommend(query_item, top_n=K)
                
            rec_titles = [r['title'] for r in recs_raw]
            ndcgs.append(ndcg_at_k(rec_titles, relevant_items, K))
            maps.append(average_precision_at_k(rec_titles, relevant_items, K))
            
        avg_n = np.mean(ndcgs)
        avg_m = np.mean(maps)
        results.append((name, avg_n, avg_m))
        print(f"[{name:20s}] NDCG@{K}: {avg_n:.4f} | MAP@{K}: {avg_m:.4f}")
        
    print("\nBenchmark Complete.")

if __name__ == '__main__':
    run_benchmark()
