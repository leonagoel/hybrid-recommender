"""
Benchmarking Pipeline for Hybrid Recommender
Compares baseline models against the Semantic-Hybrid approach.
"""
import os
import sys
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.evaluation import (
    ndcg_at_k,
    average_precision_at_k,
    _mean_reciprocal_rank,
    _hit_rate,
    _catalog_coverage,
    _intra_list_diversity,
    _build_test_data,
)
from src.model.causal_config import CausalConfig

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

    # Plain hybrid baseline (correlation only)
    hybrid_model = HybridRecommender(
        content_model, collab_model, item_df,
        alpha=0.4, beta=0.4, gamma=0.2,
    )

    # Causal hybrid — IPS debiasing with conservative config
    # Uses CausalConfig.conservative() so the benchmark reflects a realistic
    # production setting rather than an extreme λ=1.0 stress test.
    causal_hybrid_model = HybridRecommender(
        content_model, collab_model, item_df,
        alpha=0.4, beta=0.4, gamma=0.2,
        causal_config=CausalConfig.conservative(),
    )

    random_model  = RandomRecommender(item_df['title'].unique())
    popular_model = PopularityRecommender(item_df)
    
    models = {
        "Random":            random_model,
        "Popularity":        popular_model,
        "Semantic-Content":  content_model,
        "SVD-Collaborative": collab_model,
        "Semantic-Hybrid":   hybrid_model,
        # Causal variant — same weights as Semantic-Hybrid but with IPS debiasing
        "Causal-Hybrid":     causal_hybrid_model,
    }
    
    K = 10
    results = []
    
    print(f"\nRunning Benchmark on {len(test_pairs)} test users (Top-K={K})...")
    
    for name, model in models.items():
        if model is None:
            continue
            
        ndcgs = []
        maps = []
        mrrs = []
        hits = []
        ilds = []
        all_recs = []
        
        for user_id, query_item, relevant_items in test_pairs:
            if hasattr(model, 'predict_for_user') and name == "SVD-Collaborative":
                recs_raw = model.predict_for_user(user_id, top_n=K)
            elif name in ("Semantic-Hybrid", "Causal-Hybrid"):
                # Both hybrid variants use the same recommend() interface;
                # the causal one applies IPS internally before returning results
                recs_raw = model.recommend(query_item, user_id=user_id, top_n=K)
            else:
                recs_raw = model.recommend(query_item, top_n=K)
                
            rec_titles = [r['title'] for r in recs_raw]
            all_recs.append(rec_titles)

            ndcgs.append(ndcg_at_k(rec_titles, relevant_items, K))
            maps.append(average_precision_at_k(rec_titles, relevant_items, K))
            mrrs.append(_mean_reciprocal_rank(rec_titles, relevant_items, K))
            hits.append(_hit_rate(rec_titles, relevant_items, K))

            # ILD uses the content model's embeddings; content_model may be
            # available via closure from _build_test_data. If not, skip ILD.
            try:
                ild = _intra_list_diversity(rec_titles, item_df, content_model.matrix)
            except Exception:
                ild = _intra_list_diversity(rec_titles, item_df, None)
            ilds.append(ild)
            
        avg_n = np.mean(ndcgs)
        avg_m = np.mean(maps)
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        avg_hit = np.mean(hits) if hits else 0.0
        avg_ild = np.mean(ilds) if ilds else 0.0
        cov = _catalog_coverage(all_recs, len(item_df)) if all_recs else 0.0

        results.append((name, avg_n, avg_m, avg_mrr, avg_hit, cov, avg_ild))
        print(
            f"[{name:20s}] NDCG@{K}: {avg_n:.4f} | MAP@{K}: {avg_m:.4f} | "
            f"MRR@{K}: {avg_mrr:.4f} | Hit@{K}: {avg_hit:.4f} | "
            f"Cov: {cov:.4f} | ILD: {avg_ild:.4f}"
        )
        
    print("\nBenchmark Complete.")

if __name__ == '__main__':
    run_benchmark()
