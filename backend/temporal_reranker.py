import numpy as np
import pandas as pd

class TemporalDecayReRanker:
    """
    Advanced scoring pipeline that applies exponential time-decay penalties
    to historical interactions and balances recommendations via UCB1 exploration.
    """
    def __init__(self, half_life_days: float = 30.0, exploration_weight: float = 0.1):
        # Calculate the decay constant lambda based on the desired half-life
        # half_life = ln(2) / lambda -> lambda = ln(2) / half_life
        self.lam = np.log(2) / (half_life_days * 24 * 60 * 60) # Normalized to seconds
        self.c = exploration_weight # Exploration hyperparameter (UCB factor)

    def apply_temporal_decay(self, interaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates scalar decay multipliers based on item interaction age.
        Expects a DataFrame containing ['user_id', 'item_id', 'rating', 'timestamp'].
        """
        df = interaction_df.copy()
        current_time = time_seconds = df['timestamp'].max() # Base relative anchor to the newest log
        
        # Calculate delta time in seconds
        delta_t = current_time - df['timestamp']
        
        # Apply vectorized exponential time-decay function: W(t) = W_0 * e^(-lambda * delta_t)
        df['decay_weight'] = np.exp(-self.lam * delta_t)
        df['adjusted_rating'] = df['rating'] * df['decay_weight']
        
        return df

    def compute_dynamic_rerank(self, base_scores: dict, total_system_impressions: int, item_click_counts: dict) -> list:
        """
        Applies an exploitation-exploration balance layer using UCB1 re-ranking thresholds.
        Formula: Score = Base_Score + c * sqrt(ln(Total_System_Impressions) / (Item_Clicks + 1))
        
        Args:
            base_scores: dict of {item_id: raw_decayed_hybrid_score}
            total_system_impressions: Total global processing passes
            item_click_counts: dict of {item_id: total_historical_interactions}
        """
        reranked_items = []
        log_total_impressions = np.log(max(total_system_impressions, 1))

        for item_id, base_score in base_scores.items():
            clicks = item_click_counts.get(item_id, 0)
            
            # Compute Upper Confidence Bound variance modifier to protect cold-start items
            exploration_bonus = self.c * np.sqrt(log_total_impressions / (clicks + 1))
            final_score = base_score + exploration_bonus
            
            reranked_items.append({
                "item_id": item_id,
                "base_score": base_score,
                "exploration_bonus": exploration_bonus,
                "final_score": final_score
            })
            
        # Sort items in descending order based on the integrated final score
        reranked_items.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked_items

# ============================================================================
# PERFORMANCE VERIFICATION TEST SUITE
# ============================================================================
if __name__ == "__main__":
    import time

    # 1. Mock user raw interaction histories spanning over 90 days
    now_seconds = time.time()
    day_in_seconds = 24 * 60 * 60

    mock_data = {
        'user_id': [1001, 1001, 1001],
        'item_id': [501, 502, 503], # 501: old interaction, 502: recent interaction, 503: brand new
        'rating': [5.0, 4.0, 4.0],
        'timestamp': [
            now_seconds - (60 * day_in_seconds), # 60 Days Old (Should decay significantly)
            now_seconds - (5 * day_in_seconds),  # 5 Days Old (Should retain high weight)
            now_seconds - (0.1 * day_in_seconds) # Brand New
        ]
    }
    df_interactions = pd.DataFrame(mock_data)

    print("[TEMPORAL ENGINE] Running Matrix Weight Transformations...")
    ranker = TemporalDecayReRanker(half_life_days=30.0, exploration_weight=0.15)
    
    # Run exponential decay function
    decayed_df = ranker.apply_temporal_decay(df_interactions)
    print("\n--- Processed Decay Results ---")
    print(decayed_df[['item_id', 'rating', 'decay_weight', 'adjusted_rating']].to_string(index=False))

    # 2. Simulate final pipeline scoring re-rank maps with a cold-start item
    # Item 501 has massive clicks but is stale; Item 504 is a fresh entry with 0 clicks
    mock_base_hybrid_scores = {501: 0.85, 502: 0.78, 504: 0.45}
    mock_global_item_clicks = {501: 1500, 502: 300, 504: 0}
    
    final_recommendations = ranker.compute_dynamic_rerank(
        base_scores=mock_base_hybrid_scores,
        total_system_impressions=10000,
        item_click_counts=mock_global_item_clicks
    )

    print("\n--- Re-Ranked Exploitation/Exploration Pipeline Outputs ---")
    for rank, item in enumerate(final_recommendations, 1):
        print(f"Rank {rank}: Item {item['item_id']} | Base: {item['base_score']:.3f} | "
              f"UCB Bonus: {item['exploration_bonus']:.3f} -> Final Score: {item['final_score']:.3f}")