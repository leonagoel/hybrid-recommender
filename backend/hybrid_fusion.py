import numpy as np # type: ignore

class EntropyWeightedFusion:
    """
    Optimizes hybrid recommendation blending logic by dynamically shifting weights
    based on the prediction entropy (uncertainty) of individual model outputs.
    """
    def __init__(self, epsilon=1e-9):
        # Epsilon prevents log(0) calculation errors during entropy matrix shifts
        self.epsilon = epsilon

    def _calculate_shannon_entropy(self, scores: np.ndarray) -> float:
        """
        Computes the normalized Shannon Entropy H(X) of a prediction score array.
        Lower entropy means sharp, confident predictions (fewer items favored).
        Higher entropy means flat, uncertain predictions (uniform distribution).
        """
        # 1. Ensure array is flattened and copy to prevent mutating raw data
        s = np.clip(scores.flatten(), self.epsilon, None)
        
        # 2. Convert raw scores to a clean probability distribution P(x) summing to 1.0
        total_sum = np.sum(s)
        if total_sum == 0 or np.isnan(total_sum):
            return 1.0 # Max uncertainty fallback for empty vectors
            
        p = s / total_sum
        
        # 3. Calculate Shannon Entropy: H(X) = -sum(P(x) * log2(P(x)))
        entropy = -np.sum(p * np.log2(p))
        
        # 4. Normalize entropy against maximum possible layout entropy: log2(N)
        max_entropy = np.log2(len(s)) if len(s) > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)

    def fuse_hybrid_matrices(self, collaborative_scores: np.ndarray, content_scores: np.ndarray) -> np.ndarray:
        """
        Executes dynamic inverse-entropy blending.
        Automatically favors content-based paths for sparse profiles (cold start) 
        and collaborative matrices for dense history data fields.
        """
        # Convert inputs to guaranteed NumPy arrays
        collab_arr = np.nan_to_num(np.array(collaborative_scores, dtype=np.float64))
        content_arr = np.nan_to_num(np.array(content_scores, dtype=np.float64))
        
        # 1. Extract structural entropy profiles
        h_collab = self._calculate_shannon_entropy(collab_arr)
        h_content = self._calculate_shannon_entropy(content_arr)
        
        # 2. Compute inverse scores (lower uncertainty = higher weight priority)
        # We invert the metric using (1.0 - H) or 1 / (H + epsilon)
        inv_collab = 1.0 - h_collab
        inv_content = 1.0 - h_content
        
        # Enforce non-negative constraint boundaries
        inv_collab = max(inv_collab, self.epsilon)
        inv_content = max(inv_content, self.epsilon)
        
        # 3. Normalize inverse values to compute final weights summing precisely to 1.0
        weight_sum = inv_collab + inv_content
        w_collab = inv_collab / weight_sum
        w_content = inv_content / weight_sum
        
        # 4. Synthesize final prediction array via optimized matrix dot scaling
        final_scores = (w_collab * collab_arr) + (w_content * content_arr)
        
        # Log algorithmic distribution metrics for transparency testing
        print(f"[HYBRID_FUSION] Tracking Metrics -> H_Collab: {h_collab:.4f} | H_Content: {h_content:.4f}")
        print(f"[HYBRID_FUSION] Scaling Weights -> W_Collab: {w_collab:.4f} | W_Content: {w_content:.4f}")
        
        return final_scores

# ============================================================================
# VERIFICATION UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    fusion_engine = EntropyWeightedFusion()
    
    print("--- TEST CASE 1: COLD-START PROFILE USER (Sparse Collab, Sharp Content) ---")
    # Collaborative array is uniform/sparse (uncertain), Content array has clear target hits (confident)
    sparse_collab = np.array([0.1, 0.1, 0.1, 0.1, 0.1]) 
    sharp_content = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
    
    result_1 = fusion_engine.fuse_hybrid_matrices(sparse_collab, sharp_content)
    print(f"Resulting Blend Matrix Sample: {result_1[:3]}\n")

    print("--- TEST CASE 2: HIGH-DENSITY USER (Confident Collab, Flat Content) ---")
    # Collaborative array has strong history cues, Content matching is broad/flat
    confident_collab = np.array([0.85, 0.05, 0.05, 0.05, 0.0])
    flat_content = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    result_2 = fusion_engine.fuse_hybrid_matrices(confident_collab, flat_content)
    print(f"Resulting Blend Matrix Sample: {result_2[:3]}")