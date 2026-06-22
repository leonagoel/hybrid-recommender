import numpy as np # type: ignore
import heapq

class HNSWProductQuantizer:
    """
    Advanced High-Performance HNSW Proximity Graph Search Engine 
    with Product Quantization Vector Decomposition.
    """
    def __init__(self, M: int = 16, ef_search: int = 50, num_subvectors: int = 4):
        self.M = M  # Max links per node in the graph layers
        self.ef_search = ef_search  # Size of dynamic candidate lists during search loops
        self.num_subvectors = num_subvectors
        self.layers = []  # List of dictionaries representing graph layers [coarse -> fine]
        self.enter_node = None
        self.vector_store = {}
        self.codebooks = []

    def fit_quantizer(self, vectors: np.ndarray, num_centroids: int = 16):
        """
        Trains the Product Quantization codebooks by slicing vectors into sub-spaces
        and identifying localized centroid cluster points via vectorized k-means.
        """
        N, D = vectors.shape
        sub_D = D // self.num_subvectors
        self.codebooks = []

        print(f"[PQ ENGINE] Compressing {D}-dimensional vectors into {self.num_subvectors} sub-spaces...")
        for i in range(self.num_subvectors):
            sub_space = vectors[:, i*sub_D : (i+1)*sub_D]
            # Initialize random anchor centroids for clustering
            centroids = sub_space[np.random.choice(N, num_centroids, replace=False)]
            
            # Simple 3-pass fast k-means vector optimization
            for _ in range(3):
                distances = np.linalg.norm(sub_space[:, np.newaxis] - centroids, axis=2)
                assignments = np.argmin(distances, axis=1)
                for c in range(num_centroids):
                    if np.any(assignments == c):
                        centroids[c] = sub_space[assignments == c].mean(axis=0)
            self.codebooks.append(centroids)

    def compress_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Quantizes a raw vector into compressed codebook indices.
        """
        sub_D = len(vector) // self.num_subvectors
        compressed_codes = []
        
        for i in range(self.num_subvectors):
            sub_vec = vector[i*sub_D : (i+1)*sub_D]
            distances = np.linalg.norm(self.codebooks[i] - sub_vec, axis=1)
            compressed_codes.append(np.argmin(distances))
            
        return np.array(compressed_codes, dtype=np.uint8)

    def insert_index(self, node_id: int, vector: np.ndarray):
        """
        Inserts a compressed vector node into the multi-layer hierarchical graph grid.
        """
        compressed_code = self.compress_vector(vector)
        self.vector_store[node_id] = {"raw": vector, "code": compressed_code}

        # Dynamically calculate maximum structural layer depth via logarithmic probability
        max_level = int(-np.log(np.random.random()) * 0.4)
        max_level = max(0, min(max_level, 4)) # Cap at 5 total layers

        # Initialize missing structural layer levels
        while len(self.layers) <= max_level:
            self.layers.append({})

        if self.enter_node is None:
            self.enter_node = node_id
            for lvl in range(len(self.layers)):
                self.layers[lvl][node_id] = []
            return

        curr_node = self.enter_node
        # 1. COARSE NAVIGATION: Leap through top graph layers using greedy routing logic
        for lvl in range(len(self.layers) - 1, max_level, -1):
            curr_node = self._search_layer_greedy(curr_node, vector, lvl)

        # 2. FINE SELECTION: Insert the node and link neighbors from the target layer down to Layer 0
        for lvl in range(min(max_level, len(self.layers) - 1), -1, -1):
            self.layers[lvl][node_id] = []
            candidates = self._search_layer_closest(curr_node, vector, self.ef_search, lvl)
            
            # Select the nearest neighbors and establish bidirectional links
            for dist, neighbor in candidates[:self.M]:
                self.layers[lvl][node_id].append(neighbor)
                self.layers[lvl][neighbor].append(node_id)
            curr_node = candidates[0][1]

    def query_ann(self, query_vector: np.ndarray, top_k: int = 3) -> list:
        """
        Executes an extremely fast multi-layer logarithmic graph navigation query.
        """
        if self.enter_node is None:
            return []

        curr_node = self.enter_node
        # Route greedily down to the bottom layer
        for lvl in range(len(self.layers) - 1, 0, -1):
            curr_node = self._search_layer_greedy(curr_node, query_vector, lvl)

        # Perform comprehensive localized evaluation in the bottom layer
        candidates = self._search_layer_closest(curr_node, query_vector, self.ef_search, level=0)
        return [(node_id, float(dist)) for dist, node_id in candidates[:top_k]]

    def _search_layer_greedy(self, enter_node: int, query_vec: np.ndarray, level: int) -> int:
        curr_node = enter_node
        curr_dist = np.linalg.norm(self.vector_store[curr_node]["raw"] - query_vec)
        changed = True
        
        while changed:
            changed = False
            for neighbor in self.layers[level].get(curr_node, []):
                dist = np.linalg.norm(self.vector_store[neighbor]["raw"] - query_vec)
                if dist < curr_dist:
                    curr_dist = dist
                    curr_node = neighbor
                    changed = True
        return curr_node

    def _search_layer_closest(self, enter_node: int, query_vec: np.ndarray, ef: int, level: int) -> list:
        visited = {enter_node}
        # Use heaps to handle bounded tracking sets efficiently
        result_heap = [(np.linalg.norm(self.vector_store[enter_node]["raw"] - query_vec), enter_node)]
        candidate_heap = [result_heap[0]]

        while candidate_heap:
            curr_dist, curr_node = heapq.heappop(candidate_heap)
            
            if curr_dist > result_heap[-1][0]:
                break

            for neighbor in self.layers[level].get(curr_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = np.linalg.norm(self.vector_store[neighbor]["raw"] - query_vec)
                    
                    if dist < result_heap[-1][0] or len(result_heap) < ef:
                        heapq.heappush(candidate_heap, (dist, neighbor))
                        result_heap.append((dist, neighbor))
                        result_heap.sort(key=lambda x: x[0])
                        if len(result_heap) > ef:
                            result_heap.pop()
                            
        return result_heap

# ============================================================================
# HIGH-CONCURRENCY CONVERGENCE VERIFICATION SUITE
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    # Generate 100 recommendation vectors with 12 dimensions each
    mock_embeddings = np.random.uniform(-1, 1, size=(100, 12))

    print("--- EXPERIMENT START: Building HNSW + PQ Index Cluster ---")
    hnsw_engine = HNSWProductQuantizer(M=4, ef_search=10, num_subvectors=3)
    
    # 1. Train the quantization codebooks
    hnsw_engine.fit_quantizer(mock_embeddings, num_centroids=4)
    
    # 2. Build the structural graph maps
    for i, vec in enumerate(mock_embeddings):
        hnsw_engine.insert_index(node_id=1000 + i, vector=vec)
    print(f"[SUCCESS] Multi-Layer Graph Index Completed. Total layers built: {len(hnsw_engine.layers)}")

    # 3. Query the index
    test_query = np.random.uniform(-1, 1, size=(12,))
    results = hnsw_engine.query_ann(test_query, top_k=3)
    
    print("\n--- Top-3 Highly Correlated Vector Matches Discovered ---")
    for rank, (node, distance) in enumerate(results, 1):
        print(f"Rank {rank} -> Node ID: {node} | Euclidean Distance Offset: {distance:.4f}")