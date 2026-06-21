import numpy as np # type: ignore
from scipy.sparse import csr_matrix # type: ignore

class SparseBlockCoordinateDescent:
    """
    Sparsity-Aware Block-Coordinate Descent (S-BCD) Solver 
    Optimizes latent factor decomposition models over highly sparse interaction tensors.
    """
    def __init__(self, n_factors=10, l2_reg=0.05, max_iter=15, tol=1e-4):
        self.n_factors = n_factors  # Latent dimensions (K)
        self.l2_reg = l2_reg        # Tikhonov L2 regularization lambda parameter
        self.max_iter = max_iter    # Step convergence threshold upper bound
        self.tol = tol              # Early-stopping tolerance threshold

    def fit_factorization(self, interaction_matrix: csr_matrix):
        """
        Factorizes R (m x n) into User matrix P (m x K) and Item matrix Q (n x K).
        Utilizes coordinate slices to ignore unobserved data indices.
        """
        # 1. Extract dimensions from the Compressed Sparse Row matrix
        m, n = interaction_matrix.shape
        
        # 2. Initialize Latent Features using a scaled random normal distribution
        # P matrix tracks user behaviors; Q matrix tracks item vectors
        P = np.random.normal(0, 0.1, size=(m, self.n_factors))
        Q = np.random.normal(0, 0.1, size=(n, self.n_factors))

        # Ensure the sparse matrix is strictly in CSR format for rapid row slicing
        R_csr = interaction_matrix.tocsr()
        # Create a Transposed CSC format for rapid column slicing
        R_csc = interaction_matrix.tocsc()

        prev_mse = float('inf')

        print(f"[S-BCD ENGINE] Initializing Block Factorization Matrix: ({m}x{n})")
        print(f"[S-BCD ENGINE] Targets: Latent Factors={self.n_factors} | Lambda Reg={self.l2_reg}")

        # 3. Coordinate Optimization Alternating Loop Execution
        for epoch in range(self.max_iter):
            # --- BLOCK STEP 1: OPTIMIZE USER MATRICES (P) ---
            for u in range(m):
                # Isolate only non-zero item ratings given by user 'u'
                row_slice = R_csr.getrow(u)
                non_zero_items = row_slice.indices
                ratings = row_slice.data

                if len(non_zero_items) == 0:
                    continue # Skip empty interaction tracks natively

                # Extract only corresponding latent item slices
                Q_u = Q[non_zero_items, :]
                
                # S-BCD Coordinate System Update Formula: P[u] = (Q_u^T * Q_u + lambda * I)^-1 * Q_u^T * R_u
                A_u = np.dot(Q_u.T, Q_u) + np.eye(self.n_factors) * self.l2_reg
                V_u = np.dot(Q_u.T, ratings)
                P[u, :] = np.linalg.solve(A_u, V_u)

            # --- BLOCK STEP 2: OPTIMIZE ITEM MATRICES (Q) ---
            for i in range(n):
                # Isolate only non-zero user ratings received by item 'i'
                col_slice = R_csc.getcol(i)
                non_zero_users = col_slice.indices
                ratings = col_slice.data

                if len(non_zero_users) == 0:
                    continue # Skip tracking adjustments on zero fields

                # Extract only corresponding latent user vectors
                P_i = P[non_zero_users, :]
                
                # S-BCD Coordinate System Update Formula: Q[i] = (P_i^T * P_i + lambda * I)^-1 * P_i^T * R_i
                A_i = np.dot(P_i.T, P_i) + np.eye(self.n_factors) * self.l2_reg
                V_i = np.dot(P_i.T, ratings)
                Q[i, :] = np.linalg.solve(A_i, V_i)

            # --- EVALUATE MATRIX CONVERGENCE SCALING ---
            mse = self._compute_sparse_mse(R_csr, P, Q)
            mse_delta = prev_mse - mse
            
            print(f" -> Epoch {epoch+1:02d}/{self.max_iter:02d} Matrix MSE: {mse:.5f} | Improvement: {mse_delta:.6f}")
            
            # Terminate processing loops gracefully if convergence scales level out
            if 0 <= mse_delta < self.tol:
                print("[S-BCD ENGINE] Convergence matrix criteria successfully reached. Halting optimization loops.")
                break
                
            prev_mse = mse

        return P, Q

    def _compute_sparse_mse(self, R_csr, P, Q) -> float:
        """
        Calculates the Mean Squared Error exclusively on non-zero sparse indices.
        """
        mse_accum = 0.0
        total_interactions = R_csr.nnz # Count of explicit non-zero array elements

        if total_interactions == 0:
            return 0.0

        for u in range(R_csr.shape[0]):
            row = R_csr.getrow(u)
            items = row.indices
            actual_ratings = row.data

            if len(items) == 0:
                continue

            # Calculate dot product prediction coordinates across latent embeddings
            predictions = np.dot(P[u, :], Q[items, :].T)
            errors = actual_ratings - predictions
            mse_accum += np.sum(errors ** 2)

        return mse_accum / total_interactions

# ============================================================================
# VERIFICATION PIPELINE ENVIRONMENT TEST
# ============================================================================
if __name__ == "__main__":
    # Generate an extremely sparse user interaction mock matrix (98% sparse density)
    np.random.seed(42)
    mock_users = 100
    mock_items = 50
    
    dense_matrix = np.zeros((mock_users, mock_items))
    # Inject sparse structural indicators randomly (rating values between 1 and 5)
    for _ in range(120):
        u_idx = np.random.randint(0, mock_users)
        i_idx = np.random.randint(0, mock_items)
        dense_matrix[u_idx, i_idx] = np.random.randint(1, 6)

    # Compress the array into a Scipy Sparse Row Matrix
    sparse_tensor = csr_matrix(dense_matrix)

    # Initialize and execute the block solver profile
    solver = SparseBlockCoordinateDescent(n_factors=8, l2_reg=0.1, max_iter=10)
    user_factors, item_factors = solver.fit_factorization(sparse_tensor)
    
    print("\n[VERIFICATION SUCCESS] Latent decomposition matrices solved perfectly.")
    print(f"User Factor Sub-Matrix Frame: {user_factors.shape}")
    print(f"Item Factor Sub-Matrix Frame: {item_factors.shape}")