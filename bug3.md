## Bug Description
In the federated learning module (`src/model/federated_learning.py`), there is a mathematical scaling bug in how global item updates are aggregated. During `compute_local_item_updates`, each client computes its local gradient update for an item, which includes subtracting a regularization term: `error * self.user_factor - reg * v_i`.
When the server aggregates these updates in `aggregate_updates` via `np.mean(updates, axis=0)`, it effectively divides the regularization term (`reg * v_i`) by the number of clients who rated the item. This means that popular items rated by many clients will receive exponentially less regularization per client than rare items, leading to under-regularization of popular items and potential divergence or overfitting of the global item factors.

## Steps to Reproduce
1. Initialize the `FederatedServer` and simulate federated training with a dataset having a highly skewed item popularity distribution.
2. Set a high regularization coefficient `reg`.
3. Inspect the magnitude of the `global_item_factors` for popular items vs. niche items after several epochs.
4. Observe that popular items exhibit unconstrained growth (overfitting) compared to niche items due to the decayed regularization term in the averaged update.

## Expected Behavior
The server should either scale the regularization term back up by the number of contributing clients, or the clients should omit the regularization term and the server should apply the global `reg * v_i` penalty itself during the update step `self.global_item_factors[:, idx] += ...`.

## Actual Behavior
The regularization penalty is divided by the number of contributing clients for each item, causing a decay in regularization strength proportional to the item's popularity.

## Screenshots / Error Logs
N/A

## Environment
- OS: Any
- Python version: Any

## Additional Context
Advanced level machine learning bug in `src/model/federated_learning.py` affecting the stability of the federated collaborative filtering model.
