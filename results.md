# Hybrid Weight Evaluation Results

## Tested Alpha Configurations

| Configuration | Alpha (Content) | Beta (Collaborative) | Gamma (Sentiment) |
|---------------|-----------------|----------------------|-------------------|
| Alpha 0.3     | 0.3             | 0.7                  | 0.0               |
| Alpha 0.5     | 0.5             | 0.5                  | 0.0               |
| Alpha 0.7     | 0.7             | 0.3                  | 0.0               |

---

## Evaluation Setup

The evaluation pipeline was updated to support configurable hybrid blending weights for content-based and collaborative filtering.

The following metrics are evaluated:

- Precision@10
- Recall@10
- NDCG@10

---

## Dataset Adjustments

The original dataset contained only one unique user, which was insufficient for collaborative filtering evaluation.

To enable proper benchmarking:
- Synthetic users were generated for testing
- Interaction data was expanded to 50 users
- Each synthetic user was assigned sampled interactions

Evaluation statistics:

- Interaction rows: 2000
- Unique users: 50
- Average interactions per user: 40

---

## Current Status

The evaluation framework now successfully:
- loads datasets
- generates synthetic collaborative interactions
- supports configurable alpha testing
- runs hybrid evaluation experiments

This setup can now be extended with larger datasets for deeper benchmarking and metric comparison.