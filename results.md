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

---

## Recommendation Diversification

To prevent the "filter bubble" effect where a user is only shown highly similar items, we have implemented a Diversity Metric and Re-ranking Engine.

### 1. Diversity Score Calculation
The overall diversity of a returned result set is calculated dynamically using **TF-IDF Vectorization** and **Cosine Similarity** on the items' text metadata (titles/descriptions).

**Formula:**  
`Diversity Score = 1 - avg(pairwise cosine similarity)`

* A score of `1.0` (100%) means the items share almost no textual overlap (highly diverse).
* A score of `0.0` (0%) means the items are nearly identical.
* Self-similarity (an item compared to itself) is excluded from the average.

### 2. Diversification Algorithm (`?diversify=true`)
When the diversification flag is activated via the API, the system fetches a larger candidate pool and uses a **Greedy Selection Algorithm** to re-rank items:
1. The most relevant item (from the Hybrid Recommender) is selected first.
2. For remaining slots, the system evaluates all unselected candidates.
3. It calculates a sub-matrix of similarities between unselected items and *already selected* items.
4. It picks the candidate with the **lowest maximum similarity** to the already-selected list.
5. This process loops until the `limit` is reached, ensuring a perfect balance between query relevance and result variance.