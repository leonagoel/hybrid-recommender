"""
A/B Testing Module — Compare Recommendation Weight Configurations

Runs two weight configurations (A and B) through the hybrid recommender
and measures precision@K, recall@K, and diversity score to determine
which configuration produces better recommendations.
"""
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import precision_at_k, recall_at_k, ndcg_at_k
from hybrid_model import HybridRecommender


class ABTestRunner:
    """Run A/B tests comparing two hybrid recommender weight configurations.

    Parameters
    ----------
    content_model : ContentRecommender
        Trained content-based recommender.
    collab_model : CollaborativeRecommender or None
        Trained collaborative recommender (may be None).
    item_df : pd.DataFrame
        Item metadata dataframe with titles, categories, ratings, etc.
    config_a : dict
        Config A weights: ``{"alpha": 0.5, "beta": 0.3, "gamma": 0.2}``.
    config_b : dict
        Config B weights: ``{"alpha": 0.3, "beta": 0.5, "gamma": 0.2}``.
    k : int
        Number of top recommendations to evaluate (default 10).
    """

    DEFAULT_CONFIG_A = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    DEFAULT_CONFIG_B = {"alpha": 0.3, "beta": 0.5, "gamma": 0.2}

    def __init__(
        self,
        content_model,
        collab_model,
        item_df,
        config_a=None,
        config_b=None,
        k=10,
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.item_df = item_df
        self.config_a = config_a or self.DEFAULT_CONFIG_A
        self.config_b = config_b or self.DEFAULT_CONFIG_B
        self.k = k
        self.results = {}

    # ── Metrics ──────────────────────────────────────────────────────

    @staticmethod
    def diversity_score(recommendations):
        """Compute category diversity of a recommendation list.

        Returns a score in [0, 1] representing the proportion of unique
        categories among the recommended items.  Higher = more diverse.
        """
        if not recommendations:
            return 0.0
        categories = [r.get("category", "") for r in recommendations]
        unique = set(c for c in categories if c)
        return len(unique) / len(recommendations) if recommendations else 0.0

    # ── Core runner ──────────────────────────────────────────────────

    def _evaluate_config(self, config, test_pairs):
        """Evaluate a single weight configuration across all test pairs.

        Returns a dict with averaged precision, recall, ndcg, and diversity.
        """
        hybrid = HybridRecommender(
            self.content_model,
            self.collab_model,
            self.item_df,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=config["gamma"],
        )

        precisions, recalls, ndcgs, diversities = [], [], [], []

        for _user_id, query_item, relevant_items in test_pairs:
            recs = hybrid.recommend(query_item, top_n=self.k)
            rec_titles = [r["title"] for r in recs]

            precisions.append(precision_at_k(rec_titles, relevant_items, self.k))
            recalls.append(recall_at_k(rec_titles, relevant_items, self.k))
            ndcgs.append(ndcg_at_k(rec_titles, relevant_items, self.k))
            diversities.append(self.diversity_score(recs))

        return {
            "precision": round(float(np.mean(precisions)), 4),
            "recall": round(float(np.mean(recalls)), 4),
            "ndcg": round(float(np.mean(ndcgs)), 4),
            "diversity": round(float(np.mean(diversities)), 4),
        }

    def run(self, test_pairs):
        """Run the A/B test on the given test pairs.

        Parameters
        ----------
        test_pairs : list of (user_id, query_item, relevant_items)
            Evaluation data — one entry per test user.

        Returns
        -------
        dict with keys ``config_a``, ``config_b``, ``winner``.
        """
        if not test_pairs:
            print("  ⚠  No test pairs provided. Cannot run A/B test.")
            return {}

        print(f"\n  Running Config A  (α={self.config_a['alpha']}, "
              f"β={self.config_a['beta']}, γ={self.config_a['gamma']}) ...")
        metrics_a = self._evaluate_config(self.config_a, test_pairs)

        print(f"  Running Config B  (α={self.config_b['alpha']}, "
              f"β={self.config_b['beta']}, γ={self.config_b['gamma']}) ...")
        metrics_b = self._evaluate_config(self.config_b, test_pairs)

        # Determine winner by NDCG (primary), then precision (tiebreak)
        if metrics_a["ndcg"] > metrics_b["ndcg"]:
            winner = "A"
        elif metrics_b["ndcg"] > metrics_a["ndcg"]:
            winner = "B"
        elif metrics_a["precision"] >= metrics_b["precision"]:
            winner = "A"
        else:
            winner = "B"

        self.results = {
            "config_a": {**self.config_a, **metrics_a},
            "config_b": {**self.config_b, **metrics_b},
            "winner": winner,
            "k": self.k,
            "test_users": len(test_pairs),
        }

        return self.results

    # ── Display ──────────────────────────────────────────────────────

    def print_results(self):
        """Print a formatted comparison table to stdout."""
        if not self.results:
            print("  No results yet. Call run() first.")
            return

        r = self.results
        a = r["config_a"]
        b = r["config_b"]
        k = r["k"]

        print(f"\n{'=' * 72}")
        print(f"  A/B TEST RESULTS — Top-{k} Recommendations")
        print(f"  Test users: {r['test_users']}")
        print(f"{'=' * 72}\n")

        header = f"  {'Metric':<20s} {'Config A':>12s} {'Config B':>12s} {'Better':>10s}"
        print(header)
        print(f"  {'-' * 56}")

        rows = [
            ("Weights (α/β/γ)",
             f"{a['alpha']}/{a['beta']}/{a['gamma']}",
             f"{b['alpha']}/{b['beta']}/{b['gamma']}",
             ""),
            (f"Precision@{k}", f"{a['precision']:.4f}", f"{b['precision']:.4f}",
             "A" if a["precision"] > b["precision"] else ("B" if b["precision"] > a["precision"] else "Tie")),
            (f"Recall@{k}", f"{a['recall']:.4f}", f"{b['recall']:.4f}",
             "A" if a["recall"] > b["recall"] else ("B" if b["recall"] > a["recall"] else "Tie")),
            (f"NDCG@{k}", f"{a['ndcg']:.4f}", f"{b['ndcg']:.4f}",
             "A" if a["ndcg"] > b["ndcg"] else ("B" if b["ndcg"] > a["ndcg"] else "Tie")),
            ("Diversity", f"{a['diversity']:.4f}", f"{b['diversity']:.4f}",
             "A" if a["diversity"] > b["diversity"] else ("B" if b["diversity"] > a["diversity"] else "Tie")),
        ]

        for label, va, vb, better in rows:
            print(f"  {label:<20s} {va:>12s} {vb:>12s} {better:>10s}")

        print(f"\n  ★ Winner: Config {r['winner']}")
        print(f"{'=' * 72}\n")

    # ── results.md generation ────────────────────────────────────────

    def write_results_md(self, filepath=None):
        """Write A/B test results to a Markdown file.

        Parameters
        ----------
        filepath : str or None
            Path to write the results file.  Defaults to ``results.md``
            in the project root.
        """
        if not self.results:
            print("  No results to write. Call run() first.")
            return

        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "results.md")

        r = self.results
        a = r["config_a"]
        b = r["config_b"]
        k = r["k"]

        lines = [
            f"# A/B Test Results",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Test users:** {r['test_users']}  ",
            f"**Top-K:** {k}",
            f"",
            f"## Weight Configurations",
            f"",
            f"| Config | Alpha (Content) | Beta (Collab) | Gamma (Sentiment) |",
            f"|--------|:-:|:-:|:-:|",
            f"| **A** | {a['alpha']} | {a['beta']} | {a['gamma']} |",
            f"| **B** | {b['alpha']} | {b['beta']} | {b['gamma']} |",
            f"",
            f"## Metrics Comparison",
            f"",
            f"| Metric | Config A | Config B | Winner |",
            f"|--------|:-:|:-:|:-:|",
        ]

        metrics = [
            (f"Precision@{k}", a["precision"], b["precision"]),
            (f"Recall@{k}", a["recall"], b["recall"]),
            (f"NDCG@{k}", a["ndcg"], b["ndcg"]),
            ("Diversity", a["diversity"], b["diversity"]),
        ]

        for label, va, vb in metrics:
            if va > vb:
                winner = "✅ A"
            elif vb > va:
                winner = "✅ B"
            else:
                winner = "Tie"
            lines.append(f"| {label} | {va:.4f} | {vb:.4f} | {winner} |")

        lines += [
            f"",
            f"## Conclusion",
            f"",
            f"**Config {r['winner']}** performed better overall "
            f"(primary metric: NDCG@{k}).",
            f"",
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  Results written to {filepath}")
