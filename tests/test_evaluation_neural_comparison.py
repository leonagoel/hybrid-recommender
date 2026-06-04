import numpy as np
import pandas as pd

from src.evaluation.evaluation import run_evaluation


def test_run_evaluation_reports_neural_and_mixed_comparison_metrics(tmp_path, monkeypatch):
    dataset = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": [1, 2, 3, 4, 5, 6],
            "title": [
                "Alpha Headphones",
                "Beta Earbuds",
                "Gamma Speaker",
                "Delta Desk",
                "Epsilon Lamp",
                "Zeta Chair",
            ],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Home",
                "Home",
                "Home",
            ],
            "rating": [4.8, 4.1, 4.5, 4.2, 4.0, 4.7],
            "sentiment_score": [0.9, 0.4, 0.6, 0.2, 0.3, 0.5],
        }
    )
    data_path = tmp_path / "interactions.csv"
    svd_path = tmp_path / "svd_matrix.npy"
    dataset.to_csv(data_path, index=False)
    np.save(svd_path, np.random.default_rng(42).normal(size=(len(dataset), 4)))

    monkeypatch.setenv("SVD_CACHE", str(svd_path))
    monkeypatch.setenv("TFIDF_CACHE", str(tmp_path / "tfidf_matrix.npz"))

    results = run_evaluation(k=2, mode="all", data_path=str(data_path))

    assert {"hybrid", "neural", "mixed"}.issubset(results)
    for mode in ("hybrid", "neural", "mixed"):
        assert "precision" in results[mode]
        assert "recall" in results[mode]
        assert "ndcg" in results[mode]
        assert "catalog_coverage" in results[mode]
        assert "latency_ms" in results[mode]
