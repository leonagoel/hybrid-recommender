import math

from evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)

def test_precision_at_k_basic():
    recommended = ["A", "B", "C"]
    relevant = ["A", "C"]

    score = precision_at_k(recommended, relevant, 2)

    assert score == 0.5


def test_precision_at_k_empty_relevant():
    recommended = ["A", "B"]
    relevant = []

    score = precision_at_k(recommended, relevant, 2)

    assert score == 0.0


def test_precision_at_k_zero_k():
    recommended = ["A", "B"]
    relevant = ["A"]

    score = precision_at_k(recommended, relevant, 0)

    assert score == 0


def test_precision_at_k_large_k():
    recommended = ["A"]
    relevant = ["A"]

    score = precision_at_k(recommended, relevant, 10)

    assert score == 0.1


def test_recall_at_k_basic():
    recommended = ["A", "B", "C"]
    relevant = ["A", "C", "D"]

    score = recall_at_k(recommended, relevant, 3)

    assert math.isclose(score, 2 / 3)


def test_recall_at_k_empty_relevant():
    recommended = ["A", "B"]
    relevant = []

    score = recall_at_k(recommended, relevant, 2)

    assert score == 0


def test_recall_at_k_zero_k():
    recommended = ["A", "B"]
    relevant = ["A"]

    score = recall_at_k(recommended, relevant, 0)

    assert score == 0


def test_ndcg_at_k_basic():
    recommended = ["A", "B", "C"]
    relevant = ["A", "C"]

    score = ndcg_at_k(recommended, relevant, 3)

    assert 0 <= score <= 1


def test_ndcg_at_k_perfect_ranking():
    recommended = ["A", "B"]
    relevant = ["A", "B"]

    score = ndcg_at_k(recommended, relevant, 2)

    assert math.isclose(score, 1.0)


def test_ndcg_at_k_empty_relevant():
    recommended = ["A", "B"]
    relevant = []

    score = ndcg_at_k(recommended, relevant, 2)

    assert score == 0