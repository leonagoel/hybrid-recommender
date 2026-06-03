import sys
import numpy as np

# ── helper: copied logic from _diversity_rerank ──────────────────────
def diversity_rerank(results, top_n, diversity=0.0, serendipity=0.0):
    if not results:
        return results
    if diversity == 0.0 and serendipity == 0.0:
        return results[:top_n]
    selected = []
    remaining = results.copy()
    seen_categories = []
    while len(selected) < top_n and remaining:
        best = None
        best_score = -1
        for item in remaining:
            score = item['hybrid_score']
            times_seen = seen_categories.count(item.get('category', 'unknown'))
            score -= diversity * times_seen * 0.2
            score += serendipity * np.random.uniform(0, 0.3)
            if score > best_score:
                best_score = score
                best = item
        selected.append(best)
        remaining.remove(best)
        seen_categories.append(best.get('category', 'unknown'))
    return selected

# ── sample data ───────────────────────────────────────────────────────
sample_results = [
    {'title': 'iPhone 14',   'hybrid_score': 0.95, 'category': 'phones'},
    {'title': 'iPhone 13',   'hybrid_score': 0.90, 'category': 'phones'},
    {'title': 'Samsung S23', 'hybrid_score': 0.85, 'category': 'phones'},
    {'title': 'Nike Shoes',  'hybrid_score': 0.80, 'category': 'shoes'},
    {'title': 'Adidas Shoes','hybrid_score': 0.75, 'category': 'shoes'},
    {'title': 'Dell Laptop', 'hybrid_score': 0.70, 'category': 'laptops'},
    {'title': 'HP Laptop',   'hybrid_score': 0.65, 'category': 'laptops'},
]

# ── tests ─────────────────────────────────────────────────────────────
def test_no_diversity():
    result = diversity_rerank(sample_results, top_n=5)
    titles = [r['title'] for r in result]
    assert titles == ['iPhone 14', 'iPhone 13', 'Samsung S23',
                      'Nike Shoes', 'Adidas Shoes']
    print('PASS - no diversity returns top scores as-is')

def test_diversity_spreads_categories():
    result = diversity_rerank(sample_results, top_n=5, diversity=0.8)
    categories = [r['category'] for r in result]
    unique = len(set(categories))
    assert unique >= 3, f"Expected 3+ categories, got {unique}: {categories}"
    print(f'PASS - diversity spread across {unique} categories: {categories}')

def test_serendipity_returns_correct_count():
    np.random.seed(42)
    result = diversity_rerank(sample_results, top_n=5, serendipity=1.0)
    assert len(result) == 5
    print(f'PASS - serendipity returned 5 items: {[r["title"] for r in result]}')

def test_empty_input():
    result = diversity_rerank([], top_n=5, diversity=0.5)
    assert result == []
    print('PASS - empty input returns empty list')

if __name__ == '__main__':
    test_no_diversity()
    test_diversity_spreads_categories()
    test_serendipity_returns_correct_count()
    test_empty_input()
    print()
    print('All 4 tests passed!')
    