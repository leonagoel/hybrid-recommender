import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_matrix(items):
    """Generates a cosine similarity matrix based on item text."""
    texts = [str(item.get('description', item.get('title', ''))) for item in items]
    
    if not any(texts):
        return np.zeros((len(items), len(items)))
        
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)
    except ValueError:
        return np.zeros((len(items), len(items)))

def calculate_diversity_score(items):
    """Calculates: 1 - avg(pairwise cosine similarity)"""
    if len(items) <= 1:
        return 1.0 
        
    sim_matrix = get_similarity_matrix(items)
    
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    avg_sim = np.mean(sim_matrix[upper_tri_indices])
    
    return round(float(1.0 - avg_sim), 4)

def diversify_results(items, top_n=10):
    """Greedy algorithm to maximize diversity in the top_n results."""
    if not items: return []
    if len(items) == 1: return items
    
    sim_matrix = get_similarity_matrix(items)
    
    selected_indices = [0]
    unselected_indices = list(range(1, len(items)))
    
    while len(selected_indices) < top_n and unselected_indices:
        sub_matrix = sim_matrix[np.ix_(unselected_indices, selected_indices)]
        
        max_sims = np.max(sub_matrix, axis=1)
        
        best_idx_relative = np.argmin(max_sims)
        best_idx_absolute = unselected_indices[best_idx_relative]
        
        selected_indices.append(best_idx_absolute)
        unselected_indices.pop(best_idx_relative)
        
    return [items[i] for i in selected_indices]