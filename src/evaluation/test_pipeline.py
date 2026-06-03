"""Quick smoke test for the full pipeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

print("1. Loading dataset...")
from src.data.dataset_manager import DatasetManager
dm = DatasetManager()
dm.load_csv('datasets/sample_products.csv')
interaction_df, item_df = dm.merge_all()
print(f"   Items: {len(item_df)}, Interactions: {len(interaction_df)}")

print("2. Running NLP sentiment...")
from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
interaction_df = batch_analyze(interaction_df, 'review_text')
sa = aggregate_sentiment_by_item(interaction_df)
item_df = item_df.merge(sa, on='title', how='left')
item_df['avg_sentiment'] = item_df['avg_sentiment'].fillna(0)
print(f"   Avg sentiment: {item_df['avg_sentiment'].mean():.4f}")

print("3. Building content model...")
from src.model.content_model import ContentRecommender
cm = ContentRecommender(item_df)

print("4. Building collaborative model (SVD)...")
from src.model.collaborative_model import CollaborativeRecommender
collab = CollaborativeRecommender(interaction_df)

print("5. Building hybrid model...")
from src.model.hybrid_model import HybridRecommender
hm = HybridRecommender(cm, collab, item_df)

print("6. Getting recommendations...")
title = item_df['title'].iloc[0]
print(f"   Query: {title}")
recs = hm.recommend(title, top_n=5)
for i, r in enumerate(recs):
    print(f"   #{i+1} {r['title']} — Hybrid: {r['hybrid_score']:.4f}")

print("\n7. Testing LLM Explanations...")
from src.model.llm_explainer import get_explainer
explainer = get_explainer()
print(f"   LLM client initialized: {explainer.client is not None}")

# Generate explanations for first 3 recommendations
explained_recs = explainer.explain_multiple(recs[:3], title)
for i, rec in enumerate(explained_recs, start=1):
    explanation = rec.get("llm_explanation", "No explanation")
    # Truncate for display
    explanation_short = (explanation[:80] + "...") if len(explanation) > 80 else explanation
    print(f"   #{i} {rec['title']}")
    print(f"      → {explanation_short}")

print("\n8. Testing search...")
results = cm.search("Premium", top_n=3)
for r in results:
    print(f"   Found: {r['title']} (score: {r['score']:.4f})")

print("\n✅ All pipeline tests passed!")
