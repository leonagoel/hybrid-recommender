"""
Generate and store embeddings for all products using sentence-transformers.
Run once after enabling pgvector and adding the vector column.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from db import get_supabase_admin
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 50

def main():
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    try:
        sb = get_supabase_admin()
    except Exception as e:
        print(f"Failed to connect to Supabase: {e}")
        print("Make sure your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY")
        return

    print("Fetching products without embeddings...")
    result = sb.table('products').select('id, title, description').is_('embedding', 'null').execute()
    products = result.data
    if not products:
        print("No products without embeddings found.")
        return

    print(f"Generating embeddings for {len(products)} products...")
    total = len(products)
    for i in range(0, total, BATCH_SIZE):
        batch = products[i:i+BATCH_SIZE]
        texts = [f"{p['title']} {p.get('description','')}" for p in batch]
        embeddings = model.encode(texts, show_progress_bar=True)

        for p, emb in zip(batch, embeddings):
            sb.table('products').update({'embedding': emb.tolist()}).eq('id', p['id']).execute()

        print(f"Processed {min(i+BATCH_SIZE, total)}/{total}")

    print("Embeddings generation completed.")

if __name__ == "__main__":
    main()
