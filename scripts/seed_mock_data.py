"""
Seed mock users and purchase history into Supabase.
Creates realistic user-product interactions to solve the cold start problem.

Usage:
    python scripts/seed_mock_data.py
    python scripts/seed_mock_data.py --users 50 --purchases 2000
"""
import os
import sys
import random
import argparse
import secrets
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from src.data.db import get_supabase_admin


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Avery", "Quinn",
    "Blake", "Devon", "Harper", "Sage", "Rowan", "Eden", "Finley", "Emery",
    "Skyler", "Phoenix", "Reese", "Dakota", "Hayden", "Cameron", "Drew", "Sam",
    "Jamie", "Kendall", "Peyton", "Charlie", "Frankie", "Jesse", "Remy", "Shay",
    "Lane", "Silver", "Storm", "River", "Wren", "Aspen", "Cedar", "Indigo",
]

REVIEW_TEMPLATES = [
    "Great product, really enjoyed it!",
    "Not what I expected, but decent.",
    "Absolutely loved this! Highly recommend.",
    "Average quality, nothing special.",
    "Fantastic value for the price.",
    "Wouldn't buy again.",
    "Perfect for what I needed.",
    "Good but could be better.",
    "One of the best purchases I've made.",
    "Decent product, fast delivery.",
    "Quality is outstanding.",
    "Meh, it's okay I guess.",
    "Exceeded my expectations!",
    "A bit overpriced but works well.",
    "Would recommend to friends.",
]


def generate_mock_password(length=24):
    """Generate a high-entropy password for throwaway mock accounts."""
    if length < 16:
        raise ValueError("Mock user passwords must be at least 16 characters.")

    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    required = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*()-_=+"),
    ]
    remaining = [secrets.choice(alphabet) for _ in range(length - len(required))]
    chars = required + remaining
    secrets.SystemRandom().shuffle(chars)
    return ''.join(chars)


def seed_mock_data(num_users=100, num_purchases=5000):
    sb = get_supabase_admin()

    # Get available products
    print("  Fetching product catalog...")
    products_result = sb.table('products').select('id, category').limit(5000).execute()
    products = products_result.data
    if not products:
        print("  ✗ No products found. Run import_to_supabase.py first.")
        return

    product_ids = [p['id'] for p in products]
    categories = list(set(p['category'] for p in products if p.get('category')))
    category_products = {}
    for p in products:
        cat = p.get('category', 'Other')
        category_products.setdefault(cat, []).append(p['id'])

    print(f"  Found {len(products):,} products in {len(categories)} categories")

    # Create mock users
    print(f"\n  Creating {num_users} mock users...")
    mock_users = []
    for i in tqdm(range(num_users), desc="  Users"):
        name = random.choice(FIRST_NAMES)
        suffix = ''.join(random.choices(string.digits, k=4))
        email = f"mock_{name.lower()}_{suffix}@demo.hybridrec.test"
        password = generate_mock_password()
        try:
            user_resp = sb.auth.admin.create_user({
                "email": email,
                "password": password,
                "email_confirm": True,
                "user_metadata": {"display_name": name, "is_mock": True},
                "app_metadata": {"is_mock": True},
            })
            mock_users.append({
                'id': user_resp.user.id,
                'name': name,
                'fav_categories': random.sample(categories, k=min(2, len(categories))),
            })
        except Exception as e:
            if i < 3:
                print(f"  ⚠ User creation error: {str(e)[:100]}")

    print(f"  ✅ Created {len(mock_users)} mock users")

    if not mock_users:
        print("  ✗ No users created. Check Supabase service key.")
        return

    # Generate purchases with realistic patterns
    print(f"\n  Generating {num_purchases:,} purchases...")
    purchases_data = []

    # Power-law distribution: some users buy a lot
    user_weights = [random.paretovariate(1.5) for _ in mock_users]
    total_weight = sum(user_weights)
    user_weights = [w / total_weight for w in user_weights]

    for _ in tqdm(range(num_purchases), desc="  Purchases"):
        user = random.choices(mock_users, weights=user_weights, k=1)[0]

        # 70% chance to pick from favorite category
        if random.random() < 0.7 and user['fav_categories']:
            fav_cat = random.choice(user['fav_categories'])
            pool = category_products.get(fav_cat, product_ids)
        else:
            pool = product_ids

        product_id = random.choice(pool)

        # Rating: normal distribution centered at 3.5
        rating = max(0.5, min(5.0, round(random.gauss(3.5, 1.0) * 2) / 2))

        review = random.choice(REVIEW_TEMPLATES) if random.random() < 0.4 else ''

        purchases_data.append({
            'user_id': user['id'],
            'product_id': product_id,
            'rating': rating,
            'review_text': review,
        })

    # Batch insert purchases
    print("  Inserting purchases...")
    batch_size = 500
    inserted = 0
    for start in tqdm(range(0, len(purchases_data), batch_size), desc="  Batches"):
        batch = purchases_data[start:start + batch_size]
        try:
            sb.table('purchases').insert(batch).execute()
            inserted += len(batch)
        except Exception as e:
            print(f"  ⚠ Batch error: {str(e)[:150]}")

    # Also insert some reviews (deduped by user+product)
    print("  Inserting reviews...")
    seen_reviews = set()
    reviews_data = []
    for p in purchases_data:
        if p['review_text'] and (p['user_id'], p['product_id']) not in seen_reviews:
            seen_reviews.add((p['user_id'], p['product_id']))
            reviews_data.append({
                'user_id': p['user_id'],
                'product_id': p['product_id'],
                'rating': p['rating'],
                'review_text': p['review_text'],
                'sentiment': 0.5 if p['rating'] >= 3 else -0.3,
            })

    for start in range(0, len(reviews_data), batch_size):
        batch = reviews_data[start:start + batch_size]
        try:
            sb.table('reviews').upsert(batch, on_conflict='user_id,product_id').execute()
        except Exception as e:
            print(f"  Warning: Failed to upsert reviews batch: {str(e)[:100]}")

    print(f"\n  {'='*50}")
    print(f"  ✅ Seeded {len(mock_users)} users, {inserted:,} purchases, {len(reviews_data):,} reviews")
    print(f"  {'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Seed mock users and purchases')
    parser.add_argument('--users', type=int, default=100)
    parser.add_argument('--purchases', type=int, default=5000)
    args = parser.parse_args()

    seed_mock_data(args.users, args.purchases)


if __name__ == '__main__':
    main()
