"""
Generate a synthetic product-review dataset for testing the recommender.
Output: datasets/sample_products.csv  (~2000 rows)

Optimized via Issue #490: Implements strict pathlib absolute context mappings 
to prevent relative lookup path anomalies across multi-tier runtime environments.
"""
import os
import sys
import random
import csv

# --- Output paths ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_products.csv")

# --- Configuration ---
NUM_PRODUCTS = 200
NUM_USERS = 100
REVIEWS_PER_PRODUCT = (5, 15)  # min, max reviews per product

CATEGORIES = [
    'Electronics', 'Books', 'Clothing', 'Home & Kitchen', 'Sports',
    'Toys', 'Beauty', 'Automotive', 'Garden', 'Health',
]

ADJECTIVES = [
    'Premium', 'Ultra', 'Classic', 'Smart', 'Eco', 'Pro',
    'Compact', 'Advanced', 'Deluxe', 'Essential', 'Portable',
    'Wireless', 'Organic', 'Solar', 'Digital', 'Ergonomic',
]

NOUNS = {
    'Electronics': ['Headphones', 'Charger', 'Speaker', 'Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Hub', 'Cable', 'Tablet'],
    'Books': ['Novel', 'Guide', 'Handbook', 'Textbook', 'Journal', 'Anthology', 'Manual', 'Workbook', 'Cookbook', 'Atlas'],
    'Clothing': ['Jacket', 'Shirt', 'Sneakers', 'Hoodie', 'Jeans', 'Cap', 'Scarf', 'Boots', 'Shorts', 'Dress'],
    'Home & Kitchen': ['Blender', 'Toaster', 'Lamp', 'Mug', 'Rug', 'Pillow', 'Knife Set', 'Pan', 'Curtains', 'Clock'],
    'Sports': ['Yoga Mat', 'Dumbbell', 'Resistance Band', 'Jump Rope', 'Water Bottle', 'Gloves', 'Jersey', 'Helmet', 'Goggles', 'Racket'],
    'Toys': ['Building Blocks', 'Puzzle', 'Action Figure', 'Board Game', 'Doll', 'RC Car', 'Kite', 'Play Dough', 'Train Set', 'Plush Toy'],
    'Beauty': ['Moisturizer', 'Sunscreen', 'Lip Balm', 'Serum', 'Face Wash', 'Hair Oil', 'Perfume', 'Nail Polish', 'Eye Cream', 'Mask'],
    'Automotive': ['Car Mount', 'Dash Cam', 'Seat Cover', 'Air Freshener', 'Tool Kit', 'Jump Starter', 'Tire Inflator', 'Phone Holder', 'Floor Mat', 'Wiper'],
    'Garden': ['Planter', 'Hose', 'Pruner', 'Soil Mix', 'Sprinkler', 'Seed Kit', 'Fence', 'Bird Feeder', 'Shovel', 'Mulch'],
    'Health': ['Thermometer', 'Vitamins', 'First Aid Kit', 'Hand Sanitizer', 'Face Mask', 'Scale', 'BP Monitor', 'Heating Pad', 'Ice Pack', 'Pulse Oximeter'],
}

POSITIVE_REVIEWS = [
    "Absolutely love this product! Best purchase I've made this year.",
    "Great quality and super fast delivery. Highly recommend!",
    "Exceeded my expectations. Works perfectly and looks amazing.",
    "Five stars! This is exactly what I was looking for.",
    "Outstanding value for the price. Would buy again.",
    "My whole family loves it. Very well made and durable.",
    "Perfect gift idea. The recipient was thrilled!",
    "Sleek design and works flawlessly. Very impressed.",
    "Can't believe how good this is for the price. Amazing deal!",
    "Top notch quality. You won't be disappointed.",
]

NEUTRAL_REVIEWS = [
    "It's okay. Does the job but nothing special.",
    "Average product. Meets basic expectations.",
    "Decent quality for the price range. Not bad.",
    "Works as described. No complaints, no surprises.",
    "It's fine. Standard product, standard experience.",
    "Neither great nor terrible. Just average.",
    "Functional but could use some improvements.",
    "Get what you pay for. Acceptable quality.",
]

NEGATIVE_REVIEWS = [
    "Very disappointed. Broke after a week of use.",
    "Poor quality. Doesn't match the description at all.",
    "Waste of money. Would not recommend to anyone.",
    "Terrible experience. Customer service was unhelpful.",
    "Cheaply made and arrived damaged. Want a refund.",
    "Does not work as advertised. Very frustrating.",
    "Returned immediately. Complete ripoff.",
    "Horrible quality control. Mine had defects.",
]


def generate_product_name(category):
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS[category])
    brand_suffix = random.choice(['X', 'Pro', 'Plus', 'Lite', 'Max', 'One', 'V2', 'SE', '360'])
    return f"{adj} {noun} {brand_suffix}"


def generate_description(name, category):
    templates = [
        f"High-quality {category.lower()} product. The {name} offers excellent performance and reliability for everyday use.",
        f"The {name} is a top-rated {category.lower()} item designed for modern users who value quality and convenience.",
        f"Discover the {name} — a premium {category.lower()} product featuring cutting-edge technology and sleek design.",
        f"Upgrade your lifestyle with the {name}. Built for durability and engineered for superior {category.lower()} experience.",
    ]
    return random.choice(templates)


def generate_review_and_rating():
    """Generate a correlated review + rating pair."""
    sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[50, 30, 20])[0]
    if sentiment == 'positive':
        return random.choice(POSITIVE_REVIEWS), random.uniform(4.0, 5.0)
    elif sentiment == 'neutral':
        return random.choice(NEUTRAL_REVIEWS), random.uniform(2.5, 3.9)
    else:
        return random.choice(NEGATIVE_REVIEWS), random.uniform(1.0, 2.4)


def main():
    # Enforce safe directory presence before initialization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    rows = []
    product_names = set()

    for i in range(NUM_PRODUCTS):
        category = random.choice(CATEGORIES)
        name = generate_product_name(category)
        # Ensure unique names
        while name in product_names:
            name = generate_product_name(category)
        product_names.add(name)

        item_id = f"ITEM_{i+1:04d}"
        description = generate_description(name, category)
        num_reviews = random.randint(*REVIEWS_PER_PRODUCT)

        for _ in range(num_reviews):
            user_id = f"user_{random.randint(1, NUM_USERS):03d}"
            review_text, rating = generate_review_and_rating()
            rating = round(rating, 1)
            views = random.randint(0, 500)
            purchases = random.randint(0, views // 3) if views > 0 else 0

            rows.append({
                'item_id': item_id,
                'title': name,
                'description': description,
                'category': category,
                'user_id': user_id,
                'rating': rating,
                'review_text': review_text,
                'views': views,
                'purchases': purchases,
            })

    # Write CSV using verified absolute target references
    fieldnames = ['item_id', 'title', 'description', 'category', 'user_id', 'rating', 'review_text', 'views', 'purchases']
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows across {NUM_PRODUCTS} products and {NUM_USERS} users.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()