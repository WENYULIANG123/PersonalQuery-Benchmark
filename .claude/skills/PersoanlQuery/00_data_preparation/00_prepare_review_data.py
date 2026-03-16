#!/usr/bin/env python3
"""
Stage 0: Data Preparation
Load review data for target user and other users - NO LLM calls

Input: User ID from selected users list
Output: Review data JSON with all reviews (target + other users) for each product
"""
import os
import sys
import json
import gzip
import argparse
from datetime import datetime
from typing import Dict, List

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_user_products(user_id: str, review_file: str) -> List[Dict]:
    """Load all products reviewed by target user"""
    products = []
    seen_asins = set()

    log_with_timestamp(f"Loading products for user {user_id}...")

    with gzip.open(review_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                if review.get('reviewerID') == user_id:
                    asin = review.get('asin')
                    if asin and asin not in seen_asins:
                        seen_asins.add(asin)
                        products.append({
                            'asin': asin,
                            'product_title': review.get('title', '')
                        })
            except:
                continue

    log_with_timestamp(f"Found {len(products)} products for user {user_id}")
    return products

def load_reviews_for_product(review_file: str, asin: str, target_user_id: str) -> Dict:
    """Load ALL reviews (target + other users) for a product"""
    target_review = None
    other_reviews = []

    with gzip.open(review_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                if review.get('asin') != asin:
                    continue

                reviewer_id = review.get('reviewerID', '')
                if reviewer_id == target_user_id:
                    target_review = review
                else:
                    other_reviews.append(review)
            except:
                continue

    return {
        'target_review': target_review,
        'other_reviews': other_reviews
    }

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Data Preparation")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--review-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0: Data Preparation - Loading Review Data")
    log_with_timestamp("=" * 80)

    # Load products for target user
    products = load_user_products(args.user_id, args.review_file)

    # Load reviews for each product
    results = []
    for i, product in enumerate(products):
        asin = product['asin']
        log_with_timestamp(f"Processing {i+1}/{len(products)}: {asin}")

        reviews_data = load_reviews_for_product(args.review_file, asin, args.user_id)

        result = {
            'asin': asin,
            'product_title': product['product_title'],
            'target_user_id': args.user_id,
            'target_review_found': reviews_data['target_review'] is not None,
            'target_review': reviews_data['target_review'],
            'other_reviews_count': len(reviews_data['other_reviews']),
            'other_reviews': reviews_data['other_reviews']
        }
        results.append(result)

    # Save results
    output_data = {
        'user_id': args.user_id,
        'timestamp': datetime.now().isoformat(),
        'total_products': len(results),
        'results': results
    }

    output_file = os.path.join(args.output_dir, f'reviews_{args.user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"Saved to {output_file}")
    log_with_timestamp(f"Total products: {len(results)}")
    log_with_timestamp(f"Total other reviews: {sum(r['other_reviews_count'] for r in results)}")
    log_with_timestamp("Stage 0 Complete!")

if __name__ == "__main__":
    main()
