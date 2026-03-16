#!/usr/bin/env python3
"""
Enhanced Stage 0 v2: Extract and label preferences from ALL reviews

Key improvement:
- Clearly mark target_user vs other_users preferences
- Ensure target user's preferences are extracted separately
"""

import json
import gzip
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_target_user_review(review_file: str, asin: str, target_user_id: str):
    """Load target user's specific review."""
    try:
        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    if review.get('asin') == asin and review.get('reviewerID') == target_user_id:
                        return review
                except:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error loading target review: {e}")
    return None

def load_other_reviews(review_file: str, asin: str, target_user_id: str, max_reviews: int = None):
    """Load other users' reviews. If max_reviews is None, load all reviews."""
    reviews = []
    try:
        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    if review.get('asin') == asin:
                        reviewer_id = review.get('reviewerID', '')
                        if reviewer_id != target_user_id:
                            reviews.append(review)
                            if max_reviews and len(reviews) >= max_reviews:
                                break
                except:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error loading other reviews: {e}")
    return reviews

def extract_preferences_from_review(review: Dict, product_title: str, user_type: str):
    """Extract preferences from a review and mark user type."""
    review_text = review.get('reviewText', '')
    summary = review.get('summary', '')
    reviewer_id = review.get('reviewerID', '')

    if not review_text:
        return None

    prompt = f"""Analyze the following product review and extract user preferences.

Product: {product_title}
Review: {review_text}
Summary: {summary}

Extract preferences in JSON format:
{{"preferences": {{"Category1": [{{"entity": "attribute", "sentiment": "positive", "original_text": "quote", "improvement_wish": ""}}]}}}}

Rules:
1. Extract specific, meaningful attributes only
2. Use descriptive category names
3. Include original text quotes
4. Return ONLY valid JSON

Output JSON:"""

    try:
        client = LLMClient()
        response = client.call(prompt, max_tokens=2000)

        import re
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        result = json.loads(response.strip())

        # Mark user type
        prefs = result.get('preferences', {})
        for category, entities in prefs.items():
            for entity in entities:
                entity['reviewer_id'] = reviewer_id
                entity['user_type'] = user_type  # NEW: mark as target or other

        return prefs
    except Exception as e:
        log_with_timestamp(f"  Error extracting: {e}")
        return None

def process_product(asin: str, product_title: str, review_file: str, target_user_id: str, max_other_reviews: int = None):
    """Process a single product: extract preferences from target user and other users."""
    log_with_timestamp(f"Processing {asin}")

    result = {
        'asin': asin,
        'product_title': product_title,
        'target_user_id': target_user_id,
        'target_user_review_found': False,
        'target_user_preferences': {},
        'other_users_preferences': {},
        'other_users_count': 0,
        'total_reviews_processed': 0,
        'preference_breakdown': {
            'target_user': {'categories': 0, 'entities': 0},
            'other_users': {'categories': 0, 'entities': 0}
        }
    }

    # Step 1: Extract target user's preferences
    log_with_timestamp(f"  Loading target user's review...")
    target_review = load_target_user_review(review_file, asin, target_user_id)

    if target_review:
        log_with_timestamp(f"  Found target user's review")
        result['target_user_review_found'] = True
        target_prefs = extract_preferences_from_review(target_review, product_title, 'target')

        if target_prefs:
            result['target_user_preferences'] = target_prefs
            result['preference_breakdown']['target_user']['categories'] = len(target_prefs)
            result['preference_breakdown']['target_user']['entities'] = sum(len(entities) for entities in target_prefs.values())
            log_with_timestamp(f"  Extracted {result['preference_breakdown']['target_user']['categories']} categories from target user")
    else:
        log_with_timestamp(f"  Target user's review not found in sample")

    # Step 2: Extract other users' preferences
    log_with_timestamp(f"  Loading other users' reviews...")
    other_reviews = load_other_reviews(review_file, asin, target_user_id, max_reviews=max_other_reviews)

    result['other_users_count'] = len(other_reviews)
    log_with_timestamp(f"  Found {len(other_reviews)} other reviews")

    if other_reviews:
        log_with_timestamp(f"  Extracting preferences from other users (max 3 concurrent)...")

        all_other_prefs = {}
        completed_count = 0

        # Process with max 3 concurrent threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_review = {
                executor.submit(extract_preferences_from_review, review, product_title, 'other'): review
                for review in other_reviews
            }

            # Process completed tasks
            for future in as_completed(future_to_review):
                completed_count += 1
                if completed_count % 3 == 0:
                    log_with_timestamp(f"    Processed {completed_count}/{len(other_reviews)}...")

                try:
                    prefs = future.result()
                    if prefs:
                        for category, entities in prefs.items():
                            if category not in all_other_prefs:
                                all_other_prefs[category] = []
                            all_other_prefs[category].extend(entities)
                except Exception as e:
                    log_with_timestamp(f"    Error processing review: {e}")

        result['other_users_preferences'] = all_other_prefs
        result['preference_breakdown']['other_users']['categories'] = len(all_other_prefs)
        result['preference_breakdown']['other_users']['entities'] = sum(len(entities) for entities in all_other_prefs.values())

        log_with_timestamp(f"  Extracted {result['preference_breakdown']['other_users']['categories']} categories from other users")

    result['total_reviews_processed'] = (1 if target_review else 0) + len(other_reviews)

    log_with_timestamp(f"  Summary: Target={result['preference_breakdown']['target_user']['entities']}, Other={result['preference_breakdown']['other_users']['entities']}")

    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferences-file", required=True)
    parser.add_argument("--reviews-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-reviews", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("="*80)
    log_with_timestamp("Enhanced Stage 0 v2: Extract with Clear User Labels")
    log_with_timestamp("="*80)

    with open(args.preferences_file, 'r') as f:
        user_data = json.load(f)

    user_id = user_data.get('user_id')
    products = user_data.get('results', [])

    log_with_timestamp(f"User: {user_id}")
    log_with_timestamp(f"Products: {len(products)}")

    # Process ALL products
    results = []

    for i, product in enumerate(products):
        asin = product.get('asin')
        title = product.get('product_title', '')

        if not asin:
            continue

        result = process_product(asin, title, args.reviews_file, user_id, args.max_reviews)
        results.append(result)
        log_with_timestamp(f"Completed {i + 1}/{len(products)}")
        log_with_timestamp("")

    output_file = os.path.join(args.output_dir, f'preferences_{user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_products': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"Saved to {output_file}")
    log_with_timestamp("Stage 0 Complete!")

if __name__ == "__main__":
    main()
