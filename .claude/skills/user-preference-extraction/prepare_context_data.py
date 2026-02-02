#!/usr/bin/env python3
"""
Prepare Context Data Script
Fetches reviews and KB data to generate 'input_material.json' for the User Preference Extraction Skill.
Aggregates ALL products for a user into a SINGLE JSON file.
"""

import sys
import os
import json
import argparse

print("🚀 Script starting...", flush=True)

# Add path to stark/code/user_perference to import helpers
STARK_CODE_DIR = "/home/wlia0047/ar57/wenyu/stark/code/user_perference"
sys.path.append(STARK_CODE_DIR)

print("📦 Importing helper modules...", flush=True)
try:
    from kb_helper import get_kb_instance
    from user_preference_extraction import load_user_reviews
    print("✅ Helper modules imported.", flush=True)
except ImportError as e:
    print(f"❌ Error importing helper modules from {STARK_CODE_DIR}: {e}", flush=True)
    sys.exit(1)

print("Setting up paths...", flush=True)
DEFAULT_OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/preference_extraction"
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "input_material.json")

def main():
    parser = argparse.ArgumentParser(description="Generate input material for User Preference Extraction.")
    parser.add_argument("--user_id", default="A13OFOB1394G31", help="Target User ID (default: A13OFOB1394G31)")
    parser.add_argument("--asin", help="Target Product ASIN (Optional). If provided, only fetches this one.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to save aggregated input_material.json")
    parser.add_argument("--reviews_file", default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/attribute_kb/user_product_reviews.json", 
                        help="Path to the preprocessed user_product_reviews.json file")
    
    args = parser.parse_args()
    
    print(f"🔍 Fetching data for User: {args.user_id}...")
    
    # 1. Load All User Reviews
    reviews_file = args.reviews_file
    
    try:
        if os.path.exists(reviews_file):
            print(f"   Loading reviews from file: {reviews_file}")
            with open(reviews_file, 'r') as f:
                data = json.load(f)
                user_data = data.get(args.user_id, {})
                all_reviews = user_data.get('reviews', [])
                # Normalize keys for compatibility (review_text -> reviewText)
                for r in all_reviews:
                    if 'review_text' in r and 'reviewText' not in r:
                        r['reviewText'] = r['review_text']
        else:
            print(f"   ⚠️ Preprocessed file not found, falling back to raw scan...")
            all_reviews = load_user_reviews(args.user_id)
    except Exception as e:
        print(f"❌ Failed to load reviews: {e}")
        sys.exit(1)
        
    # Group by ASIN
    reviews_by_asin = {}
    for r in all_reviews:
        asin = r.get('asin')
        if asin:
            if asin not in reviews_by_asin:
                reviews_by_asin[asin] = []
            reviews_by_asin[asin].append(r)
            
    print(f"   Found {len(all_reviews)} reviews across {len(reviews_by_asin)} unique products.")

    # Determine targets
    if args.asin:
        target_asins = [args.asin]
    else:
        target_asins = list(reviews_by_asin.keys())
        print(f"   Processing ALL {len(target_asins)} products.")

    # Fetch KB Data
    kb = get_kb_instance()
    products_data = []
    
    for i, asin in enumerate(target_asins):
        try:
            # KB Data
            known_attributes = kb.get_product_attributes(asin)
            product_info = kb.get_product_unstructured_info(asin)
            
            # Construct Item
            item = {
                "user_id": args.user_id,
                "asin": asin,
                "reviews": reviews_by_asin.get(asin, []),
                "known_attributes": known_attributes,
                "product_info": product_info
            }
            products_data.append(item)
            
            if (i+1) % 10 == 0:
                print(f"   Processed {i+1}/{len(target_asins)} products...")
                
        except Exception as e:
            print(f"❌ Error processing ASIN {asin}: {e}")

    # Save to single aggregated file
    output_data = {
        "user_id": args.user_id,
        "products": products_data
    }
    
    # Ensure output dir exists
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Aggregated input material saved to: {args.output}")
    print(f"   Total products included: {len(products_data)}")

if __name__ == "__main__":
    main()
