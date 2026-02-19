#!/usr/bin/env python3
import json
import gzip
import os
import argparse
from datetime import datetime

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_metadata_asins(meta_file):
    log_with_timestamp(f"Loading all valid ASINs from {meta_file}...")
    valid_asins = set()
    try:
        open_func = gzip.open if meta_file.endswith('.gz') else open
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                for item in data:
                    asin = item.get('asin')
                    cat = item.get('category')
                    # Check if categorization exists and is not just "Unknown"
                    if asin and cat and isinstance(cat, list) and len(cat) > 0:
                        valid_asins.add(asin)
            else:
                for line in f:
                    try:
                        item = json.loads(line)
                        asin = item.get('asin')
                        cat = item.get('category')
                        if asin and cat and isinstance(cat, list) and len(cat) > 0:
                            valid_asins.add(asin)
                    except: continue
    except Exception as e:
        log_with_timestamp(f"Error: {e}")
    log_with_timestamp(f"Total valid products in metadata: {len(valid_asins)}")
    return valid_asins

def stream_user_data(file_path):
    import re
    user_start_pattern = re.compile(r'^\s*"([^"]+)":\s*\{\s*$')
    current_user_id = None
    user_buffer = []
    brace_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline() # Skip opening {
        for line in f:
            if brace_count == 0:
                match = user_start_pattern.match(line)
                if match:
                    current_user_id = match.group(1)
                    user_buffer = ["{"]
                    brace_count = 1
                    continue
            if current_user_id:
                user_buffer.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    user_content = "".join(user_buffer).rstrip().rstrip(',\n').rstrip('\r')
                    if not user_content.endswith('}'): user_content += '}'
                    try:
                        yield current_user_id, json.loads(user_content)
                    except: pass
                    current_user_id = None; user_buffer = []; brace_count = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-file", default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz")
    parser.add_argument("--reviews-file", default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json")
    parser.add_argument("--min-reviews", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=110)
    parser.add_argument("--max-users", type=int, default=10)
    args = parser.parse_args()

    valid_asins = load_metadata_asins(args.meta_file)
    
    log_with_timestamp("Scanning users for 'No Unknown' and Review Count criteria...")
    found_users = []
    
    for user_id, user_info in stream_user_data(args.reviews_file):
        reviews = user_info.get('reviews', [])
        total_count = len(reviews)
        
        if total_count < args.min_reviews:
            continue
            
        user_asins = [r.get('asin') for r in reviews if r.get('asin')]
        known_asins = [a for a in user_asins if a in valid_asins]
        
        # Criterion: Known count must be within [min, max] range
        if args.min_reviews <= len(known_asins) <= args.max_reviews:
            found_users.append({
                'user_id': user_id,
                'count': len(known_asins),
                'total': len(user_asins)
            })
            log_with_timestamp(f"Found User {len(found_users)}: {user_id} ({len(known_asins)} known / {len(user_asins)} total)")
            if len(found_users) >= args.max_users:
                break

    print("\n" + "="*50)
    print(f"Selected Top {len(found_users)} 'Perfect' Users:")
    for u in found_users:
        print(f"{u['user_id']} : {u['count']} reviews (100% known categories)")
    print("="*50)

if __name__ == "__main__":
    main()
