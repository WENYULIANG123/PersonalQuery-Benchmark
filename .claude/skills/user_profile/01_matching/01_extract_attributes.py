#!/usr/bin/env python3
"""
Generate Match Tasks Script
Part 1 of the Matching Workflow: Data Preparation

This script prepares the data needed for the 3-step matching process.
It avoids making any LLM calls and focuses on:
1. Loading User Preferences & Metadata
2. Identifying "Neighbor" products (same category)
3. Constructing a task file with all necessary context for execution

Output: match_tasks_[USER_ID].json
"""
import os
import sys
import json
import argparse
import gzip
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_product_metadata(meta_file: str, needed_asins: set) -> Dict:
    """Load product metadata for specified ASINs."""
    log_with_timestamp(f"Loading metadata for {len(needed_asins)} products...")
    metadata = {}
    
    try:
        open_func = gzip.open if meta_file.endswith('.gz') else open
        
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON list format
                all_meta = json.load(f)
                for item in all_meta:
                    asin = item.get('asin')
                    if asin in needed_asins:
                        metadata[asin] = item
            else:
                # Line delimited JSON format
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        asin = item.get('asin')
                        if asin in needed_asins:
                            metadata[asin] = item
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log_with_timestamp(f"Error loading metadata: {e}")
        
    log_with_timestamp(f"Successfully loaded metadata for {len(metadata)} products.")
    return metadata

def prepare_step1_data(product: Dict) -> Dict:
    """Prepare data for Step 1: User Direct Preferences"""
    preferences = product.get('preferences', {})
    user_prefs = []
    
    for category, items in preferences.items():
        if category == "Product Category":
            continue
        if isinstance(items, list):
            for item in items:
                sentiment = item.get('sentiment', '')
                if sentiment in ['positive', 'neutral']:
                    user_prefs.append({
                        'category': category,
                        'entity': item.get('entity', ''),
                        'sentiment': sentiment,
                        'original_text': item.get('original_text', '')[:200]
                    })
    
    return {
        'asin': product.get('asin'),
        'title': product.get('product_title', 'Unknown'),
        'user_prefs_candidates': user_prefs
    }



def prepare_step3_data(product: Dict, metadata: Dict) -> Dict:
    """Prepare data for Step 3: Product Metadata"""
    asin = product.get('asin')
    target_meta = metadata.get(asin, {})
    
    return {
        'asin': asin,
        'title': target_meta.get('title', 'N/A'),
        'brand': target_meta.get('brand', 'N/A'),
        'feature': target_meta.get('feature', []),  # Correct key is singular 'feature'
        'description': target_meta.get('description', []),
        'category': target_meta.get('category', []),
        'price': target_meta.get('price', ''),
        'details': target_meta.get('details', {}),
        'tech1': target_meta.get('tech1', ''),
        'tech2': target_meta.get('tech2', ''),
        'fit': target_meta.get('fit', '')
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Match Tasks (Data Preparation)")
    parser.add_argument("--input", required=True, help="Path to preferences_[USER_ID].json")
    parser.add_argument("--meta-file", required=True, help="Path to product metadata file")
    parser.add_argument("--output-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/match_tasks")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load User Preferences
    log_with_timestamp(f"Loading user preferences from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
        
    user_id = user_data.get('user_id')
    products = user_data.get('results', [])
    needed_asins = {p.get('asin') for p in products}
    
    # Load Metadata
    metadata_dict = load_product_metadata(args.meta_file, needed_asins)
    
    tasks = []
    
    log_with_timestamp(f"Generating tasks for {len(products)} products...")
    
    for i, product in enumerate(products):
        asin = product.get('asin')
        
        # Prepare data for matching
        step1_data = prepare_step1_data(product)
        # Step 2 (Neighbor Insights) is removed as requested
        step3_data = prepare_step3_data(product, metadata_dict)
        
        tasks.append({
            'asin': asin,
            'step1_input': step1_data,
            'step3_input': step3_data
        })
        
        if (i + 1) % 10 == 0:
             log_with_timestamp(f"Prepared {i + 1}/{len(products)} tasks.")

    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'total_tasks': len(tasks),
        'tasks': tasks
    }
    
    output_file = os.path.join(args.output_dir, f"match_tasks_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    log_with_timestamp(f"DONE! Generated {len(tasks)} match tasks at {output_file}")

if __name__ == "__main__":
    main()
