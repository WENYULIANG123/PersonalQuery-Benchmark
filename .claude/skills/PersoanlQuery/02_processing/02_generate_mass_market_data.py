#!/usr/bin/env python3
"""
Stage 3.5: Generate Mass Market Persona Data

Aggregates other_users_preferences from Stage 1 preference extraction results
to create mass market persona data (大众画像) in Stage 3 format.

Input: result/personal_query/01_preference_extraction/preferences_{user_id}.json
Output: result/personal_query/02_processing/{user_id}/mass_market/{category}.json

Note: 直接读取 Stage 1 输出，不再依赖 Stage 2
"""

import json
import os
import sys
import gzip
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_category(cat):
    return cat.replace('&', 'and').replace(',', '').strip()


def load_product_metadata(meta_file: str, needed_asins: set) -> dict:
    metadata = {}
    
    try:
        open_func = gzip.open if meta_file.endswith('.gz') else open
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                all_meta = json.load(f)
                for item in all_meta:
                    asin = item.get('asin')
                    if asin in needed_asins:
                        metadata[asin] = item
            else:
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
    
    return metadata


def convert_preferences_to_attributes(prefs_data: dict) -> list:
    attributes = []
    
    if not isinstance(prefs_data, dict):
        return attributes
    
    for category, category_data in prefs_data.items():
        if not isinstance(category_data, dict):
            continue
        for dimension, entities in category_data.items():
            if not isinstance(entities, list):
                continue
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                attr = {
                    "attribute": entity.get("entity", ""),
                    "dimension": dimension,
                    "sentiment": entity.get("sentiment", "neutral"),
                    "original_text": entity.get("original_text", ""),
                    "improvement_wish": entity.get("improvement_wish", ""),
                    "source": "direct_preference",
                    "validation_passed": True,
                    "category": category
                }
                attributes.append(attr)
    
    return attributes


def aggregate_mass_market_data(preferences_file, metadata_file, output_dir, persona_dir=None):
    if persona_dir is None:
        persona_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/02_processing"
    
    user_id = Path(preferences_file).stem.replace('preferences_', '')
    
    with open(preferences_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preferences_results = data.get('results', [])
    
    needed_asins = {p.get('asin') for p in preferences_results if p.get('asin')}
    
    log_with_timestamp(f"  Loading metadata for {len(needed_asins)} products...")
    metadata = load_product_metadata(metadata_file, needed_asins)
    log_with_timestamp(f"  Loaded metadata for {len(metadata)} products")
    
    results = []
    for product in preferences_results:
        asin = product.get('asin', '')
        title = product.get('product_title', '')
        
        category = None
        if asin in metadata:
            meta = metadata[asin]
            cat_list = meta.get('category', [])
            if cat_list and isinstance(cat_list, list) and len(cat_list) > 0:
                category = cat_list[-1]
        
        other_prefs = product.get('other_users_preferences', {})
        public_attributes = convert_preferences_to_attributes(other_prefs)
        
        results.append({
            'asin': asin,
            'product_title': title,
            'category': category,
            'public_attributes': {
                'selected_attributes': public_attributes
            }
        })
    
    target_categories = set()
    user_persona_dir = Path(persona_dir) / user_id / "persona"
    if user_persona_dir.exists():
        for f in user_persona_dir.glob("*.json"):
            category = f.stem.replace("_", " ")
            target_categories.add(normalize_category(category))
    
    log_with_timestamp(f"Target categories: {target_categories}")
    
    # Aggregate by category -> dimension -> attributes
    category_data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        category = r.get('category', 'Unknown')
        normalized_cat = normalize_category(category)
        
        # Skip categories not in target user personas
        if normalized_cat not in target_categories:
            continue
        if normalized_cat not in target_categories:
            continue
        
        public_attrs = r.get('public_attributes', {}).get('selected_attributes', [])
        
        for attr in public_attrs:
            dimension = attr.get('dimension', 'Unknown')
            category_data[category][dimension].append(attr)
    
    # Create output files for each category
    output_files = []
    
    for category, dimensions in category_data.items():
        safe_cat_name = category.replace(' ', '_').replace(',', '').replace('&', 'and')
        
        # Count total products in this category
        product_count = sum(1 for r in results if r.get('category') == category)
        
        # Get ASINs
        asins = [r.get('asin') for r in results if r.get('category') == category]
        
        # Build attributes_by_dimension (max 10 attributes per dimension)
        attributes_by_dimension = {}
        total_attrs = 0
        
        for dimension, attrs in dimensions.items():
            if attrs:
                attributes_by_dimension[dimension] = attrs[:10]
                total_attrs += min(len(attrs), 10)
        
        # Create output data (matching Stage 3 format)
        output_data = {
            'user_id': 'mass_market',
            'category': category,
            'product_count': product_count,
            'asins': asins,
            'total_attributes': total_attrs,
            'dimensions_summary': list(attributes_by_dimension.keys()),
            'attributes': [],  # Flattened list not needed for mass market
            'attributes_by_dimension': attributes_by_dimension
        }
        
        user_dir = os.path.join(output_dir, user_id)
        mass_market_dir = os.path.join(user_dir, "mass_market")
        os.makedirs(mass_market_dir, exist_ok=True)
        
        output_file = os.path.join(mass_market_dir, f"{safe_cat_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        output_files.append({
            'category': category,
            'file': output_file,
            'dimensions': len(attributes_by_dimension),
            'products': product_count,
            'total_attrs': total_attrs
        })
        
        log_with_timestamp(f"  Created: {user_id}/mass_market/{safe_cat_name}.json ({len(attributes_by_dimension)} dims, {product_count} products)")
    
    return output_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate mass market persona data from Stage 1 preferences")
    parser.add_argument("--preferences-file", required=True,
                        help="Input preferences file from Stage 1 (preferences_{USER_ID}.json)")
    parser.add_argument("--metadata-file", required=True,
                        help="Product metadata file for category extraction")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/02_processing",
                        help="Output directory")
    parser.add_argument("--persona-dir",
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/02_processing",
                        help="Directory containing target user persona files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp(f"Reading: {args.preferences_file}")
    output_files = aggregate_mass_market_data(
        args.preferences_file, 
        args.metadata_file, 
        args.output_dir, 
        args.persona_dir
    )
    
    log_with_timestamp(f"\nGenerated {len(output_files)} mass market category files:")
    for of in output_files:
        log_with_timestamp(f"  - {of['category']}: {of['dimensions']} dimensions, {of['products']} products, {of['total_attrs']} attributes")


if __name__ == "__main__":
    main()
