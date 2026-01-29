#!/usr/bin/env python3
"""
Entity Normalization using LLM (Refactored to use shared utils)
"""

import sys
import os
import json
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm as async_tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_llm_with_config, get_all_api_keys_in_order
from normalization_utils import (
    normalize_entity_key, SCHEMA, build_generic_classification_prompt,
    execute_llm_classification, process_item_async
)

def classify_wrapper(llm_model, entities, product_id):
    """Sync wrapper for classification."""
    prompt = build_generic_classification_prompt(entities)
    return execute_llm_classification(llm_model, prompt, product_id, "entity_normalization")

async def process_product_async(executor, llm_model, idx: int, product: dict) -> dict:
    product_entities = product.get("product_entities", {})
    if not isinstance(product_entities, dict):
        return {"index": idx, "mappings": {}}
    
    # Prepare entities
    normalized_entities = {}
    for key, values in product_entities.items():
        norm_key = normalize_entity_key(key)
        if norm_key in SCHEMA and isinstance(values, list):
            normalized_entities[norm_key] = [str(v).strip() for v in values if v]
    
    if not normalized_entities:
        return {"index": idx, "mappings": {}}
    
    product_id = product.get("asin", f"product_{idx}")
    
    try:
        mappings, raw_response = await process_item_async(
            executor, classify_wrapper, llm_model, normalized_entities, product_id
        )
    except Exception as e:
        # print(f"Error for {product_id}: {e}")
        return {"index": idx, "mappings": {}}

    product_with_norm = product.copy()
    product_with_norm["normalized_entities"] = mappings

    return {
        "index": idx,
        "mappings": mappings,
        "product_with_norm": product_with_norm
    }

async def process_all_products(llm_model, products: list, max_concurrent: int = 100):
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    tasks = [process_product_async(executor, llm_model, idx, p) for idx, p in enumerate(products)]
    
    results = []
    print(f"Processing {len(products)} products with {max_concurrent} workers...")
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)
    
    executor.shutdown(wait=True)
    results.sort(key=lambda x: x["index"])
    
    global_mapping = {cat: {} for cat in SCHEMA.keys()}
    products_with_norm = []
    
    for r in results:
        mappings = r.get("mappings", {})
        for cat, val_map in mappings.items():
            if cat in global_mapping:
                global_mapping[cat].update(val_map)
        
        if "product_with_norm" in r:
            products_with_norm.append(r["product_with_norm"])
            
    # Return matched signature: (global_mapping, raw_responses, products_with_norm)
    # Raw responses are now auto-saved by normalization_utils/model.py, so we return empty list to satisfy signature or backward compat
    return global_mapping, [], products_with_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_products", default="product_entities_normalized.json")
    parser.add_argument("--max_concurrent", type=int, default=102)
    args = parser.parse_args()

    all_api_keys = get_all_api_keys_in_order()
    llm_model = create_llm_with_config(all_api_keys[0])

    print(f"Loading products from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    products = data.get("products", [])
    
    loop = asyncio.get_event_loop()
    full_mapping, _, normalized_products = loop.run_until_complete(
        process_all_products(llm_model, products, args.max_concurrent)
    )

    print(f"Saving mapping to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    output_prod_path = args.output_products
    if not os.path.isabs(output_prod_path):
        output_prod_path = os.path.join(os.path.dirname(args.output), output_prod_path)
        
    print(f"Saving products to {output_prod_path}...")
    with open(output_prod_path, 'w') as f:
        json.dump({"products": normalized_products}, f, indent=2)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
