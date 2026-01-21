#!/usr/bin/env python3
"""
Entity Normalization using LLM
Reads unique entity values and uses LLM to classify them into the STARK entity schema.
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Set
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry, APIErrorException
from utils import create_llm_with_config, get_all_api_keys_in_order

# Schema Definition (The target categories)
SCHEMA = {
    "Color": ["Blue", "Green", "Red", "Yellow", "Purple", "Brown", "Gray", "White", "Black", "Metallic", "Other"],
    "Material": ["Textile", "Metal", "Plastic", "Wood", "Paper", "Mineral", "Glass", "Medium", "Other"],
    "Usage": ["Art", "Craft", "Sew", "Write", "Card", "Technical", "Office", "Storage", "Other"],
    "Dimensions": ["Small", "Medium", "Large", "Other"],
    "Quantity": ["Single", "Bulk", "Other"],
    "Safety/Certification": ["Safe", "Professional", "Toxic", "Other"],
    "Design": ["Shape", "Pattern", "Style", "Format", "Other"],
    "Selling Point": ["Quality", "Feature", "Portable", "Origin", "Other"]
}

def get_unique_values(input_file: str) -> Dict[str, Set[str]]:
    """Extract all unique values for each entity type from product_entities.json"""
    print(f"Loading entities from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    products = data.get("products", [])
    unique_values = {k: set() for k in SCHEMA.keys()}
    
    for p in products:
        entities = p.get("product_entities", {})
        if not isinstance(entities, dict):
            continue
            
        for key, values in entities.items():
            # Normalize key
            norm_key = key
            if key.lower() in ["color/finish", "colour/finish"]: norm_key = "Color"
            elif key.lower() == "selling_point": norm_key = "Selling Point"
            elif key.lower() == "size": norm_key = "Dimensions"
            
            if norm_key in unique_values:
                for v in values:
                    if v and isinstance(v, str):
                        unique_values[norm_key].add(v.strip())
                        
    return unique_values

def classify_batch(llm_model, category: str, values: List[str]) -> Dict[str, str]:
    """Classify a batch of values for a specific category using LLM"""
    if not values:
        return {}

    target_classes = SCHEMA[category]
    prompt = f"""
You are an expert data normalization assistant.
Your task is to classify raw e-commerce attribute values into a standardized schema.

**Target Category:** {category}
**Allowed Classes:** {json.dumps(target_classes)}

**Instructions:**
1. For each input value, choose the BEST matching class from the Allowed Classes list.
2. If none match perfectly, choose "Other".
3. Return a JSON object where keys are the input values and values are the chosen classes.

**Input Values to Classify:**
{json.dumps(values)}

**Output Format:**
Strict JSON object only: {{"input_value": "SelectedClass", ...}}
"""
    
    try:
        response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_normalization")
        if success and response_str:
            # Simple cleanup for json parsing
            cleanup_str = response_str.strip()
            if cleanup_str.startswith("```json"):
                cleanup_str = cleanup_str.split("```json")[1].split("```")[0].strip()
            elif cleanup_str.startswith("```"):
                cleanup_str = cleanup_str.split("```")[1].split("```")[0].strip()
                
            result = json.loads(cleanup_str)
            return result
    except Exception as e:
        print(f"Error classifying batch for {category}: {e}")
        return {v: "Other" for v in values} # Fallback
        
    return {v: "Other" for v in values} # Fallback if parsing fails

def main():
    parser = argparse.ArgumentParser(description="Normalize entities using LLM")
    parser.add_argument("--input", required=True, help="Path to product_entities.json")
    parser.add_argument("--output", required=True, help="Path to save mapping json")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of items per LLM call")
    args = parser.parse_args()

    # Load API config
    all_api_keys = get_all_api_keys_in_order()
    if not all_api_keys:
        raise RuntimeError("No API keys available")
    api_config = all_api_keys[0]  # Use first available key
    llm_model = create_llm_with_config(api_config)

    # 1. Get unique values
    unique_map = get_unique_values(args.input)
    
    full_mapping = {} # Nested dict: {Category: {OriginalVal: NormalizedVal}}

    # 2. Process each category
    for category, val_set in unique_map.items():
        val_list = sorted(list(val_set))
        print(f"Processing {category}: {len(val_list)} unique values...")
        
        category_mapping = {}
        
        # Batch processing
        for i in tqdm(range(0, len(val_list), args.batch_size)):
            batch = val_list[i : i + args.batch_size]
            
            # Use LLM to classify
            batch_result = classify_batch(llm_model, category, batch)
            
            # Update mapping
            for v in batch:
                # Use result from LLM, default to Other if missing
                category_mapping[v] = batch_result.get(v, "Other")
                
        full_mapping[category] = category_mapping

    # 3. Save result
    print(f"Saving mapping to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(full_mapping, f, indent=2)

if __name__ == "__main__":
    main()
