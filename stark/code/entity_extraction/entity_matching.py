import json
import os
import re
import threading
import concurrent.futures
from typing import List, Dict, Any, Tuple
import ast

from utils import log_with_timestamp, get_all_api_keys_in_order, create_llm_with_config, try_api_keys_with_fallback
from model import call_llm_with_retry, APIErrorException

def _normalize_entity_category_dict(entities: Any) -> Dict[str, List[Any]]:
    """Helper to ensure entity dictionary is in a consistent {category: [items]} format."""
    if not isinstance(entities, dict):
        return {}
    
    normalized = {}
    for k, v in entities.items():
        if isinstance(v, list):
            normalized[k] = v
        elif isinstance(v, (str, dict)):
            normalized[k] = [v]
        else:
            normalized[k] = []
    return normalized

def match_product_and_user_entities_one_call(product_entities: Dict, user_entities: Dict, llm_model) -> Dict[str, List[Dict]]:
    """
    Using a single LLM call to match product entities with user preference entities.
    Returns matched entities with sentiment.
    """
    
    prompt = f"""
You are an expert at matching product features with user preferences.

Given two JSON objects:
- Product entities (category -> list of entity strings): {json.dumps(product_entities, ensure_ascii=False)}
- User preference entities (category -> list of objects with "entity" and "sentiment"): {json.dumps(user_entities, ensure_ascii=False)}

Task:
1) For each category, select the BEST matching product entity/entities that satisfy the user preference entities.
2) IMPORTANT: Each matched entity MUST include the sentiment from the corresponding user preference entity.
3) Output MUST be a JSON object where:
   - Keys are CATEGORY NAMES from the input list (e.g., "Brand", "Color", "Material", "Accessories") 
   - Values are ARRAYS of objects.
   - Each object MUST have: {"entity": "the matched product entity", "sentiment": "the user's sentiment"}

⚠️ CRITICAL RULE: NEVER use "entity", "sentiment", "value", or "category" as a top-level JSON key. Use the actual product category names (like "Brand", "Usage", "Material").

✅ CORRECT: {{"Color": [{{"entity": "Red", "sentiment": "positive"}}]}}
❌ WRONG: {{"entity": [{{"entity": "Red", "sentiment": "positive"}}], "sentiment": "positive"}}

If a category has no good match, omit the key or use an empty array.

Output requirement:
Return ONLY valid JSON. No explanations.
"""
    
    try:
        response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_matching_one_call")
        if success and response_str:
            return process_entity_matching_dict_response_with_sentiment(response_str)
        return {}
    except Exception as e:
        print(f"Error in match_product_and_user_entities_one_call: {e}", flush=True)
        return {}

def process_entity_matching_dict_response_with_sentiment(response_str: str) -> Dict[str, List[Any]]:
    """Robustly parse the JSON response from entity matching with sentiment."""
    if not response_str:
        return {}

    s = response_str.strip()
    
    # Try to extract the JSON block
    potential_json_objs = []
    stack = []
    start_idx = -1
    for i, char in enumerate(s):
        if char == '{':
            if not stack: start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    potential_json_objs.append(s[start_idx:i+1])
    
    obj = {}
    if potential_json_objs:
        # Try from the end (QwQ usually puts JSON at the end)
        for block in reversed(potential_json_objs):
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    break
            except:
                continue
    else:
        # Simple cleanup as fallback
        if "```json" in s:
            s = s.split("```json")[-1].split("```")[0].strip()
        elif "```" in s:
            s = s.split("```")[-1].split("```")[0].strip()
        try:
            obj = json.loads(s)
        except:
            return {}

    if not isinstance(obj, dict):
        return {}

    forbidden_keys = {"entity", "sentiment", "value", "category", "matched_entities", "matching"}
    out = {}
    
    for k, v in obj.items():
        key = str(k).strip()
        if key in forbidden_keys:
            print(f"⚠️ Skipping hallucinated top-level key: {key}", flush=True)
            continue
            
        if not v or not isinstance(v, list):
            continue
            
        cleaned_list = []
        for item in v:
            if isinstance(item, dict):
                ent = item.get("entity", "")
                sent = item.get("sentiment", "neutral")
                if ent:
                    cleaned_list.append({"entity": str(ent), "sentiment": str(sent)})
            elif isinstance(item, str):
                cleaned_list.append({"entity": item, "sentiment": "neutral"})
        
        if cleaned_list:
            out[key] = cleaned_list
            
    return out

def generate_formatted_product_output(product, idx, total_products):
    """生成格式化的产品输出字符串"""
    asin = product.get('asin', 'Unknown')
    product_title = product.get('product_title', 'Unknown Product')
    
    product_entities_raw = product.get('product_entities', {})
    user_entities_raw = product.get('user_preference_entities', {})
    matched_entities_raw = product.get('matched_entities', {})

    # Calculate counts
    p_count = sum(len(v) if isinstance(v, list) else 1 for v in product_entities_raw.values()) if isinstance(product_entities_raw, dict) else 0
    u_count = sum(len(v) if isinstance(v, list) else 1 for v in user_entities_raw.values()) if isinstance(user_entities_raw, dict) else 0
    m_count = sum(len(v) if isinstance(v, list) else 1 for v in matched_entities_raw.values()) if isinstance(matched_entities_raw, dict) else 0

    # Format matched entities for display
    m_display = []
    if isinstance(matched_entities_raw, dict):
        for cat, items in matched_entities_raw.items():
            for it in items:
                if isinstance(it, dict):
                    m_display.append(f"{cat}:{it.get('entity','')} ({it.get('sentiment','')})")
                else:
                    m_display.append(f"{cat}:{it}")

    output_lines = [
        f"[{idx+1}/{total_products}] Product: {product_title}",
        f"ASIN: {asin}",
        f"Product Entities Count: {p_count}",
        f"User Preference Entities Count: {u_count}",
        f"Matched Entities ({m_count}): {', '.join(m_display) if m_display else 'None'}",
        ""
    ]
    return "\n".join(output_lines)

def perform_entity_matching(products: List[Dict], max_workers: int = 102) -> List[Dict]:
    """执行产品实体和用户偏好实体的匹配（并发版本）"""
    log_with_timestamp(f"🔗 Starting entity matching for {len(products)} products with {max_workers} workers...")

    if not products:
        log_with_timestamp("⚠️ No products found for matching")
        return products

    all_api_keys = get_all_api_keys_in_order()
    total_products = len(products)
    
    progress_counter = {'completed': 0, 'matched': 0}
    progress_lock = threading.Lock()

    def process_single_product(product_with_idx):
        idx, product = product_with_idx
        asin = product.get('asin', 'Unknown')
        
        try:
            product_entities = _normalize_entity_category_dict(product.get('product_entities', {}))
            user_entities = _normalize_entity_category_dict(product.get('user_preference_entities', {}))

            if not product_entities or not user_entities:
                product['matched_entities'] = {}
            else:
                def matching_operation(api_config, provider_name, key_index):
                    llm_model = create_llm_with_config(api_config)
                    return match_product_and_user_entities_one_call(product_entities, user_entities, llm_model)

                matched, success = try_api_keys_with_fallback(all_api_keys, matching_operation, f"{asin} matching")
                product['matched_entities'] = matched if success else {}

            product['formatted_output'] = generate_formatted_product_output(product, idx, total_products)

            with progress_lock:
                progress_counter['completed'] += 1
                if product['matched_entities']: progress_counter['matched'] += 1
                current = progress_counter['completed']
                if current % 10 == 0 or current == total_products:
                    log_with_timestamp(f'📊 Progress: {current}/{total_products} matched')
            
            return product
        except Exception as e:
            log_with_timestamp(f'❌ Exception for {asin}: {e}')
            product['matched_entities'] = {}
            product['formatted_output'] = generate_formatted_product_output(product, idx, total_products)
            return product

    products_with_idx = [(i, p) for i, p in enumerate(products)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_product, products_with_idx))

    log_with_timestamp(f"✅ Completed! {progress_counter['matched']}/{total_products} matched.")
    return results