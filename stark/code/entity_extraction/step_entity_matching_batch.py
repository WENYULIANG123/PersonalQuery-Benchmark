import os
import json
from typing import List, Dict, Any
from utils import log_with_timestamp
from pipeline_config import (
    PRODUCT_ENTITIES_FILE, 
    USER_PREFERENCES_FILE, 
    MATCHED_ENTITIES_FILE,
    TARGET_USER
)
from model import submit_batch_inference, wait_for_batch_results
from entity_matching import (
    _normalize_entity_category_dict, 
    process_entity_matching_dict_response_with_sentiment,
    generate_formatted_product_output
)

def run_entity_matching_batch(config):
    """
    Performs entity matching using the SiliconFlow Batch API.
    Uses product_entities.json and user_preference_entities.json as inputs.
    """
    log_with_timestamp('🔗 Starting entity matching (BATCH MODE)...')
    
    if not os.path.exists(PRODUCT_ENTITIES_FILE):
        log_with_timestamp(f"❌ Product entities file not found: {PRODUCT_ENTITIES_FILE}")
        return None
    if not os.path.exists(USER_PREFERENCES_FILE):
        log_with_timestamp(f"❌ User preferences file not found: {USER_PREFERENCES_FILE}")
        return None

    # 1. Load Data
    try:
        with open(PRODUCT_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            prod_data = json.load(f)
        with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
            pref_data = json.load(f)
            
        prod_map = {p['asin']: p for p in prod_data.get('products', []) if 'asin' in p}
        pref_map = {p['asin']: p for p in pref_data.get('products', []) if 'asin' in p}
        
        all_asins = sorted(list(set(prod_map.keys()) & set(pref_map.keys())))
        log_with_timestamp(f"📋 Found {len(all_asins)} products with both product entities and user preferences.")
        
        if not all_asins:
            log_with_timestamp("⚠️ No products to match.")
            return None

        # 1.5 Load Cache
        asin_to_matched = {}
        if os.path.exists(MATCHED_ENTITIES_FILE):
            try:
                with open(MATCHED_ENTITIES_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                for p in cache_data.get('products', []):
                    asin = p.get('asin')
                    matched = p.get('matched_entities')
                    # We consider it cached if there's a match result (even empty if user preferences were actually empty)
                    # For safety, let's only skip if it has entries or if we've processed it before.
                    # Usually, if matched_entities is present, it's cached.
                    if asin and isinstance(matched, dict):
                         asin_to_matched[asin] = matched
                log_with_timestamp(f"📦 Loaded {len(asin_to_matched)} cached matched products.")
            except Exception as e:
                log_with_timestamp(f"⚠️ Error loading matching cache: {e}")

        # 2. Prepare Batch Prompts
        prompts = []
        ordered_asins = []
        
        for asin in all_asins:
            if asin in asin_to_matched:
                continue

            p_entry = prod_map[asin]
            u_entry = pref_map[asin]
            
            product_entities = _normalize_entity_category_dict(p_entry.get('product_entities', {}))
            user_entities = _normalize_entity_category_dict(u_entry.get('user_preference_entities', {}))
            
            if not product_entities or not user_entities:
                continue

            prompt = f"""
You are an expert at matching product features with user preferences.

Given two JSON objects:
- Product entities (category -> list of entity strings): {json.dumps(product_entities, ensure_ascii=False)}
- User preference entities (category -> list of objects with "entity" and "sentiment"): {json.dumps(user_entities, ensure_ascii=False)}

Task:
1) For each category, select the BEST matching product entity/entities that satisfy the user preference entities.
2) IMPORTANT: Each matched entity MUST include the sentiment from the corresponding user preference entity.
3) Output MUST be a JSON object where:
   - Keys are categories
   - Values are arrays of objects with "entity" (the matched PRODUCT entity string) and "sentiment" (from user preference: "positive", "negative", or "neutral")

Example output format:
{{
  "Color": [{{"entity": "Blue", "sentiment": "positive"}}],
  "Material": [{{"entity": "Cotton", "sentiment": "negative"}}],
  "Quantity": [{{"entity": "12", "sentiment": "positive"}}]
}}

If a category has no good match, omit the key or use an empty array.

Output requirement:
Return ONLY valid JSON. No explanations.
"""
            prompts.append(prompt)
            ordered_asins.append(asin)

        if not prompts:
            log_with_timestamp("✅ All entities are already matched according to cache.")
        else:
            # 3. Submit Batch
            log_with_timestamp(f'🚀 Submitting entity matching batch for {len(prompts)} products...')
            batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
            log_with_timestamp(f'⏳ Batch submitted! ID: {batch_id}. Waiting for completion...')
            
            # 4. Wait and Retrieve
            batch_results = wait_for_batch_results(batch_id, poll_interval=30)
            
            # 5. Parse and Merge Results
            for res in batch_results:
                custom_id = res.get('custom_id', '')
                try:
                    idx = int(custom_id.split('-')[1])
                    asin = ordered_asins[idx]
                    
                    content = ""
                    if 'response' in res and 'body' in res['response']:
                        choices = res['response']['body'].get('choices', [])
                        if choices: content = choices[0]['message'].get('content', '')
                    
                    if content:
                        try:
                            matched_entities = process_entity_matching_dict_response_with_sentiment(content)
                            # Drop empty arrays
                            matched_entities = {k: v for k, v in matched_entities.items() if isinstance(v, list) and len(v) > 0}
                            asin_to_matched[asin] = matched_entities
                        except Exception as e:
                            log_with_timestamp(f"⚠️ Error parsing response for {asin}: {e}")
                except Exception as e:
                    log_with_timestamp(f"⚠️ Error processing result item: {e}")

        # 6. Final Assemblies
        final_products = []
        for asin in all_asins:
            p_entry = prod_map[asin].copy()
            u_entry = pref_map[asin]
            
            # Merge user preferences into the product entry for the final output
            p_entry['user_preference_entities'] = u_entry.get('user_preference_entities', {})
            p_entry['matched_entities'] = asin_to_matched.get(asin, {})
            
            # Format output string
            p_entry['formatted_output'] = generate_formatted_product_output(
                p_entry, len(final_products), len(all_asins)
            )
            final_products.append(p_entry)

        # 7. Save
        output_data = {'user_id': TARGET_USER, 'products': final_products}
        with open(MATCHED_ENTITIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        log_with_timestamp(f'✅ Entity matching results saved to {MATCHED_ENTITIES_FILE}')
        return MATCHED_ENTITIES_FILE

    except Exception as e:
        log_with_timestamp(f'⚠️ Batch entity matching failed: {e}')
        import traceback
        traceback.print_exc()
        return None
