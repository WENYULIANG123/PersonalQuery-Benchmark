import os
import json
import argparse
from typing import Dict, List, Any
from utils import log_with_timestamp
from pipeline_config import MATCHED_ENTITIES_FILE, MATCHED_ENTITIES_NORMALIZED_FILE, TARGET_USER
from model import submit_batch_inference, wait_for_batch_results
from normalization_utils import (
    normalize_entity_key, build_generic_classification_prompt, SCHEMA
)

def run_matched_normalization_batch(config):
    """
    Normalizes the entities in the matched_entities field of matched_entities.json.
    Uses the same logic and schema as product normalization.
    """
    log_with_timestamp('⚖️ Starting matched entity normalization (BATCH MODE)...')
    
    if not os.path.exists(MATCHED_ENTITIES_FILE):
        log_with_timestamp(f"❌ Matched entities file not found: {MATCHED_ENTITIES_FILE}")
        return None

    # 1. Load Data
    try:
        with open(MATCHED_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data.get('products', [])
        if not products:
             log_with_timestamp("⚠️ No products to normalize in matched results.")
             return None

        # 1.5 Load Cache (from the normalized file if it exists)
        cached_normalized_items = {}
        if os.path.exists(MATCHED_ENTITIES_NORMALIZED_FILE):
            try:
                with open(MATCHED_ENTITIES_NORMALIZED_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                for p in cache_data.get('products', []):
                    asin = p.get('asin')
                    m_entities = p.get('matched_entities')
                    if asin and m_entities:
                        cached_normalized_items[asin] = m_entities
            except Exception as e:
                log_with_timestamp(f"⚠️ Error loading matched normalization cache: {e}")

        # 2. Prepare Batch Prompts
        prompts = []
        product_asins = []
        
        for p in products:
            asin = p.get("asin")
            matched_entities = p.get("matched_entities", {})
            if not isinstance(matched_entities, dict) or not matched_entities:
                continue
            
            # If we already have this product in the normalized file, use that.
            if asin in cached_normalized_items:
                # Optional: Check if the entities still match. For now, we trust the cache.
                continue
                
            # Prepare entities for this specific product
            # For matching, we have {category: [{"entity": "...", "sentiment": "..."}]}
            # The normalization needs {category: [list of strings]}
            entities_to_normalize = {}
            for key, items in matched_entities.items():
                norm_key = normalize_entity_key(key)
                if norm_key in SCHEMA and isinstance(items, list):
                    values = []
                    for item in items:
                        if isinstance(item, dict):
                            val = item.get("entity", "").strip()
                        else:
                            val = str(item).strip()
                        if val:
                            values.append(val)
                    if values:
                        entities_to_normalize[norm_key] = values
            
            if not entities_to_normalize:
                continue
            
            prompt = build_generic_classification_prompt(entities_to_normalize)
            prompts.append(prompt)
            product_asins.append(p.get("asin", "unknown"))

        if not prompts:
            log_with_timestamp("✅ No matched entities need normalization.")
            return MATCHED_ENTITIES_FILE

        # 3. Submit Batch
        log_with_timestamp(f'🚀 Submitting matched normalization batch for {len(prompts)} products...')
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        log_with_timestamp(f'⏳ Batch submitted! ID: {batch_id}. Waiting for completion...')
        
        # 4. Wait and Retrieve
        batch_results = wait_for_batch_results(batch_id, poll_interval=30)
        
        # 5. Parse Results
        asin_to_normalized = {}
        for res in batch_results:
            custom_id = res.get('custom_id', '')
            try:
                idx = int(custom_id.split('-')[1])
                asin = product_asins[idx]
                
                content = ""
                if 'response' in res and 'body' in res['response']:
                    choices = res['response']['body'].get('choices', [])
                    if choices: content = choices[0]['message'].get('content', '')
                
                if content:
                    clean_str = content.strip()
                    if "```json" in clean_str:
                        clean_str = clean_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_str:
                        clean_str = clean_str.split("```")[1].split("```")[0].strip()
                    
                    try:
                        mappings = json.loads(clean_str)
                        asin_to_normalized[asin] = mappings
                    except Exception as e:
                        log_with_timestamp(f"⚠️ JSON parse error for {asin}: {e}")
            except:
                continue

        # 6. Apply & Merge Back
        final_products = []
        for p in products:
            asin = p.get("asin")
            new_p = p.copy()
            
            if asin in asin_to_normalized:
                mappings = asin_to_normalized[asin]
                matched_entities = new_p.get("matched_entities", {})
                
                for cat, items in matched_entities.items():
                    norm_cat = normalize_entity_key(cat)
                    if norm_cat in mappings:
                        cat_mappings = mappings[norm_cat]
                        for item in items:
                            entity_name = item.get("entity") if isinstance(item, dict) else str(item)
                            if entity_name in cat_mappings:
                                if isinstance(item, dict):
                                    item["NormalizedClass"] = cat_mappings[entity_name]
            elif asin in cached_normalized_items:
                # Use previously normalized entities
                new_p["matched_entities"] = cached_normalized_items[asin]
                
            final_products.append(new_p)

        # 7. Save to the NEW file
        output_data = {"user_id": TARGET_USER, "products": final_products}
        with open(MATCHED_ENTITIES_NORMALIZED_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log_with_timestamp(f'✅ Matched entities normalized and saved to {MATCHED_ENTITIES_NORMALIZED_FILE}')
        return MATCHED_ENTITIES_NORMALIZED_FILE

    except Exception as e:
        log_with_timestamp(f'⚠️ Batch matched normalization failed: {e}')
        return None
