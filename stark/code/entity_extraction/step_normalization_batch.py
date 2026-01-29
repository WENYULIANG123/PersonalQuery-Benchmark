import os
import json
import time
from typing import Dict, List, Any
from utils import log_with_timestamp
from pipeline_config import PRODUCT_ENTITIES_FILE, PRODUCT_ENTITIES_NORMALIZED_FILE, TARGET_USER
from normalization_utils import (
    normalize_entity_key, SCHEMA, build_generic_classification_prompt
)
from model import submit_batch_inference, wait_for_batch_results

def run_normalization_batch(config):
    """
    Runs entity normalization using the SiliconFlow Batch API.
    Dimension normalization is removed per user request as it's included in presets.
    """
    log_with_timestamp('⚖️ Starting entity normalization (BATCH MODE)...')
    
    if not os.path.exists(PRODUCT_ENTITIES_FILE):
        log_with_timestamp(f"❌ Product entities file not found: {PRODUCT_ENTITIES_FILE}")
        return None

    # 1. Load Data
    try:
        with open(PRODUCT_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data.get('products', [])
        if not products:
             log_with_timestamp("⚠️ No products to normalize.")
             return None

        # 1.5 Load Cache
        cached_normalized = {}
        if os.path.exists(PRODUCT_ENTITIES_NORMALIZED_FILE):
            try:
                with open(PRODUCT_ENTITIES_NORMALIZED_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                for p in cache_data.get('products', []):
                    asin = p.get('asin')
                    norm = p.get('normalized_entities')
                    if asin and norm:
                        cached_normalized[asin] = norm
                log_with_timestamp(f"📦 Loaded {len(cached_normalized)} cached normalized products.")
            except Exception as e:
                log_with_timestamp(f"⚠️ Error loading normalization cache: {e}")

        # 2. Prepare Batch Prompts
        prompts = []
        product_asins = []
        
        for p in products:
            asin = p.get("asin", "unknown")
            if asin in cached_normalized:
                continue

            product_entities = p.get("product_entities", {})
            if not isinstance(product_entities, dict):
                continue
                
            # Prepare entities for this specific product
            normalized_entities_input = {}
            for key, values in product_entities.items():
                norm_key = normalize_entity_key(key)
                if norm_key in SCHEMA and isinstance(values, list):
                    normalized_entities_input[norm_key] = [str(v).strip() for v in values if v]
            
            if not normalized_entities_input:
                continue
            
            # Use the classification prompt from normalization_utils
            prompt = build_generic_classification_prompt(normalized_entities_input)
            prompts.append(prompt)
            product_asins.append(p.get("asin", "unknown"))

        if not prompts:
            log_with_timestamp("✅ All entities are already normalized according to cache.")
            # Even if all are cached, we should ensure the file is in sync with product_entities.json
            # in case items were removed or the order changed.
            # (Logic handled in Step 6 & 7)
        else:
            # 3. Submit Batch
            log_with_timestamp(f'🚀 Submitting normalization batch for {len(prompts)} products...')
            batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
            log_with_timestamp(f'⏳ Batch submitted! ID: {batch_id}. Waiting for completion...')
            
            # 4. Wait and Retrieve
            batch_results = wait_for_batch_results(batch_id, poll_interval=30)
            if not batch_results:
                log_with_timestamp("❌ Batch inference failed or returned no results.")
                return None

            # 5. Parse and Apply Results
            for res in batch_results:
                custom_id = res.get('custom_id', '')
                try:
                    idx = int(custom_id.split('-')[1])
                    asin = product_asins[idx]
                    
                    content = ""
                    if 'response' in res and 'body' in res['response']:
                        choices = res['response']['body'].get('choices', [])
                        if choices:
                            content = choices[0]['message'].get('content', '')
                    
                    if content:
                        # Parse the JSON response
                        clean_str = content.strip()
                        if "```json" in clean_str:
                            clean_str = clean_str.split("```json")[1].split("```")[0].strip()
                        elif "```" in clean_str:
                            clean_str = clean_str.split("```")[1].split("```")[0].strip()
                        
                        try:
                            mappings = json.loads(clean_str)
                            cached_normalized[asin] = mappings
                        except Exception as e:
                            log_with_timestamp(f"⚠️ JSON parse error for {asin}: {e}")
                    else:
                        log_with_timestamp(f"⚠️ Empty response for {asin}")
                except:
                    continue

        # 6. Merge Back
        normalized_products = []
        for p in products:
            asin = p.get("asin")
            new_p = p.copy()
            if asin in cached_normalized:
                new_p["normalized_entities"] = cached_normalized[asin]
            else:
                new_p["normalized_entities"] = {}
            normalized_products.append(new_p)

        # 7. Save
        output_data = {"user_id": TARGET_USER, "products": normalized_products}
        with open(PRODUCT_ENTITIES_NORMALIZED_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log_with_timestamp(f'✅ Normalized entities saved to {PRODUCT_ENTITIES_NORMALIZED_FILE}')
        return PRODUCT_ENTITIES_NORMALIZED_FILE

    except Exception as e:
        log_with_timestamp(f'⚠️ Batch normalization failed: {e}')
        import traceback
        traceback.print_exc()
        return None
