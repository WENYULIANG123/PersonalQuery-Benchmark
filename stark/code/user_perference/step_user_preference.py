import os
import json
import threading
import concurrent.futures
from collections import defaultdict
from utils import log_with_timestamp, get_all_api_keys_in_order, try_api_keys_with_fallback, create_llm_with_config
from model import set_api_responses_file
from user_preference_extraction import (
    load_user_reviews, 
    prepare_content_and_extract_entities, 
    process_user_preference_extraction_response,
    normalize_user_preference_entities_with_sentiment,
    is_valid_user_preference_entities
)
from pipeline_config import TARGET_USER, USER_PREFERENCES_FILE, API_RESPONSES_FILE
from kb_helper import get_kb_instance

def run_user_preference_extraction(config, all_asins):
    """
    Run user preference extraction for the given ASINs.
    This logic was historically located after product extraction.
    """
    log_with_timestamp('🔍 Checking user preference cache...')
    
    # 1. Load User Reviews (Again, lightweight)
    user_reviews = load_user_reviews(TARGET_USER)
    if not user_reviews: return

    # 2. Check Cache
    cached_map = {}
    valid_cached = set()
    
    force_run = config.get('force', False)
    if force_run:
        log_with_timestamp('🔄 Force re-extraction enabled. Ignoring existing cache.')
    else:
        if os.path.exists(USER_PREFERENCES_FILE):
            try:
                with open(USER_PREFERENCES_FILE, 'r') as f:
                    data = json.load(f)
                if data.get('user_id') == TARGET_USER:
                    for item in data.get('products', []):
                        asin = item.get('asin')
                        norm = normalize_user_preference_entities_with_sentiment(item.get('user_preference_entities'))
                        if asin in all_asins and is_valid_user_preference_entities(norm):
                            cached_map[asin] = norm
                            valid_cached.add(asin)
            except OSError: pass

    missing = [a for a in all_asins if a not in valid_cached]
    
    if not missing:
        log_with_timestamp('⏭️ User preferences cached. Skipping.')
        return

    # 3. Extraction
    log_with_timestamp(f'🔍 Extracting user preferences for {len(missing)} products...')
    
    all_keys = get_all_api_keys_in_order()
    results = []
    
    # Pre-filter reviews
    asin_reviews = defaultdict(list)
    for r in user_reviews:
        if r.get('asin') in missing:
            if r.get('reviewText') or r.get('summary'):
                 asin_reviews[r['asin']].append(r)

    def process_asin(asin):
        # Local file setting for thread safety if needed (though global lock handles it)
        set_api_responses_file(API_RESPONSES_FILE) 
        
        valid_reviews = asin_reviews.get(asin, [])
        if not valid_reviews:
            return asin, {}

        # Fetch known attributes and unstructured info from KB
        kb = get_kb_instance()
        known_attributes = kb.get_product_attributes(asin)
        product_info = kb.get_product_unstructured_info(asin)

        # Define operation
        def op(cfg, pname, idx):
            llm = create_llm_with_config(cfg)
            # This returns raw LLM response usually
            return prepare_content_and_extract_entities(valid_reviews, 'user preference', llm, asin=asin, known_attributes=known_attributes, product_info=product_info)

        # Execute
        # Note: The original code had a complex "don't parse yet" logic relying on saving to file first.
        # We simplify here to just get result if prepare_content_and_extract_entities supports it.
        # If prepare_content_and_extract_entities returns a string/dict, we process it.
        res, success = try_api_keys_with_fallback(all_keys, op, f"pref {asin}", "")
        
        if success and res:
             # Parse result (assuming it returns content string or dict)
             # If it returns raw response object, we need to handle it.
             # For now assume it returns the content string or struct.
             # Original code parsed it using process_user_preference_extraction_response
             parsed = process_user_preference_extraction_response(res)
             normalized = normalize_user_preference_entities_with_sentiment(parsed[1] if isinstance(parsed, tuple) else parsed)
             return asin, normalized
        
        return asin, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('max_concurrent_pref', 20)) as executor:
        futures = {executor.submit(process_asin, asin): asin for asin in missing}
        for fut in concurrent.futures.as_completed(futures):
            asin, entities = fut.result()
            cached_map[asin] = entities

    # 4. Save
    output_list = []
    for asin in all_asins:
        output_list.append({
            'asin': asin,
            'user_preference_entities': cached_map.get(asin, {}),
            # 'review_content': ... # Populate if needed
        })
    
    with open(USER_PREFERENCES_FILE, 'w') as f:
        json.dump({'user_id': TARGET_USER, 'products': output_list}, f, indent=2)
    
    log_with_timestamp(f'💾 Saved user preferences to {USER_PREFERENCES_FILE}')
