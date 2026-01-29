import os
import json
from collections import defaultdict
from typing import List, Dict, Any
from utils import log_with_timestamp
from pipeline_config import TARGET_USER, USER_PREFERENCES_FILE
from user_preference_extraction import (
    load_user_reviews, 
    clean_html_content, 
    process_user_preference_extraction_response,
    prepare_and_clean_review_content,
    construct_user_preference_prompt
)
from model import submit_batch_inference, wait_for_batch_results
from kb_helper import get_kb_instance

def run_user_preference_extraction_batch(config, all_asins):
    """
    Orchestrates the extraction of user preferences using Batch API.
    Returns the path to the saved user preferences file.
    """
    log_with_timestamp('🔍 Checking user preference cache (BATCH MODE)...')
    
    # 1. Load User Reviews
    user_reviews = load_user_reviews(TARGET_USER)
    if not user_reviews:
        log_with_timestamp(f"❌ No reviews found for user {TARGET_USER}")
        return None

    # 2. Check Cache
    cached_map = {}
    asin_to_min_category = {}
    valid_cached = set()
    
    force_run = config.get('force', False)
    
    if not force_run and os.path.exists(USER_PREFERENCES_FILE):
        try:
            with open(USER_PREFERENCES_FILE, 'r') as f:
                data = json.load(f)
            if data.get('user_id') == TARGET_USER:
                for item in data.get('products', []):
                    asin = item.get('asin')
                    if asin in all_asins:
                        cached_map[asin] = item.get('user_preference_entities', {})
                        if 'min_category' in item:
                            asin_to_min_category[asin] = item['min_category']
                        valid_cached.add(asin)
        except Exception: pass

    missing = [a for a in all_asins if a not in valid_cached]
    
    if force_run:
        log_with_timestamp(f'🔄 Force re-extraction for all {len(all_asins)} products (Cache ignored).')
    elif not missing:
        log_with_timestamp('⏭️ User preferences cached (BATCH). Skipping.')
        return USER_PREFERENCES_FILE
    else:
        log_with_timestamp(f'🔍 Found {len(valid_cached)} products in cache, extracting {len(missing)} missing items.')

    # ... (Prompts preparation remains same) ...
    asin_reviews = defaultdict(list)
    for r in user_reviews:
        asin = r.get('asin')
        if asin in missing:
            if r.get('reviewText') or r.get('summary'):
                 asin_reviews[asin].append(r)

    prompts = []
    asin_order = []
    
    kb = get_kb_instance() # Initialize KB outside the loop
    for asin in missing:
        reviews = asin_reviews.get(asin, [])
        if not reviews:
            continue
            
        # Combine and clean reviews
        content = prepare_and_clean_review_content(reviews)
        
        # Fetch known attributes and unstructured info from KB
        known_attributes = kb.get_product_attributes(asin)
        product_info = kb.get_product_unstructured_info(asin)
        
        # Construct prompt
        prompt = construct_user_preference_prompt(content, known_attributes, product_info)
        prompts.append(prompt)
        asin_order.append(asin)

    if prompts:
        log_with_timestamp(f'🚀 Submitting batch for {len(prompts)} products...')
        batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
        log_with_timestamp(f'⏳ Batch submitted! ID: {batch_id}. Waiting...')
        
        import asyncio
        batch_results = asyncio.run(wait_for_batch_results(batch_id, poll_interval=30))
        
        for res in batch_results:
            custom_id = res.get('custom_id', '')
            try:
                idx = int(custom_id.split('-')[1])
                asin = asin_order[idx]
                
                content = ""
                if 'response' in res and 'body' in res['response']:
                    choices = res['response']['body'].get('choices', [])
                    if choices: content = choices[0]['message'].get('content', '')
                
                if content:
                    try:
                        _, entities_dict, llm_category = process_user_preference_extraction_response(content)
                        cached_map[asin] = entities_dict
                        if llm_category:
                            asin_to_min_category[asin] = llm_category
                    except Exception as e:
                        log_with_timestamp(f"⚠️ Error parsing response for {asin}: {e}")
            except Exception as e:
                log_with_timestamp(f"⚠️ Error processing result item: {e}")

    # 7. Final Output Construction
    output_list = []
    for asin in all_asins:
        prod_reviews = [r for r in user_reviews if r.get('asin') == asin]
        
        # Use LLM-selected category if available, otherwise fallback to KB heuristic
        min_category = asin_to_min_category.get(asin)
        if not min_category:
            min_category = get_kb_instance().get_min_category(asin)
        
        output_list.append({
            'asin': asin,
            'user_preference_entities': cached_map.get(asin, {}),
            'review_content': prod_reviews,
            'min_category': min_category
        })
    
    with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump({'user_id': TARGET_USER, 'products': output_list}, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f'💾 Saved user preferences to {USER_PREFERENCES_FILE}')
    return USER_PREFERENCES_FILE
