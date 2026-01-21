#!/usr/bin/env python3
"""
主函数文件
整合商品实体提取、用户偏好实体提取和实体匹配模块
"""

import os
import json
import sys
import threading
import concurrent.futures
from collections import defaultdict

 # Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_all_api_keys_in_order, set_api_responses_file

# Import modules
from product_extraction import (
    log_with_timestamp, clean_html_content, load_product_metadata, extract_product_entities_only
)

from user_preference_extraction import (
    load_user_reviews,
    prepare_content_and_extract_entities,
    TARGET_USER,
    process_user_preference_extraction_response,
    normalize_category_label,
)

from utils import (
    try_api_keys_with_fallback, create_llm_with_config
)

from entity_matching import (
    perform_entity_matching
)

# Dimension normalization for matched entities
from normalize_dimensions import process as normalize_dimensions_process

# from query_generation import generate_queries_for_matched_products

def report_progress(current, total, report_interval=10, message_template="📊 Progress: {current} / {total} {unit} processed"):
    """
    通用进度报告函数

    Args:
        current: 当前进度
        total: 总数
        report_interval: 报告间隔
        message_template: 消息模板
    """
    if current % report_interval == 0 or current == total:
        log_with_timestamp(message_template.format(current=current, total=total, unit="products"))

def is_valid_user_preference_entities(entities):
    """Check user preference entities contain usable values."""
    if not entities:
        return False
    if isinstance(entities, dict):
        for value in entities.values():
            if isinstance(value, list) and any(str(item).strip() for item in value):
                return True
            if isinstance(value, str) and value.strip():
                return True
    elif isinstance(entities, list):
        return any(str(item).strip() for item in entities)
    return False


def normalize_user_preference_entities_with_sentiment(entities):
    """
    Normalize user preference entities into {category: [{"entity": ..., "sentiment": ...}]}.
    - Strings are kept with neutral sentiment.
    - Dict items keep sentiment/polarity when valid, otherwise default to neutral.
    - Deduplicates by (category, entity, sentiment).
    """

    def _coerce_item(item):
        if isinstance(item, str):
            text = item.strip()
            if text:
                return {"entity": text, "sentiment": "neutral"}
            return None
        if isinstance(item, dict):
            text = str(item.get("entity") or item.get("text") or item.get("name") or "").strip()
            if not text:
                return None
            sentiment = str(item.get("sentiment") or item.get("polarity") or "").strip().lower()
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"
            return {"entity": text, "sentiment": sentiment}
        return None

    if not entities:
        return {}

    normalized = {}
    seen = set()

    if isinstance(entities, dict):
        for category, raw_items in entities.items():
            cat = normalize_category_label(category)
            # Accept single item or list
            items = raw_items if isinstance(raw_items, list) else [raw_items]
            for item in items:
                coerced = _coerce_item(item)
                if not coerced:
                    continue
                dedupe_key = (cat, coerced["entity"], coerced["sentiment"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                normalized.setdefault(cat, []).append(coerced)
    elif isinstance(entities, list):
        for item in entities:
            coerced = _coerce_item(item)
            if not coerced:
                continue
            dedupe_key = ("General", coerced["entity"], coerced["sentiment"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.setdefault("General", []).append(coerced)

    return normalized

def print_entity_matching_results():
    """打印实体匹配的完整结果"""
    log_with_timestamp("📋 Printing complete entity matching results...")

    # Get the workspace root directory (parent of stark directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    matched_entities_file = os.path.join(result_dir, "entity_matching_results.json")

    try:
        with open(matched_entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading results file for printing: {e}")
        return

    products = data.get('products', [])
    if not products:
        log_with_timestamp("⚠️ No products found in results file")
        return

    print(f'\\n📋 Complete Entity Matching Results ({len(products)} products):', flush=True)
    print('=' * 90, flush=True)

    sorted_products = sorted(products, key=lambda x: x.get('asin', ''))

    for idx, product in enumerate(sorted_products, 1):
        asin = product.get('asin', 'Unknown')
        product_title = product.get('product_title', 'Unknown')
        product_entities = product.get('product_entities', {})
        user_entities = product.get('user_preference_entities', {})
        matched_entities = product.get('matched_entities', {})
        reviews = product.get('reviews', [])
        metadata = product.get('metadata', {})

        # 打印产品信息
        progress_info = f" ({idx}/{len(products)})"
        print(f'Product {asin} ({product_title[:50]}...){progress_info}:', flush=True)

        # 打印review内容
        if reviews:
            # 去重review内容
            unique_reviews = []
            seen_contents = set()
            for review in reviews:
                title = review.get('summary', '').strip()
                text = review.get('reviewText', '').strip()
                # 移除文本中的换行符，用空格替换
                text = ' '.join(text.split())
                review_content = f"{title} {text}".strip()
                if review_content and review_content not in seen_contents:
                    seen_contents.add(review_content)
                    unique_reviews.append(review_content)

            print(f'  Reviews ({len(unique_reviews)} unique):', flush=True)
            for i, review_content in enumerate(unique_reviews[:3], 1):  # 只显示前3个unique review
                if review_content:
                    print(f'    Review {i}: {review_content}', flush=True)
            if len(unique_reviews) > 3:
                print(f'    ... and {len(unique_reviews) - 3} more unique reviews', flush=True)
        else:
            print('  Reviews: None found', flush=True)

        # 打印产品实体
        if product_entities:
            total_product_entities = sum(len(entities) for entities in product_entities.values())
            print(f'  Product Entities ({len(product_entities)} categories, {total_product_entities} total):', flush=True)
            for category, entities in product_entities.items():
                print(f'    {category}: {", ".join(entities)}', flush=True)
        else:
            print('  Product Entities: None extracted', flush=True)

        def _format_entities_with_sentiment(entity_list):
            formatted = []
            for item in entity_list:
                if isinstance(item, dict):
                    entity_text = str(item.get('entity') or item.get('text') or item.get('name') or "").strip()
                    sentiment = str(item.get('sentiment') or item.get('polarity') or "").strip().lower()
                    if entity_text:
                        formatted.append(f"{entity_text} ({sentiment})" if sentiment else entity_text)
                elif isinstance(item, str):
                    item = item.strip()
                    if item:
                        formatted.append(item)
            return formatted

        # 打印用户偏好实体
        if user_entities:
            total_user_entities = sum(len(entities) for entities in user_entities.values())
            print(f'  User Preference Entities ({len(user_entities)} categories, {total_user_entities} total):', flush=True)
            for category, entities in user_entities.items():
                if isinstance(entities, list):
                    formatted_entities = _format_entities_with_sentiment(entities)
                    print(f'    {category}: {", ".join(formatted_entities)}', flush=True)
                else:
                    print(f'    {category}: {entities}', flush=True)
        else:
            print('  User Preference Entities: None extracted', flush=True)

        # 打印匹配实体
        if matched_entities:
            total_matched = sum(len(entities) for entities in matched_entities.values())
            print(f'  Matched Entities ({len(matched_entities)} categories, {total_matched} total):', flush=True)
            for category, entities in matched_entities.items():
                if isinstance(entities, list):
                    formatted_entities = _format_entities_with_sentiment(entities)
                    print(f'    {category}: {", ".join(formatted_entities)}', flush=True)
                else:
                    print(f'    {category}: {entities}', flush=True)
        else:
            print('  Matched Entities: No matches found', flush=True)

        # 打印生成的查询
        generated_query = product.get('generated_query', '')
        if generated_query:
            print(f'  Generated Query: {generated_query}', flush=True)
        else:
            print('  Generated Query: None generated', flush=True)

        # 打印metadata
        print('  Metadata:', flush=True)
        for key, value in metadata.items():
            print(f'    {key}: {value}', flush=True)
        print()

    failed_products = [p for p in products if not p.get('matched_entities') or not any(matches for matches in p.get('matched_entities', {}).values())]
    if failed_products:
        print(f'\\n❌ Products with No Matches ({len(failed_products)}):', flush=True)
        for product in failed_products:
            asin = product.get('asin', 'Unknown')
            product_entities = product.get('product_entities', {})
            user_entities = product.get('user_preference_entities', {})
            product_count = sum(len(entities) for entities in product_entities.values())
            user_count = sum(len(entities) for entities in user_entities.values())
            print(f'  Product {asin}: Product entities ({len(product_entities)} categories, {product_count} total), User entities ({len(user_entities)} categories, {user_count} total)', flush=True)

def main():
    log_with_timestamp('Starting product entity extraction with API key fallback...')

    # Get the workspace root directory (parent of stark directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Set up API responses file for saving raw API data
    api_responses_file = os.path.join(result_dir, "api_raw_responses.json")
    set_api_responses_file(api_responses_file)
    log_with_timestamp(f'💾 API raw responses will be saved to {api_responses_file}')

    user_reviews = load_user_reviews(TARGET_USER)

    if not user_reviews:
        log_with_timestamp(f"❌ No reviews found for user {TARGET_USER}")
        return

    log_with_timestamp(f"✅ Found {len(user_reviews)} reviews for user {TARGET_USER}")

    user_asins = set(review.get('asin') for review in user_reviews if review.get('asin'))
    product_metadata = load_product_metadata(user_asins)

    if not product_metadata:
        log_with_timestamp(f"❌ No product metadata found for user {TARGET_USER}'s reviewed products")
        return

    log_with_timestamp(f"✅ Found metadata for {len(product_metadata)} products reviewed by user {TARGET_USER}")

    log_with_timestamp(f'Extracting product entities for {len(product_metadata)} products...')

    all_api_keys = get_all_api_keys_in_order()

    all_asins = list(product_metadata.keys())

    if not all_asins:
        log_with_timestamp("❌ No products found")
        return

    reviewed_asins = set()
    reviews_by_asin = {}
    for review in user_reviews:
        asin = review.get('asin')
        if asin and asin in product_metadata:
            reviewed_asins.add(asin)
            if asin not in reviews_by_asin:
                reviews_by_asin[asin] = []
            reviews_by_asin[asin].append(review)

    all_asins = sorted(list(reviewed_asins))
    total_products = len(all_asins)
    log_with_timestamp(f'🔍 Selected ASINs for processing: {all_asins}')

    if not all_asins:
        log_with_timestamp("❌ No reviewed products found")
        return

    # Get the workspace root directory (parent of stark directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    product_entities_file = os.path.join(result_dir, "product_entities.json")
    user_preferences_file = os.path.join(result_dir, "user_preference_entities.json")

    # Load cached product entities if exists
    cached_product_data = {}
    if os.path.exists(product_entities_file):
        try:
            with open(product_entities_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('user_id') == TARGET_USER:
                    for product in cached_data.get('products', []):
                        cached_asin = product.get('asin')
                        if cached_asin and product.get('product_entities'):
                            cached_product_data[cached_asin] = product
                    log_with_timestamp(f'📦 Loaded {len(cached_product_data)} cached product entities from {product_entities_file}')
        except Exception as e:
            log_with_timestamp(f'⚠️ Error loading cached product entities: {e}, will re-extract all products')

    all_results = []

    # Count how many products need extraction vs cached
    cached_count = sum(1 for asin in all_asins if asin in cached_product_data)
    need_extraction_count = total_products - cached_count
    if cached_count > 0:
        log_with_timestamp(f'📦 Found {cached_count} products in cache, {need_extraction_count} products need extraction')
    
    log_with_timestamp(f'🔄 Processing {len(all_asins)} products concurrently with 5 workers...')

    progress_counter = {'completed': 0}
    progress_lock = threading.Lock()

    def process_single_product(asin):
        try:
            # Check if we have cached data for this ASIN
            if asin in cached_product_data:
                cached_product = cached_product_data[asin]
                #log_with_timestamp(f'📦 Using cached data for product {asin}')
                
                # Convert cached format to result format
                product_info = product_metadata.get(asin, {})
                result = {
                    'asin': asin,
                    'product_title': cached_product.get('product_title', product_info.get('title', f'Product {asin}')),
                    'product_entities': cached_product.get('product_entities', {}),
                    'product_info': cached_product.get('product_info', {}),
                    'metadata': cached_product.get('metadata', {}),
                    'metadata_lines': [f"    {k}: {v}" for k, v in cached_product.get('metadata', {}).items()],
                    'success': True
                }
            else:
                # Get API keys for processing
                ordered_keys = get_all_api_keys_in_order()

                def extract_operation(api_config, provider_name, key_index):
                    result = extract_product_entities_only(asin, product_metadata, api_config, total_products)
                    if result and result.get('success', False):
                        return result
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                        raise Exception(f"Extraction failed: {error_msg}")

                product_result, success = try_api_keys_with_fallback(
                    ordered_keys,
                    extract_operation,
                    f"product {asin}",
                    "✅ Successfully processed {context} with {provider} Key #{key_num}"
                )

                if success:
                    result = product_result
                else:
                    result = {
                        'asin': asin,
                        'error': 'All API keys failed',
                        'success': False
                    }

            # Update progress
            with progress_lock:
                progress_counter['completed'] += 1
                current_count = progress_counter['completed']
                if current_count % 10 == 0 or current_count == total_products:
                    log_with_timestamp(f'📊 Progress: {current_count}/{total_products} products processed')

            return result

        except Exception as e:
            log_with_timestamp(f'❌ Error processing product {asin}: {e}')
            return {
                'asin': asin,
                'error': str(e),
                'success': False
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_asin = {executor.submit(process_single_product, asin): asin for asin in all_asins}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_asin):
            asin = future_to_asin[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                log_with_timestamp(f'❌ Exception processing {asin}: {e}')
                all_results.append({
                    'asin': asin,
                    'error': str(e),
                    'success': False
                })

    successful_count = len([r for r in all_results if r.get('success', False)])
    if successful_count == len(all_asins):
        log_with_timestamp('✅ All products processed successfully!')
    else:
        log_with_timestamp(f'⚠️  {successful_count}/{len(all_asins)} products processed successfully, {len(all_asins) - successful_count} failed')


    log_with_timestamp(f'🔍 Found reviews for {len(reviews_by_asin)} products')
    log_with_timestamp(f'🔍 Reviews by ASIN keys: {sorted(list(reviews_by_asin.keys()))[:10]}...')

    total_reviews_to_process = sum(len(reviews) for reviews in reviews_by_asin.values())
    log_with_timestamp(f'📊 Total user preference reviews to process: {total_reviews_to_process}')

    successful_products = [result for result in all_results if result.get('success', False) and 'error' not in result]

    # 用户偏好实体缓存检测，验证有效性并按需跳过提取
    cached_user_preferences_map = {}
    user_preferences_data = []
    valid_cached_asins = set()

    if os.path.exists(user_preferences_file):
        try:
            with open(user_preferences_file, 'r', encoding='utf-8') as f:
                cached_user_pref = json.load(f)

            if cached_user_pref.get('user_id') == TARGET_USER:
                cached_products = cached_user_pref.get('products', [])
                for item in cached_products:
                    asin = item.get('asin')
                    entities = item.get('user_preference_entities')
                    normalized_entities = normalize_user_preference_entities_with_sentiment(entities)
                    if asin and asin in all_asins and is_valid_user_preference_entities(normalized_entities):
                        cached_user_preferences_map[asin] = normalized_entities
                        valid_cached_asins.add(asin)
                        user_preferences_data.append({
                            'asin': asin,
                            'user_preference_entities': normalized_entities,
                            'review_content': item.get('review_content', [])
                        })
                missing_asins = [asin for asin in all_asins if asin not in valid_cached_asins]
                if not missing_asins:
                    log_with_timestamp(f'📦 Using valid cached user preference entities for all {len(valid_cached_asins)} products from {user_preferences_file}')
                else:
                    log_with_timestamp(f'⚠️ Cached user preferences missing or invalid for {len(missing_asins)} ASIN(s), will re-extract for: {missing_asins}')
            else:
                log_with_timestamp(f'⚠️ Cached user preference entities belong to {cached_user_pref.get("user_id")}, expected {TARGET_USER}, will re-extract')
        except Exception as e:
            log_with_timestamp(f'⚠️ Error loading cached user preference entities: {e}, will re-extract')

    product_user_entities_map = dict(cached_user_preferences_map)  # asin -> user_entities
    missing_asins = [asin for asin in all_asins if asin not in valid_cached_asins]
    sum(len(reviews_by_asin.get(asin, [])) for asin in missing_asins)
    user_pref_cache_valid = len(missing_asins) == 0

    if user_pref_cache_valid:
        log_with_timestamp('⏭️ Valid cached user preference entities found. Skipping extraction and proceeding to entity matching.')
    else:
        log_with_timestamp(f'🔍 Starting user preference entity extraction for {len(missing_asins)} products concurrently...')

        user_pref_progress_counter = {'completed_reviews': 0}
        user_pref_progress_lock = threading.Lock()

        target_products = [p for p in successful_products if p.get('asin') in missing_asins]
        log_with_timestamp(f'🔍 Processing {len(target_products)} products needing user preference extraction: {[p["asin"] for p in target_products[:3]]}...')

        # Store ASIN -> prompt mapping for later parsing
        asin_prompt_map = {}
        asin_review_map = {}
        
        def process_user_preferences(result):
            asin = result['asin']
            log_with_timestamp(f'🔍 Starting user preference processing for {asin}')

            try:
                # Rebuild reviews_by_asin for this ASIN to avoid concurrency issues
                product_reviews = [r for r in user_reviews if r.get('asin') == asin]
                log_with_timestamp(f'🔍 Found {len(product_reviews)} reviews for {asin}')
                if product_reviews:
                    # Check if reviews have actual content
                    valid_reviews = []
                    for review in product_reviews:
                        text = review.get('reviewText', '').strip()
                        title = review.get('summary', '').strip()
                        if text or title:
                            valid_reviews.append(review)

                    if not valid_reviews:
                        log_with_timestamp(f'⚠️ No valid content in {len(product_reviews)} reviews for {asin}')
                        product_user_entities_map[asin] = {}
                        with user_pref_progress_lock:
                            user_pref_progress_counter['completed_reviews'] += len(product_reviews)
                            current_reviews = user_pref_progress_counter['completed_reviews']
                            progress_total = target_reviews_to_process if target_reviews_to_process else total_reviews_to_process
                            if current_reviews % 100 == 0 or current_reviews == progress_total:
                                log_with_timestamp(f'📊 User preference progress: {current_reviews}/{progress_total} reviews processed')
                        return asin, []

                # Get API keys for processing
                ordered_keys = all_api_keys
                log_with_timestamp(f'🔍 Using {len(ordered_keys)} API keys for {asin}')

                def preference_operation(api_config, provider_name, key_index):
                    llm_model = create_llm_with_config(api_config)
                    # Just call LLM and return raw response - no JSON parsing here
                    raw_response = prepare_content_and_extract_entities(valid_reviews, 'user preference', llm_model, asin=asin)
                    # Store ASIN -> prompt mapping for later parsing
                    # We'll use the prompt to match responses from api_raw_responses.json
                    return raw_response

                try:
                    # Just call LLM - response will be saved to api_raw_responses.json
                    # We'll parse all responses later
                    raw_result = try_api_keys_with_fallback(
                        ordered_keys,
                        preference_operation,
                        f"{asin} user preference extraction"
                    )
                    
                    # Store the mapping for later parsing
                    # We'll use this to match responses from api_raw_responses.json
                    if raw_result:
                        # Create a simple key from reviews content for matching
                        review_content_key = ' '.join([r.get('summary', '') + ' ' + r.get('reviewText', '') for r in valid_reviews[:2]])[:200]
                        asin_prompt_map[asin] = review_content_key
                    
                    # Return placeholder - will be filled later from api_raw_responses.json
                    return asin, valid_reviews

                except Exception as api_error:
                    log_with_timestamp(f'❌ Exception in user preference extraction for {asin}: {api_error}')
                    return asin, []

            except Exception as e:
                log_with_timestamp(f'❌ Error processing user preferences for {asin}: {e}')
                return asin, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks - just call LLM, don't parse JSON yet
            future_to_result = {executor.submit(process_user_preferences, result): result for result in target_products}

            # Collect ASIN -> review_content mappings as requests complete
            asin_review_map = {}
            for future in concurrent.futures.as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    future_result = future.result()
                    if len(future_result) == 2:
                        asin, review_content = future_result
                        asin_review_map[asin] = review_content
                    else:
                        asin = result['asin']
                        log_with_timestamp(f'⚠️ Unexpected result format for {asin}: {future_result}')
                        asin_review_map[asin] = []
                except Exception as e:
                    asin = result['asin']
                    log_with_timestamp(f'❌ Exception processing user preferences for {asin}: {e}')
                    asin_review_map[asin] = []

        log_with_timestamp(f'✅ All LLM requests completed. Now parsing responses from {api_responses_file}...')
        
        # Now parse all responses from api_raw_responses.json
        # We need to match responses to ASINs by prompt content
        if not os.path.exists(api_responses_file):
            log_with_timestamp(f'❌ API responses file not found: {api_responses_file}')
        else:
            try:
                with open(api_responses_file, 'r', encoding='utf-8') as f:
                    all_responses = json.load(f)
            except Exception as e:
                log_with_timestamp(f'❌ Error reading API responses file: {e}')
                all_responses = []
            
            # Filter responses by context and success
            filtered_responses = [r for r in all_responses 
                                 if r.get('context') == 'user_preference_extraction' 
                                 and r.get('success', False)]
            
            log_with_timestamp(f'📋 Found {len(filtered_responses)} successful responses to parse')
            
            # Match responses to ASINs:
            # Only do exact match by meta.asin saved at request time.
            used_response_indices = set()
            asin_to_response_indices = defaultdict(list)

            for idx, response_data in enumerate(filtered_responses):
                meta = response_data.get('meta') or {}
                meta_asin = meta.get('asin')
                if isinstance(meta_asin, str) and meta_asin.strip():
                    asin_to_response_indices[meta_asin.strip().upper()].append(idx)
            
            for asin, review_content in asin_review_map.items():
                matched_entities = {}
                normalized_entities = {}
                
                # Exact match by meta.asin
                if asin in asin_to_response_indices:
                    # Pick the latest unused response for this ASIN (file append order is chronological)
                    candidates = [i for i in asin_to_response_indices[asin] if i not in used_response_indices]
                    if candidates:
                        idx = candidates[-1]
                        response_data = filtered_responses[idx]
                        try:
                            raw_response = response_data.get('raw_response', {})
                            content = raw_response.get('content', '')
                            if content:
                                entities_result = process_user_preference_extraction_response(content)
                                if isinstance(entities_result, tuple):
                                    entities_list, entities_dict = entities_result
                                    matched_entities = entities_dict if entities_dict else entities_list
                                else:
                                    matched_entities = entities_result
                                normalized_entities = normalize_user_preference_entities_with_sentiment(matched_entities)
                                used_response_indices.add(idx)
                            else:
                                log_with_timestamp(f'⚠️ Empty content in response {idx} for {asin} (meta.asin match)')
                        except Exception as e:
                            log_with_timestamp(f'⚠️ Error parsing response for {asin} (meta.asin match): {e}')
                if not matched_entities:
                    log_with_timestamp(
                        f'⚠️ No matching response found for {asin} by meta.asin '
                        f'(searched {len(filtered_responses)} responses, {len(used_response_indices)} already used)'
                    )
                
                product_user_entities_map[asin] = normalized_entities if matched_entities else {}
                
                user_pref_item = {
                    'asin': asin,
                    'user_preference_entities': normalized_entities if matched_entities else {},
                    'review_content': review_content
                }
                user_preferences_data.append(user_pref_item)
                
                if matched_entities:
                    entity_container = normalized_entities if normalized_entities else matched_entities
                    total_entities = sum(len(v) if isinstance(v, list) else 1 for v in entity_container.values()) if isinstance(entity_container, dict) else len(entity_container) if isinstance(entity_container, list) else 0
                    log_with_timestamp(f'✅ Parsed entities for {asin}: {len(entity_container) if isinstance(entity_container, dict) else 1} categories, {total_entities} total entities')
                else:
                    log_with_timestamp(f'⚠️ No entities parsed for {asin}')

    # Ensure every ASIN has a user preference entry (cache or extracted)
    existing_user_pref_asins = {item.get('asin') for item in user_preferences_data}
    for asin in all_asins:
        if asin not in existing_user_pref_asins:
            fallback_entities = product_user_entities_map.get(asin, {})
            user_preferences_data.append({
                'asin': asin,
                'user_preference_entities': fallback_entities if fallback_entities else {},
                'review_content': []
            })

    log_with_timestamp(f'✅ Completed entity extraction for {len(all_results)} products.')

    log_with_timestamp('💾 Saving extracted entity data...')

    # Get the workspace root directory (parent of stark directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    product_entities_file = os.path.join(result_dir, "product_entities.json")
    product_entities_data = {
        'user_id': TARGET_USER,
        'products': []
    }

    def filter_product_info(product_info):
        relevant_fields = {
            'title': 'title',
            'brand': 'brand',
            'description': 'description',
            'feature': 'feature',
            'category': 'category',
            'main_cat': 'main_cat'
        }
        filtered = {}
        for key, new_key in relevant_fields.items():
            if key in product_info:
                value = product_info[key]
                # 对文本字段进行HTML清洗
                if key in ['description', 'feature'] and isinstance(value, list):
                    # 如果是列表，对每个元素进行清洗
                    filtered[new_key] = [clean_html_content(str(item)) for item in value if clean_html_content(str(item))]
                elif isinstance(value, str):
                    # 如果是字符串，直接清洗
                    filtered[new_key] = clean_html_content(value)
                else:
                    # 其他类型保持不变
                    filtered[new_key] = value
        return filtered

    for result in successful_products:
        asin = result['asin']

        # 处理产品实体，支持新格式（字典）和旧格式（列表）
        product_entities = result['product_entities']
        if isinstance(product_entities, dict):
            # 新格式：{category: [entities]}
            # 展平所有实体用于后续处理
            cleaned_entities = []
            for category_entities in product_entities.values():
                if isinstance(category_entities, list):
                    cleaned_entities.extend(category_entities)
                else:
                    cleaned_entities.append(str(category_entities))
        else:
            # 旧格式：实体列表
            # 过滤产品实体，移除包含类别前缀的实体
            cleaned_entities = []
            for entity in product_entities:
                # 如果实体包含冒号，提取冒号后的部分
                if ':' in entity and len(entity.split(':', 1)) == 2:
                    prefix, value = entity.split(':', 1)
                    cleaned_entity = value.strip()
                    if cleaned_entity:  # 确保不为空
                        cleaned_entities.append(cleaned_entity)
                else:
                    cleaned_entities.append(entity)

        # 应用原子化过滤
        atomic_entities = []
        for entity in cleaned_entities:
            entity_lower = entity.lower()
            if (',' in entity or
                ' and ' in entity_lower or
                ' with ' in entity_lower or
                ' or ' in entity_lower or
                ' for ' in entity_lower or
                '&' in entity):
                continue
            atomic_entities.append(entity)

        product_info = result['product_info']
        product_info_filtered = filter_product_info(product_info)

        # 保存字典格式的product_entities
        if isinstance(product_entities, dict):
            # 新格式：{category: [entities]} - 直接使用
            saved_product_entities = product_entities
        else:
            # 旧格式：实体列表 - 保持原样用于向后兼容
            saved_product_entities = product_entities

        product_data = {
            'asin': asin,
            'product_title': result['product_title'],
            'product_entities': saved_product_entities,
            'product_info': product_info_filtered,
            'metadata': {}
        }

        # 解析metadata_lines添加到metadata中
        for line in result['metadata_lines']:
            if line.startswith('    '):
                line = line.strip()
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    product_data['metadata'][key.lower()] = value

        product_entities_data['products'].append(product_data)

    with open(product_entities_file, 'w', encoding='utf-8') as f:
        json.dump(product_entities_data, f, indent=2, ensure_ascii=False)
    log_with_timestamp(f'💾 Saved product entities to {product_entities_file}')

    # 保存用户偏好实体数据到新的JSON文件（若未使用缓存则写入）
    if user_pref_cache_valid:
        log_with_timestamp(f'📦 Cached user preference entities reused, no need to resave {user_preferences_file}')
    else:
        try:
            user_pref_save_data = {
                'user_id': TARGET_USER,
                'products': user_preferences_data
            }
            with open(user_preferences_file, 'w', encoding='utf-8') as f:
                json.dump(user_pref_save_data, f, indent=2, ensure_ascii=False)
            log_with_timestamp(f'💾 Saved user preference entities to {user_preferences_file}')
        except Exception as e:
            log_with_timestamp(f'❌ Error saving user preference entities: {e}')

    log_with_timestamp('✅ Product entity extraction completed. Proceeding to entity matching...')

    # 准备商品数据用于实体匹配
    save_data = {
        'user_id': TARGET_USER,
        'products': product_entities_data['products']  # 使用已提取的商品数据
    }

    log_with_timestamp('🎯 Entity extraction phase finished.')

    # 加载用户偏好数据并合并到商品数据中
    # user_preferences_file already defined above
    try:
        with open(user_preferences_file, 'r', encoding='utf-8') as f:
            user_pref_data = json.load(f)

        # 创建用户偏好数据的映射 {asin: user_entities}
        user_entities_map = {}
        for product in user_pref_data.get('products', []):
            asin = product.get('asin')
            user_entities = normalize_user_preference_entities_with_sentiment(
                product.get('user_preference_entities', {})
            )
            if asin:
                user_entities_map[asin] = user_entities

        # 将用户偏好数据合并到商品数据中
        for product in save_data['products']:
            asin = product.get('asin')
            if asin in user_entities_map:
                product['user_preference_entities'] = user_entities_map[asin]

        log_with_timestamp(f'✅ Loaded user preference data for {len(user_entities_map)} products')

    except Exception as e:
        log_with_timestamp(f'❌ Error loading user preference data: {e}')
        # 如果加载失败，继续处理，但没有用户偏好数据

    # 实体匹配及结果缓存逻辑（支持跳过已存在且有效的结果）
    matched_entities_file = os.path.join(result_dir, "entity_matching_results.json")
    matched_products = None
    matched_data = None

    if os.path.exists(matched_entities_file):
        try:
            with open(matched_entities_file, 'r', encoding='utf-8') as f:
                cached_matched = json.load(f)

            cached_user = cached_matched.get('user_id')
            cached_products = cached_matched.get('products', [])
            cached_asins = {p.get('asin') for p in cached_products if p.get('asin')}
            expected_asins = {p.get('asin') for p in save_data['products'] if p.get('asin')}

            if (
                cached_user == TARGET_USER
                and cached_products
                and expected_asins
                and cached_asins == expected_asins
            ):
                log_with_timestamp(
                    f'📦 Using cached matched entity data for {len(cached_products)} products from {matched_entities_file}'
                )
                matched_products = cached_products
                matched_data = cached_matched
        except Exception as e:
            log_with_timestamp(f'⚠️ Error loading cached matched entities: {e}, will recompute')

    # 如果没有有效缓存，则重新执行实体匹配
    if matched_products is None:
        log_with_timestamp('🎯 No valid cached matched entities found, running entity matching...')
        matched_products = perform_entity_matching(save_data['products'])
        matched_data = {
            'user_id': TARGET_USER,
            'products': matched_products
        }
        try:
            with open(matched_entities_file, 'w', encoding='utf-8') as f:
                json.dump(matched_data, f, indent=2, ensure_ascii=False)
            log_with_timestamp(f'💾 Saved matched entity data to {matched_entities_file}')
        except Exception as e:
            log_with_timestamp(f'❌ Error saving matched entity data: {e}')

    # 在实体匹配完成（无论是缓存还是新计算）后，进行维度规格标准化
    try:
        normalize_dimensions_process()
        log_with_timestamp('✅ Standardized dimensions for matched entities.')
    except Exception as e:
        log_with_timestamp(f'⚠️ Failed to normalize dimensions: {e}')

    # 为匹配实体类别大于等于3个的产品生成查询语句
    # try:
    #     products_with_queries = generate_queries_for_matched_products(matched_data, get_all_api_keys_in_order())
    #
    #     # 保存生成的查询到单独的文件
    #     if products_with_queries:
    #         generated_queries_file = os.path.join(result_dir, "generated_queries.json")
    #         queries_data = {
    #             'user_id': TARGET_USER,
    #             'products': products_with_queries
    #         }
    #         with open(generated_queries_file, 'w', encoding='utf-8') as f:
    #             json.dump(queries_data, f, indent=2, ensure_ascii=False)
    #         log_with_timestamp(f'💾 Saved generated queries for {len(products_with_queries)} products to {generated_queries_file}')
    #     else:
    #         log_with_timestamp('⚠️ No queries were generated')
    # except Exception as e:
    #     log_with_timestamp(f'❌ Error generating queries: {e}')

    print_entity_matching_results()

    log_with_timestamp('🏁 All processing completed successfully!')

if __name__ == '__main__':
    main()