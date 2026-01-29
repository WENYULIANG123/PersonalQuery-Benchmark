import os
import json
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from user_perference.user_preference_extraction import load_user_reviews
from product_extraction import load_product_metadata, extract_product_entities_only, log_with_timestamp, clean_html_content
from utils import get_all_api_keys_in_order, try_api_keys_with_fallback
from pipeline_config import TARGET_USER, PRODUCT_ENTITIES_FILE

def run_product_extraction(config):
    """
    Orchestrates the extraction of product entities.
    Returns the path to the saved product entities file.
    """
    log_with_timestamp('Starting product entity extraction...')

    # 1. Load User Reviews to identify target products
    user_reviews = load_user_reviews(TARGET_USER)
    if not user_reviews:
        log_with_timestamp(f"❌ No reviews found for user {TARGET_USER}")
        return None

    user_asins = set(review.get('asin') for review in user_reviews if review.get('asin'))
    
    # 2. Load Metadata for these products
    product_metadata = load_product_metadata(user_asins)
    if not product_metadata:
        log_with_timestamp(f"❌ No metadata found for products reviewed by {TARGET_USER}")
        return None

    all_asins = sorted(list(product_metadata.keys()))
    total_products = len(all_asins)
    log_with_timestamp(f'🔍 Selected {total_products} ASINs for processing')

    # 3. Load Cache
    cached_product_data = {}
    if os.path.exists(PRODUCT_ENTITIES_FILE):
        try:
            with open(PRODUCT_ENTITIES_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('user_id') == TARGET_USER:
                    for product in cached_data.get('products', []):
                        cached_asin = product.get('asin')
                        # Check basic validity
                        if cached_asin and product.get('product_entities'):
                            cached_product_data[cached_asin] = product
                    log_with_timestamp(f'📦 Loaded {len(cached_product_data)} cached product entities')
        except Exception as e:
            log_with_timestamp(f'⚠️ Error loading cache: {e}')

    cached_count = sum(1 for asin in all_asins if asin in cached_product_data)
    need_extraction_count = total_products - cached_count
    log_with_timestamp(f'📦 Cached: {cached_count}, Need Extraction: {need_extraction_count}')

    # 4. Processing
    all_results = []
    progress_counter = {'completed': 0}
    progress_lock = threading.Lock()
    
    # Use reduced concurrency to avoid 429 errors
    max_workers = config.get('max_concurrent_product', 20)
    log_with_timestamp(f'🔄 Processing with {max_workers} workers...')

    def process_single_product(asin):
        result = {}
        try:
            if asin in cached_product_data:
                # Use cached
                cached_product = cached_product_data[asin]
                product_info = product_metadata.get(asin, {})
                result = {
                    'asin': asin,
                    'product_title': cached_product.get('product_title', product_info.get('title', f'Product {asin}')),
                    'product_entities': cached_product.get('product_entities', {}),
                    'product_info': cached_product.get('product_info', {}),
                    'metadata': cached_product.get('metadata', {}),
                    'metadata_lines': [], # Reconstructed if needed, or ignored
                    'success': True,
                    'from_cache': True
                }
            else:
                # Extract
                ordered_keys = get_all_api_keys_in_order()
                
                def op(api_config, pname, idx):
                    return extract_product_entities_only(asin, product_metadata, api_config, total_products)

                product_result, success = try_api_keys_with_fallback(
                    ordered_keys, op, f"product {asin}", ""
                )
                
                if success:
                    result = product_result
                    result['from_cache'] = False
                else:
                    result = {'asin': asin, 'error': 'All API keys failed', 'success': False}

        except Exception as e:
            result = {'asin': asin, 'error': str(e), 'success': False}
        
        with progress_lock:
            progress_counter['completed'] += 1
            curr = progress_counter['completed']
            if curr % 10 == 0 or curr == total_products:
                log_with_timestamp(f'📊 Progress: {curr}/{total_products} products processed')
        
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_asin = {executor.submit(process_single_product, asin): asin for asin in all_asins}
        for future in concurrent.futures.as_completed(future_to_asin):
            all_results.append(future.result())

    # 5. Saving
    successful_products = [r for r in all_results if r.get('success', False)]
    log_with_timestamp(f'✅ {len(successful_products)}/{total_products} successful.')

    product_entities_data = {'user_id': TARGET_USER, 'products': []}
    
    for result in successful_products:
        # Reconstruct product object structure
        p_data = {
            'asin': result['asin'],
            'product_title': result.get('product_title', ''),
            'product_entities': result.get('product_entities', {}),
            'product_info': _filter_product_info(result.get('product_info', {})),
            'metadata': result.get('metadata', {})
        }
        
        # If we have raw metadata lines from extraction (not cache), parse them
        if not result.get('from_cache', False) and 'metadata_lines' in result:
             for line in result['metadata_lines']:
                if line.strip().startswith(('Size:', 'Color:', 'Material:', 'Brand:')): # Heuristic
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        p_data['metadata'][parts[0].strip().lower()] = parts[1].strip()

        product_entities_data['products'].append(p_data)

    with open(PRODUCT_ENTITIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(product_entities_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f'💾 Saved to {PRODUCT_ENTITIES_FILE}')
    return PRODUCT_ENTITIES_FILE

def _filter_product_info(product_info):
    relevant_fields = {
        'title': 'title', 'brand': 'brand', 'description': 'description',
        'feature': 'feature', 'category': 'category', 'main_cat': 'main_cat'
    }
    filtered = {}
    for key, new_key in relevant_fields.items():
        if key in product_info:
            val = product_info[key]
            if isinstance(val, list):
                filtered[new_key] = [clean_html_content(str(v)) for v in val if clean_html_content(str(v))]
            elif isinstance(val, str):
                filtered[new_key] = clean_html_content(val)
            else:
                filtered[new_key] = val
    return filtered
