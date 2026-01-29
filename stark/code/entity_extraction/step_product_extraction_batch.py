import os
import json
from user_perference.user_preference_extraction import load_user_reviews
from product_extraction import load_product_metadata, clean_html_content, process_product_extraction_response
from utils import log_with_timestamp
from pipeline_config import TARGET_USER, PRODUCT_ENTITIES_FILE
from model import submit_batch_inference, wait_for_batch_results

def run_product_extraction_batch(config):
    """
    Orchestrates the extraction of product entities using Batch API from model.py.
    Returns the path to the saved product entities file.
    """
    log_with_timestamp('Starting product entity extraction (BATCH MODE using model.py)...')

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
                        if cached_asin and product.get('product_entities'):
                            cached_product_data[cached_asin] = product
                    log_with_timestamp(f'📦 Loaded {len(cached_product_data)} cached product entities')
        except Exception as e:
            log_with_timestamp(f'⚠️ Error loading cache: {e}')

    asins_to_extract = [asin for asin in all_asins if asin not in cached_product_data]
    log_with_timestamp(f'📦 Cached: {len(cached_product_data)}, Need Extraction: {len(asins_to_extract)}')

    if not asins_to_extract:
        log_with_timestamp('✅ All products found in cache.')
        return PRODUCT_ENTITIES_FILE

    # 4. Prepare Batch Prompts
    prompts = []
    asin_order = []
    
    for asin in asins_to_extract:
        product_info = product_metadata.get(asin, {})
        content_parts = []
        if 'title' in product_info and product_info['title']:
            content_parts.append(f"Title: {product_info['title']}")
        if 'brand' in product_info and product_info['brand']:
            content_parts.append(f"Brand: {product_info['brand']}")
        if 'description' in product_info and product_info['description']:
            desc = product_info['description']
            content_parts.append(f"Description: {clean_html_content(' '.join(desc) if isinstance(desc, list) else desc)}")
        if 'feature' in product_info and product_info['feature']:
            feat = product_info['feature']
            content_parts.append(f"Features: {clean_html_content(' '.join(feat) if isinstance(feat, list) else feat)}")
        if 'category' in product_info and product_info['category']:
            cat = product_info['category']
            content_parts.append(f"Category: {' > '.join(cat) if isinstance(cat, list) else cat}")
        
        content = "\n".join(content_parts)
        
        # Build prompt
        prompt = f"""
You are an e-commerce data expert. Please extract key entities from the following product information.

**Input Product Information:** {content}

**Entity Classification Requirements:**
For each extracted entity, it must be classified into one of the following categories:
[Brand, Material, Dimensions, Quantity, Color, Design, Usage, Selling Point, Safety/Certification, Accessories]

**Extraction Rules:**
1. **Strictly Prohibit Outputting Category Names**: Do not use "Brand", "Color", etc., as values.
2. **Extract Only Specific Values**: Output only specific brand names (e.g., "Apple"). If no specific value exists, omit it.
3. **Usage vs. Target Audience**: ONLY extract actual use cases (e.g., "Sewing", "Illustration"). **STRICTLY PROHIBIT** extracting target user groups (e.g., "fine artists", "commercial artists", "beginners") as Usage.
4. **Dimensions Require Context/Units**: Never extract pure numbers for Dimensions. **MUST include units or context** (e.g., "8.7 yds", "Size 8", "4 mm", "22 Gauge"). Pure numbers like "12", "8" are prohibited.
5. **Concise Selling Points**: Avoid long descriptive sentences. Extract only the core attribute as a concise phrase (e.g., instead of "High strength for high speed embroidery", extract "High Strength"). 
6. **Negative List (Generic Adjectives)**: Do NOT extract generic, non-distinctive adjectives such as: "Soft", "Smooth", "Versatile", "Easy", "Convenient", "High Quality", "Value Pack", "Premium", "Durable", "Best", "Top".
7. **Atomic Only**: Each entity must be a single independent attribute. Split compound phrases like "Red and Blue" -> ["Red", "Blue"].
8. **Synonym Consolidation**: If multiple terms mean the same thing (e.g., "Genuine Leather", "Top-grain Leather", "Leather"), extract only the most specific one ("Top-grain Leather").
9. **Digital File Formats**: Place file formats (e.g., "PES", "DST") under "Accessories".
10. **Color Splits**: One color per element. "Black & White" -> ["Black", "White"].
11. **Comprehensive Extraction**: If a product explicitly lists multiple items (e.g. a set listing specific colors "Red, Blue, Green"), extract ALL of them as individual entities. Do NOT just extract the count ("30 colors").

**Important:** Entities should be keywords or short phrases. Better to miss than include noisy or generic marketing filler.

**Output Format:**
Return a JSON object where keys are category names and values are arrays of entities for that category. Multiple entities of the same category should be placed in the same array.

Example:
{{
  "Brand": ["Apple"],
  "Design": ["iPhone 15", "Smartphone"],
  "Quantity": ["256GB"],
  "Color": ["Blue", "Space Gray"],
  "Material": ["Aluminum"],
  "Selling Point": ["Waterproof", "Face ID"]
}}

Return only valid JSON object, no other explanations.
"""
        prompts.append(prompt)
        asin_order.append(asin)

    # 5. Submit Batch using model.submit_batch_inference
    log_with_timestamp(f'🚀 Submitting batch for {len(prompts)} products...')
    batch_id = submit_batch_inference(prompts, model="Qwen/QwQ-32B")
    log_with_timestamp(f'⏳ Batch submitted! ID: {batch_id}. Waiting for completion...')
    
    # 6. Wait and Retrieve using model.wait_for_batch_results
    batch_results = wait_for_batch_results(batch_id, poll_interval=30)
    if not batch_results:
        log_with_timestamp("❌ Batch inference failed or returned no results.")
        return None

    # 7. Post-process and Merge
    extracted_data = {}
    for res in batch_results:
        custom_id = res.get('custom_id', '')
        try:
            idx = int(custom_id.split('-')[1])
            asin = asin_order[idx]
            
            content = ""
            if 'response' in res and 'body' in res['response']:
                choices = res['response']['body'].get('choices', [])
                if choices:
                    content = choices[0]['message'].get('content', '')
            
            if content:
                try:
                    # Use existing parser from product_extraction.py
                    entities_list, entities_dict = process_product_extraction_response(content)
                    
                    product_info = product_metadata.get(asin, {})
                    extracted_data[asin] = {
                        'asin': asin,
                        'product_title': product_info.get('title', f'Product {asin}'),
                        'product_entities': entities_dict if entities_dict else entities_list,
                        'product_info': product_info,
                        'success': True
                    }
                except Exception as e:
                    log_with_timestamp(f"⚠️ Error parsing response for {asin}: {e}")
            else:
                log_with_timestamp(f"⚠️ Empty response for {asin}")
        except:
            continue

    # 8. Save
    final_products = []
    for asin in all_asins:
        if asin in cached_product_data:
            cached_p = cached_product_data[asin]
            product_info = product_metadata.get(asin, {})
            p_data = {
                'asin': asin,
                'product_title': cached_p.get('product_title', product_info.get('title', '')),
                'product_entities': cached_p.get('product_entities', {}),
                'product_info': _filter_product_info(product_info),
                'metadata': cached_p.get('metadata', {})
            }
            final_products.append(p_data)
        elif asin in extracted_data:
            res = extracted_data[asin]
            p_data = {
                'asin': asin,
                'product_title': res['product_title'],
                'product_entities': res['product_entities'],
                'product_info': _filter_product_info(res['product_info']),
                'metadata': {}
            }
            final_products.append(p_data)

    output_data = {'user_id': TARGET_USER, 'products': final_products}
    with open(PRODUCT_ENTITIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f'✅ {len(final_products)}/{total_products} successful. Saved to {PRODUCT_ENTITIES_FILE}')
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
