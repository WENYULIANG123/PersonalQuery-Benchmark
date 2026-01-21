#!/usr/bin/env python3
"""
查询生成模块
负责为匹配实体足够丰富的商品生成自然语言查询语句
"""

import os
import sys
import threading
from typing import List
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry
from utils import try_api_keys_with_fallback, create_llm_with_config

def log_with_timestamp(message: str):
    """Log message with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def generate_query_from_matched_entities(matched_entities: List[str], llm_model) -> str:
    """根据匹配的实体生成用户查询语句"""
    if not matched_entities:
        return ""

    prompt = f"""Based on the following matched entities, generate a natural user search query that someone might ask when looking for products with these features.

MATCHED ENTITIES: {matched_entities}

Please generate a natural, conversational query that sounds like a real customer searching for products. Make it sound like a question or natural search phrase that someone would actually type when shopping online.

Examples of good queries:
- "I'm looking for watercolor crayons that work on dark paper"
- "What are good firm pastel sticks for detailed illustrations?"
- "Need soft leather bags for organizing art supplies"
- "Looking for luminous paints in red yellow and blue"
- "Where can I find water-soluble sketching pencils?"

Examples of bad queries (avoid these):
- "Water-soluble sketching pencils for figure drawing" (too descriptive, sounds like a product title)
- "Firm pastel sticks that can be sharpened to a fine point and are durable" (too formal and feature-list like)

Make it sound like a real person asking a question or describing what they're looking for. Generate a single natural query:"""

    # Retry up to 3 times
    for attempt in range(3):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="query_generation")
            if success and response_str:
                # Clean up the response
                generated_query = response_str.strip()
                # Remove quotes if present
                if generated_query.startswith('"') and generated_query.endswith('"'):
                    generated_query = generated_query[1:-1]
                if generated_query.startswith("'") and generated_query.endswith("'"):
                    generated_query = generated_query[1:-1]
                return generated_query.strip()
        except Exception as e:
            print(f"LLM error in query generation: {e}", flush=True)
            if attempt < 2:
                continue

    return ""

def generate_queries_for_matched_products(data, all_api_keys):
    """为匹配实体类别大于等于3个的产品批量生成查询语句，返回包含查询的产品列表"""
    log_with_timestamp("🤖 Starting batch query generation for products with >=3 matched entity categories...")

    products = data.get('products', [])

    # 筛选出匹配实体类别超过3个的产品
    products_with_many_matches = []
    for p in products:
        matched_entities = p.get('matched_entities', {})
        if matched_entities:
            # 计算有多少个类别有匹配的实体
            matched_categories_count = len(matched_entities)
            if matched_categories_count >= 3:
                products_with_many_matches.append(p)

    log_with_timestamp(f"📊 Found {len(products_with_many_matches)} products with >=3 matched entity categories out of {len(products)} total products")

    if not products_with_many_matches:
        log_with_timestamp("⚠️ No products with >=3 matched entity categories found, skipping query generation")
        return []

    log_with_timestamp(f"📝 Generating queries for {len(products_with_many_matches)} products with >=3 matched entity categories concurrently...")

    # Thread-safe progress counter for query generation
    query_progress_counter = {'completed': 0}
    query_progress_lock = threading.Lock()

    def process_query_generation(product, idx):
        try:
            asin = product.get('asin', 'Unknown')
            matched_entities_dict = product.get('matched_entities', {})
            product.get('product_entities', {})

            # Convert matched entities dict to flat list for query generation
            matched_entities_list = []
            for category_matches in matched_entities_dict.values():
                matched_entities_list.extend(category_matches)


            def query_generation_operation(api_config, provider_name, key_index):
                llm_model = create_llm_with_config(api_config)
                return generate_query_from_matched_entities(matched_entities_list, llm_model)

            generated_query, query_success = try_api_keys_with_fallback(
                all_api_keys,
                query_generation_operation,
                f"{asin} query generation"
            )

            if query_success and generated_query:
                product['generated_query'] = generated_query
                result = generated_query
            else:
                product['generated_query'] = ""
                result = ""

            # Update progress
            with query_progress_lock:
                query_progress_counter['completed'] += 1
                current_count = query_progress_counter['completed']
                if current_count % 10 == 0 or current_count == len(products_with_many_matches):
                    log_with_timestamp(f'📊 Query generation progress: {current_count}/{len(products_with_many_matches)} products processed')

            return product, result

        except Exception as e:
            log_with_timestamp(f'❌ Error generating query for {product.get("asin", "Unknown")}: {e}')
            product['generated_query'] = ""
            return product, ""

    # Process query generation concurrently with 102 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=102) as executor:
        # Submit all tasks with index
        future_to_product = {executor.submit(process_query_generation, product, idx): (product, idx)
                           for idx, product in enumerate(products_with_many_matches, 1)}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_product):
            product, idx = future_to_product[future]
            try:
                updated_product, generated_query = future.result()
                # Update the original product in the data
                products_with_many_matches[idx-1] = updated_product
                if generated_query:
                    asin = updated_product.get('asin', 'Unknown')
                    log_with_timestamp(f"✅ Generated query for {asin}: {generated_query[:60]}...")
                else:
                    asin = updated_product.get('asin', 'Unknown')
                    log_with_timestamp(f"⚠️ Failed to generate query for {asin}")
            except Exception as e:
                asin = product.get('asin', 'Unknown')
                log_with_timestamp(f'❌ Exception generating query for {asin}: {e}')
                products_with_many_matches[idx-1]['generated_query'] = ""

    log_with_timestamp("🎉 Batch query generation completed!")

    # 返回包含生成查询的产品列表
    return products_with_many_matches