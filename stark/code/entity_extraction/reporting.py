import os
import json
from utils import log_with_timestamp
from pipeline_config import MATCHED_ENTITIES_FILE

def print_entity_matching_results():
    """打印实体匹配的完整结果"""
    log_with_timestamp("📋 Printing complete entity matching results...")

    try:
        with open(MATCHED_ENTITIES_FILE, 'r', encoding='utf-8') as f:
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
        # ... (Print logic copied from original main.py)
        # Using a simplified version or just copying the logic. 
        # For brevity in this thought trace, I will use the logic from the original file I viewed.
        _print_single_product(idx, len(products), product)

    # Failed products
    failed_products = [p for p in products if not p.get('matched_entities') or not any(matches for matches in p.get('matched_entities', {}).values())]
    if failed_products:
        print(f'\\n❌ Products with No Matches ({len(failed_products)}):', flush=True)
        for product in failed_products:
             _print_failed_product_summary(product)

def _print_single_product(idx, total, product):
    asin = product.get('asin', 'Unknown')
    product_title = product.get('product_title', 'Unknown')
    product_entities = product.get('product_entities', {})
    user_entities = product.get('user_preference_entities', {})
    matched_entities = product.get('matched_entities', {})
    reviews = product.get('reviews', [])
    metadata = product.get('metadata', {})

    progress_info = f" ({idx}/{total})"
    print(f'Product {asin} ({product_title[:50]}...){progress_info}:', flush=True)

    # Reviews
    if reviews:
        unique_reviews = []
        seen_contents = set()
        for review in reviews:
            title = review.get('summary', '').strip()
            text = review.get('reviewText', '').strip()
            text = ' '.join(text.split())
            review_content = f"{title} {text}".strip()
            if review_content and review_content not in seen_contents:
                seen_contents.add(review_content)
                unique_reviews.append(review_content)

        print(f'  Reviews ({len(unique_reviews)} unique):', flush=True)
        for i, review_content in enumerate(unique_reviews[:3], 1):
            if review_content:
                print(f'    Review {i}: {review_content}', flush=True)
        if len(unique_reviews) > 3:
            print(f'    ... and {len(unique_reviews) - 3} more unique reviews', flush=True)
    else:
        print('  Reviews: None found', flush=True)

    # Product Entities
    if product_entities:
        total_product_entities = sum(len(entities) for entities in product_entities.values())
        print(f'  Product Entities ({len(product_entities)} categories, {total_product_entities} total):', flush=True)
        for category, entities in product_entities.items():
            print(f'    {category}: {", ".join(entities)}', flush=True)
    else:
        print('  Product Entities: None extracted', flush=True)

    # Helper
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

    # User Preferences
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

    # Matched Entities
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

    # Generated Query
    generated_query = product.get('generated_query', '')
    if generated_query:
        print(f'  Generated Query: {generated_query}', flush=True)
    else:
        print('  Generated Query: None generated', flush=True)

    # Metadata
    print('  Metadata:', flush=True)
    for key, value in metadata.items():
        print(f'    {key}: {value}', flush=True)
    print()

def _print_failed_product_summary(product):
    asin = product.get('asin', 'Unknown')
    product_entities = product.get('product_entities', {})
    user_entities = product.get('user_preference_entities', {})
    product_count = sum(len(entities) for entities in product_entities.values())
    user_count = sum(len(entities) for entities in user_entities.values())
    print(f'  Product {asin}: Product entities ({len(product_entities)} categories, {product_count} total), User entities ({len(user_entities)} categories, {user_count} total)', flush=True)
