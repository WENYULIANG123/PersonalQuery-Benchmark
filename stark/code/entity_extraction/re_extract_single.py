import json
import os
import sys

# Add path for imports
# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_perference.user_preference_extraction import load_user_reviews, prepare_content_and_extract_entities, process_user_preference_extraction_response
from user_perference.kb_helper import get_kb_instance
from utils import get_all_api_keys_in_order, create_llm_with_config, log_with_timestamp
from pipeline_config import USER_PREFERENCES_FILE, TARGET_USER

def re_extract_single_asin(target_asin):
    log_with_timestamp(f"🔄 Re-extracting for ASIN: {target_asin}")
    
    # 1. Load reviews for this ASIN
    all_reviews = load_user_reviews(TARGET_USER)
    asin_reviews = [r for r in all_reviews if r.get('asin') == target_asin]
    
    if not asin_reviews:
        log_with_timestamp(f"❌ No reviews found for ASIN {target_asin}")
        return

    # 2. Extract
    all_keys = get_all_api_keys_in_order()
    llm = create_llm_with_config(all_keys[0])
    
    log_with_timestamp(f"🚀 Calling LLM for {target_asin}...")
    
    # Fetch KB attributes
    kb = get_kb_instance()
    known_attributes = kb.get_product_attributes(target_asin)
    
    try:
        raw_response = prepare_content_and_extract_entities(asin_reviews, 'user preference', llm, asin=target_asin, known_attributes=known_attributes)
        print("\n\n" + "="*50)
        print("RAW RESPONSE FROM LLM:")
        print(raw_response)
        print("="*50 + "\n\n")
        
        # 3. Parse using the newly fixed parser
        _, entities_dict = process_user_preference_extraction_response(raw_response)
        log_with_timestamp(f"✅ Successfully extracted entities for {target_asin}")
        
        # 4. Update the JSON file
        if os.path.exists(USER_PREFERENCES_FILE):
            with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            found = False
            for product in data.get('products', []):
                if product.get('asin') == target_asin:
                    product['user_preference_entities'] = entities_dict
                    found = True
                    break
            
            if not found:
                data['products'].append({
                    'asin': target_asin,
                    'user_preference_entities': entities_dict,
                    'review_content': asin_reviews
                })
            
            with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log_with_timestamp(f"💾 Updated {USER_PREFERENCES_FILE}")
        else:
            log_with_timestamp(f"⚠️ {USER_PREFERENCES_FILE} not found. Creating new.")
            new_data = {
                "user_id": TARGET_USER,
                "products": [{
                    "asin": target_asin,
                    "user_preference_entities": entities_dict,
                    "review_content": asin_reviews
                }]
            }
            with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        log_with_timestamp(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    re_extract_single_asin("B000BGSZFU")
