import json
import os
from pipeline_config import USER_PREFERENCES_FILE, MATCHED_ENTITIES_NORMALIZED_FILE
from utils import log_with_timestamp

def fix_user_preferences_schema():
    """
    Reads USER_PREFERENCES_FILE, converts 'value' keys to 'entity' keys in entity objects,
    and saves the file.
    """
    if not os.path.exists(USER_PREFERENCES_FILE):
        log_with_timestamp(f"File not found: {USER_PREFERENCES_FILE}")
        return

    log_with_timestamp(f"Reading {USER_PREFERENCES_FILE}...")
    with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_count = 0
    
    if 'products' in data:
        for product in data['products']:
            pref_entities = product.get('user_preference_entities', {})
            for category, entities in pref_entities.items():
                if isinstance(entities, list):
                    for entity_obj in entities:
                        if isinstance(entity_obj, dict):
                            # Fix keys
                            if "value" in entity_obj and "entity" not in entity_obj:
                                entity_obj["entity"] = entity_obj.pop("value")
                                fixed_count += 1
                                # Also fix normalization if needed (e.g. sentiment)
                                if "sentiment" not in entity_obj and "polarity" in entity_obj:
                                     entity_obj["sentiment"] = entity_obj.pop("polarity")

    log_with_timestamp(f"Fixed {fixed_count} entity objects.")
    
    with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_with_timestamp(f"Saved fixed schema to {USER_PREFERENCES_FILE}")

def fix_result_schema():
    # Also fix the final result file if it exists, to show immediate results
    if os.path.exists(MATCHED_ENTITIES_NORMALIZED_FILE):
        log_with_timestamp(f"Reading {MATCHED_ENTITIES_NORMALIZED_FILE}...")
        with open(MATCHED_ENTITIES_NORMALIZED_FILE, 'r', encoding='utf-8') as f:
            root_data = json.load(f)
            
        fixed_count = 0
        products = []
        
        # Handle both dict (with 'products' key) and list formats
        if isinstance(root_data, dict) and 'products' in root_data:
            products = root_data['products']
        elif isinstance(root_data, list):
            products = root_data
            
        for product in products:
                # Fix user_preference_entities
                pref_entities = product.get('user_preference_entities', {})
                for category, entities in pref_entities.items():
                    if isinstance(entities, list):
                        for entity_obj in entities:
                            if isinstance(entity_obj, dict):
                                if "value" in entity_obj and "entity" not in entity_obj:
                                    entity_obj["entity"] = entity_obj.pop("value")
                                    fixed_count += 1
                
                # Fix matched_entities (structure: Category -> List[ProductEntity + matched info])
                # Matched entities usually have "entity" (from product extraction), 
                # but let's check just in case.
                matched = product.get('matched_entities', {})
                for category, entities in matched.items():
                     if isinstance(entities, list):
                        for entity_obj in entities:
                             if isinstance(entity_obj, dict):
                                if "value" in entity_obj and "entity" not in entity_obj:
                                    entity_obj["entity"] = entity_obj.pop("value")
                                    fixed_count += 1

        with open(MATCHED_ENTITIES_NORMALIZED_FILE, 'w', encoding='utf-8') as f:
            json.dump(root_data, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"Saved fixed result to {MATCHED_ENTITIES_NORMALIZED_FILE} (Fixed {fixed_count} items)")

if __name__ == "__main__":
    fix_user_preferences_schema()
    fix_result_schema()
