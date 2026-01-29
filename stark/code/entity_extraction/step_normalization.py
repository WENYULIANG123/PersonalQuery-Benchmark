import os
import json
import asyncio
from utils import log_with_timestamp, get_all_api_keys_in_order, create_llm_with_config
from pipeline_config import PRODUCT_ENTITIES_FILE, PRODUCT_ENTITIES_NORMALIZED_FILE, PRODUCT_ENTITIES_WITH_DIMS_FILE, TARGET_USER
from normalize_entities_with_llm import process_all_products
from normalize_dimensions import process as normalize_dimensions_process

def run_product_normalization(config):
    """
    Runs entity normalization and dimension normalization.
    """
    log_with_timestamp('⚖️ Starting entity normalization...')
    
    if not os.path.exists(PRODUCT_ENTITIES_FILE):
        log_with_timestamp(f"❌ Product entities file not found: {PRODUCT_ENTITIES_FILE}")
        return

    # 1. General Entity Normalization (Color, Material, etc.)
    try:
        with open(PRODUCT_ENTITIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data.get('products', [])
        if not products:
             log_with_timestamp("⚠️ No products to normalize.")
             return

        # Initialize LLM
        all_keys = get_all_api_keys_in_order()
        llm_model = create_llm_with_config(all_keys[0])

        # Run async process
        # Use reduced concurrency
        max_workers = config.get('max_concurrent_norm', 50)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _, _, normalized_products = loop.run_until_complete(
            process_all_products(llm_model, products, max_concurrent=max_workers)
        )
        loop.close()

        # Save
        with open(PRODUCT_ENTITIES_NORMALIZED_FILE, 'w', encoding='utf-8') as f:
            json.dump({"user_id": TARGET_USER, "products": normalized_products}, f, indent=2, ensure_ascii=False)
        
        log_with_timestamp(f'✅ Entity normalization saved to {PRODUCT_ENTITIES_NORMALIZED_FILE}')

    except Exception as e:
        log_with_timestamp(f'⚠️ Entity normalization failed: {e}')
        # return # Verify if we should stop here? Usually yes.

    # 2. Dimension Normalization
    # Note: Dimension normalization usually runs on the output of entity extraction (before or after normalization).
    # The original script ran it on `product_entities_file`.
    log_with_timestamp('📏 Starting dimension normalization...')
    try:
        # We process the ORIGINAL extracted entities for dimensions, or the NORMALIZED ones?
        # Original main.py used `product_entities_file` as input.
        # normalize_dimensions.py reads input, adds 'standardized_dimensions' field, saves to output.
        
        normalize_dimensions_process(
            input_path=PRODUCT_ENTITIES_FILE, 
            output_path=PRODUCT_ENTITIES_WITH_DIMS_FILE
        )
        
        log_with_timestamp(f'✅ Dimension normalization saved to {PRODUCT_ENTITIES_WITH_DIMS_FILE}')
    except Exception as e:
        log_with_timestamp(f'⚠️ Dimension normalization failed: {e}')
