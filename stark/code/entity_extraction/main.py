#!/usr/bin/env python3
"""
Pipeline Runner for Entity Extraction
"""

import argparse
import sys
import os
import json

# Ensure we can import from local modules and parent modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import API_RESPONSES_FILE
from step_product_extraction import run_product_extraction
from step_product_extraction_batch import run_product_extraction_batch
from step_normalization import run_product_normalization
from step_normalization_batch import run_normalization_batch
from step_normalization_batch import run_normalization_batch
# Imports from user_perference package
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_perference'))
from user_perference.step_user_preference import run_user_preference_extraction
from user_perference.step_user_preference_batch import run_user_preference_extraction_batch
from entity_matching import perform_entity_matching
from step_entity_matching_batch import run_entity_matching_batch
from step_matched_normalization_batch import run_matched_normalization_batch
from pipeline_config import PRODUCT_ENTITIES_FILE, USER_PREFERENCES_FILE, TARGET_USER
from model import set_api_responses_file
from utils import log_with_timestamp

def main():
    parser = argparse.ArgumentParser(description='Entity Extraction Pipeline')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip product extraction step')
    parser.add_argument('--skip-normalization', action='store_true', help='Skip normalization step')
    parser.add_argument('--max-concurrent-product', type=int, default=102, help='Max workers for product extraction')
    parser.add_argument('--max-concurrent-norm', type=int, default=102, help='Max workers for normalization')
    parser.add_argument('--max-concurrent-pref', type=int, default=102, help='Max workers for user preference extraction')
    parser.add_argument('--skip-user-pref', action='store_true', help='Skip user preference extraction step')
    parser.add_argument('--skip-matching', action='store_true', help='Skip entity matching step')
    parser.add_argument('--skip-matched-norm', action='store_true', help='Skip matched entities normalization')
    parser.add_argument('--use-batch', action='store_true', help='Use SiliconFlow Batch API for processing')
    args = parser.parse_args()

    # Global Setup
    # Clear API log if starting fresh extraction
    if not args.skip_extraction:
        set_api_responses_file(API_RESPONSES_FILE, overwrite=True)
        log_with_timestamp(f'💾 API logs will be cleared and saved to {API_RESPONSES_FILE}')
    else:
        set_api_responses_file(API_RESPONSES_FILE, overwrite=False)

    config = vars(args)

    # Step 1: Product Entity Extraction
    if not args.skip_extraction:
        if args.use_batch:
            product_entities_path = run_product_extraction_batch(config)
        else:
            product_entities_path = run_product_extraction(config)
            
        if not product_entities_path:
            log_with_timestamp("❌ Product extraction failed or returned no data. Aborting.")
            return
    else:
        log_with_timestamp("⏭️ Skipping product extraction step.")

    # Step 2: Entity & Dimension Normalization
    if not args.skip_normalization:
        if args.use_batch:
            run_normalization_batch(config)
        else:
            run_product_normalization(config)
    else:
        log_with_timestamp("⏭️ Skipping normalization step.")

    # Step 3: User Preference Extraction
    if not args.skip_user_pref:
        # Load ASINs from product entities file
        if os.path.exists(PRODUCT_ENTITIES_FILE):
             with open(PRODUCT_ENTITIES_FILE, 'r') as f:
                 prod_data = json.load(f)
             all_asins = [p.get('asin') for p in prod_data.get('products', []) if p.get('asin')]
             
             if args.use_batch:
                 run_user_preference_extraction_batch(config, all_asins)
             else:
                 run_user_preference_extraction(config, all_asins)
        else:
             log_with_timestamp(f"❌ Cannot run user preference extraction: {PRODUCT_ENTITIES_FILE} not found.")
    else:
        log_with_timestamp("⏭️ Skipping user preference extraction step.")

    # Step 4: Entity Matching
    if not args.skip_matching:
        if args.use_batch:
            run_entity_matching_batch(config)
        else:
            # For non-batch, we need to load data first
            if os.path.exists(PRODUCT_ENTITIES_FILE) and os.path.exists(USER_PREFERENCES_FILE):
                with open(PRODUCT_ENTITIES_FILE, 'r') as f:
                    prod_data = json.load(f)
                with open(USER_PREFERENCES_FILE, 'r') as f:
                    pref_data = json.load(f)
                
                # Merge logic similar to batch but using perform_entity_matching
                prod_map = {p['asin']: p for p in prod_data.get('products', [])}
                pref_map = {p['asin']: p for p in pref_data.get('products', [])}
                common_asins = sorted(list(set(prod_map.keys()) & set(pref_map.keys())))
                
                merged_products = []
                for asin in common_asins:
                    p = prod_map[asin].copy()
                    p['user_preference_entities'] = pref_map[asin].get('user_preference_entities', {})
                    merged_products.append(p)
                
                perform_entity_matching(merged_products, max_workers=config.get('max_concurrent_norm', 102))
            else:
                log_with_timestamp("❌ Cannot run matching: input files missing.")
    else:
        log_with_timestamp("⏭️ Skipping entity matching step.")

    # Step 5: Matched Entity Normalization
    if not args.skip_matched_norm:
        if args.use_batch:
            run_matched_normalization_batch(config)
        else:
            log_with_timestamp("⚠️ Matched normalization currently only supports BATCH MODE.")
    else:
        log_with_timestamp("⏭️ Skipping matched entity normalization step.")

    log_with_timestamp("🏁 Pipeline completed.")

if __name__ == "__main__":
    main()