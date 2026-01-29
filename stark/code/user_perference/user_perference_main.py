#!/usr/bin/env python3
"""
Standalone runner for User Preference Extraction in user_perference directory.
"""

import argparse
import sys
import os
import json

# Add parent directory (stark/code) to path to reach modules in sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add entity_extraction to path so that 'import utils' works as it does in the original scripts
# since step_user_preference.py imports 'utils' directly.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'entity_extraction'))

# Now we can import from entity_extraction modules
from entity_extraction.pipeline_config import PRODUCT_ENTITIES_FILE, API_RESPONSES_FILE
from entity_extraction.utils import log_with_timestamp
from model import set_api_responses_file

# Import the steps from current directory
from step_user_preference import run_user_preference_extraction
from step_user_preference_batch import run_user_preference_extraction_batch

def main():
    parser = argparse.ArgumentParser(description='User Preference Extraction Runner')
    parser.add_argument('--use-batch', action='store_true', default=True, help='Use SiliconFlow Batch API (default: True)')
    parser.add_argument('--no-batch', action='store_false', dest='use_batch', help='Disable SiliconFlow Batch API')
    parser.add_argument('--force', action='store_true', default=True, help='Force re-extraction even if cache exists (default: True)')
    parser.add_argument('--no-force', action='store_false', dest='force', help='Disable force re-extraction')
    parser.add_argument('--max-concurrent-pref', type=int, default=20, help='Max concurrency for non-batch mode')
    args = parser.parse_args()

    config = vars(args)

    # Setup API log
    # Save raw responses to a separate file for user preference extraction
    # Overwrite=True to clear previous runs
    user_pref_log_file = os.path.join(os.path.dirname(API_RESPONSES_FILE), "user_perference_api_raw.json")
    set_api_responses_file(user_pref_log_file, overwrite=True) 

    log_with_timestamp("🚀 Starting User Preference Extraction specific runner")

    # Load ASINs
    try:
        from kb_helper import get_kb_instance
        kb = get_kb_instance()
        kb.load()
        
        # In the context of user preference extraction, we usually care about products 
        # that the target user has interacted with. However, the request is to "directly use SKB".
        # If we just dump ALL products from SKB, it might be huge (300k+).
        # But if the user says "Don't use product_entities.json, directly use data inside SKB", 
        # it likely implies we should look at what SKB has.
        #
        # BUT, wait. User preference extraction is about extraction USER PREFERENCES from USER REVIEWS.
        # We need to know WHICH ASINs the user has reviewed.
        # This list comes from `user_reviews` (loaded in step_user_preference.py).
        #
        # So, instead of reading `PRODUCT_ENTITIES_FILE` to get the list of ASINs,
        # we should just load the user reviews and get the unique ASINs from there.
        # AND THEN use SKB to get attributes for those ASINs.
        
        # Let's verify step_user_preference logic.
        # It calls `load_user_reviews(TARGET_USER)` and then filters `all_asins` against it?
        # No, `run_user_preference_extraction(config, all_asins)` takes `all_asins` as input.
        # It says "Extraction logic... for the given ASINs".
        # Inside, it does: `user_reviews = load_user_reviews(TARGET_USER)`
        # Then it checks `if asin in all_asins` inside the loop.
        
        # So, if we want to process ALL reviews for the user, we should just
        # generate `all_asins` from the user reviews themselves.
        
        from user_preference_extraction import load_user_reviews, TARGET_USER
        
        user_reviews = load_user_reviews(TARGET_USER)
        if not user_reviews:
            log_with_timestamp(f"❌ No reviews found for user {TARGET_USER}")
            return

        all_asins = sorted(list(set(r.get('asin') for r in user_reviews if r.get('asin'))))
        
        if not all_asins:
            log_with_timestamp(f"⚠️ No ASINs found in user reviews.")
            return

        log_with_timestamp(f"📋 Found {len(all_asins)} distinct products reviewed by {TARGET_USER}")

        if args.use_batch:
            run_user_preference_extraction_batch(config, all_asins)
        else:
            run_user_preference_extraction(config, all_asins)
            
    except Exception as e:
        log_with_timestamp(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
