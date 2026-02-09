#!/usr/bin/env python3
"""
Generate prompts for user-product combinations from existing user review data.
Handles large standard JSON files by parsing top-level user keys.
Optimized for faster metadata retrieval by indexing required ASINs in a single pass.
Outputs one JSON file per user containing all their product prompts.
"""
import os
import json
import gzip
import sys
import re
import argparse
from datetime import datetime

# Path constants
DATA_DIR = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018"
META_FILE = os.path.join(DATA_DIR, "raw/meta_Arts_Crafts_and_Sewing.json.gz")
USER_REVIEWS_FILE = os.path.join(DATA_DIR, "processed/user_reviews/user_product_reviews.json")

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def get_metadata_batch(asins, meta_file):
    """Get metadata for a batch of ASINs in a single pass over the metadata file"""
    log_with_timestamp(f"Indexing metadata for {len(asins)} ASINs...")
    metadata_map = {}
    target_asins = set(asins)
    
    try:
        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    asin = item.get('asin')
                    if asin in target_asins:
                        metadata_map[asin] = item
                        if len(metadata_map) == len(target_asins):
                            break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error reading metadata file: {e}")
    
    log_with_timestamp(f"Found metadata for {len(metadata_map)}/{len(target_asins)} ASINs")
    return metadata_map

def clean_html_content(text) -> str:
    if not text: return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def construct_user_product_prompt(user_id, product_info, review_data):
    """Construct prompt for a specific user's review of a specific product"""
    # Prepare product context
    title = product_info.get('title', '')
    desc = product_info.get('description', '')
    if isinstance(desc, list): desc = " ".join(desc)
    features = product_info.get('feature', [])
    if isinstance(features, list): features = "\n".join([f"- {f}" for f in features])
    
    # Extract category information
    categories = product_info.get('category', [])
    if isinstance(categories, list) and categories:
        product_category = categories[-1] if categories else "Unknown"
    else:
        product_category = str(categories) if categories else "Unknown"
    
    product_context = f"**Product Title:** {title}\n"
    product_context += f"**Product Category:** {product_category}\n"
    if desc: product_context += f"**Product Description:** {clean_html_content(desc)}\n"
    if features: product_context += f"**Product Features:**\n{clean_html_content(features)}\n"

    # Prepare user's review
    review_text = review_data.get('review_text', '').strip()
    review_summary = review_data.get('summary', '').strip()
    review_rating = review_data.get('rating', 0)
    
    review_content = f"""**User's Review:**
Rating: {review_rating}/5
Summary: {review_summary}
Review Text: {review_text}
"""

    prompt = f"""Analyze this user's review and extract their preferences for this specific product.

**User ID:** {user_id}

**Product Information:**
{product_context}

{review_content}

**Task:**
You are an expert product analyst. Extract this specific user's preferences from their review of this product.
For each preference, identify the entity (attribute name/value), the original text it came from, and the sentiment (positive, negative, or neutral).

**CRITICAL REQUIREMENTS:**
1. **Direct Quotes Only**: The 'original_text' field MUST be a direct quote from the review text above. Do NOT paraphrase, summarize, or rewrite. Copy the exact words from the review.
2. **Mandatory Improvement Wish**: For EVERY negative sentiment, you MUST provide a specific 'improvement_wish' that addresses the user's concern.
3. **Evidence-Based Entities**: The 'entity' must be based on actual mentions in the review, not generic assumptions.
4. **User-Specific**: Focus on what THIS USER values or dislikes about THIS PRODUCT.

**Output Format:**
Return a JSON object with the following structure:
{{
    "user_id": "{user_id}",
    "product_category": "{product_category}",
    "preferences": {{
        "CategoryName": [
            {{
                "entity": "Standardized attribute value",
                "original_text": "EXACT quote from review - must be verbatim",
                "sentiment": "positive/negative/neutral",
                "improvement_wish": "Required for negative sentiment, empty string otherwise"
            }}
        ]
    }}
}}

Return ONLY the JSON object.
"""
    return prompt

def save_user_prompts(user_id, product_prompts, output_dir):
    """Save all prompts for a user into a single JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'user_id': user_id,
        'total_prompts': len(product_prompts),
        'prompts': product_prompts,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = os.path.join(output_dir, f"prompt_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_file

def stream_user_data(file_path, target_user_id=None, max_users=None):
    """
    Generator to stream user data from the large JSON file.
    """
    import re
    # Pattern to match "USER_ID": {
    user_start_pattern = re.compile(r'^\s*"([^"]+)":\s*\{\s*$')
    
    current_user_id = None
    user_buffer = []
    brace_count = 0
    users_yielded = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip the first '{'
        first_line = f.readline()
        
        for line in f:
            if brace_count == 0:
                match = user_start_pattern.match(line)
                if match:
                    potential_user_id = match.group(1)
                    if target_user_id and potential_user_id != target_user_id:
                        continue
                    
                    current_user_id = potential_user_id
                    user_buffer = ["{"]
                    brace_count = 1
                    continue
            
            if current_user_id:
                user_buffer.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count <= 0:
                    user_content = "".join(user_buffer)
                    user_content = user_content.rstrip().rstrip(',\n').rstrip('\r')
                    if not user_content.endswith('}'):
                        user_content += '}'
                    
                    try:
                        data = json.loads(user_content)
                        yield current_user_id, data
                        users_yielded += 1
                        if max_users and users_yielded >= max_users:
                            return
                    except json.JSONDecodeError as e:
                        log_with_timestamp(f"Error parsing data for user {current_user_id}: {e}")
                    
                    current_user_id = None
                    user_buffer = []
                    brace_count = 0

def main():
    parser = argparse.ArgumentParser(description="Generate prompts for user-product combinations")
    parser.add_argument('--user-id', type=str, help='Specific user ID to process')
    parser.add_argument('--max-users', type=int, help='Maximum number of users to process')
    parser.add_argument('--target-review-count', type=int, help='Target review count for users')
    parser.add_argument('--review-tolerance', type=int, default=10, help='Tolerance range for review count (default: 10)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/user_prompts',
                       help='Output directory for prompt files')
    args = parser.parse_args()
    
    if not os.path.exists(USER_REVIEWS_FILE):
        log_with_timestamp(f"User reviews file not found: {USER_REVIEWS_FILE}")
        return
    
    log_with_timestamp(f"Reading user reviews from {USER_REVIEWS_FILE}...")
    
    users_processed = 0
    total_prompts_generated = 0
    
    # First pass: collect all required ASINs if we are processing multiple users
    # For simplicity, if it's one user, we just collect their ASINs
    users_skipped = 0
    for user_id, user_info in stream_user_data(USER_REVIEWS_FILE, args.user_id, None if args.target_review_count else args.max_users):
        reviews = user_info.get('reviews', [])
        review_count = len(reviews)

        # Filter by review count if specified
        if args.target_review_count:
            min_reviews = args.target_review_count - args.review_tolerance
            max_reviews = args.target_review_count + args.review_tolerance
            if review_count < min_reviews or review_count > max_reviews:
                users_skipped += 1
                continue
            if args.max_users and users_processed >= args.max_users:
                break

        users_processed += 1
        
        user_asins = [r.get('asin') for r in reviews if r.get('asin')]
        
        # Index metadata in one pass
        metadata_map = get_metadata_batch(user_asins, META_FILE)
        
        log_with_timestamp(f"[User {users_processed}] Processing {user_id}: {len(reviews)} reviews")
        
        user_product_prompts = []
        for review_data in reviews:
            asin = review_data.get('asin')
            if not asin: continue
            
            metadata = metadata_map.get(asin, {'asin': asin, 'title': 'Unknown Product', 'category': ['Unknown']})
            prompt = construct_user_product_prompt(user_id, metadata, review_data)
            
            user_product_prompts.append({
                'asin': asin,
                'product_title': metadata.get('title', 'Unknown'),
                'review_rating': review_data.get('rating', 0),
                'prompt': prompt
            })
            total_prompts_generated += 1
            
        if user_product_prompts:
            output_file = save_user_prompts(user_id, user_product_prompts, args.output_dir)
            log_with_timestamp(f"  Saved {len(user_product_prompts)} prompts to {os.path.basename(output_file)}")
            
    if args.target_review_count:
        log_with_timestamp(f"Done! Processed {users_processed} users (skipped {users_skipped} with review count outside {args.target_review_count}±{args.review_tolerance}), generated {total_prompts_generated} total prompts")
    else:
        log_with_timestamp(f"Done! Processed {users_processed} users, generated {total_prompts_generated} total prompts")

if __name__ == "__main__":
    main()
