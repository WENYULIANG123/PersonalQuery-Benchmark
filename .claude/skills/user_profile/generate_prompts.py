#!/usr/bin/env python3
"""
Generate prompts for user profile extraction and save them as JSON files.
This script separates prompt generation from LLM execution for batch processing.
"""
import os
import json
import gzip
import sys
import re
import argparse
from collections import Counter
from datetime import datetime

# Path constants
DATA_DIR = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw"
REVIEWS_FILE = os.path.join(DATA_DIR, "Arts_Crafts_and_Sewing.json.gz")
META_FILE = os.path.join(DATA_DIR, "meta_Arts_Crafts_and_Sewing.json.gz")

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def find_most_reviewed_product(reviews_file):
    log_with_timestamp(f"Finding the most reviewed product in {reviews_file}...")
    asin_counter = Counter()
    try:
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    asin = review.get('asin')
                    if asin:
                        asin_counter[asin] += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error reading reviews file: {e}")
        return None, 0
    
    if not asin_counter:
        return None, 0
    
    most_common_asin, count = asin_counter.most_common(1)[0]
    log_with_timestamp(f"Most reviewed product: {most_common_asin} with {count} reviews.")
    return most_common_asin, count

def get_product_metadata(asin, meta_file):
    log_with_timestamp(f"Searching metadata for ASIN: {asin}...")
    try:
        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item.get('asin') == asin:
                        return item
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error reading metadata file: {e}")
    return None

def get_product_reviews(asin, reviews_file, limit=10):
    log_with_timestamp(f"Collecting up to {limit} reviews for ASIN: {asin}...")
    reviews = []
    try:
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    if review.get('asin') == asin:
                        reviews.append(review)
                        if len(reviews) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log_with_timestamp(f"Error reading reviews file: {e}")
    return reviews

def clean_html_content(text) -> str:
    if not text: return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def construct_prompt(product_info, reviews):
    # Prepare product context
    title = product_info.get('title', '')
    desc = product_info.get('description', '')
    if isinstance(desc, list): desc = " ".join(desc)
    features = product_info.get('feature', [])
    if isinstance(features, list): features = "\n".join([f"- {f}" for f in features])
    
    # Extract category information
    categories = product_info.get('category', [])
    if isinstance(categories, list) and categories:
        # Use the most specific (last) category
        product_category = categories[-1] if categories else "Unknown"
    else:
        product_category = str(categories) if categories else "Unknown"
    
    product_context = f"**Product Title:** {title}\n"
    product_context += f"**Product Category:** {product_category}\n"
    if desc: product_context += f"**Product Description:** {clean_html_content(desc)}\n"
    if features: product_context += f"**Product Features:**\n{clean_html_content(features)}\n"

    # Prepare review context
    review_content = ""
    for i, r in enumerate(reviews):
        text = r.get('reviewText', '').strip()
        summary = r.get('summary', '').strip()
        review_content += f"--- Review {i+1} ---\nSummary: {summary}\nText: {text}\n"

    prompt = f"""Analyze the user reviews below and identify mentions of product attributes (preferences).
    
**Product Unstructured Information:**
{product_context}

**User Reviews:**
{review_content}

**Task:**
You are an expert product analyst. Your goal is to extract user preferences from the reviews.
For each preference, identify the entity (attribute name/value), the original text it came from, and the sentiment (positive, negative, or neutral).

**CRITICAL REQUIREMENTS:**
1. **Direct Quotes Only**: The 'original_text' field MUST be a direct quote from the review text above. Do NOT paraphrase, summarize, or rewrite. Copy the exact words from the review.
2. **Mandatory Improvement Wish**: For EVERY negative sentiment, you MUST provide a specific 'improvement_wish' that addresses the user's concern.
3. **Evidence-Based Entities**: The 'entity' must be based on actual mentions in the reviews, not generic assumptions.

**Output Format:**
Return a JSON object with the following structure:
{{
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

def save_prompt_to_json(asin, product_info, reviews, prompt, output_dir):
    """Save prompt and metadata to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "asin": asin,
        "product_title": product_info.get('title', 'Unknown'),
        "num_reviews": len(reviews),
        "prompt": prompt,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = os.path.join(output_dir, f"prompt_{asin}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Saved prompt to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate prompts for user profile extraction")
    parser.add_argument('--asin', type=str, help='Single ASIN to process')
    parser.add_argument('--asin-list', type=str, help='File with list of ASINs (one per line)')
    parser.add_argument('--limit', type=int, default=5, help='Number of reviews per product')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/prompts',
                       help='Output directory for prompt JSON files')
    args = parser.parse_args()
    
    # Determine ASINs to process
    asins = []
    if args.asin:
        asins = [args.asin]
    elif args.asin_list:
        log_with_timestamp(f"Loading ASINs from {args.asin_list}...")
        try:
            with open(args.asin_list, 'r') as f:
                asins = [line.strip() for line in f if line.strip()]
        except Exception as e:
            log_with_timestamp(f"Error reading ASIN list: {e}")
            return
    else:
        # Default: find most reviewed product
        asin, count = find_most_reviewed_product(REVIEWS_FILE)
        if asin:
            asins = [asin]
        else:
            log_with_timestamp("No ASINs to process. Use --asin or --asin-list.")
            return
    
    log_with_timestamp(f"Processing {len(asins)} ASIN(s)...")
    
    # Process each ASIN
    for i, asin in enumerate(asins):
        log_with_timestamp(f"[{i+1}/{len(asins)}] Processing ASIN: {asin}")
        
        # Get metadata
        metadata = get_product_metadata(asin, META_FILE)
        if not metadata:
            log_with_timestamp(f"Metadata not found for ASIN: {asin}, using placeholder")
            metadata = {'asin': asin, 'title': 'Unknown Product'}
        
        # Get reviews
        reviews = get_product_reviews(asin, REVIEWS_FILE, limit=args.limit)
        if not reviews:
            log_with_timestamp(f"No reviews found for ASIN: {asin}, skipping")
            continue
        
        # Construct prompt
        prompt = construct_prompt(metadata, reviews)
        
        # Save to JSON
        save_prompt_to_json(asin, metadata, reviews, prompt, args.output_dir)
    
    log_with_timestamp("Done!")

if __name__ == "__main__":
    main()
