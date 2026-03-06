#!/usr/bin/env python3
import os
import json
import gzip
import sys
import subprocess
import re
from collections import Counter
from datetime import datetime

# Path constants
DATA_DIR = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw"
REVIEWS_FILE = os.path.join(DATA_DIR, "Arts_Crafts_and_Sewing.json.gz")
META_FILE = os.path.join(DATA_DIR, "meta_Arts_Crafts_and_Sewing.json.gz")
CLAUDE_CLI_SCRIPT = "/home/wlia0047/ar57/wenyu/stark/code/claude_code_cli.py"

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
    
    product_context = f"**Product Title:** {title}\n"
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
If the sentiment is negative, provide an 'improvement_wish'.

**Output Format:**
Return a JSON object where keys are categories (e.g., Color, Material, Size, Performance, Quality, etc.) and values are lists of objects:
{{
    "CategoryName": [
        {{
            "entity": "Standardized attribute value",
            "original_text": "Exact text from review",
            "sentiment": "positive/negative/neutral",
            "improvement_wish": "Only for negative sentiment"
        }}
    ]
}}

Return ONLY the JSON object.
"""
    return prompt

def call_claude_cli(prompt):
    log_with_timestamp("Calling Claude Code CLI...")
    try:
        # We use python3 to call the script we wrote earlier
        # Since we are in a conda environment (activated via sbatch_wrapper), 
        # python3 should be the right one and 'claude' should be in PATH.
        process = subprocess.Popen(
            ['python3', CLAUDE_CLI_SCRIPT, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            log_with_timestamp(f"Error calling Claude CLI: {stderr}")
            return None
        
        return stdout
    except Exception as e:
        log_with_timestamp(f"Exception calling Claude CLI: {e}")
        return None

def main():
    # 1. Find most reviewed product
    asin, count = find_most_reviewed_product(REVIEWS_FILE)
    if not asin:
        log_with_timestamp("No products found.")
        return

    # 2. Get metadata
    metadata = get_product_metadata(asin, META_FILE)
    if not metadata:
        log_with_timestamp(f"Metadata not found for ASIN: {asin}")
        # We can still proceed if we have reviews, but metadata helps context
        metadata = {'asin': asin, 'title': 'Unknown Product'}

    # 3. Get reviews
    reviews = get_product_reviews(asin, REVIEWS_FILE, limit=5)
    if not reviews:
        log_with_timestamp(f"No reviews found for ASIN: {asin}")
        return

    # 4. Construct prompt
    prompt = construct_prompt(metadata, reviews)
    
    # 5. Call LLM
    llm_response = call_claude_cli(prompt)
    
    if llm_response:
        print("\n=== LLM Response ===")
        print(llm_response)
        
        # Save to file
        output_file = f"/home/wlia0047/ar57/wenyu/stark/code/user_profile/preference_{asin}.json"
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(0))
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4)
                log_with_timestamp(f"Parsed preferences saved to {output_file}")
            else:
                log_with_timestamp("Could not find JSON in LLM response.")
        except Exception as e:
            log_with_timestamp(f"Error parsing/saving JSON: {e}")

if __name__ == "__main__":
    main()
