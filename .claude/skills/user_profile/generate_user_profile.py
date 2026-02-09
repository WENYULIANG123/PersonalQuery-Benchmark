#!/usr/bin/env python3
"""
Process pre-generated prompts and call LLM API to extract user preferences.
Reads consolidated prompt JSON files and saves ALL results into ONE JSON file per user.
Optimized with multi-threading.
"""
import os
import json
import sys
import re
import argparse
import concurrent.futures
import time
import random
from datetime import datetime

# Add parent directory to path for llm_client import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def call_llm_api_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            from llm_client import LLMClient
            client = LLMClient()
            response = client.call(prompt, max_tokens=2048)
            if response:
                return response
            log_with_timestamp(f"Empty response (attempt {attempt + 1})")
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) + random.random()
                log_with_timestamp(f"Rate limited (429). Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                log_with_timestamp(f"Exception calling LLM API: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    return None

def parse_llm_response(llm_response):
    """Clean and parse JSON from LLM response"""
    try:
        json_content = llm_response
        if "```json" in llm_response:
            json_content = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL).group(1)
        elif "```" in llm_response:
            json_content = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL).group(1)
        else:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
        
        return json.loads(json_content)
    except Exception:
        return None

def save_result(user_id, asin, llm_response, output_dir):
    """Fallback save for individual results if something fails"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"preference_{user_id}_{asin}.json")
    try:
        data = parse_llm_response(llm_response)
        if data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
    except:
        pass
    return False

def process_single_prompt(user_id, item, total_count, index):
    asin = item.get('asin')
    prompt = item.get('prompt')
    product_title = item.get('product_title', 'Unknown')
    
    if not asin or not prompt:
        return None
    
    time.sleep(random.uniform(0.1, 0.5))
    llm_response = call_llm_api_with_retry(prompt)
    
    if llm_response:
        parsed_data = parse_llm_response(llm_response)
        if parsed_data:
            return {
                'asin': asin,
                'product_title': product_title,
                'preferences': parsed_data.get('preferences', {}),
                'status': 'success'
            }
        else:
            return {
                'asin': asin,
                'product_title': product_title,
                'raw_response': llm_response,
                'status': 'parse_error'
            }
    else:
        return {
            'asin': asin,
            'status': 'api_error'
        }

def main():
    parser = argparse.ArgumentParser(description="Process prompts and extract user preferences via LLM API")
    parser.add_argument('--user-id', type=str, help='Specific user ID to process')
    parser.add_argument('--prompt-file', type=str, help='Path to user prompt JSON file')
    parser.add_argument('--prompt-dir', type=str, 
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/user_prompts',
                       help='Directory containing user prompt JSON files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences',
                       help='Output directory for preference JSON files')
    parser.add_argument('--max-workers', type=int, default=3, help='Maximum number of concurrent LLM calls')
    parser.add_argument('--limit', type=int, help='Limit the number of products to process per user')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine files to process
    if args.prompt_file:
        prompt_files = [args.prompt_file]
    elif args.user_id:
        target_file = os.path.join(args.prompt_dir, f"prompt_{args.user_id}.json")
        prompt_files = [target_file] if os.path.exists(target_file) else []
    else:
        if not os.path.exists(args.prompt_dir):
            log_with_timestamp(f"Prompt directory not found: {args.prompt_dir}")
            return
        # Get all consolidated prompt files (prompt_USERID.json)
        prompt_files = [
            os.path.join(args.prompt_dir, f) 
            for f in os.listdir(args.prompt_dir) 
            if f.startswith('prompt_') and f.endswith('.json')
        ]
        # Sort files to ensure deterministic order
        prompt_files.sort()

    if not prompt_files:
        log_with_timestamp("No prompt files found to process.")
        return

    log_with_timestamp(f"Found {len(prompt_files)} user prompt file(s) to process.")
    log_with_timestamp(f"Starting processing with {args.max_workers} workers...")
    
    for i, prompt_file in enumerate(prompt_files):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
        except Exception as e:
            log_with_timestamp(f"Error loading {prompt_file}: {e}")
            continue
            
        user_id = user_data.get('user_id')
        product_prompts = user_data.get('prompts', [])
        
        if args.limit:
            product_prompts = product_prompts[:args.limit]
            
        if not user_id or not product_prompts:
            continue
            
        log_with_timestamp(f"[{i+1}/{len(prompt_files)}] User {user_id}: Processing {len(product_prompts)} products...")
        
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_single_prompt, user_id, item, len(product_prompts), j+1): j 
                for j, item in enumerate(product_prompts)
            }
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                # Log progress after each product
                log_with_timestamp(f"  Processed {len(all_results)}/{len(product_prompts)} items")

        # Consolidated output for this user
        final_output = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_products': len(product_prompts),
            'results': all_results
        }
        
        output_file = os.path.join(args.output_dir, f"preferences_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"DONE! Consolidated preferences for {user_id} saved to {output_file}")

if __name__ == "__main__":
    main()
