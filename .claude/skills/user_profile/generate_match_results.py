#!/usr/bin/env python3
"""
Process Step 3 Match Prompts and call LLM API to select Top 3 attributes.
Reads match_prompts_[USER_ID].json and saves results into match_[USER_ID].json.
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
            response = client.call(prompt, max_tokens=1024)
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

def process_single_match_prompt(user_id, item, total_count, index):
    asin = item.get('asin')
    prompt = item.get('prompt')
    
    if not asin or not prompt:
        return None
    
    # Random sleep to avoid sudden bursts
    time.sleep(random.uniform(0.1, 0.5))
    llm_response = call_llm_api_with_retry(prompt)
    
    if llm_response:
        parsed_data = parse_llm_response(llm_response)
        if parsed_data:
            return {
                'asin': asin,
                'selected_attributes': parsed_data.get('selected_attributes', []),
                'reasoning': parsed_data.get('reasoning', ''),
                'status': 'success'
            }
        else:
            return {
                'asin': asin,
                'raw_response': llm_response,
                'status': 'parse_error'
            }
    else:
        return {
            'asin': asin,
            'status': 'api_error'
        }

def main():
    parser = argparse.ArgumentParser(description="Process match prompts and extract Top 3 attributes via LLM API")
    parser.add_argument('--user-id', type=str, help='Specific user ID to process')
    parser.add_argument('--prompt-file', type=str, help='Path to specific match prompt JSON file')
    parser.add_argument('--prompt-dir', type=str, 
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/preference_match',
                       help='Directory containing match_prompts_[USER_ID].json files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results',
                       help='Output directory for match result JSON files')
    parser.add_argument('--max-workers', type=int, default=3, help='Maximum number of concurrent LLM calls')
    parser.add_argument('--limit', type=int, help='Limit the number of products to process per user')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine files to process
    if args.prompt_file:
        prompt_files = [args.prompt_file]
    elif args.user_id:
        target_file = os.path.join(args.prompt_dir, f"match_prompts_{args.user_id}.json")
        prompt_files = [target_file] if os.path.exists(target_file) else []
    else:
        if not os.path.exists(args.prompt_dir):
            log_with_timestamp(f"Prompt directory not found: {args.prompt_dir}")
            return
        # Get all consolidated match prompt files
        prompt_files = [
            os.path.join(args.prompt_dir, f) 
            for f in os.listdir(args.prompt_dir) 
            if f.startswith('match_prompts_') and f.endswith('.json')
        ]
        prompt_files.sort()

    if not prompt_files:
        log_with_timestamp("No match prompt files found to process.")
        return

    log_with_timestamp(f"Found {len(prompt_files)} match prompt file(s) to process.")
    
    for i, prompt_file in enumerate(prompt_files):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
        except Exception as e:
            log_with_timestamp(f"Error loading {prompt_file}: {e}")
            continue
            
        user_id = prompt_data.get('user_id')
        items = prompt_data.get('prompts', [])
        
        if args.limit:
            items = items[:args.limit]
            
        if not user_id or not items:
            continue
            
        log_with_timestamp(f"[{i+1}/{len(prompt_files)}] User {user_id}: Processing {len(items)} products...")
        
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_single_match_prompt, user_id, item, len(items), j+1): j 
                for j, item in enumerate(items)
            }
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                log_with_timestamp(f"  Processed {len(all_results)}/{len(items)} items")

        # Consolidated output for this user
        final_output = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_products': len(items),
            'results': all_results
        }
        
        output_file = os.path.join(args.output_dir, f"match_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"DONE! Match results for {user_id} saved to {output_file}")

if __name__ == "__main__":
    main()
