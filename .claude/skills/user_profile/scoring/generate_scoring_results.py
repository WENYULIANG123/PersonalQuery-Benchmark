#!/usr/bin/env python3
"""
Process Step 6 Scoring Prompts and call LLM API to get personalization scores.
Reads scoring_prompts_[USER_ID].json and saves results into scores_[USER_ID].json.
"""
import os
import json
import sys
import argparse
import concurrent.futures
import time
import random
from datetime import datetime

# Add parent directory to path for llm_client import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def call_llm_api_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            from llm_client import LLMClient
            client = LLMClient()
            response = client.call(prompt, max_tokens=256)
            if response:
                return response.strip()
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
        import re
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

def process_single_scoring_prompt(asin, prompt):
    llm_response = call_llm_api_with_retry(prompt)
    if llm_response:
        parsed_data = parse_llm_response(llm_response)
        if parsed_data:
            return {
                "asin": asin,
                "personalization_score": parsed_data.get("personalization_score"),
                "justification": parsed_data.get("justification"),
                "status": "success"
            }
        else:
            return {"asin": asin, "raw": llm_response, "status": "parse_error"}
    return {"asin": asin, "status": "api_error"}

def main():
    parser = argparse.ArgumentParser(description="Batch process personalization scores")
    parser.add_argument('--user-id', type=str, help='Specific user ID to process')
    parser.add_argument('--prompt-dir', type=str, 
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/scoring_prompts',
                       help='Directory containing scoring_prompts_[USER_ID].json files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/scoring_results',
                       help='Output directory for scores JSON files')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent LLM calls')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.user_id:
        target_file = os.path.join(args.prompt_dir, f"scoring_prompts_{args.user_id}.json")
        prompt_files = [target_file] if os.path.exists(target_file) else []
    else:
        prompt_files = [os.path.join(args.prompt_dir, f) for f in os.listdir(args.prompt_dir) if f.startswith('scoring_prompts_')]
        prompt_files.sort()

    if not prompt_files:
        log_with_timestamp("No scoring prompt files found.")
        return

    for prompt_file in prompt_files:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            user_id = data.get('user_id')
            prompts = data.get('prompts', [])

        log_with_timestamp(f"Processing user {user_id} with {len(prompts)} queries...")
        
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_scoring_prompt, p['asin'], p['prompt']): p['asin'] for p in prompts}
            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())
                if len(all_results) % 10 == 0:
                    log_with_timestamp(f"  Progress: {len(all_results)}/{len(prompts)}")

        output_file = os.path.join(args.output_dir, f"scores_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "average_score": sum([r.get('personalization_score', 0) for r in all_results if r.get('status') == 'success']) / len([r for r in all_results if r.get('status') == 'success']) if all_results else 0,
                "results": all_results
            }, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"✅ Scores for {user_id} saved to {output_file}")

if __name__ == "__main__":
    main()
