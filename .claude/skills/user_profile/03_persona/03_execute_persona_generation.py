#!/usr/bin/env python3
"""
Batch Generate Unique Personas
Call LLM to generate highly differentiated user personas from raw preferences.
"""

import json
import os
import sys
import argparse
import concurrent.futures
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message):
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
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
            else:
                log_with_timestamp(f"LLM error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return None

def count_words(text):
    return len(text.split())

def process_user(prompt_file, output_dir):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_id = data.get('user_id')
    prompt = data.get('prompt')
    unique_attrs = data.get('top_unique_attributes', [])[:5]

    log_with_timestamp(f"Processing {user_id}...")
    log_with_timestamp(f"  Unique focus: {[a['attribute'] for a in unique_attrs]}")

    time.sleep(random.uniform(0.5, 1.0))
    response = call_llm_api_with_retry(prompt)

    if response:
        word_count = count_words(response)
        log_with_timestamp(f"  Generated: {word_count} words")

        result = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'version': 'raw_preferences',
            'word_count': word_count,
            'top_unique_attributes': unique_attrs,
            'persona': response.strip()
        }

        output_file = os.path.join(output_dir, f"persona_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return {'user_id': user_id, 'status': 'success', 'word_count': word_count}
    else:
        log_with_timestamp(f"  FAILED")
        return {'user_id': user_id, 'status': 'failed'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/persona_prompts")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/persona_results")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.startswith('persona_prompt_') and f.endswith('.json')
    ])

    log_with_timestamp(f"Found {len(prompt_files)} users to process")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_user, pf, args.output_dir): pf
            for pf in prompt_files
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Summary
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("SUMMARY")
    log_with_timestamp("="*60)

    success = [r for r in results if r.get('status') == 'success']
    for r in success:
        log_with_timestamp(f"{r['user_id']}: {r['word_count']} words")

    log_with_timestamp(f"\nTotal: {len(success)}/{len(results)} success")
    log_with_timestamp(f"Output: {args.output_dir}")

if __name__ == "__main__":
    main()
