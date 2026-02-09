#!/usr/bin/env python3
"""
Process Step 5 Persona Prompts and call LLM API to generate holistic user personas.
Reads persona_prompt_[USER_ID].json and saves results into persona_[USER_ID].json.
Each user has one persona.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def call_llm_api_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            from llm_client import LLMClient
            client = LLMClient()
            response = client.call(prompt, max_tokens=1024) # Persona might be longer
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

def process_user_persona(user_id, prompt_file, output_dir):
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompt = data.get('prompt')
            evidence_count = data.get('total_evidence_items', 0)
    except Exception as e:
        log_with_timestamp(f"Error reading {prompt_file}: {e}")
        return False

    if not prompt:
        log_with_timestamp(f"No prompt found for user {user_id}")
        return False

    log_with_timestamp(f"Calling LLM for user {user_id} (Evidence items: {evidence_count})...")
    persona_text = call_llm_api_with_retry(prompt)
    
    if persona_text:
        word_count = len(persona_text.split())
        result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "evidence_items_count": evidence_count,
            "persona": persona_text,
            "word_count": word_count
        }
        output_file = os.path.join(output_dir, f"persona_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"✅ Persona for {user_id} saved to {output_file} (Words: {word_count})")
        return True
    else:
        log_with_timestamp(f"❌ Failed to generate persona for {user_id}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate user personas via LLM API")
    parser.add_argument('--user-id', type=str, help='Specific user ID to process')
    parser.add_argument('--prompt-dir', type=str, 
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/persona_prompts',
                       help='Directory containing persona_prompt_[USER_ID].json files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wlia0047/ar57/wenyu/result/user_profile/persona_results',
                       help='Output directory for persona JSON files')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent LLM calls')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine files to process
    if args.user_id:
        target_file = os.path.join(args.prompt_dir, f"persona_prompt_{args.user_id}.json")
        prompt_files = [(args.user_id, target_file)] if os.path.exists(target_file) else []
    else:
        if not os.path.exists(args.prompt_dir):
            log_with_timestamp(f"Prompt directory not found: {args.prompt_dir}")
            return
        prompt_files = []
        for f in os.listdir(args.prompt_dir):
            if f.startswith('persona_prompt_') and f.endswith('.json'):
                user_id = f.replace('persona_prompt_', '').replace('.json', '')
                prompt_files.append((user_id, os.path.join(args.prompt_dir, f)))
        prompt_files.sort()

    if not prompt_files:
        log_with_timestamp("No persona prompt files found to process.")
        return

    log_with_timestamp(f"Found {len(prompt_files)} persona prompt file(s) to process.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_user_persona, user_id, prompt_path, args.output_dir): user_id 
            for user_id, prompt_path in prompt_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
