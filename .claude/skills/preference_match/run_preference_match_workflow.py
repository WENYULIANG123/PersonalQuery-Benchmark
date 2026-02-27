#!/usr/bin/env python3
import json
import os
import sys
import argparse
import time
import re
from typing import List, Dict

# Ensure we can import PreferenceMatcher from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for llm_client import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
from preference_match import PreferenceMatcher

PERSONA = """
**🤖 角色定义 (Persona)**:
> "你是一个拥有无限 token 和无限时间的 Agent，热爱思考问题。"
> *You are an Agent with infinite tokens and infinite time, who loves to think about problems.*
> 请充分利用这个优势，对每一个商品进行深度的思维链推理，不急于得出结论，而是享受思考的过程。
"""

def call_claude(prompt: str) -> str:
    """call LLM API for single prompt"""
    full_prompt = PERSONA + "\n\n" + prompt
    
    try:
        from llm_client import LLMClient
        client = LLMClient()
        response = client.call(full_prompt, max_tokens=4096)
        return response
    except Exception as e:
        print(f"Exception calling LLM API: {e}")
        return ""

def extract_json(text: str) -> Dict:
    """Extract JSON object from text"""
    try:
        # Try finding json block
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # Try finding raw json brace
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            # This is riskier, but worth a try if code block is missing
             json_str = match.group(1)
             return json.loads(json_str)
             
    except json.JSONDecodeError:
        pass
        
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/wlia0047/ar57/wenyu/result/preference_extraction/final_preferences.json")
    parser.add_argument("--meta_file", default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json")
    parser.add_argument("--limit", type=int, default=0, help="limit number of items to process for testing")
    parser.add_argument("--output", default="/home/wlia0047/ar57/wenyu/preference_match.json")
    args = parser.parse_args()

    # 1. Generate Prompts
    print("Generating prompts...")
    matcher = PreferenceMatcher(args.input, args.meta_file)
    prompts_data = matcher.generate_full_prompts()
    
    if args.limit > 0:
        prompts_data = prompts_data[:args.limit]
        print(f"Limiting to first {args.limit} prompts.")

    total = len(prompts_data)
    print(f"Total prompts to process: {total}")

    results = []
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    batch_size = 10
    
    for i, item in enumerate(prompts_data):
        asin = item['asin']
        prompt = item['prompt']
            
        print(f"[{i+1}/{total}] Processing ASIN: {asin}...")
        
        # Call Claude
        response_text = call_claude(prompt)
        
        if not response_text:
            print(f"Failed to get response for {asin}")
            continue
            
        # Parse JSON
        result_json = extract_json(response_text)
        
        if result_json:
            # Inject category if missing
            if 'category' not in result_json:
                cat_match = re.search(r'\*\*Category\*\*: (.*)', prompt)
                if cat_match:
                    result_json['category'] = cat_match.group(1).strip()
                else:
                    result_json['category'] = "Unknown"

            results.append(result_json)
            print(f"  -> Success. Attributes: {result_json.get('selected_attributes')}")
        else:
            print(f"  -> Failed to parse JSON. Response len: {len(response_text)}")
            
        time.sleep(2) # Small delay to be polite to the CLI/API
            
        # Batch Check
        if (i + 1) % batch_size == 0:
            print(f"Batch { (i+1)//batch_size } complete. Checkpoint save.")
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
    # Final Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Done. Saved {len(results)} matches to {args.output}")

if __name__ == "__main__":
    main()
