#!/usr/bin/env python3
"""
Generate Scoring Prompts Script
Implements Step 6.1 of the User Profile Manager workflow.
Logic:
1. Load Step 4 queries (queries_[USER_ID].json).
2. Load Step 5 persona (persona_[USER_ID].json).
3. For each query, construct a prompt asking LLM to score personalization against the persona.
4. Save consolidated prompts per user.
"""

import json
import os
import argparse
from datetime import datetime

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_scoring_prompt(user_id, persona_text, query_text):
    prompt = f"""You are an objective auditor of AI-generated content. Your task is to evaluate the **Personalization Degree** of a search query relative to a specific user's persona.

**User Persona:**
\"\"\"
{persona_text}
\"\"\"

**Generated Search Query:**
\"\"\"
{query_text}
\"\"\"

**Task:**
Score how well the search query reflects the established user persona. Does it prioritize the specific quality standards, practical constraints, and shopping intents described in the persona?

**Evaluation Criteria (1-10 Scale):**
- **1-3 (Low)**: Generic query. Does not reflect specific persona traits.
- **4-6 (Medium)**: Reflects some traits but feels formulaic or generic in tone.
- **7-9 (High)**: Deeply aligned with the persona. Captures specific nuances, constraints, and the intended "voice" of the shopper.
- **10 (Perfect)**: Indistinguishable from a highly specific query written by the actual user described.

**Output Format (JSON ONLY):**
{{
  "personalization_score": <int>,
  "justification": "<brief sentence explaining the score>"
}}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate Step 6 Scoring Prompts from Queries and Personas")
    parser.add_argument("--user-id", help="Process specific user ID")
    parser.add_argument("--query-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/query_results", help="Directory for query files")
    parser.add_argument("--persona-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/persona_results", help="Directory for persona files")
    parser.add_argument("--output-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/scoring_prompts", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine users to process
    if args.user_id:
        user_ids = [args.user_id]
    else:
        user_ids = [f.replace('queries_', '').replace('.json', '') for f in os.listdir(args.query_dir) if f.startswith('queries_')]
    
    for user_id in user_ids:
        query_file = os.path.join(args.query_dir, f"queries_{user_id}.json")
        persona_file = os.path.join(args.persona_dir, f"persona_{user_id}.json")
        
        if not os.path.exists(query_file) or not os.path.exists(persona_file):
            log_with_timestamp(f"⚠️ Missing files for user {user_id}. Skipping.")
            continue
            
        log_with_timestamp(f"Processing user {user_id}...")
        
        with open(query_file, 'r', encoding='utf-8') as f:
            query_data = json.load(f).get('results', [])
        
        with open(persona_file, 'r', encoding='utf-8') as f:
            persona_text = json.load(f).get('persona', '')
            
        if not persona_text or not query_data:
            continue
            
        prompts_data = []
        for item in query_data:
            asin = item.get('asin')
            query_text = item.get('query')
            if not asin or not query_text:
                continue
                
            prompt_text = generate_scoring_prompt(user_id, persona_text, query_text)
            prompts_data.append({
                "asin": asin,
                "prompt": prompt_text
            })
            
        output_file = os.path.join(args.output_dir, f"scoring_prompts_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id,
                "total_queries": len(prompts_data),
                "prompts": prompts_data
            }, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f"✅ Generated {len(prompts_data)} scoring prompts for user {user_id} -> {output_file}")

if __name__ == "__main__":
    main()
