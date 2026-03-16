#!/usr/bin/env python3
"""
Stage 4b: Generate Mass Market Persona Descriptions

Read mass market files from Stage 3, generate natural language descriptions for each dimension.

Input: result/personal_query/03_processing/mass_market_*.json
Output: result/personal_query/04_persona/persona_*{user_id}.json
"""

import json
import os
import argparse
import re
import concurrent.futures
from datetime import datetime
from pathlib import Path


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def clean_llm_response(response):
    if not response:
        return ""
    clean = re.sub(r'<\|.*?\|>', '', response)
    clean = re.sub(r'<thinking>[\s\S]*?</thinking>', '', clean)
    clean = re.sub(r'<tool_code>[\s\S]*?</tool_code>', '', clean)
    clean = re.sub(r'```', '', clean)
    return clean.strip()


def generate_dimension_prompt(dimension, attributes, category):
    attr_texts = [a.get('attribute', '') for a in attributes if a.get('attribute')]
    attr_list = '\n'.join([f"  - {attr}" for attr in attr_texts[:20]])

    prompt = f"""Generate a GENERALIZED user persona description (50-80 words) for this Amazon reviewer based on their preferences related to "{dimension}" in the "{category}" category.

DIMENSION: {dimension}
CATEGORY: {category}
NUMBER OF PREFERENCES: {len(attr_texts)}

THEIR SPECIFIC PREFERENCES (raw data - DO NOT copy directly):
{attr_list}

CRITICAL RULES:
1. FOCUS ON UNDERLYING PATTERNS, NOT SPECIFIC VALUES
2. Extract GENERALIZED preferences from the examples above
3. DO NOT mention specific:
   - Numbers (e.g., instead of "18 pieces", say "multiple pieces")
   - Sizes (e.g., instead of "A2 card", say "standard card sizes")
   - Brands (e.g., instead of "Cottage Cutz", say "reliable brands")
   - Specific items (e.g., instead of "Thanksgiving", say "seasonal themes")
   - Colors/measurements (e.g., instead of "3 inches", say "compact sizes")
4. Describe what they VALUE and AVOID in abstract terms
5. Use everyday language (like describing a real person to a friend)
6. 50-80 words - be concise but informative

EXAMPLES OF GOOD vs BAD:
❌ BAD: "prefers die-cuts with 18 pieces for A2 cards"
✅ GOOD: "prefers versatile die sets with multiple pieces for standard card sizes"

❌ BAD: "loves Thanksgiving-themed decorations in autumn colors"
✅ GOOD: "enjoys seasonal and themed decorations for various occasions"

Output ONLY the generalized persona description. No intro, no JSON, no markdown."""

    return prompt


def call_llm(prompt, max_tokens=512, max_retries=3):
    import sys
    skills_path = '/fs04/ar57/wenyu/.claude/skills'
    if skills_path not in sys.path:
        sys.path.insert(0, skills_path)
    
    from llm_client import LLMClient
    client = LLMClient()
    
    for attempt in range(max_retries):
        try:
            response = client.call(prompt, max_tokens=max_tokens)
            if response:
                clean_response = clean_llm_response(response)
                if len(clean_response) > 10:
                    return clean_response
        except Exception as e:
            if attempt == max_retries - 1:
                log_with_timestamp(f"      LLM error after {max_retries} retries: {e}")
            continue
    
    return None


def generate_dimension_persona(dimension, attributes, category):
    prompt = generate_dimension_prompt(dimension, attributes, category)
    persona_text = call_llm(prompt)
    return (dimension, persona_text)


def generate_persona_for_category(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    user_id = data.get('user_id', 'mass_market')
    category = data.get('category', 'Unknown')
    safe_cat_name = category.replace(' ', '_').replace(',', '').replace('&', 'and')
    attributes_by_dimension = data.get('attributes_by_dimension', {})
    
    log_with_timestamp(f"  Processing: {category} (user: {user_id}) - {len(attributes_by_dimension)} dimensions")
    
    dimension_personas = {}
    
    for dimension, attributes in sorted(attributes_by_dimension.items(), 
                                         key=lambda x: -len(x[1])):
        if not attributes:
            continue
            
        log_with_timestamp(f"    Dimension: {dimension} ({len(attributes)} attributes)")
        dim, persona_text = generate_dimension_persona(dimension, attributes, category)
        
        if persona_text:
            dimension_personas[dim] = persona_text
            log_with_timestamp(f"      ✓ Generated ({len(persona_text)} chars)")
        else:
            log_with_timestamp(f"      ✗ Failed")
    
    if dimension_personas:
        output_data = {
            'user_id': user_id,
            'category': category,
            'dimension_personas': dimension_personas
        }
        
        output_file = os.path.join(output_dir, f"persona_{safe_cat_name}_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log_with_timestamp(f"  ✓ Saved: {output_file}")
        return {'status': 'success', 'file': output_file, 'category': category, 
                'dimensions': len(dimension_personas)}
    else:
        log_with_timestamp(f"  ✗ No dimensions generated for {category}")
        return {'status': 'failed', 'category': category}


def main():
    parser = argparse.ArgumentParser(description="Generate mass market persona descriptions")
    parser.add_argument("--input-dir", 
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/03_processing",
                        help="Directory containing mass_market_*.json from Stage 3")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/04_persona",
                        help="Output directory")
    parser.add_argument("--user-id",
                        default="mass_market",
                        help="User ID (default: mass_market)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of concurrent workers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_files = list(Path(args.input_dir).glob("mass_market_*.json"))
    
    log_with_timestamp(f"Found {len(input_files)} mass market persona files")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(generate_persona_for_category, str(f), args.output_dir): f 
            for f in input_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_dims = sum(r.get('dimensions', 0) for r in results if r['status'] == 'success')
    log_with_timestamp(f"\nCompleted: {success_count}/{len(results)} categories, {total_dims} dimension personas")
    log_with_timestamp(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
