#!/usr/bin/env python3
"""
Stage 4a: Generate Target User Persona Descriptions

Read persona files from Stage 3, generate natural language descriptions for each dimension.

Input: result/personal_query/03_processing/persona_*.json
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


# Dimension descriptions
DIMENSION_DESCRIPTIONS = {
    "Brand_Preference": "Brand preferences and loyalty patterns",
    "Functionality": "What the product does and how it works",
    "Performance": "How well the product performs its intended function",
    "Product_Category": "Type of product or thematic category",
    "Size_Dimensions": "Physical size, dimensions, and scale",
    "Target_User": "Who the user is making items for",
    "Usage_Scenario": "When and where the product is used",
    "Special_Purpose": "Specific use cases or occasions",
    "Ease_of_Use": "How easy or difficult the product is to use",
    "Style_Design": "Aesthetic style, design language, and visual appeal",
    "Quality_Craftsmanship": "Build quality, durability, and material quality",
    "Appearance_Color": "Visual appearance, colors, and finish",
    "Compatibility": "What other products or systems it works with",
    "Material_Composition": "What the product is made of",
    "Packaging_Quantity": "How many items come in a package",
    "Value": "Cost-effectiveness and value for money",
    "Comfort": "Physical comfort during use",
    "Portability": "How portable or compact the product is",
    "Safety": "Safety features and considerations",
    "Special_User_Needs": "Specific accessibility or special requirements"
}


def clean_llm_response(response):
    if not response:
        return ""
    clean = re.sub(r'<\|.*?\|>', '', response)
    clean = re.sub(r'<thinking>[\s\S]*?</thinking>', '', clean)
    clean = re.sub(r'<tool_code>[\s\S]*?</tool_code>', '', clean)
    clean = re.sub(r'【.*?】', '', clean)
    clean = re.sub(r'```', '', clean)
    return clean.strip()


def generate_dimension_prompt(dimension, attributes, category):
    dim_desc = DIMENSION_DESCRIPTIONS.get(dimension, "User preferences in this dimension")
    
    # Format attributes with sentiment
    attr_entries = []
    for a in attributes[:20]:
        attr = a.get('attribute', '')
        sentiment = a.get('sentiment', 'neutral')
        sentiment_indicator = "✓" if sentiment == 'positive' else "✗" if sentiment == 'negative' else "○"
        attr_entries.append(f"  {sentiment_indicator} {attr}")
    
    attr_list = '\n'.join(attr_entries)

    # Adjust word count based on evidence amount
    word_limit = 30 if len(attributes) <= 2 else 50

    prompt = f"""Describe this Amazon reviewer's preference for "{dimension}" in the "{category}" category.

DIMENSION: {dimension} ({dim_desc})
CATEGORY: {category}

PREFERENCES (✓=likes, ✗=dislikes):
{attr_list}

REQUIREMENTS:
1. Stay CLOSE to evidence - don't invent details not in the preferences
2. Use ✓/✗ to identify what they VALUE vs AVOID
3. Generalize naturally (e.g., "18 pieces" → "multiple pieces")
4. If {len(attributes)} preference(s), keep it VERY brief (≤{word_limit} words)

EXAMPLE:
✓ "elegant" → "prefers elegant designs"
✗ "manual trimming" → "dislikes manual finishing work"

Output ONLY the description ({word_limit} words max)."""

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
    
    user_id = data.get('user_id')
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
    parser = argparse.ArgumentParser(description="Generate target user persona descriptions")
    parser.add_argument("--input-dir", 
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/03_processing",
                        help="Directory containing persona_*.json from Stage 3")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/personal_query/04_persona",
                        help="Output directory")
    parser.add_argument("--user-id", required=True,
                        help="User ID to process")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of concurrent workers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_files = list(Path(args.input_dir).glob(f"persona_*_{args.user_id}.json"))
    
    log_with_timestamp(f"Found {len(input_files)} category persona files for user {args.user_id}")
    
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
