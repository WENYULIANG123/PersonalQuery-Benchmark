#!/usr/bin/env python3
"""
Generate Persona Prompts Script (User-Centric Version)
Implements Step 5.1 of the User Profile Manager workflow.
Logic:
1. Load Step 3 match results (match_[USER_ID].json).
2. Aggregate all selected attributes and product categories to create a comprehensive view of user preferences.
3. Construct a prompt asking LLM to synthesize a ~200-word persona description.
4. Save consolidated prompts per user.
"""

import json
import os
import argparse
from datetime import datetime

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_persona_prompt(user_id, aggregated_data):
    # aggregated_data is a list of strings: "Category: ... | Attributes: ..."
    pref_summary = "\n".join([f"- {item}" for item in aggregated_data])
    
    prompt = f"""You are an expert consumer behavior analyst. Your task is to synthesize a detailed user persona based on a collection of products they have purchased and the specific attributes they prioritized for each.

**User ID:** {user_id}

**Data Evidence (Purchased Products & Prioritized Attributes):**
{pref_summary}

**Task:**
Analyze the patterns in these preferences to create a holistic and consistent user persona description. 

**Requirements:**
1. **Length**: Approximately 200 words.
2. **Style**: Professional, analytical, and insightful.
3. **Core Elements to Include**:
   - **General Interests**: What overarching themes (e.g., hobbyist, professional, gift-giver) emerge from the categories?
   - **Quality Standards**: What specific levels of quality or performance do they consistently look for?
   - **Practical Constraints**: Do they value portability, ease of use, durability, etc.?
   - **Shopping Intent**: Are they looking for specialized tools, creative materials, or utility items?
4. **Tone**: Objective third-party observation.

**Output Format:**
Output ONLY the persona description. No explanations, no prefixes.
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate Step 5 Persona Prompts from Step 3 Match Results")
    parser.add_argument("--input", required=True, help="Path to match_[USER_ID].json")
    parser.add_argument("--output-dir", default="/home/wlia0047/ar57/wenyu/result/user_profile/persona_prompts", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.input):
        log_with_timestamp(f"❌ Input file not found: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        user_id = data.get('user_id')
        results = data.get('results', [])

    if not user_id:
        log_with_timestamp("❌ User ID not found in input file.")
        return

    log_with_timestamp(f"Aggregating preference data for user {user_id} ({len(results)} products)...")

    aggregated_data = []
    for item in results:
        category = item.get('category', 'Unknown')
        attributes = item.get('selected_attributes', [])
        if attributes:
            aggregated_data.append(f"Category: {category} | Priorities: {', '.join(attributes)}")

    if not aggregated_data:
        log_with_timestamp(f"⚠️ No attributes found for user {user_id}. Skipping.")
        return

    prompt_text = generate_persona_prompt(user_id, aggregated_data)

    output_file = os.path.join(args.output_dir, f"persona_prompt_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "user_id": user_id,
            "total_evidence_items": len(aggregated_data),
            "prompt": prompt_text
        }, f, indent=2, ensure_ascii=False)
        
    log_with_timestamp(f"✅ Generated persona prompt for user {user_id} -> {output_file}")

if __name__ == "__main__":
    main()
