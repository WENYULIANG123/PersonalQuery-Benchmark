#!/usr/bin/env python3
"""
Generate dual_queries from match_results using GLM-4.7.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_client import LLMClient

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def generate_query(category, attributes, query_type="public"):
    """Generate 20-30 word query using GLM-4.7."""
    attrs_str = ", ".join(attributes[:3])
    
    if query_type == "public":
        prompt = f"""Task: Transform common product attributes into a natural, highly descriptive search query.
Category: {category}
Attributes: {attrs_str}

STRICT CONSTRAINTS:
1. LENGTH: MUST be exactly 20-30 words. No exceptions.
2. TONE: Matter-of-fact, neutral, and practical.
3. FORBIDDEN WORDS: Avoid "Alchemy", "Poetry", "Magic", "Fascinating".
4. OUTPUT: Only the query text.

Example (27 words):
"I am searching for high-quality {category.lower()} items that offer consistent reliability and professional-grade performance, ensuring they meet the diverse needs of my daily creative projects."
"""
    else:
        prompt = f"""Task: Transform user-specific technical attributes into a natural, highly sophisticated search query.
Category: {category}
Attributes: {attrs_str}

STRICT CONSTRAINTS:
1. LENGTH: MUST be exactly 20-30 words.
2. TONE: Technical, precise, and result-oriented.
3. OUTPUT: Only the query text.

Example (26 words):
"For my professional projects, I'm searching for high-quality items that offer consistent reliability and technical precision, ensuring they meet my specific operational requirements."
"""

    client = LLMClient()

    for attempt in range(5):
        query = client.call(prompt, max_tokens=128)
        if not query:
            continue

        query = query.strip().strip('"').strip("'")
        words = query.split()
        word_count = len(words)

        if 20 <= word_count <= 30:
            return query

        # Adjust prompt based on word count
        if word_count < 20:
            log_with_timestamp(f"  Retry {attempt+1}: got {word_count} words (too short)")
            # Add instruction to be more verbose
            prompt += "\n\nIMPORTANT: Your response was TOO SHORT. Add more descriptive details about the product features and usage scenarios."
        elif word_count > 30:
            log_with_timestamp(f"  Retry {attempt+1}: got {word_count} words (too long)")
            # Add instruction to be more concise
            prompt += "\n\nIMPORTANT: Your response was TOO LONG. Be more concise and focus on the key technical requirements."
    
    # Fallback
    return f"I need {category.lower()} products that provide {attrs_str} for my specific applications and requirements."

def main():
    match_dir = Path("/home/wlia0047/wenyu/result/user_profile/01_matching/results")
    output_dir = Path("/home/wlia0047/wenyu/result/user_profile/06_query")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_with_timestamp("==========================================================")
    log_with_timestamp("Dual Query Generation with GLM-4.7")
    log_with_timestamp("==========================================================")
    
    client = LLMClient()
    log_with_timestamp(f"Using model: {client.model}")
    
    total_queries = 0
    
    for match_file in sorted(match_dir.glob("match_*.json")):
        user_id = match_file.stem.replace("match_", "")
        
        with open(match_file, 'r') as f:
            match_info = json.load(f)
        
        dual_queries = []
        
        for result in match_info.get('results', [])[:20]:  # Limit for testing
            asin = result.get('asin')
            if not asin:
                continue
            
            category = result.get('category', 'Unknown')
            
            # Get selected attributes
            final_match = result.get('final_match') or {}
            selected_attrs_obj = final_match.get('selected_attributes', [])
            
            attrs = [attr.get('attribute', '') for attr in selected_attrs_obj if isinstance(attr, dict)]
            attrs = [a for a in attrs if a]
            
            if not attrs:
                continue
            
            log_with_timestamp(f"Processing {user_id} - {asin}: {', '.join(attrs[:3])}")
            
            # Generate queries
            public_query = generate_query(category, attrs, "public")
            personalized_query = generate_query(category, attrs, "personalized")
            
            # Check word counts
            pub_words = len(public_query.split())
            per_words = len(personalized_query.split())
            
            log_with_timestamp(f"  Public: {pub_words} words, Personalized: {per_words} words")
            
            dual_queries.append({
                'asin': asin,
                'category': category,
                'personalized_query': personalized_query,
                'public_query': public_query,
                'attributes': attrs[:5],
                'word_counts': {'public': pub_words, 'personalized': per_words}
            })
        
        # Save
        if dual_queries:
            output_file = output_dir / f"dual_queries_{user_id}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'user_id': user_id,
                    'total_queries': len(dual_queries),
                    'queries': dual_queries,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)
            
            log_with_timestamp(f"Generated {len(dual_queries)} queries for {user_id}")
            total_queries += len(dual_queries)
    
    log_with_timestamp("==========================================================")
    log_with_timestamp(f"Done! Total queries: {total_queries}")
    log_with_timestamp(f"Output: {output_dir}")
    log_with_timestamp("==========================================================")

if __name__ == "__main__":
    main()
