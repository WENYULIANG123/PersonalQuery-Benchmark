#!/usr/bin/env python3
"""Test MiniMax M2 Reranker with Thinking Output - Single Product Test"""

import json
import os
import sys
import re
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'utils')
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')

from utils import log_with_timestamp, load_product_metadata, build_document_text, load_cached_candidates

USER_ID = "A13OFOB1394G31"
MODEL_NAME = "minimax-m2-thinking"

BASE_DIR = "/home/wlia0047/ar57/wenyu"
QUERY_FILE = f"{BASE_DIR}/result/personal_query/07_query/dual_queries_{USER_ID}.json"
META_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json"
PERSONA_DIR = f"{BASE_DIR}/result/personal_query/04_persona"
PROCESSING_DIR = f"{BASE_DIR}/result/personal_query/03_processing"
OUTPUT_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval"
CACHE_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval/cache"
CATEGORY = "Arts_Crafts_and_Sewing"


class MiniMaxThinkingClient:
    """MiniMax API client with thinking enabled"""
    def __init__(self, model: str = "MiniMax-M2.5-highspeed"):
        self.api_key = "sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"
        self.base_url = "https://api.minimaxi.com/v1"
        self.model = model

    def call_with_thinking(self, prompt: str, max_tokens: int = 8192, temperature: float = None, max_retries: int = 3) -> dict:
        """
        Call API with thinking enabled.
        Returns dict with 'thinking' and 'content' keys.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "enable_thinking": True  # Enable thinking mode
        }

        if temperature is not None:
            payload["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                print(f"\n{'='*60}")
                print(f"[API CALL] Attempt {attempt + 1}/{max_retries}")
                print(f"{'='*60}")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180
                )

                if response.status_code == 200:
                    result = response.json()
                    message = result["choices"][0]["message"]
                    
                    # Extract thinking content if available
                    thinking = ""
                    content = message.get("content", "")
                    
                    # Check for thinking in different possible locations
                    if "thinking" in message:
                        thinking = message["thinking"]
                    elif "reasoning_content" in message:
                        thinking = message["reasoning_content"]
                    elif "thought" in message:
                        thinking = message["thought"]
                    
                    # Extract thinking from tags in content
                    # Note: MiniMax uses various thinking tags
                    think_patterns = [
                        r'<thinkxml>(.*?)</thinkxml>',
                        r'<think_xml>(.*?)</think_xml>',
                        r'<thinking>(.*?)</thinking>',
                        r'<think\s*>(.*?)</think\s*>',
                        r'<｜place▁holder▁no▐0｜>(.*?)<｜place▁holder▁no▐1｜>',
                    ]
                    for pattern in think_patterns:
                        think_match = re.search(pattern, content, re.DOTALL)
                        if think_match:
                            thinking = think_match.group(1).strip()
                            # Remove thinking tags from content to get final answer
                            content_clean = re.sub(pattern, '', content, flags=re.DOTALL).strip()
                            content = content_clean
                            break
                    
                    return {
                        "thinking": thinking,
                        "content": content,
                        "raw_response": message
                    }
                elif response.status_code in (429, 500) and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error calling LLM API: {response.status_code} - {response.text}")
                    return {"thinking": "", "content": "", "error": response.text}
            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "500" in error_str) and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling LLM API: {e}")
                return {"thinking": "", "content": "", "error": str(e)}

        return {"thinking": "", "content": "", "error": "Max retries exceeded"}


def load_single_product(asin: str):
    """Load a single product's metadata"""
    all_asins = {asin}
    products, all_metadata = load_product_metadata(META_FILE, all_asins)
    return products.get(asin), all_metadata.get(asin, {})


def load_first_query():
    """Load the first target query from the query file"""
    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    for r in results:
        tq = r.get('target_user_query', {})
        if tq.get('query'):
            return {
                'asin': r.get('asin'),
                'query': tq['query'],
                'category': r.get('category', ''),
                'selected_attributes': tq.get('selected_attributes', [])
            }
    return None


def load_processing_attrs(category: str, user_id: str) -> List:
    """Load processing attributes for persona context"""
    if not PROCESSING_DIR:
        return []
    category_filename = category.replace(" & ", "_and_").replace(" ", "_")
    processing_file = os.path.join(PROCESSING_DIR, f"persona_{category_filename}_{user_id}.json")
    if not os.path.exists(processing_file):
        return []
    try:
        with open(processing_file, 'r') as f:
            data = json.load(f)
            return data.get('attributes', [])
    except Exception as e:
        print(f"Warning: Failed to load processing attrs: {e}")
        return []


def build_persona_context(category: str, selected_attributes: List, user_id: str) -> str:
    """Build persona context string"""
    if not selected_attributes:
        return ""
    
    selected_dims = set(attr.get('dimension', '') for attr in selected_attributes if attr.get('dimension'))
    all_attrs = load_processing_attrs(category, user_id)
    
    if not all_attrs:
        contexts = [f"  - {attr.get('dimension', '')}: {attr.get('value', '')}" 
                    for attr in selected_attributes if attr.get('dimension')]
        return "User Preferences:\n" + "\n".join(contexts) if contexts else ""
    
    attrs_by_dim = {}
    for attr in all_attrs:
        dim = attr.get('dimension', '')
        if dim in selected_dims:
            if dim not in attrs_by_dim:
                attrs_by_dim[dim] = []
            attrs_by_dim[dim].append(attr)
    
    contexts = []
    for attr in selected_attributes:
        dim = attr.get('dimension', '')
        if dim in attrs_by_dim:
            for a in attrs_by_dim[dim]:
                sentiment = a.get('sentiment', 'neutral')
                attr_val = a.get('attribute', '')
                contexts.append(f"  - {dim}: {attr_val} (sentiment: {sentiment})")
    
    return "User Preferences:\n" + "\n".join(contexts) if contexts else ""


def test_single_rerank_with_thinking():
    """Test reranking a single product with full thinking output"""
    print("\n" + "="*80)
    print("MiniMax M2 Single Product Rerank Test (with Thinking)")
    print("="*80)
    
    # Load first query
    query_info = load_first_query()
    if not query_info:
        print("ERROR: No queries found!")
        return
    
    print(f"\n[QUERY INFO]")
    print(f"  ASIN: {query_info['asin']}")
    print(f"  Query: {query_info['query']}")
    print(f"  Category: {query_info['category']}")
    print(f"  Selected Attributes: {len(query_info['selected_attributes'])} items")
    
    # Load product
    product, metadata = load_single_product(query_info['asin'])
    if not product:
        print(f"ERROR: Product {query_info['asin']} not found!")
        return
    
    # Build document text
    doc_text = build_document_text(product, metadata)
    
    print(f"\n[PRODUCT INFO]")
    print(f"  Title: {product.get('title', 'N/A')[:100]}...")
    print(f"  Brand: {product.get('brand', 'N/A')}")
    print(f"  Document length: {len(doc_text)} chars")
    
    # Build persona context
    persona_context = build_persona_context(
        query_info['category'], 
        query_info['selected_attributes'],
        USER_ID
    )
    
    if persona_context:
        print(f"\n[PERSONA CONTEXT]")
        print(persona_context)
    
    # Build prompt (personalized version)
    if persona_context:
        prompt = f'''You are an expert search relevance evaluator. Score the relevance of a product to a user query on a scale from 0.0 to 1.0.

[User Profile]
{persona_context}

[Scoring Priority - STRICT ORDER]
1. PRIMARY: Does the product match the Query's explicit constraints (brand, specific item, function)? If NO, score ≤ 0.5.
2. SECONDARY: If query matches, use User Profile to adjust score (bonus for positive alignment, penalty for negative).

Query: "{query_info['query']}"
Product Info:
{doc_text}

Constraint: Output ONLY a single number between 0.0 and 1.0. No text, no explanation.
Score:'''
    else:
        prompt = f'''You are an expert search relevance evaluator. Your task is to score the relevance of a product to a user query on a continuous scale from 0.0 to 1.0.
Scoring Rubric:
- 0.0: Completely Irrelevant. Does not match the query.
- 0.5: Partially Relevant. Matches some keywords but misses core intent (e.g. wrong brand or function).
- 1.0: Perfectly Relevant. Matches all constraints (brand, function, attributes) in the query.

Query: "{query_info['query']}"
Product Info:
{doc_text}

Constraint: Output ONLY a single floating point number between 0.0 and 1.0. Do not output text.
Relevance Score:'''

    print(f"\n[PROMPT LENGTH] {len(prompt)} chars")
    
    # Call API with thinking
    client = MiniMaxThinkingClient(model="MiniMax-M2.5-highspeed")
    
    print(f"\n[CALLING MINIMAX API WITH THINKING ENABLED]")
    print(f"  Model: {client.model}")
    
    result = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.0)
    
    # Print full thinking output
    print("\n" + "="*80)
    print("THINKING OUTPUT (FULL)")
    print("="*80)
    
    if result.get("thinking"):
        print(result["thinking"])
    else:
        print("(No thinking output available)")
        print(f"\nRaw response keys: {list(result.get('raw_response', {}).keys())}")
    
    # Print final content/score
    print("\n" + "="*80)
    print("FINAL CONTENT / SCORE")
    print("="*80)
    
    content = result.get("content", "")
    print(f"Raw content: '{content}'")
    
    # Parse score - get the LAST number in content (the actual score)
    matches = re.findall(r'0\.\d+|1\.0', content)
    if matches:
        score = float(matches[-1])  # Take the last match
        print(f"\nParsed score: {score}")
    else:
        score = None
        print("\nFailed to parse score from content")
    
    # Save full output to file
    output_file = os.path.join(OUTPUT_DIR, f"test_thinking_{USER_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "query_info": query_info,
            "product": {
                "asin": product.get('asin'),
                "title": product.get('title'),
                "brand": product.get('brand')
            },
            "prompt": prompt,
            "api_result": result,
            "parsed_score": score
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OUTPUT SAVED] {output_file}")
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_single_rerank_with_thinking()
