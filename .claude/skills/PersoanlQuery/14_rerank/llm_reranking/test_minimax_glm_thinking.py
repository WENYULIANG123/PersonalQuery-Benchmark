#!/usr/bin/env python3
"""Test MiniMax M2 and GLM-4.5 Reranker with Thinking Output - Single Product Test"""

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

from utils import log_with_timestamp, load_product_metadata, build_document_text

USER_ID = "A13OFOB1394G31"

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
        self.model_name = "MiniMax-M2.5"

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
                    think_patterns = [
                        r'<thinkxml>(.*?)</thinkxml>',
                        r'<think_xml>(.*?)</think_xml>',
                        r'<thinking>(.*?)</thinking>',
                        r'<think\s*>(.*?)</think\s*>',
                    ]
                    for pattern in think_patterns:
                        think_match = re.search(pattern, content, re.DOTALL)
                        if think_match:
                            thinking = think_match.group(1).strip()
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


class GLMClient:
    """GLM-4.5V API client using Anthropic SDK (no thinking mode)"""
    def __init__(self, model: str = "GLM-4.5V"):
        import anthropic
        self.client = anthropic.Anthropic(
            base_url="https://api.z.ai/api/anthropic",
            api_key="db2682f8a0024278a672f762ce36d7cd.RC8PtxIy5xdlh8Uj"
        )
        self.model = model
        self.model_name = "GLM-4.5V"

    def call(self, prompt: str, max_tokens: int = 1024, temperature: float = None, max_retries: int = 3) -> str:
        """
        Call GLM API via Anthropic SDK.
        Returns response text directly.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if temperature is not None:
            kwargs["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling GLM API: {e}")
                return ""

        return ""


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


def test_model(client, model_name: str, prompt: str):
    """Test a single model and return results"""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print(f"{'='*80}")
    
    print(f"\n[CALLING {model_name} API WITH THINKING ENABLED]")
    
    result = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.0)
    
    # Print full thinking output
    print(f"\n{'='*80}")
    print(f"THINKING OUTPUT ({model_name})")
    print(f"{'='*80}")
    
    if result.get("thinking"):
        print(result["thinking"])
    else:
        print("(No thinking output available)")
        if result.get("raw_response"):
            print(f"\nRaw response keys: {list(result.get('raw_response', {}).keys())}")
    
    # Print final content/score
    print(f"\n{'='*80}")
    print(f"FINAL CONTENT / SCORE ({model_name})")
    print(f"{'='*80}")
    
    content = result.get("content", "")
    print(f"Raw content: '{content}'")
    
    # Parse score - look for "Final Score: X.X" pattern first
    score_match = re.search(r'Final Score:\s*(0\.\d+|1\.0)', content, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
        print(f"\nParsed score: {score}")
    else:
        # Fallback: get the LAST number in content (for MiniMax thinking mode)
        matches = re.findall(r'0\.\d+|1\.0', content)
        if matches:
            score = float(matches[-1])  # Take the last match
            print(f"\nParsed score: {score} (from fallback)")
        else:
            score = None
            print("\nFailed to parse score from content")
    
    return {
        "model": model_name,
        "result": result,
        "parsed_score": score
    }


def test_glm_model(client, model_name: str, prompt: str):
    """Test GLM model without thinking mode"""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print(f"{'='*80}")
    
    print(f"\n[CALLING {model_name} API]")
    
    content = client.call(prompt, max_tokens=1024, temperature=0.0)
    
    # Print response
    print(f"\n{'='*80}")
    print(f"RESPONSE ({model_name})")
    print(f"{'='*80}")
    
    print(f"Full response:\n{content}")
    
    # Parse score - look for "Final Score: X.X" pattern
    score_match = re.search(r'Final Score:\s*(0\.\d+|1\.0)', content, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
        print(f"\nParsed score: {score}")
    else:
        # Fallback: get the LAST number in content
        matches = re.findall(r'0\.\d+|1\.0', content)
        if matches:
            score = float(matches[-1])
            print(f"\nParsed score: {score} (from fallback)")
        else:
            score = None
            print("\nFailed to parse score from content")
    
    return {
        "model": model_name,
        "content": content,
        "parsed_score": score
    }


def main():
    """Test both MiniMax M2 (with thinking) and GLM-4.5V (without thinking) on the same product"""
    print("\n" + "="*80)
    print("MiniMax M2 (Thinking) & GLM-4.5V Comparison Test")
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

Please analyze the product relevance step by step:
1. Check if explicit constraints match (brand, items, compatibility)
2. Consider user preferences for fine-tuning
3. Provide your reasoning
4. End with "Final Score: X.X" where X.X is between 0.0 and 1.0

Analysis:'''
    else:
        prompt = f'''You are an expert search relevance evaluator. Your task is to score the relevance of a product to a user query on a continuous scale from 0.0 to 1.0.
Scoring Rubric:
- 0.0: Completely Irrelevant. Does not match the query.
- 0.5: Partially Relevant. Matches some keywords but misses core intent (e.g. wrong brand or function).
- 1.0: Perfectly Relevant. Matches all constraints (brand, function, attributes) in the query.

Query: "{query_info['query']}"
Product Info:
{doc_text}

Please analyze the product relevance:
1. Evaluate how well the product matches the query constraints
2. Provide your reasoning
3. End with "Final Score: X.X" where X.X is between 0.0 and 1.0

Analysis:'''

    print(f"\n[PROMPT LENGTH] {len(prompt)} chars")
    
    # Test MiniMax M2
    minimax_client = MiniMaxThinkingClient(model="MiniMax-M2.5-highspeed")
    minimax_result = test_model(minimax_client, "MiniMax-M2.5", prompt)
    
    print("\n" + "="*80)
    print("Waiting 3 seconds before testing GLM-4.5V...")
    print("="*80)
    time.sleep(3)
    
    # Test GLM-4.5V (no thinking mode)
    glm_client = GLMClient(model="GLM-4.5V")
    glm_result = test_glm_model(glm_client, "GLM-4.5V", prompt)
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Score':<10} {'Has Thinking':<15}")
    print("-" * 45)
    print(f"{'MiniMax-M2.5':<20} {minimax_result['parsed_score'] or 'N/A':<10} {'Yes' if minimax_result['result'].get('thinking') else 'No':<15}")
    print(f"{'GLM-4.5V':<20} {glm_result['parsed_score'] or 'N/A':<10} {'No':<15}")
    
    # Save full output to file
    output_file = os.path.join(OUTPUT_DIR, f"test_comparison_{USER_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "query_info": query_info,
            "product": {
                "asin": product.get('asin'),
                "title": product.get('title'),
                "brand": product.get('brand')
            },
            "prompt": prompt,
            "minimax_result": {
                "model": minimax_result["model"],
                "parsed_score": minimax_result["parsed_score"],
                "thinking": minimax_result["result"].get("thinking", ""),
                "content": minimax_result["result"].get("content", "")
            },
            "glm_result": {
                "model": glm_result["model"],
                "parsed_score": glm_result["parsed_score"],
                "content": glm_result["content"]
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OUTPUT SAVED] {output_file}")
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
