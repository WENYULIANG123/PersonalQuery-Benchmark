#!/usr/bin/env python3
"""
Stage 1 - Aspect Extraction (基于论文模板1)

严格按照论文 Appendix A - Figure 4 的提示模板实现

Input: reviews_{USER_ID}.json from Stage 0
Output: aspects_{USER_ID}.json with extracted aspects and sentiments
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient


def log_with_timestamp(message: str):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_aspect_extraction_prompt(few_shot_examples: Optional[List[Dict]] = None) -> str:
    """
    生成论文模板1的提示词
    
    严格按照 Appendix A - Figure 4 的格式
    """
    
    base_prompt = """You are a helpful assistant and an expert in understanding product reviews.
Your task is to extract product aspects and their associated sentiments from customer
reviews, if any are mentioned.

A product aspect refers to a specific feature, attribute, or component of a product or service
that customers mention and evaluate.

The sentiment should be classified as one of: "POSITIVE", "MIXED", or "NEGATIVE".

"""
    
    if few_shot_examples is None:
        few_shot_examples = get_default_few_shot_examples()
    
    examples_section = "Below are examples of customer reviews and the corresponding extracted aspects:\n\n"
    
    for i, example in enumerate(few_shot_examples, 1):
        examples_section += f"Example {i}:\n"
        examples_section += f"Review: \"{example['review']}\"\n"
        examples_section += f"Extracted aspects:\n"
        for aspect in example['aspects']:
            examples_section += f"  - {aspect['aspect']}: {aspect['sentiment']}\n"
        examples_section += "\n"
    
    return base_prompt + examples_section


def get_default_few_shot_examples() -> List[Dict]:
    """获取默认的few-shot示例"""
    
    return [
        {
            "review": "I love this silver glitter glue for my scrapbooking. It works beautifully with my Cuttlebug machine.",
            "aspects": [
                {"aspect": "glitter glue", "sentiment": "POSITIVE"},
                {"aspect": "functionality", "sentiment": "POSITIVE"},
                {"aspect": "compatibility", "sentiment": "POSITIVE"}
            ]
        },
        {
            "review": "The scissors broke after one week. Very poor quality for the price.",
            "aspects": [
                {"aspect": "scissors", "sentiment": "NEGATIVE"},
                {"aspect": "durability", "sentiment": "NEGATIVE"},
                {"aspect": "quality", "sentiment": "NEGATIVE"},
                {"aspect": "price", "sentiment": "NEGATIVE"}
            ]
        },
        {
            "review": "Good product but the packaging is way too much waste. Easy to use though.",
            "aspects": [
                {"aspect": "product", "sentiment": "POSITIVE"},
                {"aspect": "packaging", "sentiment": "NEGATIVE"},
                {"aspect": "ease of use", "sentiment": "POSITIVE"}
            ]
        },
        {
            "review": "Perfect size and color. Works with my existing tools but instructions could be clearer.",
            "aspects": [
                {"aspect": "size", "sentiment": "POSITIVE"},
                {"aspect": "color", "sentiment": "POSITIVE"},
                {"aspect": "compatibility", "sentiment": "POSITIVE"},
                {"aspect": "instructions", "sentiment": "MIXED"}
            ]
        }
    ]


def parse_aspect_extraction_response(response: str) -> Optional[List[Dict]]:
    try:
        import re
        
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "aspects" in data:
                    return data["aspects"]
                elif isinstance(data, list):
                    return data
        
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "aspects" in data:
                    return data["aspects"]
                elif isinstance(data, list):
                    return data
        
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "aspects" in data:
                return data["aspects"]
            elif isinstance(data, list):
                return data
        
        # 列表JSON
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
    
    except Exception as e:
        log_with_timestamp(f"Error parsing response: {e}")
    
    return None


def extract_aspects_from_review(
    review: str,
    product_title: str,
    rating: int = 0,
    max_aspects: int = 5
) -> List[Dict]:
    """
    从单个评论中提取方面
    
    严格按照论文模板1的流程
    """
    
    client = LLMClient()
    
    base_prompt = get_aspect_extraction_prompt()
    
    extraction_prompt = f"""{base_prompt}

Now, you are given a customer review. Extract up to {max_aspects} product aspects mentioned in the
review along with their corresponding sentiments.

Each aspect can be a single word or a multi-word phrase. Respond with a valid JSON.

**Product**: {product_title}
**Rating**: {rating}/5
**Review**: "{review}"

Output format (IMPORTANT - must be valid JSON):
{{
  "aspects": [
    {{
      "aspect": "aspect_name",
      "sentiment": "POSITIVE | MIXED | NEGATIVE",
      "evidence": "supporting text from review"
    }}
  ]
}}

Output ONLY the JSON, no other text."""
    
    try:
        response = client.call(extraction_prompt, max_tokens=1024)
        aspects = parse_aspect_extraction_response(response)
        
        if aspects:
            valid_sentiments = {"POSITIVE", "MIXED", "NEGATIVE"}
            for aspect in aspects:
                if aspect.get("sentiment") not in valid_sentiments:
                    aspect["sentiment"] = "NEUTRAL"
            
            return aspects[:max_aspects]
        else:
            log_with_timestamp(f"❌ Parse failed. Raw response (first 500 chars): {response[:500]}")
            return []
    
    except Exception as e:
        log_with_timestamp(f"❌ Error: {e}. Response: {response[:300] if 'response' in locals() else 'N/A'}")
        return []


def process_product_aspects(product_data: Dict) -> Dict:
    """处理单个产品的方面提取"""
    
    asin = product_data['asin']
    title = product_data['product_title']
    target_user_id = product_data['target_user_id']
    
    # 支持旧新数据格式
    target_review = product_data.get('target_review')
    if not target_review:
        target_reviews = product_data.get('target_reviews', [])
        if target_reviews:
            target_review = target_reviews[0]
    
    other_reviews = product_data.get('other_reviews', [])
    
    result = {
        'asin': asin,
        'product_title': title,
        'target_user_id': target_user_id,
        'target_aspects': [],
        'other_aspects': {},
        'metadata': {}
    }
    
    # 提取目标用户的方面
    if target_review:
        rating = target_review.get('overall', 0) if isinstance(target_review, dict) else 0
        review_text = target_review if isinstance(target_review, str) else target_review.get('reviewText', '')
        
        target_aspects = extract_aspects_from_review(review_text, title, rating)
        
        # 添加元数据
        for aspect in target_aspects:
            aspect['user_type'] = 'target'
            aspect['reviewer_id'] = target_review.get('reviewerID', '') if isinstance(target_review, dict) else ''
        
        result['target_aspects'] = target_aspects
    
    # 提取其他用户的方面（可选）
    if other_reviews:
        other_aspects_by_reviewer = {}
        for review in other_reviews:
            reviewer_id = review.get('reviewerID', 'unknown') if isinstance(review, dict) else 'unknown'
            rating = review.get('overall', 0) if isinstance(review, dict) else 0
            review_text = review if isinstance(review, str) else review.get('reviewText', '')
            
            aspects = extract_aspects_from_review(review_text, title, rating)
            
            for aspect in aspects:
                aspect['user_type'] = 'other'
                aspect['reviewer_id'] = reviewer_id
            
            other_aspects_by_reviewer[reviewer_id] = aspects
        
        result['other_aspects'] = other_aspects_by_reviewer
    
    # 添加统计元数据
    result['metadata'] = {
        'extraction_method': 'aspect_extraction_v1_paper_template',
        'template_version': 'Appendix_A_Figure_4',
        'target_aspects_count': len(result['target_aspects']),
        'other_aspects_count': sum(len(a) for a in result['other_aspects'].values()),
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Aspect Extraction (Paper Template 1)")
    parser.add_argument("--input-file", required=True, help="Input file from Stage 0")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--include-other-users", action="store_true", help="Extract aspects for other users")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1: Aspect Extraction (Paper Template 1 - Appendix A Figure 4)")
    log_with_timestamp("=" * 80)
    
    # 加载数据
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    user_id = data['user_id']
    products = data['results']
    
    log_with_timestamp(f"User: {user_id}")
    log_with_timestamp(f"Products: {len(products)}")
    log_with_timestamp(f"Include other users: {args.include_other_users}")
    log_with_timestamp("")
    
    # 并发处理
    results = []
    completed_count = [0]
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_product_aspects, product): product
            for product in products
        }
        
        for future in as_completed(futures):
            product = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed_count[0] += 1
                
                # 定期输出进度
                if completed_count[0] % 10 == 0 or completed_count[0] == len(products):
                    log_with_timestamp(f"Progress: {completed_count[0]}/{len(products)} completed")
                    
                    # 统计方面
                    aspect_stats = defaultdict(int)
                    sentiment_stats = defaultdict(int)
                    
                    for r in results:
                        for aspect in r.get('target_aspects', []):
                            aspect_name = aspect.get('aspect', 'unknown').lower()
                            aspect_stats[aspect_name] += 1
                            sentiment = aspect.get('sentiment', 'NEUTRAL')
                            sentiment_stats[sentiment] += 1
                    
                    # 输出top 5方面
                    top_aspects = sorted(aspect_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                    log_with_timestamp("  Top aspects:")
                    for aspect, count in top_aspects:
                        log_with_timestamp(f"    {aspect:<30} count={count}")
                    
                    log_with_timestamp(f"  Sentiment distribution: {dict(sentiment_stats)}")
                    log_with_timestamp("")
            
            except Exception as e:
                log_with_timestamp(f"Error processing {product['asin']}: {e}")
                completed_count[0] += 1
    
    # 保存结果
    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'template_version': 'Appendix_A_Figure_4',
        'total_products': len(results),
        'extraction_method': 'aspect_extraction_v1',
        'results': results
    }
    
    output_file = os.path.join(args.output_dir, f'aspects_{user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Saved to {output_file}")
    log_with_timestamp("Stage 1: Aspect Extraction Complete!")


if __name__ == "__main__":
    main()
