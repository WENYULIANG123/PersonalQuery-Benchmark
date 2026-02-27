#!/usr/bin/env python3
"""
Stage 0b: 自动化偏好提取

使用 LLM API 从用户评论中批量提取偏好实体。

输入：reviews_[USER_ID].json 文件
输出：preferences_[USER_ID].json 文件
"""

import json
import os
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Any

# 添加 llm_client.py 的路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import LLMClient

# 初始化 LLM 客户端
llm_client = LLMClient()

def call_llm_api(prompt, max_tokens=2000):
    """调用 LLM API (使用 GLM-4.5-Air)"""
    try:
        return llm_client.call(prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"LLM API error: {e}")
        return None

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def extract_preferences_from_review(review: Dict, product_title: str) -> Dict:
    """使用 LLM 从单条评论中提取偏好"""

    review_text = review.get('reviewText', '')
    summary = review.get('summary', '')

    if not review_text:
        return None

    prompt = f"""Analyze the following product review and extract user preferences.

Product: {product_title}
Review: {review_text}
Summary: {summary}

Extract preferences in the following JSON format:
{{
  "preferences": {{
    "Category1": [
      {{
        "entity": "specific attribute or feature",
        "sentiment": "positive/negative/neutral",
        "original_text": "quote from review",
        "improvement_wish": "suggestion if negative, empty otherwise"
      }}
    ]
  }}
}}

Rules:
1. Extract only specific, meaningful attributes (not generic like "good product")
2. Use descriptive category names (e.g., "Ease of Use", "Quality", "Design")
3. Include original text quotes
4. For negative sentiments, suggest improvements
5. Return ONLY valid JSON, no other text

JSON:"""

    response = call_llm_api(prompt, max_tokens=2000)
    if not response:
        return None

    try:
        # 尝试解析 JSON
        # 移除可能的 markdown 代码块标记
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        result = json.loads(response.strip())
        return result.get('preferences', {})
    except:
        return None

def merge_preferences(all_preferences: List[Dict]) -> Dict:
    """合并多个评论的偏好"""
    merged = {}

    for prefs in all_preferences:
        if not prefs:
            continue

        for category, entities in prefs.items():
            if category not in merged:
                merged[category] = []

            for entity in entities:
                # 避免重复
                entity_text = entity.get('entity', '')
                if entity_text and not any(e.get('entity') == entity_text for e in merged[category]):
                    merged[category].append(entity)

    return merged

# 全局元数据缓存
_metadata_cache = None

def load_metadata(meta_file: str) -> Dict[str, Dict]:
    """加载商品元数据文件，返回 ASIN -> metadata 的映射"""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    log_with_timestamp(f"加载元数据文件: {meta_file}")
    metadata = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                asin = item.get('asin')
                if asin:
                    metadata[asin] = item
            except:
                continue
    _metadata_cache = metadata
    log_with_timestamp(f"加载了 {len(metadata)} 个商品的元数据")
    return metadata

def process_user_reviews(user_id: str, reviews_file: str, output_dir: str, metadata: Dict = None):
    """处理单个用户的评论"""

    log_with_timestamp(f"处理用户 {user_id}...")

    # 加载评论
    with open(reviews_file) as f:
        data = json.load(f)

    reviews = data.get('reviews', [])
    log_with_timestamp(f"  共 {len(reviews)} 条评论")

    # 处理每条评论
    all_preferences = []
    products = {}

    for i, review in enumerate(reviews):
        asin = review.get('asin')
        # 从元数据获取商品标题，而不是从评论中获取
        if metadata and asin in metadata:
            product_title = metadata[asin].get('title', 'Unknown')
        else:
            product_title = 'Unknown'

        if asin not in products:
            products[asin] = {
                'asin': asin,
                'product_title': product_title,
                'reviews': []
            }

        products[asin]['reviews'].append(review)

    # 为每个商品提取偏好
    results = []
    total_products = len(products)

    for i, (asin, product_data) in enumerate(products.items(), 1):
        log_with_timestamp(f"  处理商品 {i}/{total_products}: {asin}")

        # 合并该商品的所有评论
        all_reviews_text = '\n'.join(
            r.get('reviewText', '') for r in product_data['reviews'] if r.get('reviewText')
        )

        if not all_reviews_text:
            continue

        # 提取偏好
        preferences = extract_preferences_from_review(
            {'reviewText': all_reviews_text},
            product_data['product_title']
        )

        if preferences:
            results.append({
                'asin': asin,
                'product_title': product_data['product_title'],
                'preferences': preferences,
                'status': 'success'
            })
        else:
            results.append({
                'asin': asin,
                'product_title': product_data['product_title'],
                'preferences': {},
                'status': 'failed'
            })

    # 保存结果
    output_file = os.path.join(output_dir, f"preferences_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_products': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in results if r['status'] == 'success')
    log_with_timestamp(f"  完成！成功: {success_count}/{total_products}")
    log_with_timestamp(f"  保存到: {output_file}")

    return success_count

def main():
    parser = argparse.ArgumentParser(description="Batch extract preferences from user reviews")
    parser.add_argument("--reviews-dir", required=True, help="Directory containing reviews_*.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory for preferences")
    parser.add_argument("--meta-file", required=True, help="Product metadata file (meta_*.json)")
    parser.add_argument("--user-id", help="Process only specific user (optional)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载元数据
    metadata = load_metadata(args.meta_file)

    # 找到所有评论文件
    if args.user_id:
        review_files = [f"reviews_{args.user_id}.json"]
    else:
        review_files = [f for f in os.listdir(args.reviews_dir) if f.startswith("reviews_") and f.endswith(".json")]

    log_with_timestamp(f"找到 {len(review_files)} 个用户评论文件")

    total_success = 0
    for review_file in review_files:
        user_id = review_file.replace("reviews_", "").replace(".json", "")
        file_path = os.path.join(args.reviews_dir, review_file)

        success = process_user_reviews(user_id, file_path, args.output_dir, metadata)
        total_success += success

    log_with_timestamp(f"\n全部完成！总成功: {total_success}")

if __name__ == "__main__":
    main()
