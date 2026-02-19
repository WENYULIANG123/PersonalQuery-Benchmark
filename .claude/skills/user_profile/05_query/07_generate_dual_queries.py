#!/usr/bin/env python3
"""
Generate Dual Queries (V5 - Final Corrected)
生成双重查询：
- 个性化查询：直接使用 preference_match_results 中已匹配的 Top 3 属性
- 大众查询：使用该商品的【其他用户评论】中出现频率最高的 Top 3 属性
"""

import json
import os
import sys
import argparse
import random
from datetime import datetime
from collections import Counter, defaultdict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def call_llm_api_with_retry(prompt, max_retries=3):
    """通用 LLM 调用函数，带重试机制"""
    try:
        from llm_client import LLMClient
        client = LLMClient()
        
        for attempt in range(max_retries):
            response = client.call(prompt, max_tokens=128)
            if response:
                # 去除可能的引号和空白
                return response.strip().strip('"').strip("'")
            import time
            time.sleep(1)
    except Exception as e:
        print(f"LLM error: {e}")
    return None

def load_match_results(match_dir, user_id):
    """加载用户的匹配结果并按 ASIN 聚合属性"""
    match_file = os.path.join(match_dir, f"match_{user_id}.json")
    if not os.path.exists(match_file):
        return None

    with open(match_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用字典按 ASIN 聚合：asin -> {category: str, attr_pool: list}
    aggregated = {}
    
    for item in data.get('results', []):
        asin = item.get('asin')
        if not asin:
            continue
        
        # 获取该记录的所有匹配属性（不仅仅是前3个，先全部合并）
        selected_attrs = item.get('selected_attributes', [])
        
        # 兼容性处理：有时属性在 final_match 中
        if not selected_attrs:
            final_match = item.get('final_match')
            if final_match:
                selected_attrs = final_match.get('selected_attributes', [])
                
        current_attrs = []
        for attr_obj in selected_attrs:
            if isinstance(attr_obj, dict):
                current_attrs.append(attr_obj.get('attribute', ''))
            else:
                current_attrs.append(str(attr_obj))
        
        if asin not in aggregated:
            aggregated[asin] = {
                'category': item.get('category', 'Unknown'),
                'attr_pool': [] # 使用列表保留原始顺序（通常代表重要度）
            }
        
        # 将新属性加入池中（去重并保留顺序）
        for a in current_attrs:
            if a and a not in aggregated[asin]['attr_pool']:
                aggregated[asin]['attr_pool'].append(a)

    # 转化为最终格式，每个 ASIN 只保留聚合后的 Top 3
    results = []
    for asin, info in aggregated.items():
        if not info['attr_pool']:
            print(f"  Warning: {asin} has no attributes after aggregation")
            continue
            
        results.append({
            'asin': asin,
            'category': info['category'],
            'attributes': info['attr_pool'][:3] # 聚合后的 Top 3
        })

    return results

def load_all_users_preferences(prefs_dir):
    """加载所有用户的偏好数据，按 ASIN 组织，并保留 original_text"""
    asin_to_users_attrs = defaultdict(list)  # asin -> list of (user_id, attributes)
    user_preferences_map = {} # user_id -> asin -> preferences dict

    for filename in os.listdir(prefs_dir):
        if not filename.startswith('preferences_') or not filename.endswith('.json'):
            continue

        user_id = filename.replace('preferences_', '').replace('.json', '')
        pref_file = os.path.join(prefs_dir, filename)

        try:
            with open(pref_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Store user preferences map
            user_preferences_map[user_id] = {}

            # 提取该用户对所有商品的偏好
            for item in data.get('results', []):
                asin = item.get('asin')
                if not asin:
                    continue
                
                # Store full preferences for detailed lookup later
                user_preferences_map[user_id][asin] = item.get('preferences', {})

                preferences = item.get('preferences', {})
                user_attrs = []
                for pref_category, entities in preferences.items():
                    if isinstance(entities, list):
                        for ent in entities:
                            entity = ent.get('entity', '')
                            if entity:
                                user_attrs.append(entity)

                if user_attrs:
                    asin_to_users_attrs[asin].append({
                        'user_id': user_id,
                        'attributes': user_attrs
                    })
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")
            continue

    return asin_to_users_attrs, user_preferences_map


def get_top_attributes_for_product(asin, target_user_id, asin_to_users_attrs):
    """
    获取某个商品的大众 Top 3 属性（从其他用户的评论）
    """
    if asin not in asin_to_users_attrs:
        return []

    # 获取该商品的所有用户偏好（排除目标用户）
    other_users_data = [u for u in asin_to_users_attrs[asin] if u['user_id'] != target_user_id]

    if not other_users_data:
        return []

    # 合并所有其他用户的属性
    all_other_attrs = []
    for user_data in other_users_data:
        all_other_attrs.extend(user_data['attributes'])

    # 计算频率最高的 Top 3
    attr_counter = Counter(all_other_attrs)
    top_attrs = [attr for attr, count in attr_counter.most_common(3)]

    return top_attrs

def generate_public_query(asin, category, public_attrs):
    """生成语义化的大众查询（基于 LLM 进行自然语言包装）"""
    attrs_str = ", ".join(public_attrs)
    
    prompt = f"""Task: Transform common product attributes into a natural, highly descriptive search query.
Category: {category}
Attributes: {attrs_str}

CONSTRAINTS:
1. LENGTH: Aim for 25-30 words.
2. TONE: Matter-of-fact, neutral, and practical.
3. FORBIDDEN WORDS: Avoid "Alchemy", "Poetry", "Magic", "Fascinating". Use clear product-focused language.
4. CONTENT: Describe a realistic search intent for common product needs.
5. FORMAT: Output ONLY the query text.

Example (26 words):
"I am searching for high-quality {category} items that offer consistent reliability and professional-grade performance, ensuring they meet the diverse needs of my daily creative projects."
"""
    query = call_llm_api_with_retry(prompt)
    
    if not query:
        # Fallback if LLM fails
        query = f"I'm searching for {category} products that feature {attrs_str}, looking for reliable performance and quality for my upcoming projects that I am planning now."
    
    # 优化后的长度控制：避免简单重复
    words = query.split()
    if len(words) < 20:
        fillers = [
            "that meet the specific requirements of my upcoming projects",
            "to ensure consistent results in my daily creative work",
            "to achieve the reliable standards needed for my collection",
            "as I need durable performance for my current workshop tasks"
        ]
        query += " " + random.choice(fillers)
        words = query.split()
    
    if len(words) > 30:
        query = " ".join(words[:30])

    return {
        'asin': asin,
        'category': category,
        'public_attributes': public_attrs,
        'public_query': query,
        'word_count': len(query.split())
    }

def generate_personalized_query(asin, category, product_attrs, public_attrs=None):
    """生成语义化的个性化查询（由 LLM 进行语义变换，不使用原词）"""
    # 过滤掉与大众属性重复的，保留差异化
    public_set = set()
    if public_attrs:
        public_set = {a.lower().strip() for a in public_attrs if a}

    diff_attrs = [a for a in product_attrs if a and a.lower().strip() not in public_set]
    if not diff_attrs:
        diff_attrs = product_attrs
        
    attrs_str = ", ".join(diff_attrs[:3])
    
    prompt = f"""Task: Transform user-specific technical attributes into a natural, highly sophisticated search query.
Category: {category}
Attributes: {attrs_str}

STRICT CONSTRAINTS:
1. GROUNDED SEMANTIC TRANSFORMATION: Convert attributes into technical requirements or operational needs. DO NOT use flowery metaphors (NO "alchemy", "poetry", "soul", "magic").
2. LENGTH: Aim for 25-30 words.
3. TONE: Technical, precise, and result-oriented.
4. FORMAT: Output ONLY the query text.

Example: If attribute is "Acid-free", describe "guaranteeing long-term chemical neutrality and archival stability for high-value preservation tasks".
"""
    query = call_llm_api_with_retry(prompt)
    
    if not query:
        query = f"For my {category} projects, I'm searching for items offering {attrs_str} that will meet my specific needs for my professional collection."

    # 优化后的长度控制
    words = query.split()
    if len(words) < 20:
        fillers = [
            "to ensure repeatable results for my specialized assembly tasks",
            "matching the technical specifications of my current projects",
            "to maintain the high output standards for my professional work",
            "while providing the reliable performance required for complex assemblies"
        ]
        query += " " + random.choice(fillers)
        words = query.split()
        
    if len(words) > 30:
        query = " ".join(words[:30])

    return {
        'asin': asin,
        'category': category,
        'product_attributes': product_attrs,
        'personalized_query': query,
        'word_count': len(query.split())
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate Dual Queries (V5 - Final Corrected)",
        epilog="生成双重查询：大众查询（该商品的其他用户评论Top 3属性）+ 个性化查询（匹配结果的Top 3属性）"
    )

    parser.add_argument("--match-results-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results",
                        help="Directory containing match results (Top 3 attributes)")
    parser.add_argument("--preferences-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences",
                        help="Directory containing user preference results")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/dual_queries",
                        help="Output directory for dual queries")
    parser.add_argument("--user-id",
                        default=None,
                        help="Process only specific user ID (optional)")
    parser.add_argument("--holdout-dir",
                        default=None,
                        help="Directory containing holdout query files (to filter products)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 60)
    log_with_timestamp("Dual Query Generation (V5 - Final Corrected)")
    log_with_timestamp("=" * 60)

def process_user(user_id, args, asin_to_users_attrs, user_prefs_map):
    # 步骤1：加载匹配结果（个性化查询用）
    log_with_timestamp(f"Loading match results for user {user_id}...")
    match_data = load_match_results(args.match_results_dir, user_id)
    if not match_data:
        log_with_timestamp(f"  Error: No match data found for user {user_id}")
        return

    # 步骤2.5：加载 Holdout 数据（用于过滤和修正类目）
    holdout_asins = set()
    asin_to_category = {}
    if args.holdout_dir and user_id:
        holdout_file = os.path.join(args.holdout_dir, f"query_{user_id}.json")
        if os.path.exists(holdout_file):
            try:
                with open(holdout_file, 'r', encoding='utf-8') as f:
                    h_data = json.load(f)
                    
                    # 获取类目映射和 ASIN 列表
                    items = []
                    # 兼容 V2 Split 格式
                    if isinstance(h_data, dict) and 'holdout_asins' in h_data:
                         # 直接是 ASIN 列表
                        asins = h_data.get('holdout_asins', [])
                        for a in asins:
                            if a:
                                holdout_asins.add(a)
                                # 从 match data 中找对应的 category，这里先不做映射
                        log_with_timestamp(f"  Loaded {len(holdout_asins)} holdout ASINs from V2 format")
                    
                    # 兼容旧格式
                    elif isinstance(h_data, list):
                        items = h_data
                    elif isinstance(h_data, dict):
                        items = h_data.get('query_results', []) or h_data.get('query_items', [])
                    
                    if not holdout_asins: # 如果上面没加载到，尝试从 items 加载
                        for item in items:
                            a = item.get('asin')
                            c = item.get('category')
                            if a:
                                holdout_asins.add(a)
                                if c and c != "Unknown":
                                    asin_to_category[a] = c
                log_with_timestamp(f"  Filtering for {len(holdout_asins)} holdout products")
            except Exception as e:
                log_with_timestamp(f"  Error loading holdout file: {e}")

    # 步骤3：生成查询
    log_with_timestamp(f"\nGenerating queries for {len(match_data)} products...")

    queries = []
    public_attrs_summary = defaultdict(int)  # 统计所有使用的大众属性

    if not holdout_asins:
        log_with_timestamp(f"  Warning: No holdout data found for user {user_id}. Skipping generation.")
        return

    for item in match_data:
        asin = item['asin']
        
        # 严格过滤：必须在 Holdout 集中
        if asin not in holdout_asins:
            continue
            
        # 优先使用 Holdout 中的确切类目
        category = asin_to_category.get(asin, item.get('category', 'Unknown'))
        
        # 如果还是 Unknown，尝试从画像偏好中修复（保底）
        if not category or category == "Unknown":
             category = "Arts & Crafts"
             
        product_attrs = item['attributes']  # 目标用户的 Top 3（个性化查询）

        # 过滤：个性化属性必须不少于3个，确保个性化信息足够丰富
        if len(product_attrs) < 3:
            # log_with_timestamp(f"  Skipping {asin}: not enough personalized attributes ({len(product_attrs)})")
            continue

        # 获取该商品的大众 Top 3 属性（从其他用户评论）
        public_attrs = get_top_attributes_for_product(asin, user_id, asin_to_users_attrs)
        if not public_attrs:
            log_with_timestamp(f"  Warning: No other user attributes found for {asin}, using generic")
            public_attrs = ["Quality", "Easy to use", "Versatility"]  # 默认值

        # 统计大众属性
        for attr in public_attrs:
            public_attrs_summary[attr] += 1

        # 生成大众查询（基于该商品的其他用户评论）
        pub_query = generate_public_query(asin, category, public_attrs)
        if not pub_query:
            continue

        # 生成个性化查询（基于目标用户的 Top 3 属性，且过滤掉大众属性，带上下文增强）
        # 获取该用户对该商品的详细偏好（用于提取 original_text）
        user_specific_prefs = user_prefs_map.get(user_id, {}).get(asin, {})
        per_query = generate_personalized_query(asin, category, product_attrs, public_attrs)
        if not per_query:
            continue

        # 保存双重查询结果
        queries.append({
            'asin': asin,
            'category': category,
            'public_attributes': public_attrs,  # 该商品的大众 Top 3
            'public_query': pub_query['public_query'],
            'public_word_count': pub_query['word_count'],
            'product_attributes': product_attrs,  # 目标用户的 Top 3（个性化）
            'personalized_query': per_query['personalized_query'],
            'personalized_word_count': per_query['word_count']
        })

    # 打印大众属性统计
    log_with_timestamp(f"\nTop 10 most used public attributes:")
    for attr, count in Counter(public_attrs_summary).most_common(10):
        log_with_timestamp(f"    {attr}: {count} products")

    # 保存到文件
    output_file = os.path.join(args.output_dir, f"dual_queries_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"Saved: {output_file}")

    # 汇总
    log_with_timestamp("\n" + "=" * 60)
    log_with_timestamp("SUMMARY")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"User ID: {user_id}")
    log_with_timestamp(f"Total products: {len(match_data)}")
    log_with_timestamp(f"Queries generated: {len(queries)}")
    log_with_timestamp("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dual Queries (V5 - Final Corrected)",
        epilog="生成双重查询：大众查询（该商品的其他用户评论Top 3属性）+ 个性化查询（匹配结果的Top 3属性）"
    )

    parser.add_argument("--match-results-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/preference_match_results",
                        help="Directory containing match results (Top 3 attributes)")
    parser.add_argument("--preferences-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/user_preferences",
                        help="Directory containing user preference results")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/dual_queries",
                        help="Output directory for dual queries")
    parser.add_argument("--user-id",
                        default=None,
                        help="Process only specific user ID (optional)")
    parser.add_argument("--holdout-dir",
                        default=None,
                        help="Directory containing holdout query files (to filter products)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_with_timestamp("=" * 60)
    log_with_timestamp("Dual Query Generation (V5 - Final Corrected)")
    log_with_timestamp("=" * 60)

    # 步骤2：加载所有用户的偏好数据（按 ASIN 组织）
    # 注意：这步比较耗时，只加载一次
    log_with_timestamp(f"Loading all user preferences (organized by ASIN)...")
    asin_to_users_attrs, user_prefs_map = load_all_users_preferences(args.preferences_dir)
    log_with_timestamp(f"  Loaded {len(asin_to_users_attrs)} products with user preferences")

    if args.user_id:
        users = [args.user_id]
    else:
        # Scan directory
        users = []
        for f in os.listdir(args.match_results_dir):
            if f.startswith("match_") and f.endswith(".json") and "backup" not in f:
                uid = f.replace("match_", "").replace(".json", "")
                users.append(uid)
        users = sorted(users)
        log_with_timestamp(f"Found {len(users)} users to process")

    for user_id in users:
        process_user(user_id, args, asin_to_users_attrs, user_prefs_map)

if __name__ == "__main__":
    main()
