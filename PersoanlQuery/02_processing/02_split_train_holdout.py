#!/usr/bin/env python3
"""
Stage 2: Filter User Data

过滤偏好提取结果，只保留符合条件的商品。
- 输入: Stage 1 的偏好提取结果
- 输出: 过滤后的查询集数据

Input: result/personal_query/01_preference_extraction/preferences_{user_id}.json
Output: result/personal_query/02_processing/{user_id}/query.json

Note: 直接读取 Stage 1 输出
"""

import json
import os
import argparse
import random
import gzip
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def normalize_attribute(attr_str):
    """标准化属性字符串用于去重"""
    import re
    if not attr_str:
        return attr_str
    
    s = attr_str.lower().strip()
    
    plural_patterns = [
        (r's$', ''),
        (r'es$', ''),
        (r'ies$', 'y'),
        (r'ers$', 'er'),
        (r'ves$', 'fe'),
    ]
    
    for pattern, repl in plural_patterns:
        new_s = re.sub(pattern, repl, s)
        if new_s != s and len(new_s) >= 2:
            s = new_s
            break
    
    if s.endswith('s') and len(s) > 2 and s[-2] not in 'su':
        test_s = s[:-1]
        if len(test_s) >= 2:
            s = test_s
    
    return s


def filter_duplicate_attributes(product_attrs, category='', threshold=0.80):
    """基于相似度过滤重复属性"""
    if not product_attrs or len(product_attrs) <= 1:
        return product_attrs
    
    attr_list = []
    for attr in product_attrs:
        if isinstance(attr, dict):
            attr_list.append({
                'original': attr.get('attribute', ''),
                'dimension': attr.get('dimension', ''),
                'sentiment': attr.get('sentiment', 'neutral'),
                'source': attr
            })
        else:
            attr_list.append({
                'original': str(attr),
                'dimension': '',
                'sentiment': 'neutral',
                'source': attr
            })
    
    exact_groups = {}
    for item in attr_list:
        norm_key = normalize_attribute(item['original'])
        if norm_key not in exact_groups:
            exact_groups[norm_key] = []
        exact_groups[norm_key].append(item)
    
    final_attrs = []
    
    for norm_key, items in exact_groups.items():
        if len(items) == 1:
            final_attrs.append(items[0]['source'])
            continue
        
        items_sorted = sorted(items, key=lambda x: len(x['original']))
        representatives = [items_sorted[0]]
        
        for item in items_sorted[1:]:
            is_duplicate = False
            for rep in representatives:
                ratio = SequenceMatcher(None, 
                    item['original'].lower(), 
                    rep['original'].lower()
                ).ratio()
                if ratio >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                representatives.append(item)
        
        final_attrs.extend([r['source'] for r in representatives])
    
    return final_attrs


def load_product_metadata(meta_file: str, needed_asins: set) -> dict:
    """从元数据文件加载指定 ASIN 的产品信息"""
    metadata = {}
    
    try:
        open_func = gzip.open if meta_file.endswith('.gz') else open
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON list format
                all_meta = json.load(f)
                for item in all_meta:
                    asin = item.get('asin')
                    if asin in needed_asins:
                        metadata[asin] = item
            else:
                # Line delimited JSON format
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        asin = item.get('asin')
                        if asin in needed_asins:
                            metadata[asin] = item
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log_with_timestamp(f"Error loading metadata: {e}")
    
    return metadata


def convert_preferences_to_attributes(prefs_data: dict) -> list:
    """
    将 Stage 1 的嵌套偏好格式转换为扁平化属性列表
    
    Stage 1 格式:
    {
        "Product_Attributes": {
            "Product_Category": [{"entity": "...", "sentiment": "...", ...}]
        }
    }
    
    转换为:
    [
        {"attribute": "...", "dimension": "...", "sentiment": "...", "source": "direct_preference"}
    ]
    """
    attributes = []
    
    if not isinstance(prefs_data, dict):
        return attributes
    
    # 遍历所有 category 和 dimension
    for category, category_data in prefs_data.items():
        if not isinstance(category_data, dict):
            continue
        for dimension, entities in category_data.items():
            if not isinstance(entities, list):
                continue
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                attr = {
                    "attribute": entity.get("entity", ""),
                    "dimension": dimension,
                    "sentiment": entity.get("sentiment", "neutral"),
                    "original_text": entity.get("original_text", ""),
                    "improvement_wish": entity.get("improvement_wish", ""),
                    "source": "direct_preference",
                    "validation_passed": True,
                    "category": category
                }
                attributes.append(attr)
    
    return attributes


def convert_stage1_to_stage3_format(preferences_results: list, metadata: dict) -> list:
    """
    将 Stage 1 的偏好提取结果转换为 Stage 3 需要的格式
    
    Args:
        preferences_results: Stage 1 的 results 列表
        metadata: 产品元数据字典
    
    Returns:
        转换后的结果列表，格式与原 Stage 2 输出相同
    """
    converted_results = []
    
    for product in preferences_results:
        asin = product.get('asin', '')
        title = product.get('product_title', '')
        
        # 从 metadata 获取 category（取最细粒度的类别）
        category = None
        if asin in metadata:
            meta = metadata[asin]
            cat_list = meta.get('category', [])
            if cat_list and isinstance(cat_list, list) and len(cat_list) > 0:
                category = cat_list[-1]  # 最后一级是最细粒度的类别
        
        # 转换 target user preferences
        target_prefs = product.get('target_user_preferences', {})
        target_attributes = convert_preferences_to_attributes(target_prefs)
        
        # 转换 other users preferences
        other_prefs = product.get('other_users_preferences', {})
        public_attributes = convert_preferences_to_attributes(other_prefs)
        
        result = {
            'asin': asin,
            'product_title': title,
            'category': category,
            'target_attributes': {
                'selected_attributes': target_attributes,
                'validation_passed': True,
                'summary_reasoning': 'Converted from Stage 1 preferences',
                'stage': 'stage1_direct'
            },
            'public_attributes': {
                'selected_attributes': public_attributes,
                'validation_passed': True,
                'summary_reasoning': 'Converted from Stage 1 preferences',
                'stage': 'stage1_direct'
            },
            'status': 'success'
        }
        
        converted_results.append(result)
    
    return converted_results


def load_preferences_results(preferences_file: str, metadata_file: str):
    """
    加载 Stage 1 偏好提取结果并转换为 Stage 3 需要的格式
    
    Args:
        preferences_file: preferences_{USER_ID}.json 文件路径
        metadata_file: 产品元数据文件路径
    
    Returns:
        转换后的结果列表，或 None 如果文件不存在
    """
    if not os.path.exists(preferences_file):
        log_with_timestamp(f"Error: Preferences file not found: {preferences_file}")
        return None
    
    # 加载 Stage 1 输出
    with open(preferences_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preferences_results = data.get('results', [])
    
    # 收集所有需要的 ASIN
    needed_asins = {p.get('asin') for p in preferences_results if p.get('asin')}
    
    # 加载元数据
    log_with_timestamp(f"  Loading metadata for {len(needed_asins)} products...")
    metadata = load_product_metadata(metadata_file, needed_asins)
    log_with_timestamp(f"  Loaded metadata for {len(metadata)} products")
    
    # 转换格式
    log_with_timestamp(f"  Converting Stage 1 format to Stage 3 format...")
    converted_results = convert_stage1_to_stage3_format(preferences_results, metadata)
    log_with_timestamp(f"  Converted {len(converted_results)} products")
    
    return converted_results


def filter_user_data(results, user_id, min_attrs=5, min_other_users=3):
    """
    过滤用户数据：只保留符合条件的商品
    
    筛选条件:
    - target_attributes >= min_attrs
    """
    query_items = []
    filtered_count = 0

    for item in results:
        selected_attrs = []
        target_attrs = item.get('target_attributes', {})
        if target_attrs:
            selected_attrs = target_attrs.get('selected_attributes', [])
        
        if not selected_attrs:
            selected_attrs = item.get('selected_attributes', [])

        meets_attr_req = len(selected_attrs) >= min_attrs

        if meets_attr_req:
            query_items.append(item)
        else:
            filtered_count += 1
            item['_filter_reason'] = f"属性数不足 ({len(selected_attrs)} < {min_attrs})"

    log_with_timestamp(f"  总商品数: {len(results)}, 过滤掉: {filtered_count}个, 保留: {len(query_items)}个")

    filter_stats = {
        "total_products": len(results),
        "filtered_count": filtered_count,
        "retained_count": len(query_items)
    }

    return query_items, filter_stats


def deduplicate_item_attributes(item):
    """对单个商品的属性进行去重"""
    category = item.get('category', '')
    new_item = item.copy()
    
    target_attrs_data = item.get('target_attributes', {})
    if isinstance(target_attrs_data, dict):
        target_attrs = target_attrs_data.get('selected_attributes', [])
        deduplicated_target = filter_duplicate_attributes(
            target_attrs, 
            category=category,
            threshold=0.80
        )
        
        new_target = target_attrs_data.copy()
        new_target['selected_attributes'] = deduplicated_target
        new_item['target_attributes'] = new_target
    
    return new_item


def main():
    parser = argparse.ArgumentParser(description="Filter user data based on attribute count (reads Stage 1 preferences)")
    parser.add_argument("--preferences-dir", required=True, help="Directory containing preferences_USERID.json files from Stage 1")
    parser.add_argument("--metadata-file", required=True, help="Product metadata file for category extraction")
    parser.add_argument("--output-dir", required=True, help="Output directory for split data")
    parser.add_argument("--user-id", help="Single user ID to process")
    parser.add_argument("--min-attrs", type=int, default=5, help="Minimum attributes for query items (default: 5)")
    parser.add_argument("--min-other-users", type=int, default=3, help="Minimum public attributes from other users (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.user_id:
        user_files = [f"preferences_{args.user_id}.json"]
    else:
        user_files = [f for f in os.listdir(args.preferences_dir) if f.startswith("preferences_") and f.endswith(".json")]

    summary = []

    for f_name in user_files:
        user_id = f_name.replace("preferences_", "").replace(".json", "")
        preferences_file = os.path.join(args.preferences_dir, f_name)

        log_with_timestamp(f"Processing user {user_id}...")
        results = load_preferences_results(preferences_file, args.metadata_file)
        if not results:
            log_with_timestamp(f"  Warning: No results found for user {user_id}")
            continue

        query_items, filter_stats = filter_user_data(
            results, user_id,
            min_attrs=args.min_attrs, min_other_users=args.min_other_users
        )

        # 清理 public_attributes
        for item in query_items:
            if 'public_attributes' in item:
                del item['public_attributes']
            if '_filter_reason' in item:
                del item['_filter_reason']

        # 过滤无属性的物品
        original_query_count = len(query_items)
        query_items = [
            item for item in query_items
            if item.get('target_attributes', {}).get('selected_attributes', [])
        ]
        filtered_count = original_query_count - len(query_items)
        if filtered_count > 0:
            log_with_timestamp(f"  过滤掉 {filtered_count} 个无属性的物品")

        # 计算查询集维度统计
        query_dimensions_by_category = {}
        for item in query_items:
            cat = item.get('category', '')
            if cat not in query_dimensions_by_category:
                query_dimensions_by_category[cat] = set()
            attrs = item.get('target_attributes', {}).get('selected_attributes', [])
            for attr in attrs:
                dim = attr.get('dimension', '')
                if dim:
                    query_dimensions_by_category[cat].add(dim)

        # 创建用户目录结构
        user_dir = os.path.join(args.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # 保存输出文件
        output_file = os.path.join(user_dir, "query.json")
        data_to_save = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "filter_strategy": "min_attrs_filter",
            "min_attrs": args.min_attrs,
            "min_other_users": args.min_other_users,
            "total_products": filter_stats["total_products"],
            "filtered_count": filter_stats["filtered_count"],
            "retained_count": filter_stats["retained_count"],
            "query_count": len(query_items),
            "filter_ratio": filter_stats["retained_count"] / filter_stats["total_products"] if filter_stats["total_products"] > 0 else 0,
            "query_dimensions_by_category": {k: list(v) for k, v in query_dimensions_by_category.items()},
            "query_results": query_items
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        log_with_timestamp(f"  保留: {len(query_items)}/{filter_stats['total_products']} 商品")
        log_with_timestamp(f"  Output: {output_file}")
        
        summary.append({
            "user_id": user_id,
            "query_count": len(query_items)
        })

    with open(os.path.join(args.output_dir, "split_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
