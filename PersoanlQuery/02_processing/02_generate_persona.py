#!/usr/bin/env python3
"""
Stage 3b: Generate Persona Data

从分割后的数据中提取个人画像，进行深度去重，并输出按类目的画像文件。

Input: result/personal_query/02_processing/{user_id}/query.json
Output: result/personal_query/02_processing/{user_id}/persona/{category}.json
"""

import json
import os
import argparse
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    st_util = None

_semantic_model = None


def get_semantic_model():
    global _semantic_model
    if _semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def call_llm_deduplicate(prompt, max_retries=3):
    """调用LLM进行语义去重（预留接口，当前跳过）"""
    import sys
    skills_path = '/fs04/ar57/wenyu/.claude/skills'
    if skills_path not in sys.path:
        sys.path.insert(0, skills_path)
    
    try:
        from llm_client import LLMClient
        client = LLMClient()
        
        for attempt in range(max_retries):
                response = client.call(prompt, max_tokens=1024)
                if response and len(response) > 10:
                    return response
                import time
                time.sleep(1)
    except Exception as e:
        log_with_timestamp(f"LLM error: {e}")
    return None


def pre_filter_duplicates(attrs):
    """规则预过滤: 基于字符串相似度和包含关系去重"""
    if len(attrs) <= 1:
        return attrs
    
    groups = defaultdict(list)
    for attr in attrs:
        key = (
            attr.get('category', ''),
            attr.get('dimension', ''),
            attr.get('sentiment', 'neutral')
        )
        groups[key].append(attr)
    
    for key, group_attrs in groups.items():
        if len(group_attrs) <= 1:
            continue
        
        to_remove = set()
        
        # 品牌名去重
        for i in range(len(group_attrs)):
            if i in to_remove:
                continue
            attr_i = group_attrs[i].get('attribute', '').lower().strip()
            
            brand_name_i = None
            if attr_i.endswith(' brand'):
                brand_name_i = attr_i[:-6].strip()
            
            for j in range(i + 1, len(group_attrs)):
                if j in to_remove:
                    continue
                attr_j = group_attrs[j].get('attribute', '').lower().strip()
                
                if brand_name_i:
                    for keyword in ['prefer', 'prefers', 'like', 'likes', 'good', 'best']:
                        if attr_j.startswith(keyword) and brand_name_i in attr_j:
                            if len(attr_i) < len(attr_j):
                                to_remove.add(i)
                            else:
                                to_remove.add(j)
                            break
                
                brand_name_j = None
                if attr_j.endswith(' brand'):
                    brand_name_j = attr_j[:-6].strip()
                if brand_name_j:
                    for keyword in ['prefer', 'prefers', 'like', 'likes', 'good', 'best']:
                        if attr_i.startswith(keyword) and brand_name_j in attr_i:
                            if len(attr_i) < len(attr_j):
                                to_remove.add(i)
                            else:
                                to_remove.add(j)
                            break
        
        # 字符串相似度去重
        for i in range(len(group_attrs)):
            if i in to_remove:
                continue
            attr_i = group_attrs[i].get('attribute', '').lower()
            
            for j in range(i + 1, len(group_attrs)):
                if j in to_remove:
                    continue
                attr_j = group_attrs[j].get('attribute', '').lower()
                
                ratio = SequenceMatcher(None, attr_i, attr_j).ratio()
                if ratio >= 0.7:
                    if len(attr_i) < len(attr_j):
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        
        # 包含关系去重
        for i in range(len(group_attrs)):
            if i in to_remove:
                continue
            attr_i = group_attrs[i].get('attribute', '').lower().strip()
            
            for j in range(i + 1, len(group_attrs)):
                if j in to_remove:
                    continue
                attr_j = group_attrs[j].get('attribute', '').lower().strip()
                
                if attr_i in attr_j or attr_j in attr_i:
                    if len(attr_i) < len(attr_j):
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        
        if to_remove:
            for i in range(len(group_attrs) - 1, -1, -1):
                if i in to_remove:
                    group_attrs.pop(i)
    
    return attrs


def semantic_dedup_by_dimension(attrs, similarity_threshold=0.50):
    """语义去重: 使用sentence-transformers进行维度级别的去重"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return attrs
    
    model = get_semantic_model()
    if model is None:
        return attrs
    
    by_dimension = defaultdict(list)
    for attr in attrs:
        dimension = attr.get('dimension', 'unknown')
        by_dimension[dimension].append(attr)
    
    result = []
    
    for dimension, dim_attrs in by_dimension.items():
        if len(dim_attrs) < 2:
            result.extend(dim_attrs)
            continue
        
        log_with_timestamp(f"    [{dimension}] Semantic dedup: {len(dim_attrs)} attrs...")
        
        attr_strings = [a.get('attribute', '') for a in dim_attrs]
        
        embeddings = model.encode(
            attr_strings,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        if st_util is None:
            result.extend(dim_attrs)
            continue
        cos_scores = st_util.cos_sim(embeddings, embeddings)
        
        merged = set()
        to_remove = set()
        
        for i in range(len(dim_attrs)):
            if i in merged or i in to_remove:
                continue
            for j in range(i + 1, len(dim_attrs)):
                if j in merged or j in to_remove:
                    continue
                score = cos_scores[i][j].item()
                if score >= similarity_threshold:
                    to_remove.add(j)
                    merged.add(i)
        
        final_attrs = []
        for i, attr in enumerate(dim_attrs):
            if i not in to_remove:
                final_attrs.append(attr)
        
        log_with_timestamp(f"    [{dimension}] Reduced to {len(final_attrs)} unique attributes")
        result.extend(final_attrs)
    
    return result


def llm_deduplicate_category(category_attrs, category_name):
    """LLM语义去重（当前跳过，保留接口）"""
    # category_attrs = pre_filter_duplicates(category_attrs)
    
    groups = defaultdict(list)
    for attr in category_attrs:
        key = (
            attr.get('category', ''),
            attr.get('dimension', ''),
            attr.get('sentiment', 'neutral')
        )
        groups[key].append(attr)
    
    attrs_to_check = []
    for key, attrs in groups.items():
        if len(attrs) > 1:
            attrs_to_check.append((key, attrs))
    
    if not attrs_to_check:
        return category_attrs
    
    log_with_timestamp(f"    [{category_name}] Pre-processing done, skipping LLM dedup")
    return category_attrs


def load_query_data(query_file):
    """加载query文件"""
    if not os.path.exists(query_file):
        return None
    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_persona_by_category(persona_items):
    """按类别合并画像属性"""
    if not persona_items:
        return persona_items
    
    category_to_items = defaultdict(list)
    category_to_asins = defaultdict(list)
    for item in persona_items:
        cat = item.get('category', 'unknown')
        category_to_items[cat].append(item)
        asin = item.get('asin')
        if asin:
            category_to_asins[cat].append(asin)
    
    merged_results = []
    
    for cat, items in category_to_items.items():
        all_attrs = []
        for item in items:
            attrs = item.get('target_attributes', {}).get('selected_attributes', [])
            all_attrs.extend(attrs)
        
        merged_item = {
            "category": cat,
            "product_count": len(items),
            "asins": category_to_asins[cat],
            "target_attributes": {
                "selected_attributes": all_attrs
            }
        }
        merged_results.append(merged_item)
    
    log_with_timestamp(f"  类别合并: {len(persona_items)} 个商品 -> {len(merged_results)} 个类别")
    
    return merged_results


def deduplicate_persona_attributes(persona_items):
    """对画像属性进行多轮去重"""
    if not persona_items:
        return persona_items
    
    items_by_category = {}
    for item in persona_items:
        cat = item.get('category', 'unknown')
        if cat not in items_by_category:
            items_by_category[cat] = []
        items_by_category[cat].append(item)
    
    deduped_items = []
    
    for cat, items in items_by_category.items():
        all_attrs = []
        for item in items:
            target_attrs = item.get('target_attributes', {})
            if isinstance(target_attrs, dict):
                selected = target_attrs.get('selected_attributes', [])
                all_attrs.extend(selected)
        
        log_with_timestamp(f"    [{cat}] 合并后属性数: {len(all_attrs)}")
        
        # all_attrs = pre_filter_duplicates(all_attrs)
        # log_with_timestamp(f"    [{cat}] 前处理后属性数: {len(all_attrs)}")
        
        # all_attrs = semantic_dedup_by_dimension(all_attrs, similarity_threshold=0.50)
        # log_with_timestamp(f"    [{cat}] 语义去重后属性数: {len(all_attrs)}")
        
        # seen_attrs = set()
        final_attrs = all_attrs
        # for attr in all_attrs:
        #     key = (
        #         attr.get('category', ''),
        #         attr.get('dimension', ''),
        #         attr.get('attribute', ''),
        #         attr.get('sentiment', 'neutral')
        #     )
        #     if key not in seen_attrs:
        #         seen_attrs.add(key)
        #         final_attrs.append(attr)
        
        merged_item = items[0] if items else {}
        
        new_item = {
            "category": cat,
            "product_count": merged_item.get('product_count', 1),
            "asins": merged_item.get('asins', []),
            "target_attributes": {
                "selected_attributes": final_attrs
            }
        }
        deduped_items.append(new_item)
    
    return deduped_items


def save_persona_files(persona_items, user_id, output_dir):
    """保存按类目的画像文件"""
    # 创建用户目录结构
    user_dir = os.path.join(output_dir, user_id)
    persona_dir = os.path.join(user_dir, "persona")
    os.makedirs(persona_dir, exist_ok=True)
    
    for persona_item in persona_items:
        cat = persona_item.get('category', 'unknown')
        safe_cat_name = cat.replace(' ', '_').replace(',', '').replace('&', 'and')
        
        attrs = persona_item.get('target_attributes', {}).get('selected_attributes', [])
        
        dimensions_summary = defaultdict(int)
        for attr in attrs:
            dim = attr.get('dimension', 'unknown')
            dimensions_summary[dim] += 1
        
        attrs_by_dimension = defaultdict(list)
        for attr in attrs:
            dim = attr.get('dimension', 'unknown')
            attrs_by_dimension[dim].append(attr)
        
        category_output = {
            "user_id": user_id,
            "category": cat,
            "product_count": persona_item.get('product_count', 0),
            "asins": persona_item.get('asins', []),
            "total_attributes": len(attrs),
            "dimensions_summary": dict(sorted(dimensions_summary.items(), key=lambda x: -x[1])),
            "attributes": attrs,
            "attributes_by_dimension": {dim: attrs_list for dim, attrs_list in sorted(attrs_by_dimension.items())}
        }
        
        category_file = os.path.join(persona_dir, f"{safe_cat_name}.json")
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(category_output, f, indent=2, ensure_ascii=False)
        
        log_with_timestamp(f"    输出类目文件: {user_id}/persona/{safe_cat_name}.json ({len(attrs)} 属性, {len(dimensions_summary)} 维度)")


def main():
    parser = argparse.ArgumentParser(description="Generate persona data from split results")
    parser.add_argument("--query-file", required=True, help="Query file from Stage 3a (query_USERID.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for persona files")
    parser.add_argument("--user-id", help="User ID (will be extracted from query file if not provided)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载query数据
    log_with_timestamp(f"Loading query file: {args.query_file}")
    query_data = load_query_data(args.query_file)
    if not query_data:
        log_with_timestamp(f"Error: Could not load query file")
        return

    user_id = args.user_id or query_data.get('user_id', 'unknown')
    persona_items = query_data.get('persona_results', [])
    
    log_with_timestamp(f"User ID: {user_id}")
    log_with_timestamp(f"Persona items: {len(persona_items)}")

    if not persona_items:
        log_with_timestamp("No persona items found, nothing to process")
        return

    # Step 1: 按类别合并画像
    log_with_timestamp("Step 1: 按类别合并画像集...")
    persona_items = merge_persona_by_category(persona_items)

    # log_with_timestamp("Step 2: 对合并后的画像集进行规则去重 + 语义去重...")
    # persona_items = deduplicate_persona_attributes(persona_items)

    # log_with_timestamp("Step 3: 对合并后的画像集进行 LLM 语义去重...")
    # for i, item in enumerate(persona_items):
    #     cat = item.get('category', '')
    #     attrs = item.get('target_attributes', {}).get('selected_attributes', [])
    #     if attrs:
    #         log_with_timestamp(f"    处理类别 {i+1}/{len(persona_items)}: {cat}")
    #         deduped_attrs = llm_deduplicate_category(attrs, cat)
    #         item['target_attributes']['selected_attributes'] = deduped_attrs

    # 保存画像文件
    log_with_timestamp("Saving persona files...")
    save_persona_files(persona_items, user_id, args.output_dir)

    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
