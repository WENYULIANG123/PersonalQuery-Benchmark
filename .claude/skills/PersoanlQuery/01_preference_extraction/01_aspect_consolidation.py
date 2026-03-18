#!/usr/bin/env python3
"""
Stage 1.5 - Aspect Consolidation (基于论文模板2)

严格按照论文 Appendix A - Figure 5 的提示模板实现

Input: aspects_{USER_ID}.json from Stage 1
Output: consolidated_aspects_{USER_ID}.json with normalized aspects
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient


def log_with_timestamp(message: str):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_aspect_consolidation_prompt(
    aspects_to_consolidate: List[str],
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """
    生成论文模板2的提示词
    
    严格按照 Appendix A - Figure 5 的格式
    """
    
    base_prompt = """You are given a list of product aspects extracted from customer reviews. Some aspects are
very specific (low-level), while others are broader and more general (high-level).

Whenever possible, merge low-level aspects into more general high-level ones. If an aspect
is already high-level, leave it unchanged.

Respond with valid JSON where the keys are the original (low-level) aspects and the values
are their corresponding high-level forms.

"""
    
    if few_shot_examples is None:
        few_shot_examples = get_default_consolidation_examples()
    
    examples_section = "Below are examples of low-level aspects and their high-level counterparts:\n\n"
    
    for i, example in enumerate(few_shot_examples, 1):
        examples_section += f"Example {i}:\n"
        examples_section += f"Low-level aspects: {example['low_level']}\n"
        examples_section += f"High-level forms: {example['high_level']}\n"
        examples_section += "\n"
    
    aspects_section = f"Now, normalize the following list of aspects:\n{aspects_to_consolidate}\n\n"
    
    return base_prompt + examples_section + aspects_section


def get_default_consolidation_examples() -> List[Dict]:
    """获取默认的few-shot示例"""
    
    return [
        {
            "low_level": ["battery", "battery life", "battery duration", "battery lasted only 2 days"],
            "high_level": {
                "battery": "battery_life",
                "battery life": "battery_life",
                "battery duration": "battery_life",
                "battery lasted only 2 days": "battery_life"
            }
        },
        {
            "low_level": ["screen", "display", "monitor", "screen size", "display resolution"],
            "high_level": {
                "screen": "screen",
                "display": "screen",
                "monitor": "screen",
                "screen size": "screen_size",
                "display resolution": "screen_resolution"
            }
        },
        {
            "low_level": ["easy to use", "simple to use", "intuitive", "user friendly", "ease of use"],
            "high_level": {
                "easy to use": "ease_of_use",
                "simple to use": "ease_of_use",
                "intuitive": "ease_of_use",
                "user friendly": "ease_of_use",
                "ease of use": "ease_of_use"
            }
        },
        {
            "low_level": ["works with Cuttlebug", "compatible with Sizzix", "fits A2 cards", "compatible", "works with"],
            "high_level": {
                "works with Cuttlebug": "compatibility",
                "compatible with Sizzix": "compatibility",
                "fits A2 cards": "compatibility",
                "compatible": "compatibility",
                "works with": "compatibility"
            }
        }
    ]


def parse_consolidation_response(response: str) -> Optional[Dict[str, str]]:
    """
    解析LLM的方面合并响应
    
    期望格式：JSON对象，key是原始方面，value是规范化形式
    """
    
    try:
        import re
        
        # 尝试JSON格式
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        
        # 直接JSON对象
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    
    except Exception as e:
        log_with_timestamp(f"Error parsing consolidation response: {e}")
    
    return None


def consolidate_aspects_with_llm(
    aspects: List[Dict],
    product_title: str
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    使用LLM进行方面合并
    
    返回：(consolidation_mapping, raw_response)
    """
    
    client = LLMClient()
    
    # 收集所有唯一的方面
    unique_aspects = list(set(a.get('aspect', 'unknown') for a in aspects))
    
    if not unique_aspects:
        return {}, None
    
    prompt = get_aspect_consolidation_prompt(unique_aspects)
    
    consolidation_prompt = f"""{prompt}

**Product**: {product_title}

Output format (IMPORTANT - must be valid JSON object):
{{
  "original_aspect_1": "high_level_form_1",
  "original_aspect_2": "high_level_form_2",
  ...
}}

Output ONLY the JSON object, no other text."""
    
    try:
        response = client.call(consolidation_prompt, max_tokens=1024)
        mapping = parse_consolidation_response(response)
        
        if mapping:
            # 确保所有原始方面都在映射中
            for aspect in unique_aspects:
                if aspect not in mapping:
                    mapping[aspect] = aspect  # 默认不变
            
            return mapping, response
        else:
            log_with_timestamp("Failed to parse consolidation response")
            return {aspect: aspect for aspect in unique_aspects}, response
    
    except Exception as e:
        log_with_timestamp(f"Error in consolidation: {e}")
        # 回退：方面保持不变
        return {aspect: aspect for aspect in unique_aspects}, None


def consolidate_aspects_rule_based(aspects: List[Dict]) -> Dict[str, str]:
    """
    基于规则的方面合并（不需要LLM调用）
    
    用于测试或成本优化
    """
    
    consolidation_rules = {
        # Battery
        "battery": "battery_life",
        "battery life": "battery_life",
        "battery duration": "battery_life",
        "battery lasted": "battery_life",
        
        # Screen/Display
        "screen": "screen",
        "display": "screen",
        "monitor": "screen",
        
        # Ease of Use
        "easy to use": "ease_of_use",
        "simple to use": "ease_of_use",
        "intuitive": "ease_of_use",
        "user friendly": "ease_of_use",
        "simple": "ease_of_use",
        
        # Compatibility
        "works with": "compatibility",
        "compatible with": "compatibility",
        "compatible": "compatibility",
        "fits": "compatibility",
        
        # Quality
        "quality": "quality",
        "well made": "quality",
        "well-made": "quality",
        "durability": "durability",
        "durable": "durability",
        "broke": "durability",
        "broken": "durability",
        
        # Price
        "price": "price",
        "expensive": "price",
        "cheap": "price",
        "costly": "price",
        "pricey": "price",
        
        # Color/Appearance
        "color": "appearance",
        "appearance": "appearance",
        "design": "appearance",
        "style": "appearance",
        "beautiful": "appearance",
        "cute": "appearance",
        
        # Size
        "size": "size",
        "large": "size",
        "small": "size",
        "compact": "size",
        "dimensions": "size"
    }
    
    # 构建完整映射
    mapping = {}
    unique_aspects = list(set(a.get('aspect', 'unknown') for a in aspects))
    
    for aspect in unique_aspects:
        aspect_lower = aspect.lower()
        
        # 精确匹配
        if aspect_lower in consolidation_rules:
            mapping[aspect] = consolidation_rules[aspect_lower]
        # 部分匹配
        else:
            matched = False
            for rule_key, rule_value in consolidation_rules.items():
                if rule_key in aspect_lower:
                    mapping[aspect] = rule_value
                    matched = True
                    break
            
            if not matched:
                mapping[aspect] = aspect  # 保持原样
    
    return mapping


def apply_consolidation(
    aspects: List[Dict],
    mapping: Dict[str, str]
) -> List[Dict]:
    """
    应用合并映射到方面列表
    """
    
    consolidated = []
    
    for aspect in aspects:
        original_aspect = aspect.get('aspect', 'unknown')
        canonical_form = mapping.get(original_aspect, original_aspect)
        
        # 保存原始信息，添加规范化形式
        aspect['aspect_canonical'] = canonical_form
        aspect['aspect_original'] = original_aspect
        
        consolidated.append(aspect)
    
    return consolidated


def process_product_consolidation(
    product_data: Dict,
    use_llm: bool = True
) -> Dict:
    """处理单个产品的方面合并"""
    
    asin = product_data['asin']
    title = product_data['product_title']
    target_aspects = product_data.get('target_aspects', [])
    
    result = {
        'asin': asin,
        'product_title': title,
        'consolidated_aspects': [],
        'consolidation_mapping': {},
        'consolidation_method': 'llm' if use_llm else 'rule_based',
        'metadata': {}
    }
    
    if not target_aspects:
        result['metadata']['note'] = 'No aspects to consolidate'
        return result
    
    # 获取合并映射
    if use_llm:
        mapping, _ = consolidate_aspects_with_llm(target_aspects, title)
    else:
        mapping = consolidate_aspects_rule_based(target_aspects)
    
    result['consolidation_mapping'] = mapping
    
    # 应用合并
    consolidated = apply_consolidation(target_aspects, mapping)
    result['consolidated_aspects'] = consolidated
    
    # 统计信息
    original_unique = len(set(a.get('aspect') for a in target_aspects))
    canonical_unique = len(set(a.get('aspect_canonical') for a in consolidated))
    
    result['metadata'] = {
        'original_unique_count': original_unique,
        'canonical_unique_count': canonical_unique,
        'consolidation_rate': (original_unique - canonical_unique) / original_unique if original_unique > 0 else 0,
        'total_aspects': len(target_aspects),
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 1.5: Aspect Consolidation (Paper Template 2)")
    parser.add_argument("--input-file", required=True, help="Input file from Stage 1 (aspects)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--use-llm", action="store_true", default=False, help="Use LLM for consolidation (default: rule-based)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1.5: Aspect Consolidation (Paper Template 2 - Appendix A Figure 5)")
    log_with_timestamp(f"Method: {'LLM-based' if args.use_llm else 'Rule-based'}")
    log_with_timestamp("=" * 80)
    
    # 加载数据
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    user_id = data['user_id']
    products = data['results']
    
    log_with_timestamp(f"User: {user_id}")
    log_with_timestamp(f"Products: {len(products)}")
    log_with_timestamp("")
    
    # 处理产品
    results = []
    consolidation_stats = defaultdict(int)
    
    for idx, product in enumerate(products, 1):
        result = process_product_consolidation(product, use_llm=args.use_llm)
        results.append(result)
        
        # 收集统计
        consolidation_stats['total_processed'] += 1
        consolidation_stats['total_original_aspects'] += result['metadata'].get('original_unique_count', 0)
        consolidation_stats['total_canonical_aspects'] += result['metadata'].get('canonical_unique_count', 0)
        
        # 定期输出
        if idx % 10 == 0 or idx == len(products):
            log_with_timestamp(f"Progress: {idx}/{len(products)} products processed")
            
            consolidation_rate = (
                (consolidation_stats['total_original_aspects'] - consolidation_stats['total_canonical_aspects']) /
                consolidation_stats['total_original_aspects']
                if consolidation_stats['total_original_aspects'] > 0 else 0
            )
            
            log_with_timestamp(f"  Original unique aspects: {consolidation_stats['total_original_aspects']}")
            log_with_timestamp(f"  Canonical aspects: {consolidation_stats['total_canonical_aspects']}")
            log_with_timestamp(f"  Consolidation rate: {consolidation_rate:.1%}")
            log_with_timestamp("")
    
    # 保存结果
    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'template_version': 'Appendix_A_Figure_5',
        'consolidation_method': 'llm' if args.use_llm else 'rule_based',
        'total_products': len(results),
        'consolidation_statistics': dict(consolidation_stats),
        'results': results
    }
    
    output_file = os.path.join(args.output_dir, f'consolidated_aspects_{user_id}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"Saved to {output_file}")
    log_with_timestamp("Stage 1.5: Aspect Consolidation Complete!")


if __name__ == "__main__":
    main()
