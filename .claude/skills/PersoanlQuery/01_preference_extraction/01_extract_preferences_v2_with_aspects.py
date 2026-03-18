#!/usr/bin/env python3
"""
Stage 1 升级版 v2: Preference Extraction + Aspect-Level Analysis
基于学术最佳实践（ABSA论文、Amazon NAACL 2022、ACL 2025）

改进点：
1. 添加置信度评分（Confidence Scoring）
2. 检测隐式方面（Implicit Aspect Detection）
3. 同时输出维度级别（21维度）和方面级别的数据
4. 质量检查和数据验证

Input: reviews_{USER_ID}.json from Stage 0
Output: preferences_{USER_ID}_v2.json with dimensions + aspects + confidence scores
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient

# ============================================================================
# 工具函数
# ============================================================================

def log_with_timestamp(message: str):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_response(response: str, asin: str = "UNKNOWN") -> Optional[Dict]:
    if not response:
        log_with_timestamp(f"[{asin}] ❌ LLM 返回空响应")
        return None
    
    log_with_timestamp(f"[{asin}] 📝 解析 LLM 响应 (长度: {len(response)} 字符)")
    
    try:
        import re
        if "```json" in response:
            log_with_timestamp(f"[{asin}] 🔍 检测到 ```json 块")
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
                log_with_timestamp(f"[{asin}] ✅ ```json 块解析成功")
                return parsed
            else:
                log_with_timestamp(f"[{asin}] ⚠️  检测到 ```json 但正则失败")
        
        elif "```" in response:
            log_with_timestamp(f"[{asin}] 🔍 检测到 ``` 块")
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
                log_with_timestamp(f"[{asin}] ✅ ``` 块解析成功")
                return parsed
            else:
                log_with_timestamp(f"[{asin}] ⚠️  检测到 ``` 但正则失败")
        
        else:
            log_with_timestamp(f"[{asin}] 🔍 尝试直接正则提取 JSON")
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                log_with_timestamp(f"[{asin}] ✅ 正则提取解析成功")
                return parsed
            else:
                log_with_timestamp(f"[{asin}] ⚠️  正则提取失败，未找到 {{}} 模式")
    
    except json.JSONDecodeError as e:
        log_with_timestamp(f"[{asin}] ❌ JSON 解析错误: {str(e)}")
        log_with_timestamp(f"[{asin}] ❌ 错误位置: line {e.lineno}, col {e.colno}")
        log_with_timestamp(f"[{asin}] ❌ 问题响应 (前 500 字): {response[:500]}")
    
    except Exception as e:
        log_with_timestamp(f"[{asin}] ❌ 异常: {type(e).__name__}: {str(e)}")
    
    log_with_timestamp(f"[{asin}] ❌ 解析失败! 完整响应 (前 800 字):")
    log_with_timestamp(f"[{asin}] {response[:800]}")
    return None


# ============================================================================
# Phase 1 改进 1: 维度提取提示（保持原有）
# ============================================================================

def get_fixed_dimension_prompt():
    """返回固定的21维度schema提示"""
    
    dimensions_schema = """
## Fixed Dimension Schema (21 Dimensions, 7 Categories)

You MUST extract preferences into these FIXED dimensions ONLY. Do NOT create new dimensions.

### CRITICAL DIMENSION BOUNDARIES - Read Carefully

**Product_Category vs Material_Composition:**
- Product_Category: Product type/name (e.g., "glitter glue", "embossing folder", "die cut", "scissors")
- Material_Composition: Raw material/ingredient (e.g., "plastic", "metal", "cotton", "leather", "silver-colored")

**Ease_of_Use vs Compatibility:**
- Ease_of_Use: How easy/convenient to use (e.g., "easy to use", "simple to assemble", "intuitive")
- Compatibility: What systems/devices it works with (e.g., "works with Cuttlebug", "compatible with Sizzix")

**Functionality vs Usage_Scenario:**
- Functionality: What the product does/features (e.g., "cuts paper", "creates embossed effect")
- Usage_Scenario: Where/how the user uses it (e.g., "for greeting cards", "at home", "for scrapbooking")

### All 21 Dimensions:

**Product_Attributes:**
1. Product_Category - Product type/name
2. Functionality - Product features/capabilities
3. Material_Composition - Raw material/ingredient

**Quality_Attributes:**
4. Quality_Craftsmanship - Quality/workmanship
5. Performance - Performance effectiveness
6. Safety - Safety requirements

**Appearance_Design:**
7. Appearance_Color - Visual appearance
8. Size_Dimensions - Size fit
9. Style_Design - Style preference

**User_Experience:**
10. Comfort - Comfort level
11. Ease_of_Use - Usability
12. Portability - Portability

**Usage_Scenarios:**
13. Target_User - Intended user
14. Usage_Scenario - Where/how to use
15. Special_Purpose - Special use case

**Price_Value:**
16. Price - Price related
17. Value - Value for money
18. Packaging_Quantity - Packaging specs

**Special_Requirements:**
19. Compatibility - Device/system compatibility
20. Special_User_Needs - Special user requirements
21. Brand_Preference - Brand preference
"""
    
    output_format = """
## Output Format

You MUST output JSON with this exact structure:

{
  "Product_Attributes": {
    "Product_Category": [{"entity": "...", "sentiment": "positive/negative/neutral", "original_text": "...", "confidence": 0.95}],
    "Functionality": [...],
    "Material_Composition": [...]
  },
  "Quality_Attributes": {...},
  "Appearance_Design": {...},
  "User_Experience": {...},
  "Usage_Scenarios": {...},
  "Price_Value": {...},
  "Special_Requirements": {...},
  
  "extraction_metadata": {
    "overall_confidence": 0.92,
    "entity_count": 15,
    "dimensions_found": 8
  }
}

IMPORTANT:
- Use EXACT dimension names as shown above (use underscore "_")
- If no information for a dimension, use empty array []
- sentiment must be one of: positive, negative, neutral
- confidence for each entity should be 0-1 (e.g., 0.95 for high confidence)
- Extract specific evidence from the review text
"""
    
    return dimensions_schema, output_format


# ============================================================================
# Phase 1 改进 2: 方面级别提示（新增）
# ============================================================================

def get_aspect_extraction_prompt():
    """返回方面级别的提取提示"""
    
    aspect_prompt = """
## Aspect-Level Extraction (补充维度提取)

除了上述21维度分类，还应识别评论中的核心"方面"(aspects)。

方面定义：评论中显式或隐式提到的产品特性、属性或用户评价。

对每个主要方面，输出：
{
  "aspect": "方面表述（如 'glitter glue', 'fast drying'）",
  "aspect_sentiment": "POSITIVE / MIXED / NEGATIVE",
  "confidence": 0.95,
  "is_implicit": false,
  "evidence_spans": ["支持文本1", "支持文本2"],
  "dimension_mapping": "对应的21维度（如 Product_Category）"
}

### 显式方面示例：
- "Great glitter glue" → aspect: "glitter glue", is_implicit: false
- "Fast drying time" → aspect: "drying time", is_implicit: false

### 隐式方面示例（需要推理）：
- "It's too expensive" → aspect: "price", is_implicit: true, reasoning: 隐喻成本问题
- "Broke after one week" → aspect: "durability", is_implicit: true, reasoning: 隐喻品质问题
- "Easy to use" → aspect: "ease_of_use", is_implicit: false (可能是显式的)

### 隐式方面的关键词模式：

**Price-related (隐式价格):**
- expensive, costly, pricey, broke the bank, overpriced
- deal, bargain, value, worth, bang for buck

**Quality-related (隐式质量):**
- broke, stopped working, fell apart, defective
- lasted only X days/months, failed quickly

**Durability-related (隐式耐用性):**
- worn out, faded, deteriorated, degraded
- holding up, still going strong

**Functionality-related (隐式功能):**
- doesn't work, failed to, unable to, can't
- performs well, does the job, gets the job done

限制：最多提取5个主要方面。优先提取显式方面。

## Output Format

{
  "aspects": [
    {
      "aspect": "glitter glue",
      "aspect_sentiment": "POSITIVE",
      "confidence": 0.95,
      "is_implicit": false,
      "evidence_spans": ["Great glitter glue", "works beautifully"],
      "dimension_mapping": "Product_Category"
    },
    ...
  ]
}

IMPORTANT:
- confidence 范围: 0-1
- is_implicit: 是否为隐式方面
- 最多返回5个方面
"""
    
    return aspect_prompt


# ============================================================================
# Phase 1 改进 3: 隐式方面检测（新增）
# ============================================================================

def detect_implicit_aspects(review_text: str) -> List[Dict]:
    """
    基于规则和LLM联合检测隐式方面
    
    隐式方面：评论中没有直接提到但可推理出来的方面
    例如："It's so expensive" → implicit price aspect
    """
    
    implicit_aspects = []
    
    # 基于规则的隐式方面检测模式
    implicit_patterns = {
        "Price": {
            "keywords": ["expensive", "costly", "pricey", "broke the bank", "overpriced", "cheap", "affordable"],
            "dimension": "Price",
            "examples": ["太贵了", "很便宜", "不值这个价"]
        },
        "Durability": {
            "keywords": ["broke", "stopped working", "fell apart", "lasted only", "broke after", "failed", "defective"],
            "dimension": "Quality_Craftsmanship",
            "examples": ["一周后坏了", "很快就坏了", "耐用性差"]
        },
        "Value": {
            "keywords": ["deal", "bargain", "value", "worth", "worth the money", "bang for buck"],
            "dimension": "Value",
            "examples": ["物有所值", "超值", "不划算"]
        },
        "Functionality": {
            "keywords": ["doesn't work", "failed to", "unable to", "can't", "wouldn't", "broken"],
            "dimension": "Functionality",
            "examples": ["不能用", "功能不工作", "坏掉了"]
        }
    }
    
    review_lower = review_text.lower()
    
    for aspect_name, pattern_info in implicit_patterns.items():
        for keyword in pattern_info["keywords"]:
            if keyword.lower() in review_lower:
                # 找到隐式方面的证据
                match_position = review_lower.find(keyword.lower())
                # 提取周围的上下文（前后20个单词）
                words = review_lower.split()
                keyword_word_idx = len(review_lower[:match_position].split())
                context_start = max(0, keyword_word_idx - 5)
                context_end = min(len(words), keyword_word_idx + 5)
                evidence = " ".join(words[context_start:context_end])
                
                implicit_aspects.append({
                    "aspect": aspect_name.lower(),
                    "aspect_sentiment": "NEGATIVE" if "broke" in keyword or "fail" in keyword else "MIXED",
                    "confidence": 0.6,  # 隐式方面置信度通常较低
                    "is_implicit": True,
                    "evidence_spans": [evidence],
                    "dimension_mapping": pattern_info["dimension"],
                    "detection_method": "rule-based",
                    "keyword_matched": keyword
                })
                break  # 每个aspect只添加一次
    
    return implicit_aspects


# ============================================================================
# Phase 1 核心改进：统一的提取函数
# ============================================================================

def extract_preferences_from_review_v2(review, product_title: str, user_type: str, asin: str = "UNKNOWN") -> Dict:
    client = LLMClient()
    
    log_with_timestamp(f"[{asin}] 🔄 开始提取偏好 (user_type={user_type})")
    
    if isinstance(review, str):
        reviewer_id = ''
        review_text = review
        rating = 0
        log_with_timestamp(f"[{asin}] 📝 评论来自字符串，长度: {len(review_text)} 字符")
    else:
        reviewer_id = review.get('reviewerID', '')
        review_text = review.get('reviewText', '')
        rating = review.get('overall', 0)
        log_with_timestamp(f"[{asin}] 📝 评论来自字典，长度: {len(review_text)} 字符，评分: {rating}")
    
    if not review_text or len(review_text.strip()) < 10:
        log_with_timestamp(f"[{asin}] ⚠️  评论过短或为空: {len(review_text)} 字符")
        return {'dimensions': {}, 'aspects': []}
    
    dimensions_schema, output_format = get_fixed_dimension_prompt()
    aspect_prompt = get_aspect_extraction_prompt()
    
    combined_prompt = f"""Extract user preferences from this product review using BOTH dimension schema AND aspect extraction.

**Product**: {product_title}
**Rating**: {rating}/5
**Review**: {review_text}

{dimensions_schema}
{output_format}

{aspect_prompt}

Output format: You MUST output JSON with BOTH dimensions and aspects sections.
{{
  "dimensions": {{ ... }},  # 21维度提取结果
  "aspects": [ ... ]         # 方面提取结果
}}

Extract now. Output ONLY valid JSON, no explanation."""
    
    try:
        log_with_timestamp(f"[{asin}] 🌐 调用 LLM API (prompt_len={len(combined_prompt)})")
        response = client.call(combined_prompt, max_tokens=2048)
        log_with_timestamp(f"[{asin}] ✅ LLM 响应收到 (len={len(response)})")
        result = parse_response(response, asin)
        
        if result:
            # 检查返回格式
            if "dimensions" not in result:
                result["dimensions"] = {}
            if "aspects" not in result:
                result["aspects"] = []
            
            # 添加用户元数据
            for category, category_data in result.get("dimensions", {}).items():
                if not isinstance(category_data, dict):
                    continue
                for dimension, entities in category_data.items():
                    if not isinstance(entities, list):
                        continue
                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        entity['reviewer_id'] = reviewer_id
                        entity['user_type'] = user_type
            
            # 为方面添加元数据
            for aspect in result.get("aspects", []):
                aspect['reviewer_id'] = reviewer_id
                aspect['user_type'] = user_type
            
            # 检测隐式方面（规则基础）
            implicit_detected = detect_implicit_aspects(review_text)
            
            # 合并隐式和显式方面（避免重复）
            explicit_aspects = {a.get('aspect'): a for a in result.get('aspects', [])}
            for implicit_aspect in implicit_detected:
                aspect_key = implicit_aspect.get('aspect')
                if aspect_key not in explicit_aspects:
                    result['aspects'].append(implicit_aspect)
            
            # 添加提取元数据
            result['metadata'] = {
                'extraction_version': '2.0',
                'reviewer_id': reviewer_id,
                'user_type': user_type,
                'review_length': len(review_text),
                'rating': rating,
                'explicit_aspects_count': len([a for a in result.get('aspects', []) if not a.get('is_implicit', False)]),
                'implicit_aspects_count': len([a for a in result.get('aspects', []) if a.get('is_implicit', False)]),
                'dimensions_extraction_attempted': True,
                'aspects_extraction_attempted': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        else:
            log_with_timestamp(f"[{asin}] ❌ LLM 响应解析失败")
            return {
                'dimensions': {},
                'aspects': [],
                'metadata': {
                    'error': 'Failed to parse LLM response',
                    'extraction_version': '2.0'
                }
            }
    
    except Exception as e:
        log_with_timestamp(f"[{asin}] ❌ 提取异常: {type(e).__name__}: {str(e)}")
        import traceback
        log_with_timestamp(f"[{asin}] 堆栈跟踪:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_with_timestamp(f"[{asin}] {line}")
        return {
            'dimensions': {},
            'aspects': [],
            'metadata': {
                'error': str(e),
                'extraction_version': '2.0'
            }
        }


# ============================================================================
# 质量检查和验证
# ============================================================================

def validate_extraction_quality(extraction_result: Dict) -> Dict:
    """
    验证提取结果的质量
    
    返回：
    {
      "is_valid": True/False,
      "quality_score": 0-1,
      "issues": [...],
      "warnings": [...]
    }
    """
    
    issues = []
    warnings = []
    
    # 检查维度数据
    dimensions = extraction_result.get("dimensions", {})
    aspect_count = sum(
        len(entities) 
        for category in dimensions.values() 
        for entities in (category.values() if isinstance(category, dict) else [])
    )
    
    if aspect_count == 0:
        warnings.append("No entities extracted from dimensions")
    
    # 检查方面数据
    aspects = extraction_result.get("aspects", [])
    if len(aspects) == 0:
        warnings.append("No aspects extracted")
    
    # 检查置信度
    low_confidence_count = sum(
        1 for aspect in aspects 
        if aspect.get('confidence', 1.0) < 0.5
    )
    if low_confidence_count > 0:
        warnings.append(f"{low_confidence_count} aspects have low confidence (<0.5)")
    
    # 检查sentiment值的有效性
    valid_sentiments = {'positive', 'negative', 'neutral', 'POSITIVE', 'MIXED', 'NEGATIVE'}
    for aspect in aspects:
        sentiment = aspect.get('aspect_sentiment', '')
        if sentiment and sentiment not in valid_sentiments:
            issues.append(f"Invalid sentiment value: {sentiment}")
    
    # 计算质量评分
    quality_score = 1.0
    if issues:
        quality_score -= 0.3 * len(issues)
    if warnings:
        quality_score -= 0.1 * len(warnings)
    quality_score = max(0, min(1, quality_score))
    
    return {
        "is_valid": len(issues) == 0,
        "quality_score": quality_score,
        "issues": issues,
        "warnings": warnings,
        "entity_count": aspect_count,
        "aspect_count": len(aspects)
    }


# ============================================================================
# 处理产品（并发）
# ============================================================================

def process_product(product_data: Dict) -> Dict:
    """处理单个产品"""
    
    asin = product_data['asin']
    title = product_data['product_title']
    
    # 支持旧新格式
    target_review = product_data.get('target_review')
    if not target_review:
        target_reviews = product_data.get('target_reviews', [])
        if target_reviews and len(target_reviews) > 0:
            target_review = target_reviews[0]
        else:
            target_review = None
    
    # 固定维度类别
    fixed_categories = [
        'Product_Attributes', 'Quality_Attributes', 'Appearance_Design',
        'User_Experience', 'Usage_Scenarios', 'Price_Value', 'Special_Requirements'
    ]
    
    result = {
        'asin': asin,
        'product_title': title,
        'target_user_id': product_data['target_user_id'],
        'target_user_preferences': {},
        'target_user_aspects': [],
        'quality_check': None,
        'other_users_preferences': {}
    }
    
    # 初始化维度结构
    for cat in fixed_categories:
        result['target_user_preferences'][cat] = {}
    
    log_with_timestamp(f"[{asin}] 🚀 开始处理产品")
    log_with_timestamp(f"[{asin}] 📌 ASIN: {asin}")
    log_with_timestamp(f"[{asin}] 📌 标题: {title[:60]}")
    log_with_timestamp(f"[{asin}] 📌 用户: {product_data['target_user_id']}")
    
    if not target_review:
        log_with_timestamp(f"[{asin}] ❌ 错误: 无目标评论数据")
        result['quality_check'] = {
            'is_valid': False,
            'quality_score': 0,
            'issues': ['no_target_review']
        }
        return result
    
    log_with_timestamp(f"[{asin}] 📝 目标评论长度: {len(target_review) if isinstance(target_review, str) else len(target_review.get('reviewText', ''))} 字符")
    
    extraction = extract_preferences_from_review_v2(target_review, title, 'target', asin)
    result['target_user_preferences'] = extraction.get('dimensions', {})
    result['target_user_aspects'] = extraction.get('aspects', [])
    
    log_with_timestamp(f"[{asin}] ✅ 提取完成: {len(result['target_user_aspects'])} 个方面")
    
    result['quality_check'] = validate_extraction_quality(extraction)
    
    # 计算统计信息
    def count_entities(prefs_dict):
        total = 0
        for category, category_data in prefs_dict.items():
            if isinstance(category_data, dict):
                for dimension, entities in category_data.items():
                    if isinstance(entities, list):
                        total += len(entities)
            elif isinstance(category_data, list):
                total += len(category_data)
        return total
    
    def count_categories(prefs_dict):
        count = 0
        for category, category_data in prefs_dict.items():
            if isinstance(category_data, dict):
                if any(isinstance(v, list) and len(v) > 0 for v in category_data.values()):
                    count += 1
            elif isinstance(category_data, list) and len(category_data) > 0:
                count += 1
        return count
    
    target_count = count_entities(result['target_user_preferences'])
    
    result['preference_breakdown'] = {
        'target_user': {
            'categories': count_categories(result['target_user_preferences']),
            'entities': target_count
        },
        'target_user_aspects': len(result['target_user_aspects']),
        'other_users': {
            'categories': 0,
            'entities': 0
        }
    }
    
    return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1 v2: Preference Extraction + Aspects")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 1 v2: Preference Extraction + Aspect-Level Analysis")
    log_with_timestamp("=" * 80)
    
    # 加载数据
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    user_id = data['user_id']
    products = data['results']
    
    log_with_timestamp(f"User: {user_id}")
    log_with_timestamp(f"Products: {len(products)}")
    log_with_timestamp(f"Concurrency: {args.max_workers} products in parallel")
    log_with_timestamp("")
    
    # 并发处理产品
    results = []
    completed_count = [0]
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_product = {
            executor.submit(process_product, product): product
            for product in products
        }
        
        for future in as_completed(future_to_product):
            product = future_to_product[future]
            asin = product['asin']
            try:
                result = future.result()
                results.append(result)
                completed_count[0] += 1
                
                # 每10个产品输出一次进度
                if completed_count[0] % 10 == 0 or completed_count[0] == len(products):
                    log_with_timestamp(f"Progress: {completed_count[0]}/{len(products)} completed")
                    
                    # 统计维度和方面
                    dimension_stats = defaultdict(lambda: {'target': 0, 'explicit_aspects': 0, 'implicit_aspects': 0})
                    
                    for r in results:
                        for category, dims in r['target_user_preferences'].items():
                            if isinstance(dims, dict):
                                for dim, entities in dims.items():
                                    if isinstance(entities, list):
                                        dimension_stats[dim]['target'] += len(entities)
                        
                        # 统计方面
                        for aspect in r.get('target_user_aspects', []):
                            if aspect.get('is_implicit'):
                                dimension_stats[aspect.get('aspect', 'unknown')]['implicit_aspects'] += 1
                            else:
                                dimension_stats[aspect.get('aspect', 'unknown')]['explicit_aspects'] += 1
                    
                    log_with_timestamp(f"  Dimension extraction (top 5):")
                    sorted_dims = sorted(dimension_stats.items(),
                                       key=lambda x: x[1]['target'],
                                       reverse=True)[:5]
                    for dim, counts in sorted_dims:
                        log_with_timestamp(f"    {dim:<30} target={counts['target']:>3} aspects(E:{counts['explicit_aspects']:>2}/I:{counts['implicit_aspects']:>2})")
                    
                    total_prefs = sum(c['target'] for c in dimension_stats.values())
                    total_explicit = sum(c['explicit_aspects'] for c in dimension_stats.values())
                    total_implicit = sum(c['implicit_aspects'] for c in dimension_stats.values())
                    log_with_timestamp(f"  TOTALS: dimensions={total_prefs} explicit_aspects={total_explicit} implicit_aspects={total_implicit}")
                    log_with_timestamp("")
            
            except Exception as e:
                log_with_timestamp(f"[{asin}] ❌ 产品处理异常: {type(e).__name__}: {str(e)}")
                import traceback
                log_with_timestamp(f"[{asin}] 堆栈跟踪:")
                for line in traceback.format_exc().split('\n')[:10]:
                    if line.strip():
                        log_with_timestamp(f"[{asin}] {line}")
                completed_count[0] += 1
    
    success_count = sum(1 for r in results if r.get('target_user_aspects'))
    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("🎯 最终统计")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"总产品数: {len(products)}")
    log_with_timestamp(f"处理完成: {len(results)}")
    log_with_timestamp(f"成功提取: {success_count} (成功率: {100*success_count/len(results):.1f}%)")
    log_with_timestamp(f"失败产品: {len(results) - success_count}")
    
    if len(results) > 0:
        total_aspects = sum(len(r.get('target_user_aspects', [])) for r in results)
        avg_aspects = total_aspects / success_count if success_count > 0 else 0
        log_with_timestamp(f"总方面数: {total_aspects}")
        log_with_timestamp(f"平均方面数/产品: {avg_aspects:.1f}")
    
    log_with_timestamp("")
    
    output_data = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'total_products': len(results),
        'version': '2.0',
        'improvements': [
            'aspect-level extraction with confidence scores',
            'implicit aspect detection',
            'quality validation'
        ],
        'results': results
    }
    
    output_file = os.path.join(args.output_dir, f'preferences_{user_id}_v2.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"✅ 输出文件: {output_file}")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✅ Stage 1 v2 完成!")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
