#!/usr/bin/env python3
"""
商品实体提取模块
负责处理商品实体的提取和处理
"""

import os
import json
import gzip
import re
import sys
from typing import Dict, List, Optional, Union
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry, APIErrorException

def log_with_timestamp(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def normalize_category_label(category: str) -> str:
    """
    Normalize category labels to keep keys consistent across pipeline.
    Currently enforces: "Color/Finish" -> "Color" (and common variants).
    """
    if category is None:
        return category
    c = str(category).strip()
    if not c:
        return c
    c_lower = c.lower().strip()
    c_compact = c_lower.replace(" ", "")
    if c_compact in {"color/finish", "colour/finish", "colorfinish", "colourfinish"}:
        return "Color"
    if c_lower in {"color", "colour"}:
        return "Color"
    return c

def robust_json_loads(json_str: str):
    """
    Robust JSON parsing that handles common formatting issues like unescaped quotes.

    Attempts multiple parsing strategies in order:
    1. Standard json.loads()
    2. Fix unescaped quotes in string values (like "Hooker"s green" -> "Hooker's green")
    3. Use ast.literal_eval as fallback
    """
    import ast

    # First, try standard JSON parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second, try to fix unescaped quotes in string values
    try:
        fixed_json = fix_unescaped_quotes(json_str)
        if fixed_json != json_str:
            return json.loads(fixed_json)
    except (json.JSONDecodeError, Exception):
        pass

    # Third, try ast.literal_eval as a more permissive fallback
    try:
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        pass

    # If all methods fail, raise the original JSON error
    raise json.JSONDecodeError("All JSON parsing methods failed", json_str, 0)


def fix_unescaped_quotes(json_str: str) -> str:
    """
    Fix unescaped quotes within JSON string values.
    This handles cases like "Hooker"s green" -> "Hooker's green"
    """
    result = json_str

    # Handle common patterns of unescaped quotes in possessive forms
    # "Hooker"s green" -> "Hooker's green"
    result = result.replace('"s green"', "'s green\"")
    result = result.replace('"s blue"', "'s blue\"")
    result = result.replace('"s yellow"', "'s yellow\"")
    result = result.replace('"s red"', "'s red\"")
    result = result.replace('"s black"', "'s black\"")
    result = result.replace('"s white"', "'s white\"")

    # Handle other common contractions and possessives
    result = result.replace("don't", "don\\'t")
    result = result.replace("can't", "can\\'t")
    result = result.replace("won't", "won\\'t")
    result = result.replace("it's", "it\\'s")
    result = result.replace("I'm", "I\\'m")
    result = result.replace("you're", "you\\'re")
    result = result.replace("we're", "we\\'re")
    result = result.replace("they're", "they\\'re")

    return result


def clean_html_content(text) -> str:
    """Remove HTML tags, JavaScript, CSS and clean up content for entity extraction."""
    if not text:
        return ""

    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)

    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags and their content
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove HTML tags (but keep content between tags)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove JavaScript URLs
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)

    # Remove common Amazon UI text patterns
    text = re.sub(r'Save \d+% on.*?(?=when|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enter code [A-Z0-9]+ at checkout', '', text, flags=re.IGNORECASE)
    text = re.sub(r'restrictions apply', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Here\'s how', '', text, flags=re.IGNORECASE)

    # Remove Amazon-specific UI patterns
    text = re.sub(r'Read the full returns policy', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Go to Your Orders to start the return', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Print the return shipping label', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Ship it!', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Package Dimensions:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Shipping Weight:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ASIN:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Date first listed on Amazon:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Average Customer Review:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'customer reviews?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'stars?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'out of 5', '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)

    # Remove JSON-like data in HTML attributes
    text = re.sub(r'\{[^{}]*\}', '', text)

    # Remove remaining HTML entities
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)

    # Remove extra whitespace and normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove empty or very short strings that are likely noise
    if len(text) < 3:
        return ""

    return text

def load_data_from_gzip(file_path: str, data_type: str, filter_func=None, max_items: int = None) -> Union[List[Dict], Dict[str, Dict]]:
    """从gzip文件加载数据"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            if data_type == 'metadata':
                # Load metadata as dict keyed by asin
                data = {}
                for line_num, line in enumerate(f):
                    if max_items and line_num >= max_items:
                        break
                    try:
                        item = json.loads(line.strip())
                        asin = item.get('asin', '')
                        if asin:
                            if filter_func is None or filter_func(item):
                                data[asin] = item
                    except json.JSONDecodeError:
                        continue
                return data
            else:
                # Load reviews as list
                data = []
                for line_num, line in enumerate(f):
                    if max_items and line_num >= max_items:
                        break
                    try:
                        item = json.loads(line.strip())
                        if filter_func is None or filter_func(item):
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
                return data
    except Exception as e:
        log_with_timestamp(f"Error loading {data_type} from {file_path}: {e}")
        return {} if data_type == 'metadata' else []

def load_data(data_type: str, filter_func=None, max_items: int = None, user_products: set = None):
    """加载数据的通用函数"""
    if data_type == 'metadata':
        file_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
        # If user_products is specified, create a filter function
        if user_products is not None:
            def filter_func(item):
                return item.get('asin', '') in user_products
    elif data_type == 'reviews':
        file_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
        # If user_products is specified for reviews, create a filter function
        if user_products is not None:
            def filter_func(item):
                return item.get('asin', '') in user_products
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return load_data_from_gzip(file_path, data_type, filter_func, max_items)

def load_product_metadata(user_products: set = None) -> Dict[str, Dict]:
    """加载产品元数据，可选择只加载指定用户的商品（保持向后兼容）"""
    return load_data('metadata', user_products=user_products)

def process_product_extraction_response(response_str: str) -> List[str]:
    """
    处理商品实体提取的LLM响应

    Args:
        response_str: LLM返回的原始字符串

    Returns:
        处理后的实体列表

    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    if not response_str:
        raise APIErrorException("No response from product extraction")

    try:
        # Clean the response - product extraction expects direct JSON, no Chain of Thought
        response_str = response_str.strip()

        # Remove markdown code blocks if present
        if response_str.startswith('```') and '```' in response_str:
            # Find the first ```json or just ```
            json_start = response_str.find('```')
            if json_start != -1:
                # Find the end of the opening ```
                content_start = response_str.find('\n', json_start)
                if content_start == -1:
                    content_start = response_str.find('```', json_start + 3)
                    if content_start == -1:
                        content_start = json_start + 3
                else:
                    content_start += 1

                # Find the closing ```
                json_end = response_str.find('```', content_start)
                if json_end != -1:
                    response_str = response_str[content_start:json_end].strip()
                else:
                    response_str = response_str[content_start:].strip()

        # Try to parse as JSON with robust error handling
        result = robust_json_loads(response_str)

        # Detect compound color-like forms such as "Red/Blue" (but not fractions like "3/8")
        color_combo_slash_re = re.compile(r"[A-Za-z]\s*/\s*[A-Za-z]")

        # Handle different response formats
        if isinstance(result, dict) and not ('result' in result and isinstance(result['result'], list)):
            # New direct format: {category: [entities]}
            categorized_entities = {}
            flattened = []

            for category, entities in result.items():
                category_norm = normalize_category_label(category)
                if isinstance(entities, list):
                    # Process list of entities for this category
                    category_entities = []
                    for entity in entities:
                        if isinstance(entity, str):
                            entity_text = entity.strip()
                            if entity_text:
                                # Apply atomic filtering
                                entity_lower = entity_text.lower()
                                if not (',' in entity_text or
                                        ' and ' in entity_lower or
                                        ' with ' in entity_lower or
                                        ' or ' in entity_lower or
                                        ' for ' in entity_lower or
                                        '&' in entity_text or
                                        color_combo_slash_re.search(entity_text)):
                                    category_entities.append(entity_text)
                                    flattened.append(entity_text)

                    if category_entities:
                        categorized_entities[category_norm] = category_entities
                elif isinstance(entities, str):
                    # Handle single entity as string
                    entity_text = entities.strip()
                    if entity_text:
                        entity_lower = entity_text.lower()
                        if not (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text or
                                color_combo_slash_re.search(entity_text)):
                            categorized_entities[category_norm] = [entity_text]
                            flattened.append(entity_text)

            if flattened:
                return flattened, categorized_entities

        elif isinstance(result, list):
            # Legacy format: array of objects with entity and category
            categorized_entities = {}  # category -> [entities] mapping
            flattened = []

            for item in result:
                if isinstance(item, dict) and 'entity' in item and 'category' in item:
                    # Extract entity and category from structured format
                    entity_text = item.get('entity', '').strip()
                    category = normalize_category_label(item.get('category', '').strip())
                    if entity_text and category:
                        # Apply atomic filtering
                        entity_lower = entity_text.lower()
                        if not (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text or
                                color_combo_slash_re.search(entity_text)):
                            # Accumulate entities in lists by category
                            if category not in categorized_entities:
                                categorized_entities[category] = []
                            categorized_entities[category].append(entity_text)
                            flattened.append(entity_text)

            if flattened:
                # Return both list format (for backward compatibility) and dict format (new categorized format)
                return flattened, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from product extraction (empty result)")

        elif isinstance(result, dict) and 'result' in result and isinstance(result['result'], list):
            # Handle wrapped format: {"result": [...]}
            categorized_entities = {}  # category -> entity mapping
            flattened = []

            for item in result['result']:
                if isinstance(item, dict) and 'entity' in item and 'category' in item:
                    # Extract entity and category from structured format
                    entity_text = item.get('entity', '').strip()
                    category = normalize_category_label(item.get('category', '').strip())
                    if entity_text and category:
                        # Use category as key, entity as value
                        if category not in categorized_entities:
                            categorized_entities[category] = entity_text
                        flattened.append(entity_text)
                elif isinstance(item, dict) and 'entity_text' in item:
                    # Legacy format: extract entity_text
                    entity_text = item.get('entity_text', '').strip()
                    if entity_text:
                        flattened.append(entity_text)
                elif isinstance(item, str):
                    # Fallback: simple string format
                    flattened.append(item)
                elif isinstance(item, list):
                    # Flatten nested list but only take string elements
                    for subitem in item:
                        if isinstance(subitem, str):
                            flattened.append(subitem)

            # Filter entities to ensure atomic properties (no compound phrases)
            atomic_entities = []
            for entity in flattened:
                # Skip entities containing compound indicators
                entity_lower = entity.lower()
                if (',' in entity or
                    ' and ' in entity_lower or
                    ' with ' in entity_lower or
                    ' or ' in entity_lower or
                    ' for ' in entity_lower or
                    '&' in entity or
                    color_combo_slash_re.search(entity)):
                    continue
                atomic_entities.append(entity)

            if atomic_entities:
                # Return both list format (for backward compatibility) and dict format (new categorized format)
                return atomic_entities, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from product extraction (empty result)")

        elif isinstance(result, dict):
            # Structured format with categories
            flattened = []

            # Extract entities from all categories and flatten them
            expected_keys = ["product_core", "attributes", "usage_scenario", "target_audience"]
            for key in expected_keys:
                if key in result and isinstance(result[key], list):
                    for item in result[key]:
                        if isinstance(item, str) and item.strip():
                            flattened.append(item.strip())

            if flattened:
                return flattened
            else:
                raise APIErrorException("No valid entities extracted from product extraction (empty result)")

        else:
            raise APIErrorException("Invalid result format from product extraction")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error in product extraction: {e}", flush=True)
        print(f"Raw response content (first 500 chars): {response_str[:500]!r}", flush=True)
        raise APIErrorException("JSON parsing failed in product extraction")
    except Exception as e:
        print(f"Unexpected error processing product extraction response: {e}", flush=True)
        raise APIErrorException("Response processing failed in product extraction")

def prepare_content_and_extract_entities(data_source, data_type: str, llm_model, asin: str = None) -> List[str]:
    """通用函数：准备内容并提取实体

    Args:
        data_source: 数据源（产品信息或评论数据）
        data_type: 数据类型 ('product' 或 'user preference')
        llm_model: LLM模型
        asin: 产品ASIN（可选）

    Returns:
        提取的实体列表
    """
    if data_type == 'product':
        # Prepare product content
        content_parts = []

        # Add title
        if 'title' in data_source and data_source['title']:
            content_parts.append(f"Title: {data_source['title']}")

        # Add brand
        if 'brand' in data_source and data_source['brand']:
            content_parts.append(f"Brand: {data_source['brand']}")

        # Add description
        if 'description' in data_source and data_source['description']:
            desc = data_source['description']
            if isinstance(desc, list):
                desc = " ".join(desc)
            content_parts.append(f"Description: {clean_html_content(desc)}")

        # Add features
        if 'feature' in data_source and data_source['feature']:
            feature = data_source['feature']
            if isinstance(feature, list):
                feature = " ".join(feature)
            content_parts.append(f"Features: {clean_html_content(feature)}")

        # Add category information
        if 'category' in data_source and data_source['category']:
            category = data_source['category']
            if isinstance(category, list):
                category = " > ".join(category)
            content_parts.append(f"Category: {category}")

        # Add main category
        if 'main_cat' in data_source and data_source['main_cat']:
            content_parts.append(f"Main Category: {data_source['main_cat']}")

        content = "\n".join(content_parts)
        result = extract_product_entities(content, llm_model, asin)

        # Handle the new return format (tuple of list and dict)
        if isinstance(result, tuple):
            entities_list, entities_dict = result
            # Return the dict if available (new format), otherwise return the list
            return entities_dict if entities_dict else entities_list
        else:
            # Backward compatibility
            return result
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def extract_product_entities(content: str, llm_model, asin: str = None) -> List[str]:
    """Extract product entities using LLM - business logic implementation with JSON parsing retry."""
    # Ensure content is not None or empty
    if not content or not isinstance(content, str):
        raise APIErrorException("Invalid content for product extraction")

    prompt = f"""
你是一个电商数据专家。请从以下产品信息中提取关键实体。

**输入产品信息:** {content}

**实体分类要求:**
对于每个提取的实体，必须将其归类为以下类别之一：
[Brand, Material, Dimensions, Quantity, Color, Design, Usage, Selling Point, Safety/Certification, Accessories]

**提取规则:**
1. **严禁输出类别名称**: 严禁输出类别名称（如 'Brand', 'Color', 'Size', 'Material', 'Dimensions' 等）作为实体值。只提取具体的品牌名、具体的颜色名、具体的尺寸值等。
2. **只提取具体值**: 如果找不到具体的值，不要输出该条目。例如，不要输出"Brand"，而是要输出"Apple"或"Nike"等具体品牌名。
3. 只提取真正代表产品特征的具体实体
4. 每个实体必须能明确归类到上述类别之一
5. 避免提取通用形容词或营销词汇
6. 确保实体是产品描述中的具体内容
7. **原子化 (Atomic Only)**: 提取的实体必须是单一、独立的属性。严禁提取包含连接词(and,or,with,for,&)或逗号的复合短语
8. **颜色实体必须单色 (Color = One Color Only)**:
   - `Color` 类别中，每个数组元素只能包含 **一种** 颜色（一个颜色词或一个颜色短语）。
   - 如果原文出现多种颜色（例如 “Red/Blue”, “Blue and Green”, “Black & White”），必须 **拆分成多个元素**：["Red","Blue"]、["Blue","Green"]、["Black","White"]。
   - 严禁在单个颜色实体里包含分隔符 `/`, `&`, `and`, `or`, `,` 等把多个颜色拼在一起。

**重要:** 每个实体应该是单个词或简单短语，不要包含多个属性。宁缺毋滥，如果不确定是否为具体值，就不要提取。

**输出格式:**
返回一个JSON对象，其中键是类别名称，值是该类别对应的实体数组。相同类别的多个实体应该放在同一个数组中。

示例:
{{
  "Brand": ["Apple"],
  "Design": ["iPhone 15", "Smartphone"],
  "Quantity": ["256GB"],
  "Color": ["Blue", "Space Gray"],
  "Material": ["Aluminum"],
  "Selling Point": ["Waterproof", "Face ID"]
}}

只返回有效的JSON对象，不要其他解释。
"""

    # Retry up to 5 times for JSON parsing errors
    json_parse_retries = 5
    for attempt in range(json_parse_retries):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="product_extraction")
            if success and response_str:
                entities = process_product_extraction_response(response_str)

                # Handle the tuple return format (list, dict)
                if isinstance(entities, tuple) and len(entities) == 2:
                    entities_list, entities_dict = entities

                    # Additional deduplication for product entities list
                    if isinstance(entities_list, list):
                        seen = set()
                        deduplicated = []
                        for entity in entities_list:
                            if isinstance(entity, str) and entity.strip():
                                clean_entity = entity.strip()
                                if clean_entity not in seen:
                                    seen.add(clean_entity)
                                    deduplicated.append(clean_entity)
                        # Return the tuple format
                        return deduplicated, entities_dict

                    # If entities_list is not a list, return the tuple as-is
                    return entities

                # Handle single list return (backward compatibility)
                elif isinstance(entities, list):
                    # Remove duplicates while preserving order
                    seen = set()
                    deduplicated = []
                    for entity in entities:
                        if isinstance(entity, str) and entity.strip():
                            clean_entity = entity.strip()
                            if clean_entity not in seen:
                                seen.add(clean_entity)
                                deduplicated.append(clean_entity)
                    return deduplicated

                # Fallback for other formats
                return entities

            # If we get here, no valid entities were extracted - treat as failure
            raise APIErrorException("No valid entities extracted from product")
        except APIErrorException as e:
            # Check if this is a JSON parsing error
            error_msg = str(e)
            if "JSON parsing failed" in error_msg or "JSON parsing error" in error_msg:
                if attempt < json_parse_retries - 1:
                    print(f"JSON parsing failed (attempt {attempt + 1}/{json_parse_retries}), retrying...", flush=True)
                    continue
                else:
                    print(f"JSON parsing failed after {json_parse_retries} attempts", flush=True)
            # Re-raise API errors (including JSON parsing errors after retries)
            raise
        except Exception:
            # Let the caller handle other API errors - they will trigger key switching
            raise

def generate_entity_explanations(entities: List[str], product_info: Dict, llm_model) -> Dict[str, str]:
    """Generate explanations for why each entity was extracted from the product."""
    explanations = {}

    # Generate explanations for all extracted entities
    entities_to_explain = entities  # Process all entities
    print(f"🔍 Generating explanations for all {len(entities_to_explain)} entities", flush=True)

    for entity in entities_to_explain:
        prompt = f"""
Based on the following product information, explain why "{entity}" was identified as a key product entity/feature.

Product Information:
- Title: {product_info.get('title', 'N/A')}
- Brand: {product_info.get('brand', 'N/A')}
- Description: {product_info.get('description', 'N/A')[:500]}...
- Features: {', '.join(product_info.get('feature', [])[:3])}
- Category: {', '.join(product_info.get('category', [])[:2])}

Provide a brief explanation (1-2 sentences) of why this entity represents an important feature of this product.
"""

        try:
            response = llm_model.invoke([{"role": "user", "content": prompt}])
            explanation = str(response.content).strip() if response and hasattr(response, 'content') else "No explanation available"
            explanations[entity] = explanation
        except Exception as e:
            explanations[entity] = f"Failed to generate explanation: {str(e)}"

    return explanations

def extract_product_entities_only(asin: str, product_metadata: Dict[str, Dict], api_config: Dict, total_products: int = None) -> Optional[Dict]:
    """Extract product entities using specified API config. Returns data for later printing."""

    try:
        product_info = product_metadata.get(asin, {})
        product_title = product_info.get('title', f'Product {asin}')

        # Create LLM model with the specified API configuration
        from utils import create_llm_with_config
        llm_model = create_llm_with_config(api_config)

        product_entities_result = prepare_content_and_extract_entities(product_info, 'product', llm_model, asin)

        # Handle both old list format and new tuple format for backward compatibility
        if isinstance(product_entities_result, tuple):
            product_entities_list, product_entities_dict = product_entities_result
        else:
            # Backward compatibility: if it's just a list, create empty dict
            product_entities_list = product_entities_result
            product_entities_dict = {}

        # Use the categorized dict if available and not empty, otherwise fall back to list
        # Ensure we never return None
        if product_entities_dict:
            product_entities = product_entities_dict
        elif product_entities_list:
            product_entities = product_entities_list
        else:
            product_entities = []  # Fallback to empty list

        # Skip generating explanations for entities - no longer needed
        # entity_explanations = generate_entity_explanations(entities_for_explanations, product_info, llm_model)

        # Prepare metadata for printing (still need HTML cleaning for display)
        metadata_lines = []
        for key in ['title', 'brand', 'description', 'feature', 'category', 'main_cat']:
            if key in product_info and product_info[key]:
                value = product_info[key]
                if key in ['description', 'feature'] and isinstance(value, list):
                    content = clean_html_content(" ".join(value)) if value else ""
                    metadata_lines.append(f'    {key}: {len(value)} items - {content}')
                else:
                    clean_value = clean_html_content(value)
                    metadata_lines.append(f'    {key}: {clean_value}')

        return {
            'asin': asin,
            'product_title': product_title,
            'product_entities': product_entities,
            'product_info': product_info,
            'metadata_lines': metadata_lines,
            'success': True
        }
    except APIErrorException:
        # API-related errors should be re-raised to trigger key switching
        raise
    except Exception as e:
        # Other errors (like data issues) return error result without key switching
        error_msg = f"Product extraction failed for {asin}: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        return {
            'asin': asin,
            'error': error_msg,
            'success': False
        }