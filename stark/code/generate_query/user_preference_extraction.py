#!/usr/bin/env python3
"""
用户偏好实体提取模块
负责处理用户偏好实体的提取和处理
"""

import os
import json
import gzip
import sys
import re
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from collections import defaultdict

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

def clean_html_content(text) -> str:
    """Remove HTML tags and clean up content for entity extraction."""
    if not text:
        return ""

    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove JavaScript content
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common Amazon UI text patterns
    text = re.sub(r'Save \d+% on.*?(?=when|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enter code [A-Z0-9]+ at checkout', '', text, flags=re.IGNORECASE)
    text = re.sub(r'restrictions apply', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Here\'s how', '', text, flags=re.IGNORECASE)

    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()

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
    elif data_type == 'reviews':
        file_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return load_data_from_gzip(file_path, data_type, filter_func, max_items)

TARGET_USER = "AG7EF0SVBQOUX"
REVIEW_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
META_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
OUTPUT_FILE = "/home/wlia0047/ar57_scratch/wenyu/user_preference_queries.json"

def load_user_reviews(target_user: str = None) -> List[Dict]:
    """加载指定用户的所有评论（保持向后兼容）"""
    if target_user:
        def filter_func(data):
            user_id = data.get('user_id') or data.get('reviewerID') or data.get('reviewer_id')
            return user_id == target_user
        return load_data('reviews', filter_func, max_items=None)  # Remove max_items limit when filtering for specific user
    return load_data('reviews', max_items=100)

def process_user_preference_extraction_response(response_str: str) -> tuple:
    """
    处理用户偏好实体提取的LLM响应

    Args:
        response_str: LLM返回的原始字符串

    Returns:
        (flattened_entities_list, categorized_entities_dict) 元组

    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    if not response_str:
        raise APIErrorException("No response from user preference extraction")

    try:
        def _normalize_sentiment(val: Optional[str]) -> str:
            if not val:
                return "neutral"
            v = str(val).strip().lower()
            if v in {"positive", "pos", "+"}:
                return "positive"
            if v in {"negative", "neg", "-"}:
                return "negative"
            if v in {"neutral", "neu", "unknown", "mixed"}:
                return "neutral"
            return "neutral"

        def _coerce_entity_item(item) -> Optional[Dict[str, str]]:
            """
            Normalize entity item to: {"entity": <str>, "sentiment": <positive|negative|neutral>}
            Supports:
              - "string"
              - {"entity": "...", "sentiment": "..."} (also accepts keys: text/name, polarity)
            """
            if isinstance(item, str):
                entity_text = item.strip()
                if not entity_text:
                    return None
                return {"entity": entity_text, "sentiment": "neutral"}

            if isinstance(item, dict):
                entity_text = (
                    item.get("entity")
                    or item.get("text")
                    or item.get("name")
                    or ""
                )
                entity_text = str(entity_text).strip()
                if not entity_text:
                    return None
                sentiment = _normalize_sentiment(item.get("sentiment") or item.get("polarity"))
                return {"entity": entity_text, "sentiment": sentiment}

            return None

        # Clean the response
        response_str = response_str.strip()

        # Smart JSON extraction for Chain of Thought responses
        lines = response_str.strip().split('\n')
        json_found = False

        # Check if the last few lines contain valid JSON
        for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):  # Check last 5 lines
            line = lines[i].strip()
            if line.startswith('{') and line.endswith('}'):
                # Found JSON object at the end
                response_str = line
                json_found = True
                break

        # If no JSON found at the end, look for code blocks
        if not json_found:
            # Find the LAST json code block (in case there are multiple)
            json_blocks = []
            start = 0
            while True:
                json_start = response_str.find('```json', start)
                if json_start == -1:
                    break
                json_end = response_str.find('```', json_start + 7)
                if json_end == -1:
                    break
                content_start = response_str.find('\n', json_start) + 1
                if content_start > 0:
                    content = response_str[content_start:json_end].strip()
                    if content:
                        json_blocks.append(content)
                start = json_end + 3

            # Also handle regular ``` blocks
            if not json_blocks:
                if '```' in response_str:
                    # Find the LAST code block
                    last_triple = response_str.rfind('```')
                    first_triple = response_str.rfind('```', 0, last_triple)
                    if first_triple != last_triple:
                        content_start = response_str.find('\n', first_triple) + 1
                        if content_start > 0:
                            response_str = response_str[content_start:last_triple].strip()
                    else:
                        # Single code block
                        content_start = response_str.find('\n', first_triple) + 1
                        if content_start > 0:
                            response_str = response_str[content_start:].strip()
            
            # Use the last json block if found
            if json_blocks:
                response_str = json_blocks[-1]

        # Try to parse as JSON
        result = json.loads(response_str)

        # Handle different response formats
        if isinstance(result, dict):
            # New format:
            #  - {category: ["entity", ...]}
            #  - {category: [{"entity": "...", "sentiment": "positive|negative|neutral"}, ...]}
            categorized_entities = {}
            flattened = []

            for category, entities in result.items():
                category_norm = normalize_category_label(category)
                if isinstance(entities, list):
                    # Process list of entities for this category
                    category_entities = []
                    for entity in entities:
                        normalized = _coerce_entity_item(entity)
                        if not normalized:
                            continue
                        entity_text = normalized["entity"]

                        # Apply atomic filtering on entity text
                        entity_lower = entity_text.lower()
                        if (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text):
                            continue

                        category_entities.append(normalized)
                        flattened.append(entity_text)

                    if category_entities:
                        if category_norm not in categorized_entities:
                            categorized_entities[category_norm] = []
                        categorized_entities[category_norm].extend(category_entities)
                elif isinstance(entities, str):
                    # Handle single entity as string (legacy)
                    normalized = _coerce_entity_item(entities)
                    if normalized:
                        entity_text = normalized["entity"]
                        entity_lower = entity_text.lower()
                        if not (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text):
                            categorized_entities[category_norm] = [normalized]
                            flattened.append(entity_text)
                elif isinstance(entities, dict):
                    # Handle single entity object
                    normalized = _coerce_entity_item(entities)
                    if normalized:
                        entity_text = normalized["entity"]
                        entity_lower = entity_text.lower()
                        if not (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text):
                            categorized_entities[category_norm] = [normalized]
                            flattened.append(entity_text)

            if flattened:
                return flattened, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from user preference extraction")

        elif isinstance(result, list):
            # Legacy format: array of strings - convert to new format with neutral sentiment
            categorized_entities = {"General": []}
            flattened = []

            for item in result:
                normalized = _coerce_entity_item(item)
                if not normalized:
                    continue
                entity_text = normalized["entity"]
                # Apply atomic filtering
                entity_lower = entity_text.lower()
                if (',' in entity_text or
                        ' and ' in entity_lower or
                        ' with ' in entity_lower or
                        ' or ' in entity_lower or
                        ' for ' in entity_lower or
                        '&' in entity_text):
                    continue
                categorized_entities["General"].append(normalized)
                flattened.append(entity_text)

            if flattened:
                return flattened, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from user preference extraction")

        else:
            raise APIErrorException("Invalid result format from user preference extraction")

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in user preference extraction: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in user preference extraction")
    except Exception as e:
        print(f"Unexpected error processing user preference extraction response: {e}", flush=True)
        raise APIErrorException("Response processing failed in user preference extraction")

def prepare_content_and_extract_entities(data_source, data_type: str, llm_model, asin: str = None) -> str:
    """通用函数：准备内容并提取实体（只调用LLM，不解析JSON）

    Args:
        data_source: 数据源（产品信息字典或评论列表）
        data_type: 数据类型 ('product' 或 'user preference')
        llm_model: LLM模型
        asin: 产品ASIN（可选）

    Returns:
        原始响应字符串（将被保存到 api_raw_responses.json，稍后统一解析）
    """
    if data_type in ['user_preference', 'user preference']:
        # 处理用户评论
        user_reviews = data_source
        if not user_reviews:
            raise APIErrorException("No user reviews available for preference extraction")

        # Combine all review content
        content_parts = []
        for review in user_reviews:
            text = review.get('reviewText', '').strip()
            title = review.get('summary', '').strip()

            if text or title:
                review_parts = []
                if title:
                    review_parts.append(f"Title: {title}")
                if text:
                    review_parts.append(f"Review: {text}")
                content_parts.append(' '.join(review_parts))

        content = ' '.join(content_parts)
        # Clean content
        content = clean_html_content(content)

        return extract_user_preference_entities(content, llm_model, asin=asin)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def extract_review_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从prompt中提取标题和评论文本
    
    Args:
        prompt: API请求的prompt字符串
    
    Returns:
        (title, review_text) 元组
    """
    if not prompt:
        return None, None
    
    # 查找 "Text: Title: ... Review: ..." 格式
    text_match = re.search(r'Text:\s*Title:\s*(.+?)\s*Review:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
    if text_match:
        title = text_match.group(1).strip()
        review = text_match.group(2).strip()
        return title, review
    
    # 如果没有找到，尝试其他格式
    title_match = re.search(r'Title:\s*(.+?)(?:\n|Review:)', prompt, re.DOTALL)
    review_match = re.search(r'Review:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else None
    review = review_match.group(1).strip() if review_match else None
    
    return title, review

def match_review_to_product(title: str, review_text: str, user_preference_data: Dict) -> Optional[str]:
    """
    通过匹配评论文本找到对应的ASIN
    
    Args:
        title: 评论标题
        review_text: 评论文本
        user_preference_data: 用户偏好数据字典，包含products列表
    
    Returns:
        匹配的ASIN，如果未找到则返回None
    """
    if not title and not review_text:
        return None
    
    # 创建搜索关键词
    title_lower = title.lower().strip() if title else ""
    review_lower = review_text.lower().strip() if review_text else ""
    
    # 提取关键词（前50个字符的标题和前100个字符的评论）
    title_key = title_lower[:50] if title_lower else ""
    review_key = review_lower[:100] if review_lower else ""
    
    # 在user_preference_data中查找匹配的评论
    best_match = None
    best_score = 0
    
    for product in user_preference_data.get('products', []):
        for review in product.get('review_content', []):
            review_title = review.get('summary', '').strip().lower()
            review_text_content = review.get('reviewText', '').strip().lower()
            
            score = 0
            
            # 匹配标题
            if title_key and review_title:
                # 检查标题是否匹配（至少匹配前30个字符）
                if title_key[:30] in review_title or review_title[:30] in title_key:
                    score += 2
            
            # 匹配评论文本
            if review_key and review_text_content:
                # 检查评论是否匹配（至少匹配前80个字符）
                if review_key[:80] in review_text_content or review_text_content[:80] in review_key:
                    score += 3
            
            # 如果标题和评论都匹配，分数更高
            if score > best_score:
                best_score = score
                best_match = product.get('asin')
    
    # 如果分数足够高，返回匹配的ASIN
    if best_score >= 2:
        return best_match
    
    return None

def parse_responses_from_file(api_responses_file: str, context: str = "user_preference_extraction", 
                              user_preference_data: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    从 api_raw_responses.json 文件中解析响应
    
    Args:
        api_responses_file: API响应文件路径
        context: 要解析的上下文（默认为 "user_preference_extraction"）
        user_preference_data: 可选的用户偏好数据，用于匹配ASIN
    
    Returns:
        字典，键为prompt的hash或索引（或ASIN如果提供了user_preference_data），值为解析后的实体字典
    """
    if not os.path.exists(api_responses_file):
        log_with_timestamp(f"⚠️ API responses file not found: {api_responses_file}")
        return {}
    
    try:
        with open(api_responses_file, 'r', encoding='utf-8') as f:
            all_responses = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading API responses file: {e}")
        return {}
    
    # Filter responses by context
    filtered_responses = [r for r in all_responses if r.get('context') == context and r.get('success', False)]
    
    log_with_timestamp(f"📋 Found {len(filtered_responses)} responses with context '{context}'")
    
    parsed_results = {}
    for idx, response_data in enumerate(filtered_responses):
        try:
            raw_response = response_data.get('raw_response', {})
            content = raw_response.get('content', '')
            
            if not content:
                log_with_timestamp(f"⚠️ Empty content in response {idx}")
                continue
            
            # Parse the response
            entities_result = process_user_preference_extraction_response(content)
            
            # Handle tuple return format (list, dict)
            if isinstance(entities_result, tuple):
                entities_list, entities_dict = entities_result
                # Use the dict if available (new format), otherwise use the list
                parsed_entities = entities_dict if entities_dict else entities_list
            else:
                parsed_entities = entities_result
            
            # Prefer exact ASIN from saved meta (required for ASIN mapping)
            key = None
            meta = response_data.get('meta') or {}
            meta_asin = meta.get('asin')
            if isinstance(meta_asin, str) and meta_asin.strip():
                key = meta_asin.strip().upper()
            
            # Fallback to index-based key (no ASIN mapping)
            if not key:
                key = f"response_{idx}"
            
            parsed_results[key] = parsed_entities
            
        except Exception as e:
            log_with_timestamp(f"⚠️ Error parsing response {idx}: {e}")
            continue
    
    log_with_timestamp(f"✅ Successfully parsed {len(parsed_results)} responses")
    return parsed_results

def parse_api_responses_to_user_preferences(api_responses_file: str, user_preference_file: str, 
                                            output_file: Optional[str] = None) -> Dict:
    """
    解析API响应文件并生成完整的用户偏好实体数据
    
    Args:
        api_responses_file: API响应文件路径
        user_preference_file: 用户偏好文件路径（用于获取用户ID和评论内容）
        output_file: 可选的输出文件路径
    
    Returns:
        用户偏好实体数据字典，格式与user_preference_entities.json相同
    """
    # 读取用户偏好文件以获取用户ID和产品信息
    if not os.path.exists(user_preference_file):
        log_with_timestamp(f"⚠️ User preference file not found: {user_preference_file}")
        return {}
    
    try:
        with open(user_preference_file, 'r', encoding='utf-8') as f:
            user_preference_data = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading user preference file: {e}")
        return {}
    
    user_id = user_preference_data.get('user_id', '')
    
    # 读取API响应文件
    if not os.path.exists(api_responses_file):
        log_with_timestamp(f"⚠️ API responses file not found: {api_responses_file}")
        return {}
    
    try:
        with open(api_responses_file, 'r', encoding='utf-8') as f:
            all_responses = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading API responses file: {e}")
        return {}
    
    # 过滤成功的响应
    filtered_responses = [
        r for r in all_responses 
        if r.get('context') == 'user_preference_extraction' 
        and r.get('success', False)
    ]
    
    log_with_timestamp(f"✅ Found {len(filtered_responses)} successful responses")
    
    # 按ASIN组织响应
    asin_responses = defaultdict(list)
    
    for idx, response_data in enumerate(filtered_responses):
        try:
            # 仅使用保存的 meta.asin（精确匹配，不依赖 prompt 子串）
            asin = None
            meta = response_data.get('meta') or {}
            meta_asin = meta.get('asin')
            if isinstance(meta_asin, str) and meta_asin.strip():
                asin = meta_asin.strip().upper()
            
            if not asin:
                log_with_timestamp(f"⚠️ Cannot find ASIN for response {idx} (missing meta.asin)")
                continue
            
            # 解析响应内容
            raw_response = response_data.get('raw_response', {})
            content = raw_response.get('content', '')
            
            if not content:
                log_with_timestamp(f"⚠️ Empty content in response {idx}")
                continue
            
            # 解析实体
            try:
                entities_result = process_user_preference_extraction_response(content)
                
                # Handle tuple return format (list, dict)
                if isinstance(entities_result, tuple):
                    entities_list, entities_dict = entities_result
                    entities = entities_dict if entities_dict else entities_list
                else:
                    entities = entities_result
                
                if entities:
                    asin_responses[asin].append({
                        'entities': entities,
                        'title': title,
                        'review_text': review_text
                    })
            except Exception as e:
                log_with_timestamp(f"⚠️ Error parsing entities for response {idx}: {e}")
                continue
                
        except Exception as e:
            log_with_timestamp(f"⚠️ Error processing response {idx}: {e}")
            continue
    
    log_with_timestamp(f"✅ Successfully parsed responses for {len(asin_responses)} products")
    
    # 构建输出数据
    output_data = {
        'user_id': user_id,
        'products': []
    }
    
    # 为每个ASIN合并实体
    for asin, responses in asin_responses.items():
        # 合并所有响应中的实体
        merged_entities = {}
        merged_seen = set()  # (category, entity, sentiment)
        
        for response_info in responses:
            entities = response_info['entities']
            if isinstance(entities, dict):
                for category, entity_list in entities.items():
                    category_norm = normalize_category_label(category)
                    if isinstance(entity_list, list):
                        if category_norm not in merged_entities:
                            merged_entities[category_norm] = []
                        # 添加新实体（去重）
                        for entity in entity_list:
                            # Normalize to {"entity": ..., "sentiment": ...}
                            if isinstance(entity, str):
                                entity_text = entity.strip()
                                if not entity_text:
                                    continue
                                normalized = {"entity": entity_text, "sentiment": "neutral"}
                            elif isinstance(entity, dict):
                                entity_text = str(entity.get("entity") or entity.get("text") or entity.get("name") or "").strip()
                                if not entity_text:
                                    continue
                                sentiment = str(entity.get("sentiment") or entity.get("polarity") or "").strip().lower()
                                if sentiment not in {"positive", "negative", "neutral"}:
                                    sentiment = "neutral"
                                normalized = {"entity": entity_text, "sentiment": sentiment}
                            else:
                                continue

                            dedupe_key = (category_norm, normalized["entity"], normalized["sentiment"])
                            if dedupe_key in merged_seen:
                                continue
                            merged_seen.add(dedupe_key)
                            merged_entities[category_norm].append(normalized)
        
        # 查找对应的产品评论
        product_reviews = []
        for product in user_preference_data.get('products', []):
            if product.get('asin') == asin:
                product_reviews = product.get('review_content', [])
                break
        
        # 添加到输出
        output_data['products'].append({
            'asin': asin,
            'user_preference_entities': merged_entities,
            'review_content': product_reviews
        })
    
    # 保存输出文件（如果指定）
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            log_with_timestamp(f"💾 Saved results to {output_file}")
        except Exception as e:
            log_with_timestamp(f"❌ Error saving output file: {e}")
    
    return output_data

def extract_user_preference_entities(content: str, llm_model, asin: Optional[str] = None) -> str:
    """Extract user preference entities using LLM - only call LLM and return raw response.
    
    Returns:
        Raw response string from LLM (will be parsed later from api_raw_responses.json)
    """
    # Ensure content is not None or empty
    if not content or not isinstance(content, str):
        raise APIErrorException("Invalid content for user preference extraction")

    prompt = f"""Extract all entities mentioned in the following product review text and categorize them.
Include any products, activities, techniques, materials, tools, brands, or other relevant entities that the user mentions, regardless of whether they like them or not.

**实体分类要求:**
对于每个提取的实体，必须将其归类为以下类别之一：
[Brand, Material, Dimensions, Quantity, Color, Design, Usage, Selling Point, Safety/Certification, Accessories, Activity, Technique]

**情感标签要求:**
对每个提取出来的实体，增加一个字段 `sentiment`，表示用户对该实体的态度：
- positive: 积极/喜欢/认可/推荐/夸赞
- negative: 消极/不喜欢/抱怨/批评/不推荐
- neutral: 文本中未体现明确态度或无法判断

**输出格式:**
返回一个JSON对象，其中键是类别名称，值是该类别对应的实体数组。数组内每个元素是一个对象：
{{"entity": "<实体文本>", "sentiment": "positive|negative|neutral"}}

示例:
{{
  "Brand": [{{"entity": "Apple", "sentiment": "positive"}}, {{"entity": "Samsung", "sentiment": "negative"}}],
  "Design": [{{"entity": "smartphone", "sentiment": "neutral"}}, {{"entity": "waterproof", "sentiment": "positive"}}],
  "Selling Point": [{{"entity": "battery life", "sentiment": "positive"}}, {{"entity": "camera quality", "sentiment": "neutral"}}],
  "Usage": [{{"entity": "wireless charging", "sentiment": "positive"}}, {{"entity": "fast charging", "sentiment": "positive"}}],
  "Color": [{{"entity": "black", "sentiment": "neutral"}}, {{"entity": "blue", "sentiment": "neutral"}}]
}}

只返回有效的JSON对象，不要其他解释。

Text: {content}"""

    # Just call LLM and return raw response - no JSON parsing here
    # All responses are saved to api_raw_responses.json, will be parsed later
    response_str, success = call_llm_with_retry(
        llm_model,
        prompt,
        context="user_preference_extraction",
        meta={"asin": asin.strip().upper()} if isinstance(asin, str) and asin.strip() else None,
    )
    if success and response_str:
        return response_str
    else:
        raise APIErrorException("Failed to get response from LLM for user preference extraction")