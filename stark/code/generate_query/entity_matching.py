#!/usr/bin/env python3
"""
实体匹配模块
负责匹配产品实体和用户偏好实体
"""

import os
import json
import sys
import threading
import re
from typing import Any, Dict, List, Union
from datetime import datetime
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry, APIErrorException, ApiProvider
from utils import get_all_api_keys_in_order, create_llm_with_config, try_api_keys_with_fallback

def log_with_timestamp(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def _entity_items_to_texts(items: Any) -> List[str]:
    """
    Normalize entity containers into a list of entity text strings.
    Supports:
      - ["foo", "bar"]
      - [{"entity": "foo", "sentiment": "positive"}, ...]
      - mixed lists
    """
    if items is None:
        return []
    if isinstance(items, str):
        s = items.strip()
        return [s] if s else []
    if not isinstance(items, list):
        return []

    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            s = str(item.get("entity") or item.get("text") or item.get("name") or "").strip()
            if s:
                out.append(s)
    return out

def _normalize_category_label(category: str) -> str:
    """
    Normalize category labels so product/user dict keys match.
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

def _normalize_entity_category_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge keys that normalize to the same category label.
    Values are typically lists of entity items (str or dict), but we handle
    strings/dicts too and avoid nested lists.
    """
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        nk = _normalize_category_label(k)
        if nk not in out:
            out[nk] = v
            continue

        existing = out[nk]
        # Merge list-like values without creating nested lists
        if isinstance(existing, list):
            if isinstance(v, list):
                existing.extend(v)
            else:
                existing.append(v)
            out[nk] = existing
        else:
            if isinstance(v, list):
                out[nk] = [existing, *v]
            else:
                out[nk] = [existing, v]
    return out

def _extract_numbers(text: str) -> List[float]:
    """Extract numeric values (supports simple decimals) from text."""
    if not text:
        return []
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            continue
    return out

def _classify_pack_type(numbers: List[float]) -> str:
    """
    Classify pack type based on numeric quantity.
    - 1 (or <=1.5)          -> Single/Trial
    - 2-10 (inclusive)      -> Multi-pack
    - >10                   -> Bulk
    """
    if not numbers:
        return ""
    max_n = max(numbers)
    if max_n <= 1.5:
        return "Single/Trial"
    if max_n <= 10:
        return "Multi-pack"
    return "Bulk"

def _map_quantity_to_spec(user_entity: str, product_entities: List[str], llm_choice: str) -> (str, str):
    """
    For Quantity category, prefer mapping to concrete spec strings (e.g., '12 Pack')
    instead of returning a bare number. Use LLM choice when meaningful, otherwise
    choose the longest product entity containing the numeric token.
    Returns (spec_string, pack_type_label).
    """
    numeric_tokens = re.findall(r'\d+(?:\.\d+)?', user_entity or "")
    product_entities = [str(pe).strip() for pe in (product_entities or []) if str(pe).strip()]

    candidates = []
    for pe in product_entities:
        if numeric_tokens and any(tok in pe for tok in numeric_tokens):
            candidates.append(pe)

    chosen = llm_choice
    if llm_choice:
        enriched = [c for c in candidates if llm_choice in c or c in llm_choice]
        if enriched:
            chosen = max(enriched, key=len)
        elif llm_choice in product_entities:
            chosen = llm_choice

    if not chosen and candidates:
        chosen = max(candidates, key=len)

    numbers = _extract_numbers(chosen or "") or _extract_numbers(user_entity or "")
    pack_type = _classify_pack_type(numbers)

    return chosen or llm_choice, pack_type

def create_llm_with_config(api_config):
    """Create LLM with config based on provider."""
    from langchain_openai import ChatOpenAI

    provider = api_config.get('provider', 'siliconflow')

    if provider == 'siliconflow':
        return ChatOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_config['api_key'],
            model_name=api_config.get('model', 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'),
            temperature=0.1,
            max_tokens=4000,
        )
    else:
        # Default to OpenAI
        return ChatOpenAI(
            api_key=api_config['api_key'],
            model_name=api_config.get('model', 'gpt-3.5-turbo'),
            temperature=0.1,
            max_tokens=4000,
        )

def try_api_keys_with_fallback(api_keys: List[Dict], operation_func, context: str, success_message: str = None, error_message: str = None):
    """
    通用API key循环重试函数

    Args:
        api_keys: API key配置列表
        operation_func: 要执行的操作函数，参数为(api_config, provider_name, key_index)
        context: 上下文信息，用于日志
        success_message: 成功时的日志消息模板
        error_message: 错误时的日志消息模板

    Returns:
        (result, success) 元组，result是操作结果，success表示是否成功
    """
    for key_index, api_config in enumerate(api_keys):
        provider_name = "SiliconFlow" if api_config['provider'] == ApiProvider.SILICONFLOW else "Unknown"
        try:
            result = operation_func(api_config, provider_name, key_index)

            # 成功处理
            # if success_message:
            #     log_with_timestamp(success_message.format(
            #         context=context,
            #         provider=provider_name,
            #         key_num=api_config['key_index'] + 1,
            #     ))
            return result, True
        except APIErrorException as e:
            # API错误，继续下一个key
            if error_message:
                log_with_timestamp(error_message.format(
                    context=context,
                    provider=provider_name,
                    key_num=api_config['key_index'] + 1,
                    error=str(e)
                ))
            continue
        except Exception as e:
            # 其他错误，继续下一个key
            log_with_timestamp(f"❌ Unexpected error with {provider_name} Key #{api_config['key_index'] + 1}: {e}")
            continue

    # 所有key都失败了
    return None, False


def process_entity_matching_response(response_str: str) -> List[str]:
    """
    处理实体匹配的LLM响应

    Args:
        response_str: LLM返回的原始字符串

    Returns:
        处理后的实体列表

    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    # Debug: print raw response
    print(f"🔍 Entity matching raw response (first 500 chars): {response_str[:500]!r}", flush=True)

    # Check for markdown code blocks
    if response_str.startswith('```') and '```' in response_str:
        print("📦 Found markdown code block, extracting JSON...", flush=True)

    if not response_str:
        raise APIErrorException("No response from entity matching")

    try:
        # Clean the response
        response_str = response_str.strip()

        # Smart JSON extraction for Chain of Thought responses
        lines = response_str.strip().split('\n')
        json_found = False

        # Check if the last few lines contain valid JSON
        for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):  # Check last 5 lines
            line = lines[i].strip()
            if line.startswith('[') and line.endswith(']'):
                # Found JSON array at the end
                response_str = line
                json_found = True
                break
            elif line.startswith('{') and line.endswith('}'):
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
                elif json_blocks:
                    response_str = json_blocks[-1]  # Use the last json block

        # Try to parse as JSON
        print(f"🔄 Attempting to parse JSON: {response_str[:200]!r}", flush=True)
        result = json.loads(response_str)
        print(f"✅ JSON parsed successfully: {result}", flush=True)

        # Handle different response formats
        if isinstance(result, list):
            # Array format - expected for entity matching
            flattened = []
            for item in result:
                if isinstance(item, str):
                    flattened.append(item)
                elif isinstance(item, list):
                    # Flatten nested list but only take string elements
                    for subitem in item:
                        if isinstance(subitem, str):
                            flattened.append(subitem)

            if flattened:
                return flattened
            else:
                raise APIErrorException("No valid entities extracted from entity matching (empty result)")

        elif isinstance(result, dict):
            # If somehow returns dict, try to extract matched entities
            flattened = []
            possible_keys = ["matched_entities", "matches", "results"]
            for key in possible_keys:
                if key in result and isinstance(result[key], list):
                    for item in result[key]:
                        if isinstance(item, str) and item.strip():
                            flattened.append(item.strip())

            if flattened:
                return flattened
            else:
                raise APIErrorException("No valid entities extracted from entity matching (empty result)")

        else:
            raise APIErrorException("Invalid result format from entity matching")

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in entity matching: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in entity matching")
    except Exception as e:
        print(f"Unexpected error processing entity matching response: {e}", flush=True)
        raise APIErrorException("Response processing failed in entity matching")

def process_entity_matching_dict_response(response_str: str) -> Dict[str, List[str]]:
    """
    Parse per-product entity matching response.
    Expected output: JSON object mapping category -> list of matched product entity strings.
    """
    if not response_str:
        raise APIErrorException("No response from entity matching")

    try:
        s = response_str.strip()

        # Extract last JSON object if chain-of-thought present
        lines = s.split("\n")
        for i in range(len(lines) - 1, max(-1, len(lines) - 10), -1):
            line = lines[i].strip()
            if line.startswith("{") and line.endswith("}"):
                s = line
                break

        # Handle code blocks
        if "```" in s:
            # Prefer last ```json block
            json_start = s.rfind("```json")
            if json_start != -1:
                json_end = s.find("```", json_start + 7)
                if json_end != -1:
                    content_start = s.find("\n", json_start) + 1
                    s = s[content_start:json_end].strip()
            else:
                last_triple = s.rfind("```")
                first_triple = s.rfind("```", 0, last_triple)
                if first_triple != -1 and last_triple != -1 and first_triple != last_triple:
                    content_start = s.find("\n", first_triple) + 1
                    s = s[content_start:last_triple].strip()

        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise APIErrorException("Invalid result format from entity matching (expected dict)")

        out: Dict[str, List[str]] = {}
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, str):
                vv = v.strip()
                if vv:
                    out[str(k)] = [vv]
                continue
            if isinstance(v, list):
                cleaned = []
                for item in v:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip())
                if cleaned:
                    out[str(k)] = cleaned
        return out
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in entity matching dict: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in entity matching dict")
    except APIErrorException:
        raise
    except Exception as e:
        print(f"Unexpected error processing entity matching dict response: {e}", flush=True)
        raise APIErrorException("Response processing failed in entity matching dict")

def process_entity_matching_dict_response_with_sentiment(response_str: str) -> Dict[str, List[Any]]:
    """
    Parse per-product entity matching response with sentiment information.
    Expected output: JSON object mapping category -> list of matched product entity objects with sentiment.
    
    Each entry can be:
    - {"entity": "...", "sentiment": "positive|negative|neutral"}
    - Or a plain string (for pack_type and other non-sentiment fields)
    
    Args:
        response_str: LLM返回的原始字符串
        
    Returns:
        Dict[str, List[Any]]: 处理后的实体字典，包含sentiment信息
        
    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    if not response_str:
        raise APIErrorException("No response from entity matching")

    try:
        s = response_str.strip()

        # Extract last JSON object if chain-of-thought present
        lines = s.split("\n")
        for i in range(len(lines) - 1, max(-1, len(lines) - 10), -1):
            line = lines[i].strip()
            if line.startswith("{") and line.endswith("}"):
                s = line
                break

        # Handle code blocks
        if "```" in s:
            # Prefer last ```json block
            json_start = s.rfind("```json")
            if json_start != -1:
                json_end = s.find("```", json_start + 7)
                if json_end != -1:
                    content_start = s.find("\n", json_start) + 1
                    s = s[content_start:json_end].strip()
            else:
                last_triple = s.rfind("```")
                first_triple = s.rfind("```", 0, last_triple)
                if first_triple != -1 and last_triple != -1 and first_triple != last_triple:
                    content_start = s.find("\n", first_triple) + 1
                    s = s[content_start:last_triple].strip()

        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise APIErrorException("Invalid result format from entity matching (expected dict)")

        out: Dict[str, List[Any]] = {}
        for k, v in obj.items():
            if v is None:
                continue
            
            key = str(k)
            
            # Handle pack_type - also use entity+sentiment format for consistency
            if key == "pack_type":
                if isinstance(v, str):
                    vv = v.strip()
                    if vv:
                        out[key] = [{"entity": vv, "sentiment": "neutral"}]
                elif isinstance(v, list):
                    cleaned = []
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            cleaned.append({"entity": item.strip(), "sentiment": "neutral"})
                        elif isinstance(item, dict):
                            entity_val = str(item.get("entity", "")).strip()
                            sentiment_val = str(item.get("sentiment", "neutral")).strip().lower()
                            if sentiment_val not in {"positive", "negative", "neutral"}:
                                sentiment_val = "neutral"
                            if entity_val:
                                cleaned.append({"entity": entity_val, "sentiment": sentiment_val})
                    if cleaned:
                        out[key] = cleaned
                continue
            
            # For other categories, handle entity+sentiment objects
            if isinstance(v, str):
                # Plain string - wrap as object with neutral sentiment
                vv = v.strip()
                if vv:
                    out[key] = [{"entity": vv, "sentiment": "neutral"}]
                continue
                
            if isinstance(v, dict):
                # Single dict object
                entity_val = str(v.get("entity", "")).strip()
                sentiment_val = str(v.get("sentiment", "neutral")).strip().lower()
                if sentiment_val not in {"positive", "negative", "neutral"}:
                    sentiment_val = "neutral"
                if entity_val:
                    out[key] = [{"entity": entity_val, "sentiment": sentiment_val}]
                continue
                
            if isinstance(v, list):
                cleaned = []
                for item in v:
                    if isinstance(item, str) and item.strip():
                        item_str = item.strip()
                        # Check if string is a stringified dict (e.g., "{'entity': 'DMC', 'sentiment': 'positive'}")
                        if (item_str.startswith("{") and item_str.endswith("}") and 
                            ("'entity'" in item_str or '"entity"' in item_str)):
                            try:
                                # Try to parse as Python dict literal or JSON
                                import ast
                                parsed = ast.literal_eval(item_str)
                                if isinstance(parsed, dict):
                                    entity_val = str(parsed.get("entity", "")).strip()
                                    sentiment_val = str(parsed.get("sentiment", "neutral")).strip().lower()
                                    if sentiment_val not in {"positive", "negative", "neutral"}:
                                        sentiment_val = "neutral"
                                    if entity_val:
                                        cleaned.append({"entity": entity_val, "sentiment": sentiment_val})
                                    continue
                            except (ValueError, SyntaxError):
                                pass
                        # Plain string - wrap as object with neutral sentiment
                        cleaned.append({"entity": item_str, "sentiment": "neutral"})
                    elif isinstance(item, dict):
                        # Object with entity and sentiment
                        entity_val = str(item.get("entity", "")).strip()
                        # Check if entity_val itself is a stringified dict
                        if (entity_val.startswith("{") and entity_val.endswith("}") and
                            ("'entity'" in entity_val or '"entity"' in entity_val)):
                            try:
                                import ast
                                parsed = ast.literal_eval(entity_val)
                                if isinstance(parsed, dict):
                                    entity_val = str(parsed.get("entity", "")).strip()
                                    # Use sentiment from parsed dict if available, otherwise from outer dict
                                    sentiment_val = str(parsed.get("sentiment", item.get("sentiment", "neutral"))).strip().lower()
                                    if sentiment_val not in {"positive", "negative", "neutral"}:
                                        sentiment_val = "neutral"
                                    if entity_val:
                                        cleaned.append({"entity": entity_val, "sentiment": sentiment_val})
                                    continue
                            except (ValueError, SyntaxError):
                                pass
                        sentiment_val = str(item.get("sentiment", "neutral")).strip().lower()
                        if sentiment_val not in {"positive", "negative", "neutral"}:
                            sentiment_val = "neutral"
                        if entity_val:
                            cleaned.append({"entity": entity_val, "sentiment": sentiment_val})
                if cleaned:
                    out[key] = cleaned
                    
        return out
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in entity matching dict with sentiment: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in entity matching dict with sentiment")
    except APIErrorException:
        raise
    except Exception as e:
        print(f"Unexpected error processing entity matching dict response with sentiment: {e}", flush=True)
        raise APIErrorException("Response processing failed in entity matching dict with sentiment")


def match_product_and_user_entities_no_llm(
    product_entities: Union[Dict[str, Any], List[Any]],
    user_entities: Union[Dict[str, Any], List[Any]],
    llm_model,
) -> Dict[str, List[str]]:
    """
    使用LLM进行实体匹配：对每个用户偏好实体，在相同类别的产品实体中找到相似度最大的实体

    Args:
        product_entities: 商品实体字典 {category: [entities]}
        user_entities: 用户偏好实体字典 {category: [entities]}
        llm_model: LLM模型用于计算相似度

    Returns:
        匹配的实体字典 {category: [matched_entities]}
    """
    matched_entities = {}

    # Backward/forward compatible normalization:
    # - product_entities/user_entities can be dict{category: [..]} or list[..]
    if isinstance(product_entities, list):
        product_entities = {"General": product_entities}
    if isinstance(user_entities, list):
        user_entities = {"General": user_entities}
    if isinstance(product_entities, dict):
        product_entities = _normalize_entity_category_dict(product_entities)
    if isinstance(user_entities, dict):
        user_entities = _normalize_entity_category_dict(user_entities)

    # 遍历用户偏好实体的所有类别
    for user_category, user_entity_list in (user_entities or {}).items():
        # 如果产品实体中也存在这个类别
        if isinstance(product_entities, dict) and user_category in product_entities:
            product_entity_list = product_entities[user_category]
            matched_in_category = []

            # 对每个用户偏好实体，在产品实体中找到最相似的
            matched_product_entities = set()  # 用于去重匹配的产品实体
            pack_types: List[str] = matched_entities.get("pack_type", [])
            user_texts = _entity_items_to_texts(user_entity_list)
            product_texts = _entity_items_to_texts(product_entity_list)
            for user_entity in user_texts:
                best_match = find_most_similar_entity_with_llm(user_entity, product_texts, llm_model)
                if user_category == "Quantity":
                    best_match, pack_type = _map_quantity_to_spec(user_entity, product_texts, best_match)
                    if pack_type and pack_type not in pack_types:
                        pack_types.append(pack_type)
                if best_match and best_match not in matched_product_entities:
                    matched_in_category.append(best_match)
                    matched_product_entities.add(best_match)

            if pack_types:
                matched_entities["pack_type"] = pack_types

            if matched_in_category:
                matched_entities[user_category] = matched_in_category

    return matched_entities

def match_product_and_user_entities_one_call(
    product_entities: Union[Dict[str, Any], List[Any]],
    user_entities: Union[Dict[str, Any], List[Any]],
    llm_model,
) -> Dict[str, List[Any]]:
    """
    Single LLM call per product.
    Returns: {category: [{"entity": "...", "sentiment": "positive|negative|neutral"}], "pack_type": ["Single/Trial"|"Multi-pack"|"Bulk"]}
    """
    # Normalize inputs
    if isinstance(product_entities, list):
        product_entities = {"General": product_entities}
    if isinstance(user_entities, list):
        user_entities = {"General": user_entities}
    if isinstance(product_entities, dict):
        product_entities = _normalize_entity_category_dict(product_entities)
    if isinstance(user_entities, dict):
        user_entities = _normalize_entity_category_dict(user_entities)

    prompt = f"""
You are an expert at matching product features with user preferences.

Given two JSON objects:
- Product entities (category -> list of entity strings): {json.dumps(product_entities, ensure_ascii=False)}
- User preference entities (category -> list of objects with "entity" and "sentiment"): {json.dumps(user_entities, ensure_ascii=False)}

Task:
1) For each category, select the BEST matching product entity/entities that satisfy the user preference entities.
2) IMPORTANT: Each matched entity MUST include the sentiment from the corresponding user preference entity.
3) Output MUST be a JSON object where:
   - Keys are categories
   - Values are arrays of objects with "entity" (the matched PRODUCT entity string) and "sentiment" (from user preference: "positive", "negative", or "neutral")

Example output format:
{{
  "Color": [{{"entity": "Blue", "sentiment": "positive"}}],
  "Material": [{{"entity": "Cotton", "sentiment": "negative"}}],
  "Quantity": [{{"entity": "12", "sentiment": "positive"}}]
}}

If a category has no good match, omit the key or use an empty array.

Output requirement:
Return ONLY valid JSON. No explanations.
"""

    response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_matching_product")
    if not success or not response_str:
        raise APIErrorException("No response from per-product entity matching")

    matched = process_entity_matching_dict_response_with_sentiment(response_str)

    # Drop empty arrays
    if isinstance(matched, dict):
        matched = {k: v for k, v in matched.items() if isinstance(v, list) and len(v) > 0}

    return matched

def find_most_similar_entity_with_llm(user_entity: str, product_entities: List[str], llm_model) -> str:
    """
    使用LLM在产品实体列表中找到与用户实体最相似的实体

    Args:
        user_entity: 用户偏好实体
        product_entities: 产品实体列表
        llm_model: LLM模型

    Returns:
        最相似的产品实体，如果没有找到则返回空字符串
    """
    if not product_entities:
        return ""

    # 如果只有一个产品实体，直接返回
    if len(product_entities) == 1:
        return product_entities[0]

    prompt = f"""
You are an expert at finding semantic similarity between product features.

Given:
- User preference entity: "{user_entity}"
- Product entities to compare: {product_entities}

Find the product entity that is most semantically similar to the user preference entity.
Consider synonyms, related concepts, and contextual similarity.

**OUTPUT REQUIREMENT:**
Return ONLY the most similar product entity as a JSON string. No explanations.

Example:
- User: "24 colors" → Product entities: ["24", "12", "36"] → Output: "24"
- User: "waterproof" → Product entities: ["water resistant", "durable", "lightweight"] → Output: "water resistant"

Output format:
"most_similar_entity"
"""

    # Retry up to 3 times
    for attempt in range(3):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_similarity")
            if success and response_str:
                # 尝试解析JSON字符串
                try:
                    # 移除可能的引号包装
                    if response_str.startswith('"') and response_str.endswith('"'):
                        result = response_str[1:-1]
                    else:
                        result = response_str.strip()

                    # 检查结果是否在产品实体列表中
                    if result in product_entities:
                        return result
                    else:
                        # 如果不在列表中，尝试找到最相似的
                        for product_entity in product_entities:
                            if result.lower() in product_entity.lower() or product_entity.lower() in result.lower():
                                return product_entity

                except Exception as e:
                    print(f"Error parsing LLM response for similarity: {e}", flush=True)

        except Exception as e:
            print(f"LLM error in entity similarity: {e}", flush=True)
            if attempt < 2:  # 不是最后一次尝试
                continue

    # 如果LLM失败，返回空字符串
    return ""

def match_product_and_user_entities(product_entities: List[str], user_entities: List[str], llm_model) -> List[str]:
    """使用LLM匹配产品实体和用户偏好实体，找出匹配的实体"""
    if not product_entities or not user_entities:
        return []

    # 简化的实体匹配prompt，直接要求JSON输出
    prompt = f"""
You are an expert at matching product features with user preferences.

Given:
- Product Entities: {product_entities}
- User Preferences: {user_entities}

Find entities that appear in both lists OR are semantically equivalent (synonyms or closely related).

Special handling for dimensional specs:
- When a numeric dimension actually represents a standardized size for bedding/furniture (e.g., bed sheets, mattresses, beds, sofas), map it to the standard size label instead of returning the raw number/range.
- Examples: 200cm, 203cm → "Queen Size"; 220cm, 203x193cm → "King Size"; map to the semantic spec like "Queen Size" or "King Size" (or the locale-equivalent label) rather than "200-220cm".
- If the dimension refers to fit/size (e.g., bed sheet fit), return the standard size term (e.g., "fit_size: King Size") not the numeric span.
- Prefer semantic size words over numeric ranges whenever a standard size category exists.

**OUTPUT REQUIREMENT:**
Return ONLY a JSON array of matched entities. No explanations.

Examples:
- If "color" appears in both lists → ["color"]
- If "size" in products and "dimensions" in user preferences → ["size"]
- If user mentions 200cm bed sheet and product has "King Size" → ["King Size"]
- If no matches → []

```json

```json
[]
```

Begin your analysis now.
"""

    # Retry up to 5 times for JSON parsing errors in matching
    json_parse_retries = 5
    for attempt in range(json_parse_retries):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_matching")
            if success and response_str:
                entities = process_entity_matching_response(response_str)

                # Filter to ensure only strings and remove duplicates (specific to matching)
                matched_entities = []
                for item in entities:
                    if isinstance(item, str) and item.strip():
                        clean_item = item.strip()
                        if clean_item not in matched_entities:
                            matched_entities.append(clean_item)

                return matched_entities
        except APIErrorException as e:
            # Check if this is a JSON parsing error
            error_msg = str(e)
            if "JSON parsing failed" in error_msg or "JSON parsing error" in error_msg:
                if attempt < json_parse_retries - 1:
                    print(f"JSON parsing failed in matching (attempt {attempt + 1}/{json_parse_retries}), retrying...", flush=True)
                    continue
                else:
                    print(f"JSON parsing failed in matching after {json_parse_retries} attempts", flush=True)
            # For matching, we return empty list on error instead of raising
            return []
        except Exception as e:
            print(f"LLM error in entity matching: {e}", flush=True)
            raise  # 重新抛出异常，让API key循环处理

    return []


def perform_entity_matching(products: List[Dict], max_workers: int = 102) -> List[Dict]:
    """执行产品实体和用户偏好实体的匹配（并发版本）"""
    log_with_timestamp(f"🔗 Starting entity matching for {len(products)} products with {max_workers} workers...")

    if not products:
        log_with_timestamp("⚠️ No products found for matching")
        return products

    # 获取API keys用于LLM匹配
    all_api_keys = get_all_api_keys_in_order()

    total_products = len(products)
    matched_count = 0
    
    # 线程安全的计数器和锁
    progress_counter = {'completed': 0, 'matched': 0}
    progress_lock = threading.Lock()

    def process_single_product(product_with_idx):
        """处理单个产品的实体匹配"""
        idx, product = product_with_idx
        asin = product.get('asin', 'Unknown')
        
        try:
            product_entities = product.get('product_entities', {})
            user_entities = product.get('user_preference_entities', {})

            # 使用LLM进行实体相似度匹配
            def matching_operation(api_config, provider_name, key_index):
                llm_model = create_llm_with_config(api_config)
                return match_product_and_user_entities_one_call(product_entities, user_entities, llm_model)

            matched_entities, success = try_api_keys_with_fallback(
                all_api_keys,
                matching_operation,
                f"{asin} entity matching"
            )

            if not success:
                matched_entities = {}

            # Process matched_entities to ensure consistent format
            # matched_entities from LLM: {category: [{"entity": "...", "sentiment": "..."}] or [strings]}
            # user_entities format: {category: [{"entity": "...", "sentiment": "..."}] or [strings]}
            enriched_matched = {}
            for category, matched_values in matched_entities.items():
                if not isinstance(matched_values, list):
                    continue
                enriched_list = []
                user_cat_entities = user_entities.get(category, [])
                
                # Build a map from user entity value to sentiment
                user_sentiment_map = {}
                for user_item in user_cat_entities:
                    if isinstance(user_item, dict):
                        entity_val = str(user_item.get("entity", "")).strip()
                        sentiment_val = str(user_item.get("sentiment", "neutral")).strip().lower()
                        if entity_val:
                            user_sentiment_map[entity_val.lower()] = sentiment_val
                    elif isinstance(user_item, str):
                        entity_val = str(user_item).strip()
                        if entity_val:
                            user_sentiment_map[entity_val.lower()] = "neutral"
                
                # For each matched product entity, ensure dict format with sentiment
                for matched_val in matched_values:
                    # Handle already dict format (from LLM response)
                    if isinstance(matched_val, dict):
                        entity_str = str(matched_val.get("entity", "")).strip()
                        sentiment_str = str(matched_val.get("sentiment", "")).strip().lower()
                        if not entity_str:
                            continue
                        # Validate sentiment
                        if sentiment_str not in {"positive", "negative", "neutral"}:
                            # Try to find sentiment from user_preference_entities
                            entity_lower = entity_str.lower()
                            if entity_lower in user_sentiment_map:
                                sentiment_str = user_sentiment_map[entity_lower]
                            else:
                                # Substring match
                                for user_val, user_sent in user_sentiment_map.items():
                                    if entity_lower in user_val or user_val in entity_lower:
                                        sentiment_str = user_sent
                                        break
                                else:
                                    sentiment_str = "neutral"
                        enriched_list.append({
                            "entity": entity_str,
                            "sentiment": sentiment_str
                        })
                    elif isinstance(matched_val, str):
                        # Handle string format (legacy)
                        matched_str = matched_val.strip()
                        if not matched_str:
                            continue
                        
                        # Try to find sentiment from user_preference_entities
                        sentiment = "neutral"
                        matched_lower = matched_str.lower()
                        
                        # Exact match
                        if matched_lower in user_sentiment_map:
                            sentiment = user_sentiment_map[matched_lower]
                        else:
                            # Case-insensitive match
                            for user_val, user_sent in user_sentiment_map.items():
                                if user_val == matched_lower:
                                    sentiment = user_sent
                                    break
                            # Substring match
                            if sentiment == "neutral":
                                for user_val, user_sent in user_sentiment_map.items():
                                    if matched_lower in user_val or user_val in matched_lower:
                                        sentiment = user_sent
                                        break
                        
                        # Always store as dict format
                        enriched_list.append({
                            "entity": matched_str,
                            "sentiment": sentiment
                        })
                
                if enriched_list:
                    enriched_matched[category] = enriched_list
            
            # Add pack_type if present (also use dict format for consistency)
            if "pack_type" in matched_entities:
                pack_values = matched_entities["pack_type"]
                if isinstance(pack_values, list):
                    pack_list = []
                    for pv in pack_values:
                        if isinstance(pv, dict):
                            pack_list.append(pv)
                        elif isinstance(pv, str) and pv.strip():
                            pack_list.append({"entity": pv.strip(), "sentiment": "neutral"})
                    if pack_list:
                        enriched_matched["pack_type"] = pack_list

            # 检查是否有匹配的实体
            has_matches = any(matches for matches in enriched_matched.values() if isinstance(matches, list) and len(matches) > 0)

            # 添加匹配结果到产品数据（包含sentiment）
            product['matched_entities'] = enriched_matched

            # 生成格式化的输出字符串
            formatted_output = generate_formatted_product_output(product, idx, total_products)
            product['formatted_output'] = formatted_output

            # 更新进度（线程安全）
            with progress_lock:
                progress_counter['completed'] += 1
                if has_matches:
                    progress_counter['matched'] += 1
                current_count = progress_counter['completed']
            # 每处理10个产品或最后一批时输出进度
            if current_count % 10 == 0 or current_count == total_products:
                log_with_timestamp(f'📊 Entity matching progress: {current_count}/{total_products} products processed')

            return product

        except Exception as e:
            log_with_timestamp(f'❌ Exception in entity matching for {asin}: {e}')
            product['matched_entities'] = {}
            product['formatted_output'] = generate_formatted_product_output(product, idx, total_products)

            # 更新进度（线程安全）
            with progress_lock:
                progress_counter['completed'] += 1
                current_count = progress_counter['completed']
                if current_count % 10 == 0 or current_count == total_products:
                    log_with_timestamp(f'📊 Entity matching progress: {current_count}/{total_products} products processed')
            
            return product

    # 使用ThreadPoolExecutor并发处理
    products_with_idx = [(idx, product) for idx, product in enumerate(products)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_product = {executor.submit(process_single_product, p): p for p in products_with_idx}
        
        # 收集结果（保持原始顺序）
        results = [None] * total_products
        for future in concurrent.futures.as_completed(future_to_product):
            product_with_idx = future_to_product[future]
            try:
                result = future.result()
                idx = product_with_idx[0]
                results[idx] = result
            except Exception as e:
                idx = product_with_idx[0]
                product = product_with_idx[1]
                asin = product.get('asin', 'Unknown')
                log_with_timestamp(f'❌ Exception processing {asin}: {e}')
                results[idx] = product

    matched_count = progress_counter['matched']
    log_with_timestamp(f'✅ Entity matching completed! {matched_count}/{total_products} products have matched entities')
    return results



def generate_formatted_product_output(product, idx, total_products):
    """生成格式化的产品输出字符串"""
    asin = product.get('asin', 'Unknown')
    product_title = product.get('product_title', 'Unknown Product')
    product_entities = product.get('product_entities', [])
    user_entities = product.get('user_preference_entities', [])
    matched_entities = product.get('matched_entities', [])

    output_lines = [
        f"[{idx+1}/{total_products}] Product: {product_title}",
        f"ASIN: {asin}",
        f"Product Entities ({len(product_entities)}): {', '.join(product_entities) if product_entities else 'None'}",
        f"User Preference Entities ({len(user_entities)}): {', '.join(user_entities) if user_entities else 'None'}",
        f"Matched Entities ({len(matched_entities)}): {', '.join(matched_entities) if matched_entities else 'None'}",
        ""
    ]

    return "\n".join(output_lines)