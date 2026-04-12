#!/usr/bin/env python3
"""
Stage 9 (V2): ACL-based Noisy Query Generation

基于用户ACL错误的噪声查询生成：
1. 读取用户的ACL查询（来自acl_query.json）
2. 读取用户的错误（来自acl_error.json）
3. 如果用户有错误，调用LLM将错误注入到查询中
4. 输出noisy_acl_query.json

Input:
  - /home/wlia0047/ar57/wenyu/result/personal_query/06_query/acl_query.json
  - /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/acl_error.json

Output:
  - /home/wlia0047/ar57/wenyu/result/personal_query/07_targeted_noisy_query/noisy_acl_query.json

Usage:
  python 07_generate_acl_noisy_queries.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


# ============================================================================
# 硬编码参数
# ============================================================================

ACL_QUERY_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/acl_query.json"
ACL_ERROR_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/acl_error.json"
ACL_PROFILES_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/acl_user_profiles.json"
OUTPUT_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/07_targeted_noisy_query/noisy_acl_query.json"
MAX_WORKERS = 30
MAX_USERS = 10


# ============================================================================
# Prompt Template
# ============================================================================

def build_noisy_prompt(query: str, error_patterns: List[Dict], protected_keywords: List[str]) -> str:
    """构建噪声查询prompt"""
    pattern_str = ""
    for i, ep in enumerate(error_patterns[:10]):
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        etype = ep.get("error_type", "unknown")
        pattern_str += f"{i+1}. [{etype}] user writes: '{orig}' but correct is: '{corr}'\n"

    if not pattern_str:
        pattern_str = "No specific error patterns available."

    keywords_str = ", ".join([f"'{k}'" for k in protected_keywords]) if protected_keywords else "none"

    prompt = f"""You are an expert at injecting realistic spelling/typing errors into text, mimicking how real users make mistakes when searching.

You will receive:
1. A user's original search query (which is grammatically correct)
2. The user's typical error patterns (original=what user writes, corrected=correct form)
3. PROTECTED KEYWORDS that should NEVER be replaced

CRITICAL RULES:
- ONLY inject errors that are EXACTLY the same type as the user's error patterns
- Do NOT invent your own errors (e.g., which->that, good->nice are NOT allowed)
- NEVER replace: which, that, good, nice, provide, offer, and, or, from, of (these are not user errors)
- NEVER replace sentence starters or structural phrases (e.g., "I need", "I am", "I want", "looking for", "searching for")
- The replacement MUST preserve the sentence's grammatical structure and meaning

INJECTION STRATEGY (follow this order):
1. FIRST: Check if the query contains the "corrected" form from any error pattern. If yes, replace it with the "original" (error) form.
   - Example: Error pattern says user writes "high quality" (original) instead of "high-quality" (corrected)
     If query contains "high-quality" → replace with "high quality" ✓
   - Example: Error pattern says user writes "excelent" (original) instead of "excellent" (corrected)
     If query contains "excellent" → replace with "excelent" ✓

2. ONLY IF no direct match found: You may replace a word in the query with the user's error form (original), BUT ONLY IF:
   - The word being replaced and the error form (original) have the SAME part of speech AND similar semantic function
   - For compound_variant: only replace another compound modifier (e.g., "well-made" → "long lasting" is OK if both are adjective modifiers)
   - For spelling errors: only replace a word that is similar to the misspelled word
   - NEVER replace verbs with adjectives, or sentence openers with random words
   - If no semantically compatible replacement exists, return the ORIGINAL query unchanged with empty injected_errors

3. If no suitable injection can be made, return the original query as-is with an empty injected_errors list.

BAD EXAMPLES (DO NOT DO THIS):
- Replacing "I need" with "long lasting" (verb phrase → adjective modifier, breaks sentence structure) ✗
- Replacing "I am searching for" with "high quality" (sentence structure → adjective) ✗
- Replacing "which" with "that" (not a user error) ✗

GOOD EXAMPLES:
- Replacing "high-quality" with "high quality" (compound_variant, direct match) ✓
- Replacing "excellent" with "excelent" (spelling, direct match) ✓
- Replacing "well-made" with "long lasting" (compound_variant, both are adjective modifiers describing product quality) ✓

Protected keywords (A1-A5 product attributes): NEVER replace unless the error is a direct typo of that exact word.

Return JSON format:
{{
  "original_query": "<the original query>",
  "noisy_query": "<query with injected errors>",
  "injected_errors": [{{"replaced_word": "word that was replaced", "injected_error": "error form injected"}}]
}}

Original query to inject errors into:
{query}

Protected keywords (NEVER replace):
{keywords_str}

User's typical error patterns (ONLY use these for injection):
{pattern_str}

Inject ONLY user errors from the patterns above:
"""
    return prompt


def call_noisy_llm(query: str, error_patterns: List[Dict], protected_keywords: List[str]) -> Dict:
    """调用LLM生成噪声查询，支持速率限制重试"""
    import time
    from llm_client import ZAIAnthropicClient

    prompt = build_noisy_prompt(query, error_patterns, protected_keywords)

    wait_seconds = 5
    max_retries = 20

    for attempt in range(max_retries):
        try:
            client = ZAIAnthropicClient(model='glm-5')
            thinking, text = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.3)
            result_text = text.strip()

            if not result_text:
                return {
                    "original_query": query,
                    "noisy_query": query,
                    "injected_errors": []
                }

            try:
                if "```json" in result_text:
                    start = result_text.find("```json") + 7
                    end = result_text.find("```", start)
                    result_text = result_text[start:end].strip()
                elif "```" in result_text:
                    start = result_text.find("```") + 3
                    end = result_text.find("```", start)
                    result_text = result_text[start:end].strip()

                result = json.loads(result_text)
                return {
                    "original_query": query,
                    "noisy_query": result.get("noisy_query", query),
                    "injected_errors": result.get("injected_errors", [])
                }

            except json.JSONDecodeError:
                return {
                    "original_query": query,
                    "noisy_query": query,
                    "injected_errors": []
                }

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str or 'too many request' in error_str
            if not is_rate_limit:
                return {
                    "original_query": query,
                    "noisy_query": query,
                    "injected_errors": [],
                    "error": str(e)
                }
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit, retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                wait_seconds += 5
            else:
                return {
                    "original_query": query,
                    "noisy_query": query,
                    "injected_errors": [],
                    "error": str(e)
                }


# ============================================================================
# 数据加载
# ============================================================================

def load_user_errors(error_file: str) -> Dict[str, List[Dict]]:
    """加载用户错误数据"""
    with open(error_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_errors = {}
    for user in data.get('user_results', []):
        uid = user['user_id']
        if user['total_errors'] > 0 and user.get('detailed_results'):
            # 收集所有错误模式
            patterns = []
            for detail in user['detailed_results']:
                for err in detail.get('errors', []):
                    patterns.append({
                        "original": err.get('original', ''),
                        "corrected": err.get('corrected', ''),
                        "error_type": err.get('error_type', 'unknown'),
                        "span_text": detail.get('span_text', ''),
                        "region_type": detail.get('region_type', 'unknown')
                    })
            if patterns:
                user_errors[uid] = patterns

    logger.info(f"Loaded {len(user_errors)} users with errors")
    return user_errors


def load_user_attributes(profile_file: str) -> Dict[str, Dict[str, str]]:
    """加载用户的A1-A5属性关键词"""
    with open(profile_file, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    user_attrs = {}
    for profile in profiles:
        uid = profile.get('user_id')
        products = profile.get('products', [])
        if not products:
            continue
        # 收集该用户所有产品的属性
        for prod in products:
            asin = prod.get('asin')
            key = f"{uid}_{asin}"
            attrs = []
            # A1: product_type
            if prod.get('A1_product_type'):
                attrs.append(str(prod['A1_product_type']))
            # A2: brand
            if prod.get('A2_brand'):
                attrs.append(str(prod['A2_brand']))
            # A3: price
            if prod.get('A3_price'):
                attrs.append(str(prod['A3_price']))
            # A4: appearance (list)
            if prod.get('A4_appearance'):
                for a in prod['A4_appearance']:
                    attrs.append(str(a))
            # A5: use_case
            if prod.get('A5_use_case'):
                attrs.append(str(prod['A5_use_case']))
            if attrs:
                user_attrs[key] = attrs

    logger.info(f"Loaded attributes for {len(user_attrs)} user-product pairs")
    return user_attrs


def load_user_queries(query_file: str) -> Dict[str, List[Dict]]:
    """加载用户查询数据"""
    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_queries = {}
    for entry in data:
        uid = entry.get('user_id')
        asin = entry.get('asin')
        queries = entry.get('queries', [])

        if uid not in user_queries:
            user_queries[uid] = []

        for q in queries:
            user_queries[uid].append({
                "asin": asin,
                "acl_index": q.get('acl', 0),
                "filled_query": q.get('filled_query', ''),
                "word_count": q.get('word_count', 0)
            })

    logger.info(f"Loaded queries for {len(user_queries)} users")
    return user_queries


# ============================================================================
# 主处理逻辑
# ============================================================================

def process_single_user(user_id: str, queries: List[Dict], error_patterns: List[Dict], user_attributes: Dict[str, Dict[str, str]]) -> Dict:
    """为单个用户生成噪声查询"""
    results = []

    for q in queries:
        original_query = q['filled_query']
        asin = q.get('asin', '')
        if not original_query:
            continue

        # 获取该用户该产品的受保护关键词
        key = f"{user_id}_{asin}"
        protected_keywords = user_attributes.get(key, [])

        llm_result = call_noisy_llm(original_query, error_patterns, protected_keywords)

        results.append({
            "user_id": user_id,
            "asin": asin,
            "acl_index": q['acl_index'],
            "original_query": original_query,
            "noisy_query": llm_result['noisy_query'],
            "injected_errors": llm_result.get('injected_errors', [])
        })

    return {
        "user_id": user_id,
        "queries_generated": len(results),
        "results": results
    }


# ============================================================================
# Main
# ============================================================================

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    log_with_timestamp("="*80)
    log_with_timestamp("Stage 9 V2: ACL-based Noisy Query Generation")
    log_with_timestamp("="*80)
    log_with_timestamp(f"Query file: {ACL_QUERY_FILE}")
    log_with_timestamp(f"Error file: {ACL_ERROR_FILE}")
    log_with_timestamp(f"Output file: {OUTPUT_FILE}")
    log_with_timestamp(f"Max workers: {MAX_WORKERS}")

    # 加载数据
    user_errors = load_user_errors(ACL_ERROR_FILE)
    user_queries = load_user_queries(ACL_QUERY_FILE)
    user_attributes = load_user_attributes(ACL_PROFILES_FILE)

    # 找出有错误且有查询的用户
    target_users = set(user_errors.keys()) & set(user_queries.keys())
    log_with_timestamp(f"Users with both errors and queries: {len(target_users)}")
    if MAX_USERS:
        target_users = sorted(target_users)[:MAX_USERS]
        log_with_timestamp(f"Limited to {MAX_USERS} users for testing")

    if not target_users:
        log_with_timestamp("ERROR: No users with both errors and queries found!")
        sys.exit(1)

    # 准备处理任务
    tasks = []
    for uid in sorted(target_users):
        queries = user_queries[uid]
        errors = user_errors[uid]
        tasks.append((uid, queries, errors))

    log_with_timestamp(f"Processing {len(tasks)} users with {MAX_WORKERS} workers...")
    log_with_timestamp("="*80)

    results = []
    total_start = time.time()
    completed_count = 0

    def process_task(uid, queries, errors):
        return process_single_user(uid, queries, errors, user_attributes)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_task, uid, qs, errs): uid
            for uid, qs, errs in tasks
        }

        for future in as_completed(futures):
            uid = futures[future]
            completed_count += 1
            try:
                result = future.result()
                results.append(result)

                if result["queries_generated"] > 0:
                    log_with_timestamp(
                        f"  [{completed_count}/{len(tasks)}] "
                        f"✓ {result['user_id']}: {result['queries_generated']} noisy queries"
                    )
                else:
                    log_with_timestamp(
                        f"  [{completed_count}/{len(tasks)}] "
                        f"✗ {result['user_id']}: no queries generated"
                    )
            except Exception as e:
                log_with_timestamp(f"  [{completed_count}/{len(tasks)}] ✗ {uid}: {str(e)}")
                results.append({"user_id": uid, "queries_generated": 0, "results": [], "error": str(e)})

    total_elapsed = time.time() - total_start

    successful = [r for r in results if r.get("queries_generated", 0) > 0]

    log_with_timestamp("="*80)
    log_with_timestamp(f"Completed: {len(successful)} success, {len(results) - len(successful)} failed ({total_elapsed:.1f}s)")
    log_with_timestamp("="*80)

    # 收集所有查询结果（只保留有注入错误的）
    all_noisy_queries = []
    error_type_counts = defaultdict(int)
    total_injected = 0
    filtered_results = []

    for r in results:
        user_queries_with_errors = []
        for qr in r.get('results', []):
            if qr.get('injected_errors'):
                all_noisy_queries.append(qr)
                for err in qr.get('injected_errors', []):
                    etype = err.get('error_type', 'unknown')
                    error_type_counts[etype] += 1
                    total_injected += 1
                user_queries_with_errors.append(qr)

        # 只有当用户有至少一个注入错误的查询时才保留
        if user_queries_with_errors:
            filtered_results.append({
                "user_id": r["user_id"],
                "queries_generated": len(user_queries_with_errors),
                "results": user_queries_with_errors
            })

    log_with_timestamp(f"Total noisy queries: {len(all_noisy_queries)}")
    log_with_timestamp(f"Total injected errors: {total_injected}")
    if error_type_counts:
        log_with_timestamp("Error type distribution:")
        for etype, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
            log_with_timestamp(f"  {etype}: {count}")

    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(tasks),
        "users_with_noisy_queries": len(filtered_results),
        "total_noisy_queries": len(all_noisy_queries),
        "total_injected_errors": total_injected,
        "error_type_distribution": dict(error_type_counts),
        "results": filtered_results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    log_with_timestamp(f"Output saved to {OUTPUT_FILE}")

    log_with_timestamp("="*80)
    log_with_timestamp("ALL PROCESSING COMPLETE!")
    log_with_timestamp("="*80)


if __name__ == "__main__":
    main()
