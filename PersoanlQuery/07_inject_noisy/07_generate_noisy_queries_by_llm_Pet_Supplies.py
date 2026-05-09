#!/usr/bin/env python3
"""
基于 LLM 的查询语句噪声注入 - Pet_Supplies
"""

import sys
import json
import time
import re
import os
from datetime import datetime
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')

# ========================================
# 配置加载
# ========================================
CATEGORY = "Pet_Supplies"

# 全局配置
NOISY_CONFIG_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/07_inject_noisy/noisy_query_config.json'


# ========================================
# 加载配置和 prompt 模板
# ========================================
def get_required_config_value(config: dict, *keys):
    current = config
    current_path = []
    for key in keys:
        current_path.append(str(key))
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"配置缺少字段: {'.'.join(current_path)}")
        current = current[key]
    return current


with open(NOISY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _NOISY_CONFIG = json.load(f)

_CATEGORY_CONFIG = get_required_config_value(_NOISY_CONFIG, 'categories', CATEGORY)

NUM_USERS_TO_TEST = get_required_config_value(_NOISY_CONFIG, 'num_users_to_test')
MAX_WORKERS = get_required_config_value(_NOISY_CONFIG, 'max_workers')
USE_MINIMAXIO = get_required_config_value(_NOISY_CONFIG, 'use_minimaxio')
INJECT_ERROR_COUNT = get_required_config_value(_NOISY_CONFIG, 'inject_error_count')
EFFECTIVE_MAX_WORKERS = min(MAX_WORKERS, 8)
LLM_MAX_RETRIES = 6

QUERY_FILE = get_required_config_value(_CATEGORY_CONFIG, 'query_file')
USER_ERROR_FILE = get_required_config_value(_CATEGORY_CONFIG, 'user_error_file')
NOISY_OUTPUT_FILE = get_required_config_value(_CATEGORY_CONFIG, 'noisy_output_file')
NOISY_PROMPT_FILE = get_required_config_value(_NOISY_CONFIG, 'prompt_file')

# 加载噪声 prompt 模板
with open(NOISY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    _NOISY_PROMPTS = json.load(f)

NOISY_SYSTEM_BASE = get_required_config_value(_NOISY_PROMPTS, f"system_base_{CATEGORY}")
NOISY_USER_CONTENT_TEMPLATE = get_required_config_value(_NOISY_PROMPTS, "user_content_noisy")


# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 加载用户错误数据
# ========================================
def load_user_errors(error_file: str) -> dict:
    if not os.path.exists(error_file):
        log(f"错误文件不存在: {error_file}")
        return {}

    with open(error_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持数组格式（直接是用户列表）和字典格式（有 user_results 字段）
    if isinstance(data, list):
        users_list = data
    else:
        users_list = data.get('user_results', [])

    user_errors = {}
    for user in users_list:
        uid = user['user_id']
        if user['total_errors'] == 0 or not user.get('detailed_results'):
            continue

        acl_patterns = []
        ccomp_patterns = []

        for detail in user['detailed_results']:
            error_category = detail.get('error_category', '')
            if error_category not in ('acl', 'ccomp'):
                continue

            seen = set()
            patterns = []
            for err in detail.get('errors', []):
                orig = err.get('original', '')
                corr = err.get('corrected', '')
                key = (orig, corr)
                if key not in seen:
                    seen.add(key)
                    patterns.append({
                        'original': orig,
                        'corrected': corr,
                        'error_type': err.get('error_type', 'unknown'),
                    })

            if error_category == 'acl':
                acl_patterns.extend(patterns)
            elif error_category == 'ccomp':
                ccomp_patterns.extend(patterns)

        if acl_patterns or ccomp_patterns:
            user_errors[uid] = {'acl': acl_patterns, 'ccomp': ccomp_patterns}

    log(f"加载了 {len(user_errors)} 个有错误的用户")
    return user_errors


def filter_error_patterns(error_patterns: list) -> list:
    if not error_patterns:
        return []

    filtered = []
    for ep in error_patterns:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")

        if not orig or not corr:
            continue
        if '-' in orig or '-' in corr:
            continue
        if ' ' in orig or ' ' in corr:
            if orig.strip() != corr.strip():
                continue
        if is_pure_suffix_change(orig, corr):
            continue
        if is_pure_punctuation_change(orig, corr):
            continue
        if is_apostrophe_only_change(orig, corr):
            continue
        if is_case_only_change(orig, corr):
            continue

        filtered.append(ep)
    return filtered


def is_pure_suffix_change(orig: str, corr: str) -> bool:
    common_prefix_len = 0
    min_len = min(len(orig), len(corr))
    for i in range(min_len):
        if orig[i].lower() == corr[i].lower():
            common_prefix_len += 1
        else:
            break
    if common_prefix_len == 0:
        return False
    orig_suffix = orig[common_prefix_len:] if common_prefix_len < len(orig) else ''
    corr_suffix = corr[common_prefix_len:] if common_prefix_len < len(corr) else ''
    core_suffixes = {'ing', 'ed', 's', 'er', 'est', 'ly', 'd', 'en', 'ment', 'tion', 'ness'}
    if len(corr_suffix) == 0 and len(orig_suffix) > 0:
        return orig_suffix.lower() in core_suffixes
    if len(orig_suffix) == 0 and len(corr_suffix) > 0:
        return corr_suffix.lower() in core_suffixes
    if len(corr_suffix) > 0 and corr_suffix.lower() in core_suffixes:
        return True
    if orig_suffix.lower() in core_suffixes and corr_suffix.lower() in core_suffixes:
        return True
    return False


def is_pure_punctuation_change(orig: str, corr: str) -> bool:
    import string
    orig_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in orig)
    corr_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in corr)
    if orig_only_punct and corr_only_punct:
        return True
    return False


def is_apostrophe_only_change(orig: str, corr: str) -> bool:
    orig_clean = orig.replace("'", "").replace("'", "")
    corr_clean = corr.replace("'", "").replace("'", "")
    if orig_clean.lower() == corr_clean.lower():
        return True
    return False


def is_case_only_change(orig: str, corr: str) -> bool:
    if orig.lower() == corr.lower() and orig != corr:
        return True
    return False


# ========================================
# LLM 调用
# ========================================
_minimax_client = None
_first_request = True


def load_minimax_client():
    global _minimax_client
    if _minimax_client is None:
        from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient
        if USE_MINIMAXIO:
            _minimax_client = MiniMaxIOAnthropicClient()
            log("MiniMaxIO API 客户端初始化完成")
        else:
            _minimax_client = MiniMaxAnthropicClient()
            log("MiniMax API 客户端初始化完成")


def call_llm(prompt: str, system_base: str = None) -> str:
    global _minimax_client, _first_request
    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    if system_base and _first_request:
        log(f"[Request] system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_request = False

    log(f"[Request] user_content:\n{prompt}")

    if system_base:
        response, cache_info = _minimax_client.call_with_cache(
            system_base=system_base, user_content=prompt, max_tokens=4096, temperature=1.0, max_retries=LLM_MAX_RETRIES
        )
    else:
        response = _minimax_client.call(prompt=prompt, max_tokens=4096, temperature=1.0, max_retries=LLM_MAX_RETRIES)

    log(f"[Cache] {cache_info}")
    log(f"[Response] response:\n{response}")
    return response


def fix_incomplete_json(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if not text.startswith('{') and not text.startswith('['):
        return text
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    if open_brackets > close_brackets:
        text = text + ']' * (open_brackets - close_brackets)
    if open_braces > close_braces:
        text = text + '}' * (open_braces - close_braces)
    return text


def count_keyword_in_query(query: str, keyword: str) -> int:
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not isinstance(keyword, str) or not keyword:
        raise ValueError("keyword 必须是非空字符串")
    return len(re.findall(rf'\b{re.escape(keyword)}\b', query, flags=re.IGNORECASE))


def build_complexity_constraint_text(query_category: str, target_level: int) -> str:
    if query_category == 'acl':
        return (
            f"This query is an ACL / wide query with complexity level {target_level}.\n"
            f"- The rewritten correct query must contain exactly {target_level} occurrence(s) of the word 'which'.\n"
            "- The rewritten correct query must contain zero occurrences of the word 'that'.\n"
            "- Do not increase or decrease the ACL complexity level during rewriting.\n"
        )
    if query_category == 'ccomp':
        return (
            f"This query is a CCOMP / deep query with complexity level {target_level}.\n"
            f"- The rewritten correct query must contain exactly {target_level} occurrence(s) of the word 'that'.\n"
            "- The rewritten correct query must contain zero occurrences of the word 'which'.\n"
            "- Do not increase or decrease the CCOMP complexity level during rewriting.\n"
        )
    raise ValueError(f"未知 query_category: {query_category}")


def revised_query_matches_expected_complexity(query: str, query_category: str, target_level: int) -> bool:
    which_count = count_keyword_in_query(query, 'which')
    that_count = count_keyword_in_query(query, 'that')
    if query_category == 'acl':
        return which_count == target_level and that_count == 0
    if query_category == 'ccomp':
        return that_count == target_level and which_count == 0
    raise ValueError(f"未知 query_category: {query_category}")


def build_noisy_prompt(query: str, error_patterns: list) -> tuple:
    system_base = NOISY_SYSTEM_BASE
    if error_patterns:
        error_lines = []
        for i, ep in enumerate(error_patterns[:10], 1):
            orig = ep.get('original', '')
            corr = ep.get('corrected', '')
            err_type = ep.get('error_type', 'unknown')
            error_lines.append(f"{i}. '{corr}' -> '{orig}' (error_type: {err_type})")
        errors_section = "User's typical spelling error patterns:\n" + "\n".join(error_lines) + "\n"
    else:
        errors_section = ""

    user_content = NOISY_USER_CONTENT_TEMPLATE.format(
        query=query, errors_section=errors_section, inject_count=INJECT_ERROR_COUNT
    )
    return system_base, user_content


def build_anchor_rewrite_prompt(query: str, error_patterns: list, query_category: str, target_level: int) -> tuple:
    system_base = NOISY_SYSTEM_BASE
    error_lines = []
    for i, ep in enumerate(error_patterns[:10], 1):
        orig = ep.get('original', '')
        corr = ep.get('corrected', '')
        err_type = ep.get('error_type', 'unknown')
        error_lines.append(f"{i}. '{corr}' -> '{orig}' (error_type: {err_type})")
    errors_section = "User's typical spelling error patterns:\n" + "\n".join(error_lines) + "\n"
    complexity_section = build_complexity_constraint_text(query_category, target_level)
    user_content = (
        f"Original correct query:\n{query}\n\n"
        f"{errors_section}"
        f"{complexity_section}"
        "Task:\n"
        "1. The original query currently does not contain a usable exact anchor for the user's real spelling patterns.\n"
        "2. First minimally rewrite the correct query so that at least one exact 'correct' text from the listed patterns appears naturally.\n"
        "3. The rewritten correct query must stay grammatical, natural, and preserve the original product, brand, price, attributes, search intent, and complexity style.\n"
        f"4. Then inject 1-{INJECT_ERROR_COUNT} user spelling errors into that rewritten correct query.\n"
        "5. You must use only exact listed pairs. Do not invent approximations.\n\n"
        "Output format (JSON):\n"
        "{\n"
        '  "revised_correct_query": "...",\n'
        '  "noisy_query": "...",\n'
        '  "injected_errors": [\n'
        '    {"correct": "...", "error": "...", "error_type": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Important:\n"
        "- revised_correct_query must be a natural search query\n"
        "- noisy_query must be derived from revised_correct_query\n"
        "- injected_errors.correct must literally appear in revised_correct_query\n"
        "- Preserve product attributes, search intent, and required ACL/CCOMP complexity pattern exactly"
    )
    return system_base, user_content


def parse_noisy_response(text_content: str, original_query: str) -> dict:
    try:
        text_content = fix_incomplete_json(text_content)
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        noisy_query = data.get('noisy_query', '').strip()
        if not noisy_query:
            return None
        injected_errors = data.get('injected_errors', [])
        return {'noisy_query': noisy_query, 'injected_errors': injected_errors if isinstance(injected_errors, list) else []}
    except Exception as e:
        log(f"    [DEBUG] JSON解析失败: {e}")
        if text_content.strip():
            return {'noisy_query': text_content.strip(), 'injected_errors': []}
    return None


def parse_anchor_rewrite_response(text_content: str) -> dict:
    try:
        text_content = fix_incomplete_json(text_content)
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)
        revised_correct_query = data.get('revised_correct_query', '').strip()
        noisy_query = data.get('noisy_query', '').strip()
        injected_errors = data.get('injected_errors', [])
        if not revised_correct_query or not noisy_query:
            return None
        return {
            'revised_correct_query': revised_correct_query,
            'noisy_query': noisy_query,
            'injected_errors': injected_errors if isinstance(injected_errors, list) else [],
        }
    except Exception as e:
        log(f"    [DEBUG] 锚点改写JSON解析失败: {e}")
    return None


def build_real_error_pairs(error_patterns: list) -> set:
    real_error_pairs = set()
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        original = pattern.get('original')
        corrected = pattern.get('corrected')
        if not isinstance(original, str) or not original:
            raise ValueError(f"error_patterns[{idx}].original 必须是非空字符串")
        if not isinstance(corrected, str) or not corrected:
            raise ValueError(f"error_patterns[{idx}].corrected 必须是非空字符串")
        real_error_pairs.add((corrected, original))
        real_error_pairs.add((corrected.lower(), original.lower()))
    if not real_error_pairs:
        raise ValueError("真实错误模式不能为空")
    return real_error_pairs


def injected_errors_match_real_patterns(injected_errors: list, error_patterns: list) -> bool:
    if not isinstance(injected_errors, list) or not injected_errors:
        return False
    real_error_pairs = build_real_error_pairs(error_patterns)
    for idx, injected_error in enumerate(injected_errors):
        if not isinstance(injected_error, dict):
            raise TypeError(f"injected_errors[{idx}] 必须是 dict")
        correct = injected_error.get('correct')
        error = injected_error.get('error')
        if not isinstance(correct, str) or not isinstance(error, str):
            return False
        if (correct, error) not in real_error_pairs and (correct.lower(), error.lower()) not in real_error_pairs:
            return False
    return True


def query_contains_exact_anchor(query: str, correct_text: str) -> bool:
    if not isinstance(query, str) or not isinstance(correct_text, str) or not correct_text:
        return False
    escaped = re.escape(correct_text)
    if re.fullmatch(r"[A-Za-z0-9']+", correct_text):
        return re.search(rf"\b{escaped}\b", query, flags=re.IGNORECASE) is not None
    return re.search(escaped, query, flags=re.IGNORECASE) is not None


def query_has_any_real_anchor(query: str, error_patterns: list) -> bool:
    for idx, pattern in enumerate(error_patterns):
        if not isinstance(pattern, dict):
            raise TypeError(f"error_patterns[{idx}] 必须是 dict")
        correct = pattern.get('corrected')
        if not isinstance(correct, str) or not correct:
            raise ValueError(f"error_patterns[{idx}].corrected 必须是非空字符串")
        if query_contains_exact_anchor(query, correct):
            return True
    return False


def injected_errors_have_query_anchor(query: str, injected_errors: list) -> bool:
    if not isinstance(injected_errors, list) or not injected_errors:
        return False
    for idx, injected_error in enumerate(injected_errors):
        if not isinstance(injected_error, dict):
            raise TypeError(f"injected_errors[{idx}] 必须是 dict")
        correct = injected_error.get('correct')
        if not isinstance(correct, str):
            return False
        if not query_contains_exact_anchor(query, correct):
            return False
    return True


# ========================================
# 增量写入辅助函数
# ========================================
def write_noisy_result_incremental(result_item: dict, output_file: str, query_category: str):
    """将单个结果增量写入合并文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if 'query_info' not in result_item or not isinstance(result_item['query_info'], dict):
        raise ValueError("result_item.query_info 必须存在且为 dict")
    original_query_info = dict(result_item['query_info'])
    original_query_info['query'] = result_item['ground_truth_query']
    original_query_info['word_count'] = result_item['ground_truth_word_count']
    output_data = {
        'user_id': result_item['uid'],
        'asin': result_item['asin'],
        'query_category': query_category,
        'ground_truth_query': result_item['ground_truth_query'],
        'noisy_query': result_item['noisy_query'],
        'injected_errors': result_item['injected_errors'],
        'word_count': result_item['ground_truth_word_count'],
        'original_query_info': original_query_info,
    }
    # 追加到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        f.write('\n')


# ========================================
# 主函数
# ========================================
def main():
    log(f"=== 基于 LLM 的噪声注入开始 (Category: {CATEGORY}) ===")

    log(f"加载查询 from {QUERY_FILE}...")
    with open(QUERY_FILE, 'r', encoding='utf-8') as f:
        all_queries = json.load(f)
    log(f"加载了 {len(all_queries)} 个用户的查询")

    log(f"加载用户错误 from {USER_ERROR_FILE}...")
    raw_user_errors = load_user_errors(USER_ERROR_FILE)

    user_errors = {}
    for uid, err_data in raw_user_errors.items():
        filtered_acl = filter_error_patterns(err_data.get('acl', []))
        filtered_ccomp = filter_error_patterns(err_data.get('ccomp', []))
        if filtered_acl or filtered_ccomp:
            user_errors[uid] = {'acl': filtered_acl, 'ccomp': filtered_ccomp}
    log(f"过滤后有错误的用户数: {len(user_errors)}")

    load_minimax_client()

    # ACL
    log("\n=== 处理 ACL 查询 ===")

    # 加载已完成的用户ID，避免重复处理
    def load_completed_uids_by_category(file_path: str) -> Tuple[set, set]:
        """读取已完成的用户ID，按query_category分类返回，支持 pretty-printed JSON 和 JSON Lines 格式"""
        if not os.path.exists(file_path):
            return set(), set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return set(), set()
                acl_uids = set()
                ccomp_uids = set()
                # 尝试 JSON array 格式
                if content.startswith('['):
                    data = json.loads(content)
                    for item in data:
                        if 'user_id' not in item:
                            continue
                        cat = item.get('query_category', '')
                        if cat == 'acl':
                            acl_uids.add(item['user_id'])
                        elif cat == 'ccomp':
                            ccomp_uids.add(item['user_id'])
                # 处理 pretty-printed JSON（多个JSON对象用换行分隔，每个包含嵌套的{}）
                elif content.startswith('{'):
                    depth = 0
                    start = -1
                    for i, c in enumerate(content):
                        if c == '{':
                            if depth == 0:
                                start = i
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0 and start >= 0:
                                try:
                                    item = json.loads(content[start:i+1])
                                    if 'user_id' in item:
                                        cat = item.get('query_category', '')
                                        if cat == 'acl':
                                            acl_uids.add(item['user_id'])
                                        elif cat == 'ccomp':
                                            ccomp_uids.add(item['user_id'])
                                except:
                                    pass
                                start = -1
                return acl_uids, ccomp_uids
        except Exception as e:
            log(f"  读取已有结果失败 ({file_path}): {e}")
            return set(), set()

    completed_acl_uids, completed_ccomp_uids = load_completed_uids_by_category(NOISY_OUTPUT_FILE)
    log(f"  ACL 已完成用户数: {len(completed_acl_uids)}")
    log(f"  CCOMP 已完成用户数: {len(completed_ccomp_uids)}")

    # ACL 任务构建（从合并的 query.json 中提取 acl_query）
    acl_tasks = []
    for user_data in all_queries:
        uid = user_data['user_id']
        if uid in completed_acl_uids:
            continue
        errors = user_errors.get(uid, {})
        if not errors.get('acl'):
            continue
        # 新格式：嵌套的 acl_query 字段
        acl_query_data = user_data.get('acl_query', {})
        ground_truth_query = acl_query_data.get('query', '')
        if not ground_truth_query:
            continue
        acl_tasks.append({
            'uid': uid,
            'asin': user_data.get('asin', ''),
            'ground_truth_query': ground_truth_query,
            'errors': errors['acl'],
            'user_data': user_data,
            'query_info': acl_query_data,
            'level': acl_query_data.get('level', 0),
            'word_count': acl_query_data.get('word_count', 0),
        })
    # 限制用户数量
    acl_tasks = acl_tasks[:NUM_USERS_TO_TEST]
    log(f"ACL 任务数: {len(acl_tasks)} (去重后)")

    def process_one_acl_task(task):
        uid, query, errors = task['uid'], task['ground_truth_query'], task['errors']
        if not query_has_any_real_anchor(query, errors):
            system_base, user_content = build_anchor_rewrite_prompt(query, errors, 'acl', task['level'])
            response = call_llm(user_content, system_base=system_base)
            parsed = parse_anchor_rewrite_response(response)
            if not parsed:
                log(f"    [ERROR] ACL 用户 {uid[:20]} 锚点改写失败")
                return {'uid': uid, 'status': 'llm_error', 'original_query': query}
            revised_query = parsed['revised_correct_query']
            noisy_query = parsed['noisy_query']
            injected_errors = parsed['injected_errors']
            if not revised_query_matches_expected_complexity(revised_query, 'acl', task['level']):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'complexity_mismatch', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if revised_query == query or noisy_query == revised_query or not injected_errors:
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if not injected_errors_have_query_anchor(revised_query, injected_errors):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'no_anchor', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if not injected_errors_match_real_patterns(injected_errors, errors):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'pattern_mismatch', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                'ground_truth_word_count': len(revised_query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'success', 'user_errors': errors, 'user_data': task['user_data'],
                'query_info': task['query_info'], 'query_rewritten': revised_query != query,
            }
        system_base, user_content = build_noisy_prompt(query, errors)
        response = call_llm(user_content, system_base=system_base)
        parsed = parse_noisy_response(response, query)
        if not parsed:
            log(f"    [ERROR] ACL 用户 {uid[:20]} LLM调用或解析失败")
            return {'uid': uid, 'status': 'llm_error', 'original_query': query}

        # 检查是否成功注入错误
        noisy_query = parsed['noisy_query']
        injected_errors = parsed['injected_errors']

        if noisy_query == query or not injected_errors:
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }
        if not injected_errors_have_query_anchor(query, injected_errors):
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_anchor', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }
        if not injected_errors_match_real_patterns(injected_errors, errors):
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'pattern_mismatch', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }

        return {
            'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
            'ground_truth_word_count': len(query.split()),
            'noisy_query': noisy_query, 'injected_errors': injected_errors,
            'status': 'success', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
            'query_rewritten': False,
        }

    log("\n=== 处理 ACL 查询 ===")
    acl_results = []
    acl_llm_errors = []
    acl_no_injection = []
    acl_no_anchor = []
    acl_pattern_mismatch = []
    acl_complexity_mismatch = []
    acl_rewritten_success = 0
    with ThreadPoolExecutor(max_workers=EFFECTIVE_MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_acl_task, t): t for t in acl_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                if r['status'] == 'success':
                    acl_results.append(r)
                    if r.get('query_rewritten'):
                        acl_rewritten_success += 1
                    write_noisy_result_incremental(r, NOISY_OUTPUT_FILE, 'acl')
                elif r['status'] == 'llm_error':
                    acl_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    acl_no_injection.append(r['uid'])
                elif r['status'] == 'no_anchor':
                    acl_no_anchor.append(r['uid'])
                elif r['status'] == 'pattern_mismatch':
                    acl_pattern_mismatch.append(r['uid'])
                elif r['status'] == 'complexity_mismatch':
                    acl_complexity_mismatch.append(r['uid'])
                log(f"  [ACL] 成功:{len(acl_results)} 改写成功:{acl_rewritten_success} 无注入:{len(acl_no_injection)} 无锚点:{len(acl_no_anchor)} 模式不匹配:{len(acl_pattern_mismatch)} 复杂度不匹配:{len(acl_complexity_mismatch)} LLM错误:{len(acl_llm_errors)} user={r['uid'][:20]}")
    log(f"ACL 完成: 成功注入 {len(acl_results)}, 其中改写成功 {acl_rewritten_success}, 无注入 {len(acl_no_injection)}, 无锚点 {len(acl_no_anchor)}, 模式不匹配 {len(acl_pattern_mismatch)}, 复杂度不匹配 {len(acl_complexity_mismatch)}, LLM错误 {len(acl_llm_errors)}")

    # CCOMP
    log("\n=== 处理 CCOMP 查询 ===")

    ccomp_tasks = []
    for user_data in all_queries:
        uid = user_data['user_id']
        if uid in completed_ccomp_uids:
            continue
        errors = user_errors.get(uid, {})
        if not errors.get('ccomp'):
            continue
        # 新格式：嵌套的 ccomp_query 字段
        ccomp_query_data = user_data.get('ccomp_query', {})
        ground_truth_query = ccomp_query_data.get('query', '')
        if not ground_truth_query:
            continue
        ccomp_tasks.append({
            'uid': uid,
            'asin': user_data.get('asin', ''),
            'ground_truth_query': ground_truth_query,
            'errors': errors['ccomp'],
            'user_data': user_data,
            'query_info': ccomp_query_data,
            'level': ccomp_query_data.get('level', 0),
            'word_count': ccomp_query_data.get('word_count', 0),
        })
    # 限制用户数量
    ccomp_tasks = ccomp_tasks[:NUM_USERS_TO_TEST]
    log(f"CCOMP 任务数: {len(ccomp_tasks)} (去重后)")

    def process_one_ccomp_task(task):
        uid, query, errors = task['uid'], task['ground_truth_query'], task['errors']
        if not query_has_any_real_anchor(query, errors):
            system_base, user_content = build_anchor_rewrite_prompt(query, errors, 'ccomp', task['level'])
            response = call_llm(user_content, system_base=system_base)
            parsed = parse_anchor_rewrite_response(response)
            if not parsed:
                log(f"    [ERROR] CCOMP 用户 {uid[:20]} 锚点改写失败")
                return {'uid': uid, 'status': 'llm_error', 'original_query': query}
            revised_query = parsed['revised_correct_query']
            noisy_query = parsed['noisy_query']
            injected_errors = parsed['injected_errors']
            if not revised_query_matches_expected_complexity(revised_query, 'ccomp', task['level']):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'complexity_mismatch', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if revised_query == query or noisy_query == revised_query or not injected_errors:
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if not injected_errors_have_query_anchor(revised_query, injected_errors):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'no_anchor', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            if not injected_errors_match_real_patterns(injected_errors, errors):
                return {
                    'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                    'ground_truth_word_count': len(revised_query.split()),
                    'noisy_query': noisy_query, 'injected_errors': injected_errors,
                    'status': 'pattern_mismatch', 'user_errors': errors, 'user_data': task['user_data'],
                    'query_info': task['query_info'], 'query_rewritten': revised_query != query,
                }
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': revised_query,
                'ground_truth_word_count': len(revised_query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'success', 'user_errors': errors, 'user_data': task['user_data'],
                'query_info': task['query_info'], 'query_rewritten': revised_query != query,
            }
        system_base, user_content = build_noisy_prompt(query, errors)
        response = call_llm(user_content, system_base=system_base)
        parsed = parse_noisy_response(response, query)
        if not parsed:
            log(f"    [ERROR] CCOMP 用户 {uid[:20]} LLM调用或解析失败")
            return {'uid': uid, 'status': 'llm_error', 'original_query': query}

        # 检查是否成功注入错误
        noisy_query = parsed['noisy_query']
        injected_errors = parsed['injected_errors']

        if noisy_query == query or not injected_errors:
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }
        if not injected_errors_have_query_anchor(query, injected_errors):
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_anchor', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }
        if not injected_errors_match_real_patterns(injected_errors, errors):
            return {
                'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
                'ground_truth_word_count': len(query.split()),
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'pattern_mismatch', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
                'query_rewritten': False,
            }

        return {
            'uid': uid, 'asin': task['asin'], 'ground_truth_query': query,
            'ground_truth_word_count': len(query.split()),
            'noisy_query': noisy_query, 'injected_errors': injected_errors,
            'status': 'success', 'user_errors': errors, 'user_data': task['user_data'], 'query_info': task['query_info'],
            'query_rewritten': False,
        }

    log("\n=== 处理 CCOMP 查询 ===")
    ccomp_results = []
    ccomp_llm_errors = []
    ccomp_no_injection = []
    ccomp_no_anchor = []
    ccomp_pattern_mismatch = []
    ccomp_complexity_mismatch = []
    ccomp_rewritten_success = 0
    with ThreadPoolExecutor(max_workers=EFFECTIVE_MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_ccomp_task, t): t for t in ccomp_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                if r['status'] == 'success':
                    ccomp_results.append(r)
                    if r.get('query_rewritten'):
                        ccomp_rewritten_success += 1
                    write_noisy_result_incremental(r, NOISY_OUTPUT_FILE, 'ccomp')
                elif r['status'] == 'llm_error':
                    ccomp_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    ccomp_no_injection.append(r['uid'])
                elif r['status'] == 'no_anchor':
                    ccomp_no_anchor.append(r['uid'])
                elif r['status'] == 'pattern_mismatch':
                    ccomp_pattern_mismatch.append(r['uid'])
                elif r['status'] == 'complexity_mismatch':
                    ccomp_complexity_mismatch.append(r['uid'])
                log(f"  [CCOMP] 成功注入:{len(ccomp_results)} 改写成功:{ccomp_rewritten_success} 无注入:{len(ccomp_no_injection)} 无锚点:{len(ccomp_no_anchor)} 模式不匹配:{len(ccomp_pattern_mismatch)} 复杂度不匹配:{len(ccomp_complexity_mismatch)} LLM错误:{len(ccomp_llm_errors)} user={r['uid'][:20]}")
    log(f"CCOMP 完成: 成功注入 {len(ccomp_results)}, 其中改写成功 {ccomp_rewritten_success}, 无注入 {len(ccomp_no_injection)}, 无锚点 {len(ccomp_no_anchor)}, 模式不匹配 {len(ccomp_pattern_mismatch)}, 复杂度不匹配 {len(ccomp_complexity_mismatch)}, LLM错误 {len(ccomp_llm_errors)}")

    # 最终保存（如需额外后处理可在此添加）
    # 由于使用了增量写入，此处无需再次保存

    def safe_pct(numerator, denominator):
        """安全计算百分比，避免除零"""
        if denominator == 0:
            return "N/A"
        return f"{numerator/denominator*100:.1f}%"

    log(f"\n{'='*60}")
    log(f"==================== 统计结果 ====================")
    log(f"ACL 任务总数: {len(acl_tasks)}")
    log(f"  - 成功注入噪声: {len(acl_results)} ({safe_pct(len(acl_results), len(acl_tasks))})")
    log(f"  - 其中改写后成功: {acl_rewritten_success} ({safe_pct(acl_rewritten_success, len(acl_tasks))})")
    log(f"  - 无注入(查询未变或错误为空): {len(acl_no_injection)} ({safe_pct(len(acl_no_injection), len(acl_tasks))})")
    log(f"  - 无锚点(不写入): {len(acl_no_anchor)} ({safe_pct(len(acl_no_anchor), len(acl_tasks))})")
    log(f"  - 模式不匹配(不写入): {len(acl_pattern_mismatch)} ({safe_pct(len(acl_pattern_mismatch), len(acl_tasks))})")
    log(f"  - LLM调用/解析失败: {len(acl_llm_errors)} ({safe_pct(len(acl_llm_errors), len(acl_tasks))})")
    log(f"CCOMP 任务总数: {len(ccomp_tasks)}")
    log(f"  - 成功注入噪声: {len(ccomp_results)} ({safe_pct(len(ccomp_results), len(ccomp_tasks))})")
    log(f"  - 其中改写后成功: {ccomp_rewritten_success} ({safe_pct(ccomp_rewritten_success, len(ccomp_tasks))})")
    log(f"  - 无注入(查询未变或错误为空): {len(ccomp_no_injection)} ({safe_pct(len(ccomp_no_injection), len(ccomp_tasks))})")
    log(f"  - 无锚点(不写入): {len(ccomp_no_anchor)} ({safe_pct(len(ccomp_no_anchor), len(ccomp_tasks))})")
    log(f"  - 模式不匹配(不写入): {len(ccomp_pattern_mismatch)} ({safe_pct(len(ccomp_pattern_mismatch), len(ccomp_tasks))})")
    log(f"  - LLM调用/解析失败: {len(ccomp_llm_errors)} ({safe_pct(len(ccomp_llm_errors), len(ccomp_tasks))})")
    log(f"===============================================")
    log(f"结果保存到: {NOISY_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
