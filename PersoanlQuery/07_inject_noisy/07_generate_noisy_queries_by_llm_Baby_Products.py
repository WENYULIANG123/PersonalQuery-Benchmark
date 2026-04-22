#!/usr/bin/env python3
"""
基于 LLM 的查询语句噪声注入 - Baby_Products
"""

import sys
import json
import time
import re
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery')

# ========================================
# 配置加载
# ========================================
from config import get_category_config

CATEGORY = "Baby_Products"
CAT_CONFIG = get_category_config(CATEGORY)

# Stage 6 查询文件
ACL_QUERY_FILE = CAT_CONFIG['acl_query_file']
CCOMP_QUERY_FILE = CAT_CONFIG['ccomp_query_file']

# Stage 4 用户错误文件
USER_ERROR_FILE = CAT_CONFIG['user_error_file']

# 输出文件
ACL_NOISY_OUTPUT_FILE = CAT_CONFIG['acl_noisy_output']
CCOMP_NOISY_OUTPUT_FILE = CAT_CONFIG['ccomp_noisy_output']

# Prompt 配置
NOISY_PROMPT_FILE = '/workspace/PersonalQuery/PersoanlQuery/07_inject_noisy/noisy_query_prompts.json'

# 全局配置
NOISY_CONFIG_FILE = '/workspace/PersonalQuery/PersoanlQuery/07_inject_noisy/noisy_query_config.json'
QUERY_CONFIG_FILE = '/workspace/PersonalQuery/PersoanlQuery/06_query/query_config.json'


# ========================================
# 加载配置和 prompt 模板
# ========================================
with open(NOISY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _NOISY_CONFIG = json.load(f)
with open(QUERY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _CONFIG = json.load(f)

NUM_USERS_TO_TEST = _NOISY_CONFIG.get('num_users_to_test', 50)
MAX_WORKERS = _NOISY_CONFIG.get('max_workers', 10)
USE_MINIMAXIO = _CONFIG.get('use_minimaxio', False)
INJECT_ERROR_COUNT = _NOISY_CONFIG.get('inject_error_count', 3)

# 加载噪声 prompt 模板
with open(NOISY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    _NOISY_PROMPTS = json.load(f)

_system_base_key = f"system_base_{CATEGORY}"
if _system_base_key in _NOISY_PROMPTS:
    NOISY_SYSTEM_BASE = _NOISY_PROMPTS[_system_base_key]
else:
    NOISY_SYSTEM_BASE = _NOISY_PROMPTS.get("system_base_Baby_Products", "")

NOISY_USER_CONTENT_TEMPLATE = _NOISY_PROMPTS.get("user_content_noisy", "")


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
            system_base=system_base, user_content=prompt, max_tokens=4096, temperature=1.0
        )
    else:
        response = _minimax_client.call(prompt=prompt, max_tokens=4096, temperature=1.0)

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


# ========================================
# 主函数
# ========================================
def main():
    log(f"=== 基于 LLM 的噪声注入开始 (Category: {CATEGORY}) ===")

    log(f"加载 ACL 查询 from {ACL_QUERY_FILE}...")
    with open(ACL_QUERY_FILE, 'r', encoding='utf-8') as f:
        acl_queries = json.load(f)
    log(f"加载了 {len(acl_queries)} 个用户的 ACL 查询")

    log(f"加载 CCOMP 查询 from {CCOMP_QUERY_FILE}...")
    with open(CCOMP_QUERY_FILE, 'r', encoding='utf-8') as f:
        ccomp_queries = json.load(f)
    log(f"加载了 {len(ccomp_queries)} 个用户的 CCOMP 查询")

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
    acl_tasks = []
    for user_data in acl_queries:
        uid = user_data['user_id']
        errors = user_errors.get(uid, {})
        if not errors.get('acl'):
            continue
        # Baby_Products 使用嵌套的 acl_query/ccomp_query 格式
        acl_query_data = user_data.get('acl_query', {})
        ground_truth_query = acl_query_data.get('query', '')
        if not ground_truth_query:
            continue
        acl_tasks.append({
            'uid': uid, 'asin': user_data.get('asin', ''),
            'ground_truth_query': ground_truth_query,
            'errors': errors['acl'],
            'user_data': user_data,
            'level': acl_query_data.get('level', 0),
            'word_count': acl_query_data.get('word_count', 0),
        })
    # 限制用户数量
    acl_tasks = acl_tasks[:NUM_USERS_TO_TEST]
    log(f"ACL 任务数: {len(acl_tasks)}")

    def process_one_acl_task(task):
        uid, query, errors = task['uid'], task['ground_truth_query'], task['errors']
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
                'uid': uid, 'asin': task['asin'], 'original_query': query,
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'],
            }

        return {
            'uid': uid, 'asin': task['asin'], 'original_query': query,
            'noisy_query': noisy_query, 'injected_errors': injected_errors,
            'status': 'success', 'user_errors': errors, 'user_data': task['user_data'],
        }

    log("\n=== 处理 ACL 查询 ===")
    acl_results = []
    acl_llm_errors = []
    acl_no_injection = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_acl_task, t): t for t in acl_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                if r['status'] == 'success':
                    acl_results.append(r)
                elif r['status'] == 'llm_error':
                    acl_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    acl_no_injection.append(r['uid'])
                log(f"  [ACL] 成功:{len(acl_results)} 无注入:{len(acl_no_injection)} LLM错误:{len(acl_llm_errors)} user={r['uid'][:20]}")
    log(f"ACL 完成: 成功注入 {len(acl_results)}, 无注入 {len(acl_no_injection)}, LLM错误 {len(acl_llm_errors)}")

    # CCOMP
    log("\n=== 处理 CCOMP 查询 ===")
    ccomp_tasks = []
    for user_data in ccomp_queries:
        uid = user_data['user_id']
        errors = user_errors.get(uid, {})
        if not errors.get('ccomp'):
            continue
        # Baby_Products 使用嵌套的 acl_query/ccomp_query 格式
        ccomp_query_data = user_data.get('ccomp_query', {})
        ground_truth_query = ccomp_query_data.get('query', '')
        if not ground_truth_query:
            continue
        ccomp_tasks.append({
            'uid': uid, 'asin': user_data.get('asin', ''),
            'ground_truth_query': ground_truth_query,
            'errors': errors['ccomp'],
            'user_data': user_data,
            'level': ccomp_query_data.get('level', 0),
            'word_count': ccomp_query_data.get('word_count', 0),
        })
    # 限制用户数量
    ccomp_tasks = ccomp_tasks[:NUM_USERS_TO_TEST]
    log(f"CCOMP 任务数: {len(ccomp_tasks)}")

    def process_one_ccomp_task(task):
        uid, query, errors = task['uid'], task['ground_truth_query'], task['errors']
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
                'uid': uid, 'asin': task['asin'], 'original_query': query,
                'noisy_query': noisy_query, 'injected_errors': injected_errors,
                'status': 'no_injection', 'user_errors': errors, 'user_data': task['user_data'],
            }

        return {
            'uid': uid, 'asin': task['asin'], 'original_query': query,
            'noisy_query': noisy_query, 'injected_errors': injected_errors,
            'status': 'success', 'user_errors': errors, 'user_data': task['user_data'],
        }

    log("\n=== 处理 CCOMP 查询 ===")
    ccomp_results = []
    ccomp_llm_errors = []
    ccomp_no_injection = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_ccomp_task, t): t for t in ccomp_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                if r['status'] == 'success':
                    ccomp_results.append(r)
                elif r['status'] == 'llm_error':
                    ccomp_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    ccomp_no_injection.append(r['uid'])
                log(f"  [CCOMP] 成功注入:{len(ccomp_results)} 无注入:{len(ccomp_no_injection)} LLM错误:{len(ccomp_llm_errors)} user={r['uid'][:20]}")
    log(f"CCOMP 完成: 成功注入 {len(ccomp_results)}, 无注入 {len(ccomp_no_injection)}, LLM错误 {len(ccomp_llm_errors)}")

    # 保存结果
    os.makedirs(os.path.dirname(ACL_NOISY_OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(CCOMP_NOISY_OUTPUT_FILE), exist_ok=True)

    acl_output = []
    for r in acl_results:
        # Baby_Products 新格式直接使用 acl_query
        original_query_info = r['user_data'].get('acl_query', {})
        acl_output.append({
            'user_id': r['uid'],
            'asin': r['asin'],
            'noisy_query': {
                'level': r.get('level', original_query_info.get('level', 0)),
                'query': r['noisy_query'],
                'word_count': r.get('word_count', len(r['noisy_query'].split())),
            },
            'injected_errors': r['injected_errors'],
        })
    with open(ACL_NOISY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(acl_output, f, indent=2, ensure_ascii=False)

    ccomp_output = []
    for r in ccomp_results:
        # Baby_Products 新格式直接使用 ccomp_query
        original_query_info = r['user_data'].get('ccomp_query', {})
        ccomp_output.append({
            'user_id': r['uid'],
            'asin': r['asin'],
            'noisy_query': {
                'level': r.get('level', original_query_info.get('level', 0)),
                'query': r['noisy_query'],
                'word_count': r.get('word_count', len(r['noisy_query'].split())),
            },
            'injected_errors': r['injected_errors'],
        })
    with open(CCOMP_NOISY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ccomp_output, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"==================== 统计结果 ====================")
    log(f"ACL 任务总数: {len(acl_tasks)}")
    log(f"  - 成功注入噪声: {len(acl_results)} ({len(acl_results)/len(acl_tasks)*100:.1f}%)")
    log(f"  - 无注入(查询未变或错误为空): {len(acl_no_injection)} ({len(acl_no_injection)/len(acl_tasks)*100:.1f}%)")
    log(f"  - LLM调用/解析失败: {len(acl_llm_errors)} ({len(acl_llm_errors)/len(acl_tasks)*100:.1f}%)")
    log(f"CCOMP 任务总数: {len(ccomp_tasks)}")
    log(f"  - 成功注入噪声: {len(ccomp_results)} ({len(ccomp_results)/len(ccomp_tasks)*100:.1f}%)")
    log(f"  - 无注入(查询未变或错误为空): {len(ccomp_no_injection)} ({len(ccomp_no_injection)/len(ccomp_tasks)*100:.1f}%)")
    log(f"  - LLM调用/解析失败: {len(ccomp_llm_errors)} ({len(ccomp_llm_errors)/len(ccomp_tasks)*100:.1f}%)")
    log(f"===============================================")
    log(f"ACL 结果保存到: {ACL_NOISY_OUTPUT_FILE}")
    log(f"CCOMP 结果保存到: {CCOMP_NOISY_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
