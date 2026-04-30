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
# 增量写入辅助函数
# ========================================
def write_noisy_result_incremental(result_item: dict, output_file: str, query_category: str):
    """将单个结果增量写入合并文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 新格式：直接从 user_data 中获取对应的 query 字段
    if query_category == 'acl':
        original_query_info = result_item['user_data'].get('acl_query', {})
    else:
        original_query_info = result_item['user_data'].get('ccomp_query', {})
    output_data = {
        'user_id': result_item['uid'],
        'asin': result_item['asin'],
        'query_category': query_category,
        'ground_truth_query': result_item['original_query'],
        'noisy_query': result_item['noisy_query'],
        'injected_errors': result_item['injected_errors'],
        'word_count': len(result_item['original_query'].split()),
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
            'level': acl_query_data.get('level', 0),
            'word_count': acl_query_data.get('word_count', 0),
        })
    # 限制用户数量
    acl_tasks = acl_tasks[:NUM_USERS_TO_TEST]
    log(f"ACL 任务数: {len(acl_tasks)} (去重后)")

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
                    write_noisy_result_incremental(r, NOISY_OUTPUT_FILE, 'acl')
                elif r['status'] == 'llm_error':
                    acl_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    acl_no_injection.append(r['uid'])
                log(f"  [ACL] 成功:{len(acl_results)} 无注入:{len(acl_no_injection)} LLM错误:{len(acl_llm_errors)} user={r['uid'][:20]}")
    log(f"ACL 完成: 成功注入 {len(acl_results)}, 无注入 {len(acl_no_injection)}, LLM错误 {len(acl_llm_errors)}")

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
            'level': ccomp_query_data.get('level', 0),
            'word_count': ccomp_query_data.get('word_count', 0),
        })
    # 限制用户数量
    ccomp_tasks = ccomp_tasks[:NUM_USERS_TO_TEST]
    log(f"CCOMP 任务数: {len(ccomp_tasks)} (去重后)")

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
                    write_noisy_result_incremental(r, NOISY_OUTPUT_FILE, 'ccomp')
                elif r['status'] == 'llm_error':
                    ccomp_llm_errors.append(r['uid'])
                elif r['status'] == 'no_injection':
                    ccomp_no_injection.append(r['uid'])
                log(f"  [CCOMP] 成功注入:{len(ccomp_results)} 无注入:{len(ccomp_no_injection)} LLM错误:{len(ccomp_llm_errors)} user={r['uid'][:20]}")
    log(f"CCOMP 完成: 成功注入 {len(ccomp_results)}, 无注入 {len(ccomp_no_injection)}, LLM错误 {len(ccomp_llm_errors)}")

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
    log(f"  - 无注入(查询未变或错误为空): {len(acl_no_injection)} ({safe_pct(len(acl_no_injection), len(acl_tasks))})")
    log(f"  - LLM调用/解析失败: {len(acl_llm_errors)} ({safe_pct(len(acl_llm_errors), len(acl_tasks))})")
    log(f"CCOMP 任务总数: {len(ccomp_tasks)}")
    log(f"  - 成功注入噪声: {len(ccomp_results)} ({safe_pct(len(ccomp_results), len(ccomp_tasks))})")
    log(f"  - 无注入(查询未变或错误为空): {len(ccomp_no_injection)} ({safe_pct(len(ccomp_no_injection), len(ccomp_tasks))})")
    log(f"  - LLM调用/解析失败: {len(ccomp_llm_errors)} ({safe_pct(len(ccomp_llm_errors), len(ccomp_tasks))})")
    log(f"===============================================")
    log(f"结果保存到: {NOISY_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
