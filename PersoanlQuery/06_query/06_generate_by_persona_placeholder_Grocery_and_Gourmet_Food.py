#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（分离版本流程）
=================================
1. 每次请求只生成一个版本（acl_0 / acl_1 / acl_2 / acl_3 / ccomp_0 / ccomp_1 / ccomp_2 / ccomp_3）
2. 得到占位符模板后，回填属性词
3. 需要添加正确词时，再发送一次请求
"""

import sys
import json
import time
import re
import math
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')

# ========================================
# 硬编码参数
# ========================================
CATEGORY = "Grocery_and_Gourmet_Food"
ACL_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Grocery_and_Gourmet_Food/acl_user_profiles.json'
CCOMP_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Grocery_and_Gourmet_Food/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Grocery_and_Gourmet_Food/attr_density_user_profiles.json'
ATTR_VALUES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/Grocery_and_Gourmet_Food/attributes_Grocery_and_Gourmet_Food.json'
USER_ERROR_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Grocery_and_Gourmet_Food/acl_ccomp_error.json'
ACL_OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Grocery_and_Gourmet_Food/acl_query.json'
CCOMP_OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Grocery_and_Gourmet_Food/ccomp_query.json'

QUERY_CONFIG_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/query_config.json'
ACL_PROMPTS_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/acl_query_prompts.json'
CCOMP_PROMPTS_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/ccomp_query_prompts.json'

# ========================================
# 加载配置和 prompt 模板
# ========================================
with open(QUERY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _CONFIG = json.load(f)
NUM_USERS_TO_TEST = _CONFIG['num_users_to_test']
MAX_WORKERS = _CONFIG['max_workers']
USE_MINIMAXIO = _CONFIG.get('use_minimaxio', False)

with open(ACL_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    _ACL_PROMPTS = json.load(f)
ACL_SYSTEM_BASE = _ACL_PROMPTS['system_base']

with open(CCOMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    _CCOMP_PROMPTS = json.load(f)
CCOMP_SYSTEM_BASE = _CCOMP_PROMPTS['system_base']


# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 特征提取
# ========================================
def count_words(text: str) -> int:
    """简单分词统计词数"""
    return len(text.split())


# ========================================
# 加载用户错误数据
# ========================================
def load_user_errors(error_file: str) -> dict:
    """加载用户错误画像，返回 {uid: {'acl': [...], 'ccomp': [...]}}"""
    with open(error_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_errors = {}
    for user in data.get('user_results', []):
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
            user_errors[uid] = {
                'acl': acl_patterns,
                'ccomp': ccomp_patterns,
            }

    return user_errors


# ========================================
# 构建 Prompt
# ========================================
def build_acl_prompt(query_level: int, attrs: dict, correct_words: list = None) -> tuple:
    """为指定 ACL 版本构建 prompt

    query_level: 0, 1, 2, 3
    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含

    Returns: (system_base, user_content)
    """
    system_base = ACL_SYSTEM_BASE

    # 构建正确词 section
    if correct_words:
        correct_words_section = "Correct words that SHOULD appear in the query if they can fit naturally:\n- " + "\n- ".join(correct_words) + "\n"
    else:
        correct_words_section = ""

    user_content = _ACL_PROMPTS[f"user_content_acl_{query_level}"].format(
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


def build_acl_batch_prompt(attrs: dict, correct_words: list = None, groundtruth_level: int = None) -> tuple:
    """为 ACL 批量生成构建 prompt（K=0,1,2,3 一次返回）

    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含
    groundtruth_level: groundtruth 级别，如果为 None 则不包含正确词提示

    Returns: (system_base, user_content)
    """
    system_base = ACL_SYSTEM_BASE

    # 构建正确词 section
    if correct_words and groundtruth_level is not None:
        words_list = "\n- ".join(correct_words)
        correct_words_section = f"For level {groundtruth_level} (groundtruth level only), try to naturally incorporate these USER CORRECT WORDS if they fit. The 'that'/'which' keywords used for CCOMP/ACL sentence structure are NOT user correct words - do NOT include them in used_correct_words. If the user words cannot fit naturally, simply skip them. Do NOT return IMPOSSIBLE:\n- {words_list}\n"
    elif correct_words:
        correct_words_section = ""
    else:
        correct_words_section = ""

    user_content = _ACL_PROMPTS["user_content_acl_batch"].format(
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


def build_acl_single_prompt(level: int, attrs: dict, correct_words: list = None) -> tuple:
    """为 ACL 单个级别构建 prompt（用于 groundtruth 注入正确词）

    level: 0, 1, 2, 3
    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含

    Returns: (system_base, user_content)
    """
    system_base = ACL_SYSTEM_BASE

    # 构建正确词 section
    if correct_words:
        correct_words_section = "Correct words that SHOULD appear in the query if they can fit naturally:\n- " + "\n- ".join(correct_words) + "\n"
    else:
        correct_words_section = ""

    user_content = _ACL_PROMPTS["user_content_acl_single"].format(
        level=level,
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


def build_ccomp_prompt(query_level: int, attrs: dict, correct_words: list = None) -> tuple:
    """为指定 CCOMP 版本构建 prompt

    query_level: 0, 1, 2, 3
    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含

    Returns: (system_base, user_content)
    """
    system_base = CCOMP_SYSTEM_BASE

    # 构建正确词 section
    if correct_words:
        correct_words_section = "Correct words that SHOULD appear in the query if they can fit naturally:\n- " + "\n- ".join(correct_words) + "\n"
    else:
        correct_words_section = ""

    user_content = _CCOMP_PROMPTS[f"user_content_ccomp_{query_level}"].format(
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


def build_ccomp_batch_prompt(attrs: dict, correct_words: list = None, groundtruth_level: int = None) -> tuple:
    """为 CCOMP 批量生成构建 prompt（K=0,1,2,3 一次返回）

    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含
    groundtruth_level: groundtruth 级别，如果为 None 则不包含正确词提示

    Returns: (system_base, user_content)
    """
    system_base = CCOMP_SYSTEM_BASE

    # 构建正确词 section
    if correct_words and groundtruth_level is not None:
        words_list = "\n- ".join(correct_words)
        correct_words_section = f"For level {groundtruth_level} (groundtruth level only), try to naturally incorporate these USER CORRECT WORDS if they fit. The 'that'/'which' keywords used for CCOMP/ACL sentence structure are NOT user correct words - do NOT include them in used_correct_words. If the user words cannot fit naturally, simply skip them. Do NOT return IMPOSSIBLE:\n- {words_list}\n"
    elif correct_words:
        correct_words_section = ""
    else:
        correct_words_section = ""

    user_content = _CCOMP_PROMPTS["user_content_ccomp_batch"].format(
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


def build_ccomp_single_prompt(level: int, attrs: dict, correct_words: list = None) -> tuple:
    """为 CCOMP 单个级别构建 prompt（用于 groundtruth 注入正确词）

    level: 0, 1, 2, 3
    attrs: 属性字典 {'A1': ..., 'A2': ..., 'A3': ..., 'A4': ..., 'A5': ...}
    correct_words: 用户的正确词列表，如果为 None 则不包含

    Returns: (system_base, user_content)
    """
    system_base = CCOMP_SYSTEM_BASE

    # 构建正确词 section
    if correct_words:
        correct_words_section = "Correct words that SHOULD appear in the query if they can fit naturally:\n- " + "\n- ".join(correct_words) + "\n"
    else:
        correct_words_section = ""

    user_content = _CCOMP_PROMPTS["user_content_ccomp_single"].format(
        level=level,
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=correct_words_section
    )

    return system_base, user_content


# ========================================
# LLM 调用
# ========================================
_minimax_client = None
_first_acl_request = True
_first_ccomp_request = True
_acl_cache_warmed = False
_ccomp_cache_warmed = False


def load_minimax_client():
    """加载 MiniMax API 客户端"""
    global _minimax_client
    if _minimax_client is None:
        from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient
        if USE_MINIMAXIO:
            _minimax_client = MiniMaxIOAnthropicClient()
            log("MiniMaxIO API 客户端初始化完成")
        else:
            _minimax_client = MiniMaxAnthropicClient()
            log("MiniMax API 客户端初始化完成")


def prewarm_cache():
    """预热 cache，用通用模板创建缓存（ACL 和 CCOMP 并发执行）"""
    global _acl_cache_warmed, _ccomp_cache_warmed

    if _acl_cache_warmed and _ccomp_cache_warmed:
        log("[Cache] Cache 已预热，跳过")
        return

    # 通用属性模板（全 NONE，避免对 LLM 生成产生影响）
    generic_attrs = {
        'A1': 'None',
        'A2': 'None',
        'A3': 'None',
        'A4': 'None',
        'A5': 'None'
    }

    def _prewarm_acl():
        """预热 ACL cache"""
        global _acl_cache_warmed
        if _acl_cache_warmed:
            return
        log("[Cache] 预热 ACL cache...")
        system, user = build_acl_batch_prompt(generic_attrs)
        call_llm(user, system_base=system, is_acl=True)
        _acl_cache_warmed = True
        log("[Cache] ACL cache 预热完成")

    def _prewarm_ccomp():
        """预热 CCOMP cache"""
        global _ccomp_cache_warmed
        if _ccomp_cache_warmed:
            return
        log("[Cache] 预热 CCOMP cache...")
        system, user = build_ccomp_batch_prompt(generic_attrs)
        call_llm(user, system_base=system, is_acl=False)
        _ccomp_cache_warmed = True
        log("[Cache] CCOMP cache 预热完成")

    # 并发预热 ACL 和 CCOMP cache
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_acl = executor.submit(_prewarm_acl)
        future_ccomp = executor.submit(_prewarm_ccomp)
        future_acl.result()
        future_ccomp.result()


def call_llm(prompt: str, system_base: str = None, is_acl: bool = True) -> str:
    """调用 MiniMax API，支持可选的系统提示词缓存"""
    global _minimax_client, _first_acl_request, _first_ccomp_request

    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # ACL 第一次请求时打印 system_base（创建缓存）
    if is_acl and system_base and _first_acl_request:
        log(f"[Request] ACL system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_acl_request = False

    # CCOMP 第一次请求时打印 system_base（创建缓存）
    if not is_acl and system_base and _first_ccomp_request:
        log(f"[Request] CCOMP system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_ccomp_request = False

    # 打印 user_content
    prompt_type = "ACL" if is_acl else "CCOMP"
    log(f"[Request] {prompt_type} user_content:\n{prompt}")

    if system_base:
        response, cache_info = _minimax_client.call_with_cache(
            system_base=system_base,
            user_content=prompt,
            max_tokens=8192,
            temperature=1.0
        )
    else:
        response = _minimax_client.call(
            prompt=prompt,
            max_tokens=8192,
            temperature=1.0
        )

    # 打印所有缓存相关字段
    log(f"[Cache] {cache_info}")

    # 打印 LLM 响应
    log(f"[Response] {prompt_type} response:\n{response[:1500]}")

    return response


def fix_incomplete_json(text: str) -> str:
    """修复可能被截断的 JSON"""
    if not text:
        return text
    text = text.strip()

    if not text.startswith('{'):
        return text

    # 计算括号数量
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # 修复缺失的闭合括号
    if open_brackets > close_brackets:
        text = text + ']' * (open_brackets - close_brackets)
    if open_braces > close_braces:
        text = text + '}' * (open_braces - close_braces)

    return text


# ========================================
# 解析查询结果
# ========================================
def parse_query_response(text_content: str, query_type: str, query_level: int) -> list:
    """解析查询响应

    query_type: 'acl' 或 'ccomp'
    query_level: 0, 1, 2, 3

    Returns: [{'query': str, 'word_count': int, 'attrs_used': dict}, ...] 或 None
    """
    try:
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        # 处理新的返回格式 {"query": "...", "attrs_used": {...}}
        if 'query' in data:
            query = data.get('query', '').strip()
            attrs_used = data.get('attrs_used', {})
            if query:
                return [{
                    'query': query,
                    'word_count': count_words(query),
                    'attrs_used': attrs_used,
                }]

        # 旧格式兼容
        query_key = f"{query_type}_{query_level}"
        if query_key not in data:
            return None

        items = data[query_key]

        if isinstance(items, str):
            items = [items]
        elif not isinstance(items, list):
            return None

        variants = []
        for item in items:
            query = item.strip() if isinstance(item, str) else ''
            if query:
                variants.append({
                    'query': query,
                    'word_count': count_words(query),
                })

        if not variants:
            return None

        return variants
    except Exception as e:
        log(f"    [DEBUG] {query_type}_{query_level} JSON解析失败: {e}")
    return None


def parse_batch_query_response(text_content: str, query_type: str) -> dict:
    """解析批量查询响应（一次返回 K=0,1,2,3）

    query_type: 'acl' 或 'ccomp'

    Returns: {0: {'query': ..., 'word_count': ..., 'attrs_used': ...}, 1: {...}, 2: {...}, 3: {...}}
    """
    try:
        # 尝试提取 JSON 数组
        json_match = re.search(r'\[[\s\S]*\]', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # 尝试整个内容作为 JSON
            data = json.loads(text_content)

        if not isinstance(data, list):
            # 尝试作为单个对象处理
            if 'level' in data:
                data = [data]
            else:
                return None

        result = {}
        for item in data:
            if isinstance(item, dict) and 'level' in item:
                level = item['level']
                # 支持字符串或整数类型的 level
                if isinstance(level, str):
                    try:
                        level = int(level)
                    except (ValueError, TypeError):
                        continue
                query = item.get('query', '').strip()
                used_correct_words = item.get('used_correct_words', [])
                if query and level in [0, 1, 2, 3]:
                    result[level] = {
                        'query': query,
                        'word_count': count_words(query),
                        'attrs_used': item.get('attrs_used', {}),
                        'used_correct_words': used_correct_words if isinstance(used_correct_words, list) else [],
                    }

        if not result:
            log(f"    [DEBUG] {query_type} 批量解析失败，未找到有效结果")
            return None

        return result
    except Exception as e:
        log(f"    [DEBUG] {query_type} 批量JSON解析失败: {e}")
    return None


# ========================================
# 验证占位符
# ========================================
def validate_placeholders(query: str) -> dict:
    """验证查询中是否正确使用了所有5个占位符"""
    placeholders = {
        '${A1}': '${A1}' in query,
        '${A2}': '${A2}' in query,
        '${A3}': '${A3}' in query,
        '${A4}': '${A4}' in query,
        '${A5}': '${A5}' in query,
    }
    return placeholders


def fill_placeholders(query: str, attrs: dict) -> str:
    """将占位符替换为实际属性值"""
    result = query
    for key in ['${A1}', '${A2}', '${A3}', '${A4}', '${A5}']:
        attr_key = key[2:-1]  # ${A1} -> A1
        val = attrs.get(attr_key, '')
        if isinstance(val, list):
            val = ', '.join(str(v) for v in val)
        result = result.replace(key, str(val))
    return result


def count_which_in_query(query: str) -> int:
    """计算查询中 'which' 的数量"""
    return len(re.findall(r'\bwhich\b', query, re.IGNORECASE))


def count_that_in_query(query: str) -> int:
    """计算查询中 'that' 的数量"""
    return len(re.findall(r'\bthat\b', query, re.IGNORECASE))


def is_pure_suffix_change(orig: str, corr: str) -> bool:
    """检查是否只是后缀变化"""
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
    """检查是否只是标点符号变化"""
    import string
    orig_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in orig)
    corr_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in corr)

    if orig_only_punct and corr_only_punct:
        return True

    return False


def is_apostrophe_only_change(orig: str, corr: str) -> bool:
    """检查是否只是撇号差异（如 daughter's vs daughters）"""
    # 移除撇号后比较
    orig_clean = orig.replace("'", "").replace("'", "")
    corr_clean = corr.replace("'", "").replace("'", "")
    # 如果移除撇号后相同，则是撇号差异
    if orig_clean.lower() == corr_clean.lower():
        return True
    return False


def is_case_only_change(orig: str, corr: str) -> bool:
    """检查是否只是大小写差异（如 I vs i）"""
    if orig.lower() == corr.lower() and orig != corr:
        return True
    return False


def filter_error_patterns(error_patterns: list) -> list:
    """过滤掉不需要的错误类型"""
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


def inject_errors(query: str, error_patterns: list) -> tuple:
    """将查询中的正确词替换为错误词

    Returns: (noisy_query, injected_errors)
        noisy_query: 替换后的查询
        injected_errors: 实际被注入的错误列表 [{'correct': ..., 'error': ..., 'error_type': ...}, ...]
    """
    import re

    if not error_patterns:
        return query, []

    filtered_patterns = filter_error_patterns(error_patterns)
    if not filtered_patterns:
        return query, []

    result = query
    injected_errors = []
    for ep in filtered_patterns[:10]:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        if orig and corr:
            # 使用单词边界确保只替换完整单词，不替换子串
            # 例如 "or" 不会替换 "for" 中的 "or"
            pattern = r'\b' + re.escape(corr) + r'\b'
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, orig, result, flags=re.IGNORECASE)
                injected_errors.append(ep)

    return result, injected_errors


# ========================================
# 主函数
# ========================================
def main():
    # 加载 ACL 用户画像
    log(f"加载ACL用户画像 from {ACL_USER_PROFILES_FILE}...")
    with open(ACL_USER_PROFILES_FILE, 'r') as f:
        acl_user_profiles = json.load(f)
    log(f"加载了 {len(acl_user_profiles)} 个ACL用户画像")

    # 加载 CCOMP 用户画像
    log(f"加载CCOMP用户画像 from {CCOMP_USER_PROFILES_FILE}...")
    with open(CCOMP_USER_PROFILES_FILE, 'r') as f:
        ccomp_user_profiles = json.load(f)
    log(f"加载了 {len(ccomp_user_profiles)} 个CCOMP用户画像")

    # 加载 attr_density 用户画像（用于 words_per_attribute）
    log(f"加载attr_density用户画像 from {ATTR_DENSITY_PROFILES_FILE}...")
    with open(ATTR_DENSITY_PROFILES_FILE, 'r') as f:
        attr_density_profiles = json.load(f)
    user_wpa_map = {}
    for p in attr_density_profiles:
        uid = p.get('user_id')
        wpa = p.get('words_per_attribute')
        if uid and wpa is not None:
            user_wpa_map[uid] = float(wpa)
    log(f"加载了 {len(user_wpa_map)} 个attr_density用户画像")

    # 加载属性值文件（用于 A1-A5）
    log(f"加载属性值 from {ATTR_VALUES_FILE}...")
    with open(ATTR_VALUES_FILE, 'r') as f:
        attr_values_data = json.load(f)
    user_prod_map = {}
    if isinstance(attr_values_data, dict) and 'products' in attr_values_data:
        for p in attr_values_data['products']:
            uid = p.get('user_id')
            if uid:
                if uid not in user_prod_map:
                    user_prod_map[uid] = []
                user_prod_map[uid].append(p)
    log(f"加载了 {len(user_prod_map)} 个用户的属性产品")

    # 加载用户错误画像
    log(f"加载用户错误画像 from {USER_ERROR_FILE}...")
    user_errors = {}
    if os.path.exists(USER_ERROR_FILE):
        raw_errors = load_user_errors(USER_ERROR_FILE)
        log(f"加载了 {len(raw_errors)} 个有错误的用户")

        for uid, err_data in raw_errors.items():
            filtered_acl = filter_error_patterns(err_data.get('acl', []))
            filtered_ccomp = filter_error_patterns(err_data.get('ccomp', []))
            if filtered_acl or filtered_ccomp:
                user_errors[uid] = {
                    'acl': filtered_acl,
                    'ccomp': filtered_ccomp,
                }
        log(f"过滤后有错误的用户数: {len(user_errors)}")
    else:
        log(f"错误文件不存在，跳过错误处理")

    # 构建用户画像 map
    acl_profile_map = {p['user_id']: p for p in acl_user_profiles}
    ccomp_profile_map = {p['user_id']: p for p in ccomp_user_profiles}

    # 计算所有用户的 ground_truth 并过滤
    all_user_ids = set(acl_profile_map.keys()) & set(ccomp_profile_map.keys()) & set(user_wpa_map.keys())
    log(f"同时存在于ACL、CCOMP和attr_density的用户数: {len(all_user_ids)}")

    all_user_data = []
    for uid in all_user_ids:
        acl_profile = acl_profile_map[uid]
        ccomp_profile = ccomp_profile_map[uid]

        # 使用新的attributes文件获取产品属性
        user_products = user_prod_map.get(uid, [])
        if not user_products:
            continue
        prod = user_products[0]

        words_per_attribute = user_wpa_map.get(uid)
        if words_per_attribute is None:
            continue

        words_per_acl = acl_profile.get('words_per_acl', 100.0)
        words_per_ccomp = ccomp_profile.get('words_per_ccomp', 100.0)

        target_length = math.ceil(words_per_attribute) * 5

        if words_per_acl and words_per_acl > 0:
            ground_truth_acl = int(target_length / words_per_acl)
            ground_truth_acl = max(0, min(5, ground_truth_acl))
        else:
            ground_truth_acl = 0

        if words_per_ccomp and words_per_ccomp > 0:
            ground_truth_ccomp = int(target_length / words_per_ccomp)
            ground_truth_ccomp = max(0, min(5, ground_truth_ccomp))
        else:
            ground_truth_ccomp = 0

        if ground_truth_acl > 3 or ground_truth_ccomp > 3:
            continue

        has_acl_errors = uid in user_errors and len(user_errors[uid].get('acl', [])) > 0
        has_ccomp_errors = uid in user_errors and len(user_errors[uid].get('ccomp', [])) > 0

        all_user_data.append({
            'uid': uid,
            'acl_profile': acl_profile,
            'ccomp_profile': ccomp_profile,
            'prod': prod,
            'words_per_attribute': words_per_attribute,
            'words_per_acl': words_per_acl,
            'words_per_ccomp': words_per_ccomp,
            'target_length': target_length,
            'ground_truth_acl': ground_truth_acl,
            'ground_truth_ccomp': ground_truth_ccomp,
            'has_acl_errors': has_acl_errors,
            'has_ccomp_errors': has_ccomp_errors,
        })

    log(f"过滤后（ACL≤3 且 CCOMP≤3）的用户数: {len(all_user_data)}")

    # 按优先级选用户
    both_errors = [u for u in all_user_data if u['has_acl_errors'] and u['has_ccomp_errors']]
    only_ccomp = [u for u in all_user_data if u['has_ccomp_errors'] and not u['has_acl_errors']]
    only_acl = [u for u in all_user_data if u['has_acl_errors'] and not u['has_ccomp_errors']]
    no_errors = [u for u in all_user_data if not (u['has_acl_errors'] or u['has_ccomp_errors'])]

    # 预热 cache（确保 API 客户端已初始化）
    prewarm_cache()

    target_users = []
    remaining = NUM_USERS_TO_TEST

    if remaining > 0:
        take = min(remaining, len(both_errors))
        target_users.extend(both_errors[:take])
        remaining -= take

    if remaining > 0:
        take = min(remaining, len(only_ccomp))
        target_users.extend(only_ccomp[:take])
        remaining -= take

    if remaining > 0:
        take = min(remaining, len(only_acl))
        target_users.extend(only_acl[:take])
        remaining -= take

    if remaining > 0:
        take = min(remaining, len(no_errors))
        target_users.extend(no_errors[:take])
        remaining -= take

    has_error_count = sum(1 for u in target_users if u['has_acl_errors'] or u['has_ccomp_errors'])
    both_count = sum(1 for u in target_users if u['has_acl_errors'] and u['has_ccomp_errors'])
    ccomp_only = sum(1 for u in target_users if u['has_ccomp_errors'] and not u['has_acl_errors'])
    acl_only = sum(1 for u in target_users if u['has_acl_errors'] and not u['has_ccomp_errors'])
    log(f"目标用户: {len(target_users)} 个（ACL+CCOMP错误: {both_count}, 仅CCOMP: {ccomp_only}, 仅ACL: {acl_only}, 无错误: {len(target_users)-has_error_count}）")

    # 构建用户任务列表
    user_tasks = []
    for u in target_users:
        uid = u['uid']
        errors = user_errors.get(uid, None)

        persona_base = {
            'user_id': uid,
            'asin': u['prod'].get('asin', ''),
            'acl_sentence_ratio': u['acl_profile'].get('acl_sentence_ratio', 0.0),
            'ccomp_sentence_ratio': u['ccomp_profile'].get('ccomp_sentence_ratio', 0.0),
            'density_label': u['acl_profile'].get('density_label', 'simple'),
            'length_label': u['acl_profile'].get('length_label', 'medium'),
            'words_per_attribute': u['words_per_attribute'],
            'words_per_acl': u['words_per_acl'],
            'words_per_ccomp': u['words_per_ccomp'],
            'target_length': u['target_length'],
            'ground_truth_acl': u['ground_truth_acl'],
            'ground_truth_ccomp': u['ground_truth_ccomp'],
            'has_acl_errors': u['has_acl_errors'],
            'has_ccomp_errors': u['has_ccomp_errors'],
            'original_attrs': {
                'A1': u['prod'].get('A1_product_type', ''),
                'A2': u['prod'].get('A2_brand', ''),
                'A3': u['prod'].get('A3_price', ''),
                'A4': u['prod'].get('A4_appearance', '')[0] if isinstance(u['prod'].get('A4_appearance', ''), list) else u['prod'].get('A4_appearance', ''),
                'A5': u['prod'].get('A5_use_case', ''),
            },
            'errors': errors,
        }
        user_tasks.append(persona_base)

    def process_one_user(persona):
        """处理单个用户，批量生成 ACL/CCOMP 查询"""
        uid = persona['user_id']
        attrs = persona['original_attrs']
        errors = persona['errors']
        ground_truth_acl = persona['ground_truth_acl']
        ground_truth_ccomp = persona['ground_truth_ccomp']

        acl_results = []
        ccomp_results = []

        # ========== ACL 批量查询（1次请求生成 K=0,1,2,3）==========
        correct_words = [ep['corrected'] for ep in errors['acl'][:10] if ep.get('corrected')] if errors and errors.get('acl') else None
        system_base, user_content = build_acl_batch_prompt(attrs, correct_words=correct_words, groundtruth_level=ground_truth_acl if correct_words else None)
        response = call_llm(user_content, system_base=system_base, is_acl=True)

        # 解析批量结果
        batch_results = parse_batch_query_response(response, 'acl')

        # 用于存储最终结果，key 为 acl_level
        acl_results_dict = {}

        if batch_results:
            for acl_level in range(4):
                query_key = f"acl_{acl_level}"
                is_ground_truth = (acl_level == ground_truth_acl)

                if acl_level not in batch_results:
                    log(f"    [DEBUG] {query_key} 缺失，user={uid}")
                    continue

                parsed = batch_results[acl_level]
                query = parsed['query']
                attrs_used = parsed.get('attrs_used', {})
                used_correct_words = parsed.get('used_correct_words', [])

                # 验证 which 数量
                actual_which = count_which_in_query(query)
                if actual_which != acl_level:
                    log(f"    [DEBUG] {query_key} which数量不匹配(期望{acl_level},实际{actual_which})")
                    continue

                acl_results_dict[acl_level] = {
                    'query': query,
                    'attrs_used': attrs_used,
                    'used_correct_words': used_correct_words,
                    'is_ground_truth': is_ground_truth,
                }

        # 构建最终结果
        for acl_level in sorted(acl_results_dict.keys()):
            data = acl_results_dict[acl_level]
            query = data['query']
            attrs_used = data['attrs_used']
            used_correct_words = data['used_correct_words']
            is_ground_truth = data['is_ground_truth']

            # 如果是 ground_truth 且有错误，生成 noisy 版本
            if is_ground_truth and persona['has_acl_errors'] and errors and errors.get('acl'):
                correct_words = [ep['corrected'] for ep in errors['acl'][:10] if ep.get('corrected')]
                if correct_words:
                    noisy_query, injected_errors = inject_errors(query, errors.get('acl', []))
                    # 只有当实际注入了错误时才生成双版本
                    if noisy_query != query:
                        acl_results.append({
                            'user_id': uid,
                            'asin': persona['asin'],
                            'target_acl': acl_level,
                            'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                            'density_label': persona.get('density_label', 'simple'),
                            'length_label': persona.get('length_label', 'medium'),
                            'words_per_acl': persona.get('words_per_acl'),
                            'ground_truth_acl': ground_truth_acl,
                            'target_length': persona.get('target_length'),
                            'has_errors': True,
                            'correct_query': query,
                            'noisy_query': noisy_query,
                            'attrs_used': attrs_used,
                            'used_correct_words': used_correct_words,
                            'error_words': injected_errors,
                            'word_count': count_words(query),
                            'is_ground_truth': True,
                        })
                    else:
                        # 没有实际注入错误，只生成单版本
                        acl_results.append({
                            'user_id': uid,
                            'asin': persona['asin'],
                            'target_acl': acl_level,
                            'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                            'density_label': persona.get('density_label', 'simple'),
                            'length_label': persona.get('length_label', 'medium'),
                            'words_per_acl': persona.get('words_per_acl'),
                            'ground_truth_acl': ground_truth_acl,
                            'target_length': persona.get('target_length'),
                            'has_errors': False,
                            'query': query,
                            'attrs_used': attrs_used,
                            'used_correct_words': used_correct_words,
                            'word_count': count_words(query),
                            'is_ground_truth': True,
                        })
                else:
                    acl_results.append({
                        'user_id': uid,
                        'asin': persona['asin'],
                        'target_acl': acl_level,
                        'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                        'density_label': persona.get('density_label', 'simple'),
                        'length_label': persona.get('length_label', 'medium'),
                        'words_per_acl': persona.get('words_per_acl'),
                        'ground_truth_acl': ground_truth_acl,
                        'target_length': persona.get('target_length'),
                        'has_errors': False,
                        'query': query,
                        'attrs_used': attrs_used,
                        'used_correct_words': used_correct_words,
                        'word_count': count_words(query),
                        'is_ground_truth': is_ground_truth,
                    })
            else:
                acl_results.append({
                    'user_id': uid,
                    'asin': persona['asin'],
                    'target_acl': acl_level,
                    'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                    'density_label': persona.get('density_label', 'simple'),
                    'length_label': persona.get('length_label', 'medium'),
                    'words_per_acl': persona.get('words_per_acl'),
                    'ground_truth_acl': ground_truth_acl,
                    'target_length': persona.get('target_length'),
                    'has_errors': persona['has_acl_errors'],
                    'query': query,
                    'attrs_used': attrs_used,
                    'word_count': count_words(query),
                    'is_ground_truth': is_ground_truth,
                })

        # ========== CCOMP 批量查询（1次请求生成 K=0,1,2,3）==========
        correct_words = [ep['corrected'] for ep in errors['ccomp'][:10] if ep.get('corrected')] if errors and errors.get('ccomp') else None
        system_base, user_content = build_ccomp_batch_prompt(attrs, correct_words=correct_words, groundtruth_level=ground_truth_ccomp if correct_words else None)
        response = call_llm(user_content, system_base=system_base, is_acl=False)

        # 解析批量结果
        batch_results = parse_batch_query_response(response, 'ccomp')

        # 用于存储最终结果，key 为 ccomp_level
        ccomp_results_dict = {}

        if batch_results:
            for ccomp_level in range(4):
                query_key = f"ccomp_{ccomp_level}"
                is_ground_truth = (ccomp_level == ground_truth_ccomp)

                if ccomp_level not in batch_results:
                    log(f"    [DEBUG] {query_key} 缺失，user={uid}")
                    continue

                parsed = batch_results[ccomp_level]
                query = parsed['query']
                attrs_used = parsed.get('attrs_used', {})
                used_correct_words = parsed.get('used_correct_words', [])

                # 验证 that 数量
                actual_that = count_that_in_query(query)
                if actual_that != ccomp_level:
                    log(f"    [DEBUG] {query_key} that数量不匹配(期望{ccomp_level},实际{actual_that})")
                    continue

                ccomp_results_dict[ccomp_level] = {
                    'query': query,
                    'attrs_used': attrs_used,
                    'used_correct_words': used_correct_words,
                    'is_ground_truth': is_ground_truth,
                }

        # 构建最终结果
        for ccomp_level in sorted(ccomp_results_dict.keys()):
            data = ccomp_results_dict[ccomp_level]
            query = data['query']
            attrs_used = data['attrs_used']
            used_correct_words = data['used_correct_words']
            is_ground_truth = data['is_ground_truth']

            # 如果是 ground_truth 且有错误，生成 noisy 版本
            if is_ground_truth and persona['has_ccomp_errors'] and errors and errors.get('ccomp'):
                correct_words = [ep['corrected'] for ep in errors['ccomp'][:10] if ep.get('corrected')]
                if correct_words:
                    noisy_query, injected_errors = inject_errors(query, errors.get('ccomp', []))
                    # 只有当实际注入了错误时才生成双版本
                    if noisy_query != query:
                        ccomp_results.append({
                            'user_id': uid,
                            'asin': persona['asin'],
                            'target_ccomp': ccomp_level,
                            'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                            'density_label': persona.get('density_label', 'simple'),
                            'length_label': persona.get('length_label', 'medium'),
                            'words_per_ccomp': persona.get('words_per_ccomp'),
                            'ground_truth_ccomp': ground_truth_ccomp,
                            'target_length': persona.get('target_length'),
                            'has_errors': True,
                            'correct_query': query,
                            'noisy_query': noisy_query,
                            'attrs_used': attrs_used,
                            'used_correct_words': used_correct_words,
                            'error_words': injected_errors,
                            'word_count': count_words(query),
                            'is_ground_truth': True,
                        })
                    else:
                        # 没有实际注入错误，只生成单版本
                        ccomp_results.append({
                            'user_id': uid,
                            'asin': persona['asin'],
                            'target_ccomp': ccomp_level,
                            'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                            'density_label': persona.get('density_label', 'simple'),
                            'length_label': persona.get('length_label', 'medium'),
                            'words_per_ccomp': persona.get('words_per_ccomp'),
                            'ground_truth_ccomp': ground_truth_ccomp,
                            'target_length': persona.get('target_length'),
                            'has_errors': False,
                            'query': query,
                            'attrs_used': attrs_used,
                            'used_correct_words': used_correct_words,
                            'word_count': count_words(query),
                            'is_ground_truth': True,
                        })
                else:
                    ccomp_results.append({
                        'user_id': uid,
                        'asin': persona['asin'],
                        'target_ccomp': ccomp_level,
                        'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                        'density_label': persona.get('density_label', 'simple'),
                        'length_label': persona.get('length_label', 'medium'),
                        'words_per_ccomp': persona.get('words_per_ccomp'),
                        'ground_truth_ccomp': ground_truth_ccomp,
                        'target_length': persona.get('target_length'),
                        'has_errors': False,
                        'query': query,
                        'attrs_used': attrs_used,
                        'used_correct_words': used_correct_words,
                        'word_count': count_words(query),
                        'is_ground_truth': is_ground_truth,
                    })
            else:
                ccomp_results.append({
                    'user_id': uid,
                    'asin': persona['asin'],
                    'target_ccomp': ccomp_level,
                    'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                    'density_label': persona.get('density_label', 'simple'),
                    'length_label': persona.get('length_label', 'medium'),
                    'words_per_ccomp': persona.get('words_per_ccomp'),
                    'ground_truth_ccomp': ground_truth_ccomp,
                    'target_length': persona.get('target_length'),
                    'has_errors': persona['has_ccomp_errors'],
                    'query': query,
                    'attrs_used': attrs_used,
                    'used_correct_words': used_correct_words,
                    'word_count': count_words(query),
                    'is_ground_truth': is_ground_truth,
                })

        return {'acl': acl_results, 'ccomp': ccomp_results}

    results = {'acl': [], 'ccomp': []}
    failed_users = []
    total_start = time.time()
    total_users = len(user_tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_user, t): t for t in user_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results['acl'].extend(r['acl'])
                results['ccomp'].extend(r['ccomp'])
                done_users = len(set(x['user_id'] for x in results['acl']))
                log(f"  [{done_users}/{total_users}] user={r['acl'][0]['user_id'][:20] if r['acl'] else 'N/A'}")
            else:
                failed_users.append(futures[future]['user_id'])

    total_elapsed = time.time() - total_start

    # 按用户分组 ACL 结果
    acl_user_map = {}
    for r in results['acl']:
        uid = r['user_id']
        if uid not in acl_user_map:
            acl_user_map[uid] = {
                'asin': r.get('asin', ''),
                'target_length': r.get('target_length'),
                'words_per_acl': r.get('words_per_acl'),
                'ground_truth_acl': r.get('ground_truth_acl'),
                'queries': []
            }

        if r['has_errors'] and r.get('is_ground_truth'):
            acl_user_map[uid]['queries'].append({
                'acl': r['target_acl'],
                'correct_query': r['correct_query'],
                'noisy_query': r['noisy_query'],
                'error_words': r.get('error_words', []),
                'word_count': r['word_count'],
                'is_ground_truth': True,
            })
        else:
            acl_user_map[uid]['queries'].append({
                'acl': r['target_acl'],
                'query': r.get('query', ''),
                'word_count': r['word_count'],
                'is_ground_truth': r.get('is_ground_truth', False),
            })

    # 按用户分组 CCOMP 结果
    ccomp_user_map = {}
    for r in results['ccomp']:
        uid = r['user_id']
        if uid not in ccomp_user_map:
            ccomp_user_map[uid] = {
                'asin': r.get('asin', ''),
                'target_length': r.get('target_length'),
                'words_per_ccomp': r.get('words_per_ccomp'),
                'ground_truth_ccomp': r.get('ground_truth_ccomp'),
                'queries': []
            }

        if r['has_errors'] and r.get('is_ground_truth'):
            ccomp_user_map[uid]['queries'].append({
                'ccomp': r['target_ccomp'],
                'correct_query': r['correct_query'],
                'noisy_query': r['noisy_query'],
                'error_words': r.get('error_words', []),
                'word_count': r['word_count'],
                'is_ground_truth': True,
            })
        else:
            ccomp_user_map[uid]['queries'].append({
                'ccomp': r['target_ccomp'],
                'query': r.get('query', ''),
                'word_count': r['word_count'],
                'is_ground_truth': r.get('is_ground_truth', False),
            })

    # 排序
    for uid in acl_user_map:
        acl_user_map[uid]['queries'].sort(key=lambda x: x['acl'])
    for uid in ccomp_user_map:
        ccomp_user_map[uid]['queries'].sort(key=lambda x: x['ccomp'])

    # 输出 ACL 结果
    acl_output_data = [{'user_id': uid, 'asin': v['asin'],
                        'target_length': v.get('target_length'),
                        'words_per_acl': v.get('words_per_acl'),
                        'ground_truth_acl': v.get('ground_truth_acl'),
                        'queries': v['queries']}
                       for uid, v in acl_user_map.items()]

    os.makedirs(os.path.dirname(ACL_OUTPUT_FILE), exist_ok=True)
    with open(ACL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(acl_output_data, f, indent=2, ensure_ascii=False)

    # 输出 CCOMP 结果
    ccomp_output_data = [{'user_id': uid, 'asin': v['asin'],
                          'target_length': v.get('target_length'),
                          'words_per_ccomp': v.get('words_per_ccomp'),
                          'ground_truth_ccomp': v.get('ground_truth_ccomp'),
                          'queries': v['queries']}
                         for uid, v in ccomp_user_map.items()]

    os.makedirs(os.path.dirname(CCOMP_OUTPUT_FILE), exist_ok=True)
    with open(CCOMP_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ccomp_output_data, f, indent=2, ensure_ascii=False)

    # 统计
    acl_error_users = [uid for uid in acl_user_map if any(
        q.get('correct_query') for q in acl_user_map[uid]['queries']
    )]
    ccomp_error_users = [uid for uid in ccomp_user_map if any(
        q.get('correct_query') for q in ccomp_user_map[uid]['queries']
    )]

    log(f"\n{'='*60}")
    log(f"成功用户: {len(acl_user_map)}/{total_users}")
    log(f"失败用户: {len(failed_users)}/{total_users}")
    if failed_users:
        log(f"  失败用户ID: {failed_users[:5]}...")
    log(f"ACL: {len(results['acl'])} queries, {len(acl_user_map)} users")
    log(f"  有错误用户: {len(acl_error_users)} (correct+noisy 双版本)")
    log(f"CCOMP: {len(results['ccomp'])} queries, {len(ccomp_user_map)} users")
    log(f"  有错误用户: {len(ccomp_error_users)} (correct+noisy 双版本)")
    log(f"总计耗时: {total_elapsed:.1f}s")
    log(f"ACL saved to {ACL_OUTPUT_FILE}")
    log(f"CCOMP saved to {CCOMP_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
