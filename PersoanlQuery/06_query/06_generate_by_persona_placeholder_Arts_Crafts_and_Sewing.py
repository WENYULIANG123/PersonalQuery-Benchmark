#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（ACL+CCOMP合并版本）
=================================
一次性生成 ACL (acl_0~3) 和 CCOMP (ccomp_0~3) 共8个版本
将属性词替换为 A1-A5 占位符，用于测试语言结构
有错误的用户生成 correct + noisy 双版本
参数硬编码，勿改动
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
CATEGORY = "Arts_Crafts_and_Sewing"
BATCH_SIZE = 100
ACL_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Arts_Crafts_and_Sewing/acl_user_profiles.json'
CCOMP_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Arts_Crafts_and_Sewing/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Arts_Crafts_and_Sewing/attr_density_user_profiles.json'
USER_ERROR_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Arts_Crafts_and_Sewing/acl_ccomp_error.json'
ACL_OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Arts_Crafts_and_Sewing/acl_query.json'
CCOMP_OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Arts_Crafts_and_Sewing/ccomp_query.json'

PROMPT_TEMPLATE_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/acl_ccomp_query_prompt.json'

# ========================================
# 加载 prompt 模板
# ========================================
with open(PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    _TEMPLATE = json.load(f)

NUM_USERS_TO_TEST = _TEMPLATE['num_users_to_test']
MAX_WORKERS = _TEMPLATE['max_workers']
USE_MINIMAXIO = _TEMPLATE.get('use_minimaxio', False)
_BASE_SYSTEM_TEMPLATE = _TEMPLATE['system_base_template']
USER_CONTENT_WITH_ERRORS = _TEMPLATE['user_content_with_errors']
USER_CONTENT_NO_ERRORS = _TEMPLATE['user_content_no_errors']

# ========================================
# 类别示例映射
# ========================================
CATEGORY_EXAMPLES = {
    "Pet_Supplies": {
        "a1_examples": "dog food, cat toy, aquarium",
        "a2_examples": "Purina, Kong, Fluval",
        "a4_examples": "waterproof, large size, colorful",
        "a5_examples": "for puppies, for training, for outdoor use",
    },
    "Grocery_and_Gourmet_Food": {
        "a1_examples": "coffee, tea, spices",
        "a2_examples": "Starbucks, Lipton",
        "a4_examples": "organic, dark roast, green color",
        "a5_examples": "for baking, for cooking, for gifting",
    },
    "Arts_Crafts_and_Sewing": {
        "a1_examples": "markers, fabric, ribbon",
        "a2_examples": "Darice, Fiskars",
        "a4_examples": "red color, smooth finish, vibrant hues",
        "a5_examples": "for calligraphy, for painting, for crafting",
    },
}


def get_system_base(category: str) -> str:
    """根据类别返回填充后的 system_base"""
    examples = CATEGORY_EXAMPLES.get(category, CATEGORY_EXAMPLES["Pet_Supplies"])
    return _BASE_SYSTEM_TEMPLATE.format(**examples)


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
# 加载用户错误数据（统一文件，区分ACL/CCOMP）
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

            # 使用 set 去重 (original, corrected) 组合
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
# 构建一次性返回8个版本的Prompt（ACL+CCOMP）
# ========================================
def build_user_prompt(persona: dict, error_patterns: dict = None) -> tuple:
    """为用户构建请求8个版本的prompt（acl_0~3 + ccomp_0~3）
    Returns: (system_base, user_content)

    error_patterns: {'acl': [...], 'ccomp': [...]} 或 None
    """
    system_base = get_system_base(CATEGORY)

    has_acl_errors = error_patterns is not None and len(error_patterns.get('acl', [])) > 0
    has_ccomp_errors = error_patterns is not None and len(error_patterns.get('ccomp', [])) > 0
    has_errors = has_acl_errors or has_ccomp_errors

    target_length = persona.get('target_length', 20)

    if has_errors:
        # 有错误用户：构建正确词列表（合并ACL和CCOMP的正确词）
        correct_words_str = ""
        if has_acl_errors:
            for ep in error_patterns['acl'][:10]:
                corr = ep.get("corrected", "")
                correct_words_str += f"- \"{corr}\" (acl error)\n"
        if has_ccomp_errors:
            for ep in error_patterns['ccomp'][:10]:
                corr = ep.get("corrected", "")
                correct_words_str += f"- \"{corr}\" (ccomp error)\n"

        user_content = USER_CONTENT_WITH_ERRORS.format(
            target_length=target_length,
            correct_words=correct_words_str if correct_words_str else "(none)"
        )
    else:
        # 无错误用户：8版本
        user_content = USER_CONTENT_NO_ERRORS.format(target_length=target_length)

    return system_base, user_content


# ========================================
# LLM 调用
# ========================================
_minimax_client = None
_first_request = True


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


def call_llm(prompt: str, system_base: str = None) -> str:
    """调用 MiniMax API，支持可选的系统提示词缓存"""
    global _minimax_client, _first_request

    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # 第一次请求时打印 system_base（创建缓存）
    if system_base and _first_request:
        log(f"[Request] system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_request = False

    # 打印 user_content
    log(f"[Request] user_content:\n{prompt}")

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

    # 修复被截断的 JSON
    text = fix_incomplete_json(response)
    return text


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
# 解析8版本查询结果
# ========================================
def parse_8_versions_query(text_content: str, ground_truth_acl: int = 0, ground_truth_ccomp: int = 0) -> dict:
    """解析包含8个版本的JSON（acl_0~3 + ccomp_0~3），每个版本有3个变体

    ground_truth_acl: 用户的真实acl级别
    ground_truth_ccomp: 用户的真实ccomp级别

    每个版本返回3个变体，后处理时会选择符合要求的一个
    """
    try:
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        result = {}

        # 解析 ACL 版本（每个版本有3个变体）
        for acl_level in ['acl_0', 'acl_1', 'acl_2', 'acl_3']:
            if acl_level not in data:
                return None
            items = data[acl_level]
            acl_num = int(acl_level.split('_')[1])
            is_ground_truth = (acl_num == ground_truth_acl)

            # 支持字符串或数组格式
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

            result[acl_level] = {
                'variants': variants,
                'is_ground_truth': is_ground_truth,
            }

        # 解析 CCOMP 版本（每个版本有3个变体）
        for ccomp_level in ['ccomp_0', 'ccomp_1', 'ccomp_2', 'ccomp_3']:
            if ccomp_level not in data:
                return None
            items = data[ccomp_level]
            ccomp_num = int(ccomp_level.split('_')[1])
            is_ground_truth = (ccomp_num == ground_truth_ccomp)

            # 支持字符串或数组格式
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

            result[ccomp_level] = {
                'variants': variants,
                'is_ground_truth': is_ground_truth,
            }

        return result
    except Exception as e:
        log(f"    [DEBUG] 8版本JSON解析失败: {e}, text_content={repr(text_content[:500])}")
    return None


def count_which_in_query(query: str) -> int:
    """计算查询中 'which' 的数量"""
    return len(re.findall(r'\bwhich\b', query, re.IGNORECASE))


def count_that_in_query(query: str) -> int:
    """计算查询中 'that' 的数量"""
    return len(re.findall(r'\bthat\b', query, re.IGNORECASE))


def validate_clause_count(query: str, clause_type: str, expected_count: int) -> bool:
    """验证查询中的从词数量是否符合要求

    clause_type: 'acl' (检查which) 或 'ccomp' (检查that)
    expected_count: 期望的从词数量 (0-3)
    """
    if clause_type == 'acl':
        actual_count = count_which_in_query(query)
    else:  # ccomp
        actual_count = count_that_in_query(query)

    return actual_count == expected_count


def select_best_variant(variants: list, attrs: dict, clause_type: str, expected_count: int) -> dict:
    """从多个变体中选择第一个符合要求的变体

    返回第一个符合以下条件的变体：
    1. 占位符完整（A1-A5都存在）
    2. 从句数量符合要求

    如果找不到符合条件的变体，返回 None
    """
    for variant in variants:
        query = variant['query']

        # 验证占位符
        validation = validate_placeholders(query)
        missing = [ph for ph, present in validation.items() if not present]
        if missing:
            continue

        # 验证从句数量
        if not validate_clause_count(query, clause_type, expected_count):
            continue

        # 填充占位符
        filled_query = fill_placeholders(query, attrs)

        return {
            'query': query,
            'filled_query': filled_query,
            'word_count': variant['word_count'],
        }

    # 没有找到符合要求的变体
    return None


# ========================================
# 验证占位符
# ========================================
def validate_placeholders(query: str) -> dict:
    """验证查询中是否正确使用了所有5个占位符"""
    placeholders = {
        'A1': 'A1' in query,
        'A2': 'A2' in query,
        'A3': 'A3' in query,
        'A4': 'A4' in query,
        'A5': 'A5' in query,
    }
    return placeholders


def fill_placeholders(query: str, attrs: dict) -> str:
    """将占位符替换为实际属性值"""
    result = query
    for key in ['A1', 'A2', 'A3', 'A4', 'A5']:
        val = attrs.get(key, '')
        if isinstance(val, list):
            val = ', '.join(str(v) for v in val)
        result = result.replace(key, str(val))
    return result


def is_pure_suffix_change(orig: str, corr: str) -> bool:
    """检查是否只是后缀变化（需要过滤）

    例如：
    - "have" -> "having" (只加了ing后缀)
    - "fitted" -> "fitting" (只改了后缀)
    - "play" -> "playing" (只加了ing后缀)
    - "cat" -> "cats" (只加了s后缀)
    - "interesting" -> "interest" (只删了ing后缀)

    不算纯后缀变化（应该保留）：
    - "unhappy" -> "happy" (删除了前缀un-)
    - "rewrite" -> "write" (删除了前缀re-)
    - "fit" -> "fitted" (加了t+ed，不是纯后缀)
    - "big" -> "bigger" (加了g+er，不是纯后缀)
    """
    # 计算公共前缀
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

    # 核心后缀列表
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
    """检查是否只是标点符号变化（需要过滤）

    例如：
    - "." -> ","
    - "!" -> "."
    - ":" -> ";"

    不是纯标点变化（应该保留）：
    - "don't" -> "dont" (撇号被删除，但词干相同)
    """
    import string
    # 判断两个字符串是否只包含标点符号
    # 如果 original 和 corrected 都只包含标点符号（无字母数字），则是纯标点变化
    orig_chars = set(c for c in orig if c not in string.ascii_letters + string.digits + string.whitespace)
    corr_chars = set(c for c in corr if c not in string.ascii_letters + string.digits + string.whitespace)

    # 两个字符串都只包含标点符号
    orig_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in orig)
    corr_only_punct = all(c not in string.ascii_letters + string.digits + string.whitespace for c in corr)

    if orig_only_punct and corr_only_punct:
        return True

    return False


def filter_error_patterns(error_patterns: list) -> list:
    """过滤掉不需要的错误类型：
    - 连字符错误
    - 空格错误
    - 纯后缀变化错误
    - 纯标点变化错误
    """
    if not error_patterns:
        return []

    filtered = []
    for ep in error_patterns:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")

        # 跳过空值
        if not orig or not corr:
            continue

        # 跳过连字符错误
        if '-' in orig or '-' in corr:
            continue

        # 跳过空格错误（多余空格、少空格等）
        if ' ' in orig or ' ' in corr:
            if orig.strip() != corr.strip():
                continue

        # 跳过纯后缀变化
        if is_pure_suffix_change(orig, corr):
            continue

        # 跳过纯标点变化
        if is_pure_punctuation_change(orig, corr):
            continue

        filtered.append(ep)

    return filtered


def validate_at_least_one_correct_word(query: str, error_patterns: list) -> bool:
    """验证查询中是否至少包含一个正确词"""
    if not error_patterns:
        return True

    for ep in error_patterns:
        corr = ep.get("corrected", "")
        if corr and corr.lower() in query.lower():
            return True

    return False


def inject_errors(query: str, error_patterns: list) -> str:
    """将查询中的正确词替换为错误词（仅处理过滤后的错误）"""
    if not error_patterns:
        return query

    # 先过滤错误
    filtered_patterns = filter_error_patterns(error_patterns)
    if not filtered_patterns:
        return query

    result = query
    for ep in filtered_patterns[:10]:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        if orig and corr:
            result = re.sub(re.escape(corr), orig, result, flags=re.IGNORECASE)
    return result
    result = query
    for ep in error_patterns[:10]:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        if orig and corr:
            result = result.replace(corr, orig)
    return result


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

    # 加载 attr_density 用户画像
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

    # 加载用户错误画像
    log(f"加载用户错误画像 from {USER_ERROR_FILE}...")
    user_errors = {}
    if os.path.exists(USER_ERROR_FILE):
        raw_errors = load_user_errors(USER_ERROR_FILE)
        log(f"加载了 {len(raw_errors)} 个有错误的用户")

        # 过滤错误：去掉连字符错误、空格错误、纯后缀变化、纯标点变化
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
        log(f"错误文件不存在，跳过错误处理（所有用户按无错误处理）")

    # 构建用户画像 map
    acl_profile_map = {p['user_id']: p for p in acl_user_profiles}
    ccomp_profile_map = {p['user_id']: p for p in ccomp_user_profiles}

    # 第一步：计算所有用户的 ground_truth 并过滤
    all_user_ids = set(acl_profile_map.keys()) & set(ccomp_profile_map.keys()) & set(user_wpa_map.keys())
    log(f"同时存在于ACL、CCOMP和attr_density的用户数: {len(all_user_ids)}")

    all_user_data = []
    for uid in all_user_ids:
        acl_profile = acl_profile_map[uid]
        ccomp_profile = ccomp_profile_map[uid]

        acl_products = acl_profile.get('products', [])
        ccomp_products = ccomp_profile.get('products', [])
        if not acl_products or not ccomp_products:
            continue
        prod = acl_products[0]  # 使用同一个产品

        # 获取 words_per_attribute
        words_per_attribute = user_wpa_map.get(uid)
        if words_per_attribute is None:
            continue

        # 获取 words_per_acl
        words_per_acl = acl_profile.get('words_per_acl')
        if words_per_acl is None:
            words_per_acl = 100.0
        else:
            words_per_acl = float(words_per_acl)

        # 获取 words_per_ccomp
        words_per_ccomp = ccomp_profile.get('words_per_ccomp')
        if words_per_ccomp is None:
            words_per_ccomp = 100.0
        else:
            words_per_ccomp = float(words_per_ccomp)

        # 计算 target_length
        target_length = math.ceil(words_per_attribute) * 5

        # 计算 ground_truth_acl
        if words_per_acl and words_per_acl > 0:
            ground_truth_acl = int(target_length / words_per_acl)
            ground_truth_acl = max(0, min(5, ground_truth_acl))
        else:
            ground_truth_acl = 0

        # 计算 ground_truth_ccomp
        if words_per_ccomp and words_per_ccomp > 0:
            ground_truth_ccomp = int(target_length / words_per_ccomp)
            ground_truth_ccomp = max(0, min(5, ground_truth_ccomp))
        else:
            ground_truth_ccomp = 0

        # 过滤：ground_truth_acl > 3 或 ground_truth_ccomp > 3 的用户
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

    # 第二步：按优先级选用户
    # 优先级：ACL+CCOMP错误 > 只有CCOMP > 只有ACL > 无错误
    both_errors = [u for u in all_user_data if u['has_acl_errors'] and u['has_ccomp_errors']]
    only_ccomp = [u for u in all_user_data if u['has_ccomp_errors'] and not u['has_acl_errors']]
    only_acl = [u for u in all_user_data if u['has_acl_errors'] and not u['has_ccomp_errors']]
    no_errors = [u for u in all_user_data if not (u['has_acl_errors'] or u['has_ccomp_errors'])]

    target_users = []
    remaining = NUM_USERS_TO_TEST

    # 优先级1: ACL+CCOMP错误用户
    if remaining > 0:
        take = min(remaining, len(both_errors))
        target_users.extend(both_errors[:take])
        remaining -= take

    # 优先级2: 只有CCOMP错误用户
    if remaining > 0:
        take = min(remaining, len(only_ccomp))
        target_users.extend(only_ccomp[:take])
        remaining -= take

    # 优先级3: 只有ACL错误用户
    if remaining > 0:
        take = min(remaining, len(only_acl))
        target_users.extend(only_acl[:take])
        remaining -= take

    # 优先级4: 无错误用户
    if remaining > 0:
        take = min(remaining, len(no_errors))
        target_users.extend(no_errors[:take])
        remaining -= take

    has_error_count = sum(1 for u in target_users if u['has_acl_errors'] or u['has_ccomp_errors'])
    both_count = sum(1 for u in target_users if u['has_acl_errors'] and u['has_ccomp_errors'])
    ccomp_only = sum(1 for u in target_users if u['has_ccomp_errors'] and not u['has_acl_errors'])
    acl_only = sum(1 for u in target_users if u['has_acl_errors'] and not u['has_ccomp_errors'])
    log(f"目标用户: {len(target_users)} 个（ACL+CCOMP错误: {both_count}, 仅CCOMP: {ccomp_only}, 仅ACL: {acl_only}, 无错误: {len(target_users)-has_error_count}）")

    # 构建用户任务列表（每个用户一个任务，一次性返回8个版本）
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
                'A4': u['prod'].get('A4_appearance', ''),
                'A5': u['prod'].get('A5_use_case', ''),
            },
            'errors': errors,
        }
        user_tasks.append(persona_base)

    log(f"构建了 {len(user_tasks)} 个用户任务")
    log(f"开始处理，并发数={MAX_WORKERS}")

    def process_one_user(persona):
        """处理单个用户，一次返回8个版本（acl_0~3 + ccomp_0~3）"""
        uid = persona['user_id']
        attrs = persona['original_attrs']
        errors = persona['errors']
        ground_truth_acl = persona['ground_truth_acl']
        ground_truth_ccomp = persona['ground_truth_ccomp']

        # 构建 prompt
        system_base, user_content = build_user_prompt(persona, errors)
        text = call_llm(user_content, system_base=system_base)

        # 解析8个版本
        query_data = parse_8_versions_query(text, ground_truth_acl, ground_truth_ccomp)
        if not query_data:
            return None

        # 构建结果
        acl_results = []
        ccomp_results = []

        # 处理 ACL 版本（从3个变体中选择符合要求的）
        for acl_level in ['acl_0', 'acl_1', 'acl_2', 'acl_3']:
            target_acl = int(acl_level.split('_')[1])
            qdata = query_data[acl_level]
            has_acl_err = persona['has_acl_errors']

            # 从变体中选择最佳结果
            result = select_best_variant(
                qdata['variants'], attrs, 'acl', target_acl
            )

            if result is None:
                # 收集失败原因
                reasons = []
                for variant in qdata['variants']:
                    query = variant['query']
                    validation = validate_placeholders(query)
                    missing = [ph for ph, present in validation.items() if not present]
                    if missing:
                        reasons.append(f"缺少占位符: {missing}")
                    else:
                        actual = count_which_in_query(query)
                        reasons.append(f"which数量不匹配(期望{target_acl},实际{actual})")
                log(f"    [DEBUG] {acl_level} 所有变体都失败: {reasons[:2]}")
                return None

            selected = result

            if qdata.get('is_ground_truth') and has_acl_err and errors:
                correct_filled = selected['filled_query']
                # 验证至少有一个正确词存在
                if not validate_at_least_one_correct_word(correct_filled, errors.get('acl', [])):
                    log(f"    [DEBUG] {acl_level} 没有正确词，判定为失败")
                    return None
                else:
                    noisy_filled = inject_errors(correct_filled, errors.get('acl', []))
                    acl_results.append({
                        'user_id': uid,
                        'asin': persona['asin'],
                        'target_acl': target_acl,
                        'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                        'density_label': persona.get('density_label', 'simple'),
                        'length_label': persona.get('length_label', 'medium'),
                        'words_per_acl': persona.get('words_per_acl'),
                        'ground_truth_acl': ground_truth_acl,
                        'target_length': persona.get('target_length'),
                        'has_errors': True,
                        'correct_query': correct_filled,
                        'noisy_query': noisy_filled,
                        'error_words': [{'correct': ep['corrected'], 'error': ep['original'], 'error_type': ep.get('error_type', 'unknown')} for ep in (errors.get('acl', []) or [])[:10]],
                        'word_count': selected['word_count'],
                        'is_ground_truth': True,
                    })
            else:
                acl_results.append({
                    'user_id': uid,
                    'asin': persona['asin'],
                    'target_acl': target_acl,
                    'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                    'density_label': persona.get('density_label', 'simple'),
                    'length_label': persona.get('length_label', 'medium'),
                    'words_per_acl': persona.get('words_per_acl'),
                    'ground_truth_acl': ground_truth_acl,
                    'target_length': persona.get('target_length'),
                    'has_errors': has_acl_err,
                    'filled_query': selected['filled_query'],
                    'word_count': selected['word_count'],
                    'is_ground_truth': qdata.get('is_ground_truth', False),
                })

        # 处理 CCOMP 版本（从3个变体中选择符合要求的）
        for ccomp_level in ['ccomp_0', 'ccomp_1', 'ccomp_2', 'ccomp_3']:
            target_ccomp = int(ccomp_level.split('_')[1])
            qdata = query_data[ccomp_level]
            has_ccomp_err = persona['has_ccomp_errors']

            # 从变体中选择最佳结果
            result = select_best_variant(
                qdata['variants'], attrs, 'ccomp', target_ccomp
            )

            if result is None:
                # 收集失败原因
                reasons = []
                for variant in qdata['variants']:
                    query = variant['query']
                    validation = validate_placeholders(query)
                    missing = [ph for ph, present in validation.items() if not present]
                    if missing:
                        reasons.append(f"缺少占位符: {missing}")
                    else:
                        actual = count_that_in_query(query)
                        reasons.append(f"that数量不匹配(期望{target_ccomp},实际{actual})")
                log(f"    [DEBUG] {ccomp_level} 所有变体都失败: {reasons[:2]}")
                return None

            selected = result

            if qdata.get('is_ground_truth') and has_ccomp_err and errors:
                correct_filled = selected['filled_query']
                # 验证至少有一个正确词存在
                if not validate_at_least_one_correct_word(correct_filled, errors.get('ccomp', [])):
                    log(f"    [DEBUG] {ccomp_level} 没有正确词，判定为失败")
                    return None
                else:
                    noisy_filled = inject_errors(correct_filled, errors.get('ccomp', []))
                    ccomp_results.append({
                        'user_id': uid,
                        'asin': persona['asin'],
                        'target_ccomp': target_ccomp,
                        'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                        'density_label': persona.get('density_label', 'simple'),
                        'length_label': persona.get('length_label', 'medium'),
                        'words_per_ccomp': persona.get('words_per_ccomp'),
                        'ground_truth_ccomp': ground_truth_ccomp,
                        'target_length': persona.get('target_length'),
                        'has_errors': True,
                        'correct_query': correct_filled,
                        'noisy_query': noisy_filled,
                        'error_words': [{'correct': ep['corrected'], 'error': ep['original'], 'error_type': ep.get('error_type', 'unknown')} for ep in (errors.get('ccomp', []) or [])[:10]],
                        'word_count': selected['word_count'],
                        'is_ground_truth': True,
                    })
            else:
                ccomp_results.append({
                    'user_id': uid,
                    'asin': persona['asin'],
                    'target_ccomp': target_ccomp,
                    'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                    'density_label': persona.get('density_label', 'simple'),
                    'length_label': persona.get('length_label', 'medium'),
                    'words_per_ccomp': persona.get('words_per_ccomp'),
                    'ground_truth_ccomp': ground_truth_ccomp,
                    'target_length': persona.get('target_length'),
                    'has_errors': has_ccomp_err,
                    'filled_query': selected['filled_query'],
                    'word_count': selected['word_count'],
                    'is_ground_truth': qdata.get('is_ground_truth', False),
                })

        return {'acl': acl_results, 'ccomp': ccomp_results}

    results = {'acl': [], 'ccomp': []}
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
                err_tag = " [error user]" if r['acl'][0]['has_errors'] else ""
                log(f"  [{done_users}/{total_users}] user={r['acl'][0]['user_id'][:20]}{err_tag}")

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
                'filled_query': r['filled_query'],
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
                'filled_query': r['filled_query'],
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
    log(f"ACL: {len(results['acl'])} queries, {len(acl_user_map)} users")
    log(f"  有错误用户: {len(acl_error_users)} (correct+noisy 双版本)")
    log(f"CCOMP: {len(results['ccomp'])} queries, {len(ccomp_user_map)} users")
    log(f"  有错误用户: {len(ccomp_error_users)} (correct+noisy 双版本)")
    log(f"总计耗时: {total_elapsed:.1f}s")
    log(f"ACL saved to {ACL_OUTPUT_FILE}")
    log(f"CCOMP saved to {CCOMP_OUTPUT_FILE}")


if __name__ == '__main__':
    main()
