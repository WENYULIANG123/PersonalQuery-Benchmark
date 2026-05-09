#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（共享 Level 0 三步生成流程）
=================================
1. Step 1: 生成共享 Level 0 句子（无 which/that）
2. Step 2: 基于 Level 0 生成 ACL 1-3（添加 which 从句）
3. Step 3: 基于 Level 0 生成 CCOMP 1-3（添加 that 从句）

不再包含错误注入逻辑。
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
# 加载配置
# ========================================
from config import get_category_config, get_query_config_file, get_ccomp_prompts_file

CATEGORY = "Baby_Products"
_cat_config = get_category_config(CATEGORY)

LEVEL_FILE = _cat_config['level_file']
ACL_USER_PROFILES_FILE = _cat_config['acl_user_profiles_file']
CCOMP_USER_PROFILES_FILE = _cat_config['ccomp_user_profiles_file']
ATTR_DENSITY_PROFILES_FILE = _cat_config['attr_density_profiles_file']
ATTR_VALUES_FILE = _cat_config['attr_values_file']
OUTPUT_FILE = _cat_config['output_file']

QUERY_CONFIG_FILE = get_query_config_file()
CCOMP_PROMPTS_FILE = get_ccomp_prompts_file()

# ========================================
# 加载配置和 prompt 模板
# ========================================
with open(QUERY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _CONFIG = json.load(f)
NUM_USERS_TO_TEST = _CONFIG['num_users_to_test']
MAX_WORKERS = _CONFIG['max_workers']
USE_MINIMAXIO = _CONFIG.get('use_minimaxio', False)
REQUIRED_ATTR_COUNT = 5

with open(CCOMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    _PROMPTS = json.load(f)


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


def _format_dict_key(key: str) -> str:
    """Convert metadata dict keys into readable attribute labels."""
    return re.sub(r'\s+', ' ', key.replace('_', ' ')).strip()


def _normalize_scalar_attr(value) -> str | None:
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        raw_value = str(value).strip()
    elif isinstance(value, str):
        raw_value = value.strip()
    else:
        return None
    if not raw_value:
        return None
    if ';' in raw_value:
        raw_value = raw_value.split(';')[0].strip()
    elif ',' in raw_value:
        raw_value = raw_value.split(',')[0].strip()
    return raw_value if raw_value else None


def _normalize_attr_value(value) -> str | None:
    if isinstance(value, dict):
        parts = []
        for child_key, child_value in value.items():
            child_text = _normalize_attr_value(child_value)
            if child_text:
                parts.append(f"{_format_dict_key(str(child_key))}: {child_text}")
        return '; '.join(parts) if parts else None
    if isinstance(value, list):
        for item in value:
            item_text = _normalize_attr_value(item)
            if item_text:
                return item_text
        return None
    return _normalize_scalar_attr(value)


def _extract_attrs_from_product(prod: dict) -> dict:
    """从产品中提取最多 REQUIRED_ATTR_COUNT 个非空属性，优先选择 A1, A2...

    返回格式: {"A1": {"value": "xxx", "type": "product_type"}, ...}
    选择有效的简单属性值，并将 dict 类型属性展开成可读字符串。
    规则：
    1. 跳过 A14 (size) 类型
    2. 如果字符串属性值包含分号或逗号，只取第一个值
    """
    SKIP_KEYS = {'A14'}  # 跳过 size 类型
    ATTR_TYPES = {
        'A1': 'product_type', 'A2': 'brand', 'A3': 'price', 'A4': 'appearance',
        'A5': 'use_case', 'A6': 'detailed', 'A7': 'material', 'A8': 'safety',
        'A9': 'durability', 'A10': 'ease_of_use', 'A11': 'temperature_resistance',
        'A12': 'surface', 'A13': 'reusability', 'A14': 'size', 'A15': 'weight',
        'A16': 'compatibility', 'A17': 'flavor', 'A18': 'quality',
    }
    attr_keys = [f'A{i}' for i in range(1, 19)]  # A1-A18
    attrs = {}
    for key in attr_keys:
        if len(attrs) >= REQUIRED_ATTR_COUNT:
            break
        # 跳过 A14 (size)
        if key in SKIP_KEYS:
            continue
        prod_key = f'{key}_product_type' if key == 'A1' else f'{key}_' + {
            'A2': 'brand', 'A3': 'price', 'A4': 'appearance', 'A5': 'use_case',
            'A6': 'detailed', 'A7': 'material', 'A8': 'safety', 'A9': 'durability',
            'A10': 'ease_of_use', 'A11': 'temperature_resistance', 'A12': 'surface',
            'A13': 'reusability', 'A14': 'size', 'A15': 'weight', 'A16': 'compatibility',
            'A17': 'flavor', 'A18': 'quality',
        }.get(key, '')
        raw_value = _normalize_attr_value(prod.get(prod_key))
        if raw_value:
            attrs[key] = {'value': raw_value, 'type': ATTR_TYPES.get(key, 'unknown')}
            if len(attrs) >= REQUIRED_ATTR_COUNT:
                break
    return attrs


# ========================================
# 构建 Prompt（三步）
# ========================================
def build_shared_level0_prompt(attrs: dict) -> tuple:
    """Step 1: 生成共享 Level 0 prompt"""
    system_base = _PROMPTS[f'system_base_shared_{CATEGORY}']
    user_content = _PROMPTS["user_content_shared"].format(
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=""
    )
    return system_base, user_content


def build_acl_expand_prompt(level0_query: str, attrs: dict) -> tuple:
    """Step 2: 基于 Level 0 生成 ACL 1-3"""
    system_base = _PROMPTS[f'system_base_acl_expand_{CATEGORY}']
    user_content = _PROMPTS["user_content_acl_expand"].format(
        level0_query=level0_query,
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=""
    )
    return system_base, user_content


def build_ccomp_expand_prompt(level0_query: str, attrs: dict) -> tuple:
    """Step 3: 基于 Level 0 生成 CCOMP 1-3"""
    system_base = _PROMPTS[f'system_base_ccomp_expand_{CATEGORY}']
    user_content = _PROMPTS["user_content_ccomp_expand"].format(
        level0_query=level0_query,
        A1=attrs.get('A1', ''),
        A2=attrs.get('A2', ''),
        A3=attrs.get('A3', ''),
        A4=attrs.get('A4', ''),
        A5=attrs.get('A5', ''),
        correct_words_section=""
    )
    return system_base, user_content


def _format_attrs_for_prompt(attrs: dict) -> str:
    """格式化属性用于 prompt 显示，只显示非空的属性"""
    lines = []
    for key in attrs.keys():  # 只遍历实际选中的 key
        info = attrs.get(key)
        # 处理字符串值（预热时用）或字典值（实际运行时用）
        if isinstance(info, str):
            value = info if info and info != 'None' else ''
            attr_type = 'unknown'
        else:
            value = (info.get('value', 'None') or 'None') if info else 'None'
            attr_type = info.get('type', 'unknown') if info else 'unknown'
        if value and value != 'None':
            lines.append(f"- {key} ({attr_type}): {value}")
    return '\n'.join(lines)


def _attrs_used_from_source(attrs: dict) -> dict:
    """Return the source product attributes used for both ACL and CCOMP queries."""
    if not attrs:
        raise ValueError("source attrs must not be empty")
    if len(attrs) < REQUIRED_ATTR_COUNT:
        raise ValueError(f"source attrs must contain at least {REQUIRED_ATTR_COUNT} attributes, got {len(attrs)}")
    attrs_used = {}
    for key, info in attrs.items():
        if not isinstance(info, dict):
            raise TypeError(f"source attr {key} must be a dict")
        if 'value' not in info:
            raise KeyError(f"source attr {key} is missing value")
        value = info['value']
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source attr {key}.value must be a non-empty string")
        attrs_used[key] = value.strip()
    return attrs_used


def build_direct_level0_prompt(attrs: dict) -> tuple:
    """直接生成 Level 0 查询（简单句，无 which/that）"""
    system_base = _PROMPTS[f'system_base_shared_{CATEGORY}']
    user_content = f"""Generate a Level 0 query with ZERO 'which' and ZERO 'that' clauses.

Product attributes:
{_format_attrs_for_prompt(attrs)}

Requirements:
- Output JSON only: {{"level": 0, "query": "...", "attrs_used": {{"A1": "{{value}}", ...}}}}
- Include all 5 listed attributes in the query
- Query must be a natural e-commerce search phrase in FIRST PERSON
- Level 0 = simple sentence with NO clauses (no 'which', no 'that')
- Each attribute value appears EXACTLY ONCE in the query

Example output:
{{"level": 0, "query": "I need a product that has these features", "attrs_used": {{"A1": "...", "A2": "...", "A3": "...", "A4": "...", "A5": "..."}}}}"""
    return system_base, user_content


def build_direct_acl_prompt(target_level: int, attrs: dict) -> tuple:
    """直接生成指定 ACL 等级的查询"""
    system_base = _PROMPTS[f'system_base_shared_{CATEGORY}']
    user_content = f"""Generate an ACL level {target_level} query with EXACTLY {target_level} 'which' clauses.

Product attributes:
{_format_attrs_for_prompt(attrs)}

Requirements:
- Output JSON only: {{"level": {target_level}, "query": "...", "attrs_used": {{"A1": "{{value}}", "A2": "{{value}}", ...}}}}
- Include all 5 listed attributes in the query
- Add EXACTLY {target_level} 'which' clauses (each with ONE characteristic)
- Query must be a natural e-commerce search phrase in FIRST PERSON
- Each attribute value appears EXACTLY ONCE in the query

Example output:
{{"level": {target_level}, "query": "I need ... which is ... which is ...", "attrs_used": {{"A1": "...", "A2": "...", "A3": "...", "A4": "...", "A5": "..."}}}}"""
    return system_base, user_content


def build_direct_ccomp_prompt(target_level: int, attrs: dict) -> tuple:
    """直接生成指定 CCOMP 等级的查询"""
    system_base = _PROMPTS[f'system_base_shared_{CATEGORY}']
    user_content = f"""Generate a CCOMP level {target_level} query with EXACTLY {target_level} 'that' clauses.

Product attributes:
{_format_attrs_for_prompt(attrs)}

Requirements:
- Output JSON only: {{"level": {target_level}, "query": "...", "attrs_used": {{"A1": "...", ...}}}}
- Include all 5 listed attributes in the query
- Add EXACTLY {target_level} 'that' clauses (each with ONE user need/preference)
- Query must be a natural e-commerce search phrase in FIRST PERSON
- Each attribute value appears EXACTLY ONCE in the query

Example output:
{{"level": {target_level}, "query": "I need ... that ... that ...", "attrs_used": {{"A1": "...", "A2": "...", "A3": "...", "A4": "...", "A5": "..."}}}}"""
    return system_base, user_content


# ========================================
# LLM 调用
# ========================================
_minimax_client = None
_first_request = True
_cache_warmed = False


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
    """预热 cache，用通用模板创建缓存"""
    global _cache_warmed
    if _cache_warmed:
        log("[Cache] Cache 已预热，跳过")
        return

    generic_attrs = {
        'A1': 'None',
        'A2': 'None',
        'A3': 'None',
        'A4': 'None',
        'A5': 'None'
    }

    log("[Cache] 预热 Step1 (shared level0)...")
    system, user = build_shared_level0_prompt(generic_attrs)
    call_llm(user, system_base=system, step_name="Step1_SharedLevel0")

    log("[Cache] 预热 Step2 (ACL expand)...")
    system, user = build_acl_expand_prompt("generic level0 query example", generic_attrs)
    call_llm(user, system_base=system, step_name="Step2_ACLExpand")

    log("[Cache] 预热 Step3 (CCOMP expand)...")
    system, user = build_ccomp_expand_prompt("generic level0 query example", generic_attrs)
    call_llm(user, system_base=system, step_name="Step3_CCOMPExpand")

    # 预热直接生成
    log("[Cache] 预热直接 ACL/CCOMP/Level0 生成...")
    system, user = build_direct_acl_prompt(1, generic_attrs)
    call_llm(user, system_base=system, step_name="Step4_DirectACL")
    system, user = build_direct_ccomp_prompt(1, generic_attrs)
    call_llm(user, system_base=system, step_name="Step5_DirectCCOMP")
    system, user = build_direct_level0_prompt(generic_attrs)
    call_llm(user, system_base=system, step_name="Step6_DirectLevel0")

    _cache_warmed = True
    log("[Cache] Cache 预热完成")


def call_llm(prompt: str, system_base: str = None, step_name: str = None) -> str:
    """调用 MiniMax API"""
    global _minimax_client, _first_request

    if _minimax_client is None:
        load_minimax_client()

    cache_info = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    if system_base and _first_request:
        log(f"[Request] {step_name} system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_request = False

    log(f"[Request] {step_name} user_content:\n{prompt}")

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

    log(f"[Cache] {cache_info}")
    log(f"[Response] {step_name} response:\n{response[:1500]}")

    return response


def fix_incomplete_json(text: str) -> str:
    """修复可能被截断的 JSON"""
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


# ========================================
# 解析查询结果
# ========================================
def parse_level0_response(text_content: str) -> dict:
    """解析 Level 0 响应"""
    try:
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        if 'query' in data:
            query = data.get('query', '').strip()
            attrs_used = data.get('attrs_used', {})
            if query:
                return {
                    'query': query,
                    'word_count': count_words(query),
                    'attrs_used': attrs_used,
                }
    except Exception as e:
        log(f"    [DEBUG] Level0 JSON解析失败: {e}")
    return None


def parse_direct_response(text_content: str, target_level: int) -> dict:
    """解析直接生成的查询响应"""
    try:
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        if 'query' in data:
            query = data.get('query', '').strip()
            attrs_used = data.get('attrs_used', {})
            if query:
                return {
                    'level': target_level,
                    'query': query,
                    'word_count': count_words(query),
                    'attrs_used': attrs_used,
                }
    except Exception as e:
        log(f"    [DEBUG] Direct JSON解析失败: {e}")
    return None


def parse_expand_response(text_content: str, expand_type: str) -> dict:
    """解析扩展查询响应（ACL 1-3 或 CCOMP 1-3）

    expand_type: 'acl' 或 'ccomp'
    Returns: {1: {'query': ..., 'word_count': ..., 'attrs_used': ...}, 2: {...}, 3: {...}}
    """
    try:
        json_match = re.search(r'\[[\s\S]*\]', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        if not isinstance(data, list):
            if 'level' in data:
                data = [data]
            else:
                return None

        result = {}
        for item in data:
            if isinstance(item, dict) and 'level' in item:
                level = item['level']
                if isinstance(level, str):
                    try:
                        level = int(level)
                    except (ValueError, TypeError):
                        continue
                query = item.get('query', '').strip()
                used_correct_words = item.get('used_correct_words', [])
                if query and level in [1, 2, 3]:
                    result[level] = {
                        'query': query,
                        'word_count': count_words(query),
                        'attrs_used': item.get('attrs_used', {}),
                        'used_correct_words': used_correct_words if isinstance(used_correct_words, list) else [],
                    }

        if not result:
            log(f"    [DEBUG] {expand_type} 扩展解析失败，未找到有效结果")
            return None

        return result
    except Exception as e:
        log(f"    [DEBUG] {expand_type} 扩展JSON解析失败: {e}")
    return None


def count_which_in_query(query: str) -> int:
    """计算查询中 'which' 的数量"""
    return len(re.findall(r'\bwhich\b', query, re.IGNORECASE))


def count_that_in_query(query: str) -> int:
    """计算查询中 'that' 的数量"""
    return len(re.findall(r'\bthat\b', query, re.IGNORECASE))


# ========================================
# 主函数
# ========================================
def main():
    # 加载等级文件
    log(f"加载用户等级 from {LEVEL_FILE}...")
    with open(LEVEL_FILE, 'r') as f:
        level_data = json.load(f)
    user_level_map = {u['user_id']: {'acl_level': u['acl_level'], 'ccomp_level': u['ccomp_level']} for u in level_data}
    log(f"加载了 {len(user_level_map)} 个用户的等级")

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

    # 加载属性值文件
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

    # 构建用户画像 map
    acl_profile_map = {p['user_id']: p for p in acl_user_profiles}
    ccomp_profile_map = {p['user_id']: p for p in ccomp_user_profiles}

    # ========================================
    # 详细过滤步骤
    # ========================================
    total_users = len(user_level_map)
    log(f"\n{'='*60}")
    log(f"用户过滤详细步骤")
    log(f"{'='*60}")
    log(f"Step 0: 原始用户总数: {total_users}")

    # Step 1: 只检查是否在 level.json 中
    all_user_ids = set(user_level_map.keys())
    log(f"Step 1: 检查是否在 level.json 中")
    log(f"  - level.json 用户: {len(user_level_map)}")
    log(f"  → 存在于 level.json 的用户: {len(all_user_ids)} (过滤: {total_users - len(all_user_ids)})")

    # Step 2: 过滤无产品数据的用户
    filtered_by_no_prod = 0
    temp_user_ids = set()
    for uid in all_user_ids:
        if user_prod_map.get(uid):
            temp_user_ids.add(uid)
        else:
            filtered_by_no_prod += 1
    log(f"Step 2: 过滤无产品数据的用户")
    log(f"  - 无产品数据的用户: {filtered_by_no_prod}")
    log(f"  → 剩余用户: {len(temp_user_ids)}")

    # Step 3: 过滤无 attr_density 数据的用户
    filtered_by_no_wpa = 0
    temp2_user_ids = set()
    for uid in temp_user_ids:
        if user_wpa_map.get(uid) is not None:
            temp2_user_ids.add(uid)
        else:
            filtered_by_no_wpa += 1
    log(f"Step 3: 过滤无 attr_density 数据的用户")
    log(f"  - 无 attr_density 数据的用户: {filtered_by_no_wpa}")
    log(f"  → 剩余用户: {len(temp2_user_ids)}")

    # Step 4: 过滤 ACL > 3 或 CCOMP > 3 的用户
    filtered_by_level = 0
    filtered_acl_gt3 = 0
    filtered_ccomp_gt3 = 0
    filtered_both_gt3 = 0
    temp3_user_ids = set()
    for uid in temp2_user_ids:
        level_info = user_level_map.get(uid, {})
        acl_level = level_info.get('acl_level', 0)
        ccomp_level = level_info.get('ccomp_level', 0)
        if acl_level > 3 or ccomp_level > 3:
            filtered_by_level += 1
            if acl_level > 3 and ccomp_level > 3:
                filtered_both_gt3 += 1
            elif acl_level > 3:
                filtered_acl_gt3 += 1
            else:
                filtered_ccomp_gt3 += 1
        else:
            temp3_user_ids.add(uid)
    log(f"Step 4: 过滤 ACL > 3 或 CCOMP > 3 的用户")
    log(f"  - ACL > 3: {filtered_acl_gt3}")
    log(f"  - CCOMP > 3: {filtered_ccomp_gt3}")
    log(f"  - 两者都 > 3: {filtered_both_gt3}")
    log(f"  → 剩余用户: {len(temp3_user_ids)}")

    # Step 5: 汇总有效用户
    log(f"\n{'='*60}")
    log(f"过滤完成，总计有效用户: {len(temp3_user_ids)}")
    log(f"总过滤用户: {total_users - len(temp3_user_ids)}")
    log(f"{'='*60}\n")

    # 构建用户数据
    all_user_data = []
    for uid in temp3_user_ids:
        acl_profile = acl_profile_map[uid]
        ccomp_profile = ccomp_profile_map[uid]
        prod = user_prod_map[uid][0]
        words_per_attribute = user_wpa_map.get(uid)
        level_info = user_level_map.get(uid, {})
        acl_level = level_info.get('acl_level', 0)
        ccomp_level = level_info.get('ccomp_level', 0)

        all_user_data.append({
            'uid': uid,
            'acl_profile': acl_profile,
            'ccomp_profile': ccomp_profile,
            'prod': prod,
            'words_per_attribute': words_per_attribute,
            'acl_level': acl_level,
            'ccomp_level': ccomp_level,
        })

    log(f"过滤后（ACL≤3 且 CCOMP≤3）的用户数: {len(all_user_data)}")

    # 加载已完成的用户，跳过已存在的
    completed_user_ids = set()
    existing_results = []
    if os.path.exists(OUTPUT_FILE):
        log(f"检测到已存在的 {OUTPUT_FILE}，加载已完成的用户...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            completed_user_ids = {r['user_id'] for r in existing_results if 'user_id' in r}
            log(f"  已完成用户数: {len(completed_user_ids)}")
        except json.JSONDecodeError:
            log(f"  文件格式错误，将从头开始")
            existing_results = []
            completed_user_ids = set()

    # 详细分析已完成用户与有效用户的关系
    valid_user_ids = {u['uid'] for u in all_user_data}
    completed_in_valid = completed_user_ids & valid_user_ids
    completed_not_in_valid = completed_user_ids - valid_user_ids
    log(f"\n{'='*60}")
    log(f"已完成用户分析")
    log(f"{'='*60}")
    log(f"已完成用户总数: {len(completed_user_ids)}")
    log(f"在有效用户集中的已完成用户: {len(completed_in_valid)}")
    log(f"不在有效用户集中的已完成用户（脏数据）: {len(completed_not_in_valid)}")
    log(f"{'='*60}\n")

    # 过滤掉已完成的用户
    original_count = len(all_user_data)
    all_user_data = [u for u in all_user_data if u['uid'] not in completed_user_ids]
    log(f"去除已完成后剩余用户数: {len(all_user_data)} (过滤: {original_count - len(all_user_data)})")

    # 优先选择 ACL level == CCOMP level 的用户
    same_level_users = [u for u in all_user_data if u['acl_level'] == u['ccomp_level']]
    diff_level_users = [u for u in all_user_data if u['acl_level'] != u['ccomp_level']]
    log(f"ACL/CCOMP level 相同的用户数: {len(same_level_users)}")
    log(f"ACL/CCOMP level 不同的用户数: {len(diff_level_users)}")

    # 优先选择 level 相同的用户，不够时再选 level 不同的
    target_users = same_level_users[:NUM_USERS_TO_TEST]
    if len(target_users) < NUM_USERS_TO_TEST:
        remaining = NUM_USERS_TO_TEST - len(target_users)
        target_users.extend(diff_level_users[:remaining])
        log(f"补充 {remaining} 个 level 不同的用户")

    # 预热 cache
    prewarm_cache()
    log(f"目标用户数: {len(target_users)}")

    # 构建用户任务列表
    user_tasks = []
    skipped_by_attr_count = []
    for u in target_users:
        original_attrs = _extract_attrs_from_product(u['prod'])
        if len(original_attrs) < REQUIRED_ATTR_COUNT:
            skipped_by_attr_count.append({
                'user_id': u['uid'],
                'asin': u['prod'].get('asin', ''),
                'attr_count': len(original_attrs),
            })
            continue
        persona_base = {
            'user_id': u['uid'],
            'asin': u['prod'].get('asin', ''),
            'acl_sentence_ratio': u['acl_profile'].get('acl_sentence_ratio', 0.0),
            'ccomp_sentence_ratio': u['ccomp_profile'].get('ccomp_sentence_ratio', 0.0),
            'density_label': u['acl_profile'].get('density_label', 'simple'),
            'length_label': u['acl_profile'].get('length_label', 'medium'),
            'words_per_attribute': u['words_per_attribute'],
            'acl_level': u['acl_level'],
            'ccomp_level': u['ccomp_level'],
            'original_attrs': original_attrs,
        }
        user_tasks.append(persona_base)

    if skipped_by_attr_count:
        log(f"跳过属性数少于 {REQUIRED_ATTR_COUNT} 的用户商品: {len(skipped_by_attr_count)}")
        log(f"  示例: {skipped_by_attr_count[:5]}")
    if not user_tasks:
        raise ValueError(f"No user-product tasks have at least {REQUIRED_ATTR_COUNT} extracted attributes")

    def process_one_user(persona):
        """处理单个用户，直接生成对应等级的查询"""
        uid = persona['user_id']
        attrs = persona['original_attrs']
        source_attrs_used = _attrs_used_from_source(attrs)
        target_acl_level = persona['acl_level']
        target_ccomp_level = persona['ccomp_level']

        # ========== 直接生成 ACL 查询 ==========
        acl_query_result = None
        if target_acl_level > 0:
            system_base, user_content = build_direct_acl_prompt(target_acl_level, attrs)
            acl_response = call_llm(user_content, system_base=system_base, step_name="Step1_ACL")
            acl_data = parse_direct_response(acl_response, target_acl_level)

            if not acl_data:
                log(f"    [ERROR] ACL 查询解析失败，user={uid} 标记为失败")
                return None

            query = acl_data['query']
            actual_which = count_which_in_query(query)
            if actual_which != target_acl_level:
                log(f"    [DEBUG] ACL level {target_acl_level} which数量不匹配(期望{target_acl_level},实际{actual_which})")
                log(f"    [ERROR] ACL level {target_acl_level} 验证失败，user={uid} 标记为失败")
                return None

            acl_query_result = {
                'level': target_acl_level,
                'query': query,
                'word_count': count_words(query),
                'attrs_used': dict(source_attrs_used),
            }
        else:
            # Level 0: 生成简单句查询
            system_base, user_content = build_direct_level0_prompt(attrs)
            acl_response = call_llm(user_content, system_base=system_base, step_name="Step1_ACL")
            acl_data = parse_direct_response(acl_response, 0)

            if not acl_data:
                log(f"    [ERROR] ACL Level 0 查询解析失败，user={uid} 标记为失败")
                return None

            query = acl_data['query']
            actual_which = count_which_in_query(query)
            if actual_which != 0:
                log(f"    [DEBUG] ACL Level 0 which数量不匹配(期望0,实际{actual_which})")
                log(f"    [ERROR] ACL Level 0 验证失败，user={uid} 标记为失败")
                return None

            acl_query_result = {
                'level': 0,
                'query': query,
                'word_count': count_words(query),
                'attrs_used': dict(source_attrs_used),
            }

        # ========== 直接生成 CCOMP 查询 ==========
        ccomp_query_result = None
        if target_ccomp_level > 0:
            system_base, user_content = build_direct_ccomp_prompt(target_ccomp_level, attrs)
            ccomp_response = call_llm(user_content, system_base=system_base, step_name="Step2_CCOMP")
            ccomp_data = parse_direct_response(ccomp_response, target_ccomp_level)

            if not ccomp_data:
                log(f"    [ERROR] CCOMP 查询解析失败，user={uid} 标记为失败")
                return None

            query = ccomp_data['query']
            actual_that = count_that_in_query(query)
            if actual_that != target_ccomp_level:
                log(f"    [DEBUG] CCOMP level {target_ccomp_level} that数量不匹配(期望{target_ccomp_level},实际{actual_that})")
                log(f"    [ERROR] CCOMP level {target_ccomp_level} 验证失败，user={uid} 标记为失败")
                return None

            ccomp_query_result = {
                'level': target_ccomp_level,
                'query': query,
                'word_count': count_words(query),
                'attrs_used': dict(source_attrs_used),
            }
        else:
            # Level 0: 生成简单句查询
            system_base, user_content = build_direct_level0_prompt(attrs)
            ccomp_response = call_llm(user_content, system_base=system_base, step_name="Step2_CCOMP")
            ccomp_data = parse_direct_response(ccomp_response, 0)

            if not ccomp_data:
                log(f"    [ERROR] CCOMP Level 0 查询解析失败，user={uid} 标记为失败")
                return None

            query = ccomp_data['query']
            actual_that = count_that_in_query(query)
            if actual_that != 0:
                log(f"    [DEBUG] CCOMP Level 0 that数量不匹配(期望0,实际{actual_that})")
                log(f"    [ERROR] CCOMP Level 0 验证失败，user={uid} 标记为失败")
                return None

            ccomp_query_result = {
                'level': 0,
                'query': query,
                'word_count': count_words(query),
                'attrs_used': dict(source_attrs_used),
            }

        # ========== 构建最终结果 ==========
        user_result = {
            'user_id': uid,
            'asin': persona['asin'],
            'acl_query': acl_query_result,
            'ccomp_query': ccomp_query_result,
        }

        return user_result

    results = existing_results.copy()
    failed_users = []
    total_start = time.time()
    total_users = len(user_tasks)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_user, t): t for t in user_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)
                done = len(results)
                log(f"  [{done}/{total_users}] user={r['user_id'][:20] if r['user_id'] else 'N/A'}")
                # 流式写入：每次完成后立即写入文件
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                failed_users.append(futures[future]['user_id'])

    total_elapsed = time.time() - total_start

    # 最终统计
    log(f"\n{'='*60}")
    log(f"成功用户: {len(results)}/{total_users}")
    log(f"失败用户: {len(failed_users)}/{total_users}")
    if failed_users:
        log(f"  失败用户ID: {failed_users[:5]}...")
    log(f"总计耗时: {total_elapsed:.1f}s")
    log(f"结果 saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
