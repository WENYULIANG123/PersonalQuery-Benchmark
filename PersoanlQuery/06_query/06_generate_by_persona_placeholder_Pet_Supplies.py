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
# 硬编码参数
# ========================================
CATEGORY = "Pet_Supplies"
ACL_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/acl_user_profiles.json'
CCOMP_USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/attr_density_user_profiles.json'
ATTR_VALUES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction/Pet_Supplies/attributes_Pet_Supplies.json'
OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Pet_Supplies/shared_level0_queries.json'

QUERY_CONFIG_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/query_config.json'
CCOMP_PROMPTS_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/ccomp_query_prompts.json'

# ========================================
# 加载配置和 prompt 模板
# ========================================
with open(QUERY_CONFIG_FILE, 'r', encoding='utf-8') as f:
    _CONFIG = json.load(f)
NUM_USERS_TO_TEST = _CONFIG['num_users_to_test']
MAX_WORKERS = _CONFIG['max_workers']
USE_MINIMAXIO = _CONFIG.get('use_minimaxio', False)

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

    # 计算所有用户的 ground_truth 并过滤
    all_user_ids = set(acl_profile_map.keys()) & set(ccomp_profile_map.keys()) & set(user_wpa_map.keys())
    log(f"同时存在于ACL、CCOMP和attr_density的用户数: {len(all_user_ids)}")

    all_user_data = []
    for uid in all_user_ids:
        acl_profile = acl_profile_map[uid]
        ccomp_profile = ccomp_profile_map[uid]

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
        })

    log(f"过滤后（ACL≤3 且 CCOMP≤3）的用户数: {len(all_user_data)}")

    # 预热 cache
    prewarm_cache()

    # 选择前 N 个用户
    target_users = all_user_data[:NUM_USERS_TO_TEST]
    log(f"目标用户数: {len(target_users)}")

    # 构建用户任务列表
    user_tasks = []
    for u in target_users:
        persona_base = {
            'user_id': u['uid'],
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
            'original_attrs': {
                'A1': u['prod'].get('A1_product_type', ''),
                'A2': u['prod'].get('A2_brand', ''),
                'A3': u['prod'].get('A3_price', ''),
                'A4': u['prod'].get('A4_appearance', '')[0] if isinstance(u['prod'].get('A4_appearance', ''), list) else u['prod'].get('A4_appearance', ''),
                'A5': u['prod'].get('A5_use_case', ''),
            },
        }
        user_tasks.append(persona_base)

    def process_one_user(persona):
        """处理单个用户，三步生成查询"""
        uid = persona['user_id']
        attrs = persona['original_attrs']

        # ========== Step 1: 生成共享 Level 0 ==========
        system_base, user_content = build_shared_level0_prompt(attrs)
        level0_response = call_llm(user_content, system_base=system_base, step_name="Step1_SharedLevel0")
        level0_result = parse_level0_response(level0_response)

        if not level0_result:
            log(f"    [ERROR] Level 0 解析失败，user={uid} 标记为失败")
            return None

        level0_query = level0_result['query']
        level0_attrs = level0_result.get('attrs_used', {})

        # 验证 Level 0 无 which/that
        level0_which = count_which_in_query(level0_query)
        level0_that = count_that_in_query(level0_query)
        if level0_which != 0 or level0_that != 0:
            log(f"    [DEBUG] Level 0 包含 which/that (which={level0_which}, that={level0_that})")
            log(f"    [ERROR] Level 0 验证失败，user={uid} 标记为失败")
            return None

        # 验证 Level 0 词数（15-35词）
        level0_word_count = level0_result.get('word_count', count_words(level0_query))
        if not (15 <= level0_word_count <= 35):
            log(f"    [DEBUG] Level 0 词数不在范围内 (词数={level0_word_count}, 要求=15-35)")
            log(f"    [ERROR] Level 0 词数验证失败，user={uid} 标记为失败")
            return None

        # ========== Step 2: 生成 ACL 1-3 ==========
        system_base, user_content = build_acl_expand_prompt(level0_query, attrs)
        acl_response = call_llm(user_content, system_base=system_base, step_name="Step2_ACLExpand")
        acl_results = parse_expand_response(acl_response, 'acl')

        if not acl_results or len(acl_results) != 3:
            log(f"    [ERROR] ACL 扩展解析失败，user={uid} 标记为失败")
            return None

        # 验证 ACL 结果
        for acl_level in [1, 2, 3]:
            if acl_level not in acl_results:
                log(f"    [ERROR] ACL level {acl_level} 缺失，user={uid} 标记为失败")
                return None
            query = acl_results[acl_level]['query']
            actual_which = count_which_in_query(query)
            if actual_which != acl_level:
                log(f"    [DEBUG] ACL level {acl_level} which数量不匹配(期望{acl_level},实际{actual_which})")
                log(f"    [ERROR] ACL level {acl_level} 验证失败，user={uid} 标记为失败")
                return None

        # ========== Step 3: 生成 CCOMP 1-3 ==========
        system_base, user_content = build_ccomp_expand_prompt(level0_query, attrs)
        ccomp_response = call_llm(user_content, system_base=system_base, step_name="Step3_CCOMPExpand")
        ccomp_results = parse_expand_response(ccomp_response, 'ccomp')

        if not ccomp_results or len(ccomp_results) != 3:
            log(f"    [ERROR] CCOMP 扩展解析失败，user={uid} 标记为失败")
            return None

        # 验证 CCOMP 结果
        for ccomp_level in [1, 2, 3]:
            if ccomp_level not in ccomp_results:
                log(f"    [ERROR] CCOMP level {ccomp_level} 缺失，user={uid} 标记为失败")
                return None
            query = ccomp_results[ccomp_level]['query']
            actual_that = count_that_in_query(query)
            if actual_that != ccomp_level:
                log(f"    [DEBUG] CCOMP level {ccomp_level} that数量不匹配(期望{ccomp_level},实际{actual_that})")
                log(f"    [ERROR] CCOMP level {ccomp_level} 验证失败，user={uid} 标记为失败")
                return None

        # ========== 构建最终结果 ==========
        user_result = {
            'user_id': uid,
            'asin': persona['asin'],
            'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
            'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
            'density_label': persona.get('density_label', 'simple'),
            'length_label': persona.get('length_label', 'medium'),
            'words_per_attribute': persona.get('words_per_attribute'),
            'words_per_acl': persona.get('words_per_acl'),
            'words_per_ccomp': persona.get('words_per_ccomp'),
            'target_length': persona.get('target_length'),
            'ground_truth_acl': persona.get('ground_truth_acl'),
            'ground_truth_ccomp': persona.get('ground_truth_ccomp'),
            'level0': {
                'query': level0_query,
                'word_count': level0_result['word_count'],
                'attrs_used': level0_attrs,
            },
            'acl_queries': [],
            'ccomp_queries': [],
        }

        for acl_level in [1, 2, 3]:
            entry = acl_results[acl_level]
            is_ground_truth = (acl_level == persona.get('ground_truth_acl'))
            user_result['acl_queries'].append({
                'level': acl_level,
                'query': entry['query'],
                'word_count': entry['word_count'],
                'attrs_used': entry.get('attrs_used', {}),
                'is_ground_truth': is_ground_truth,
            })

        for ccomp_level in [1, 2, 3]:
            entry = ccomp_results[ccomp_level]
            is_ground_truth = (ccomp_level == persona.get('ground_truth_ccomp'))
            user_result['ccomp_queries'].append({
                'level': ccomp_level,
                'query': entry['query'],
                'word_count': entry['word_count'],
                'attrs_used': entry.get('attrs_used', {}),
                'is_ground_truth': is_ground_truth,
            })

        return user_result

    results = []
    failed_users = []
    total_start = time.time()
    total_users = len(user_tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_user, t): t for t in user_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)
                done = len(results)
                log(f"  [{done}/{total_users}] user={r['user_id'][:20] if r['user_id'] else 'N/A'}")
            else:
                failed_users.append(futures[future]['user_id'])

    total_elapsed = time.time() - total_start

    # 输出结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 统计
    log(f"\n{'='*60}")
    log(f"成功用户: {len(results)}/{total_users}")
    log(f"失败用户: {len(failed_users)}/{total_users}")
    if failed_users:
        log(f"  失败用户ID: {failed_users[:5]}...")
    log(f"总计耗时: {total_elapsed:.1f}s")
    log(f"结果 saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
