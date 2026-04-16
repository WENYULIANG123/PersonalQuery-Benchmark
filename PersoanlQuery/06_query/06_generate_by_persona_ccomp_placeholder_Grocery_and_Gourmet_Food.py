#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（ccomp占位符版本）
=================================
将属性词替换为 A1-A5 占位符，用于测试语言结构
有错误的用户生成 correct + noisy 双版本
参数硬编码，勿改动

CCOMP 错误类型（4类）：
1. clause-shell typo: think→thikn, believe→belive 等壳层词拼写错误
2. complement-linking error: that/if/whether 连接处错误
3. modal/auxiliary distortion: would→woudl, could→cuold 等情态动词错误
4. clause-boundary structural writing error: 从句边界结构错误（如主谓不一致）
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
BATCH_SIZE = 100
USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Grocery_and_Gourmet_Food/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Grocery_and_Gourmet_Food/attr_density_user_profiles.json'
USER_ERROR_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Grocery_and_Gourmet_Food/ccomp_error.json'
OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Grocery_and_Gourmet_Food/ccomp_query.json'
PROMPT_TEMPLATE_FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/ccomp_prompt_template.json'

# ========================================
# 加载 prompt 模板
# ========================================
with open(PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    _PROMPT_TEMPLATE = json.load(f)

NUM_USERS_TO_TEST = _PROMPT_TEMPLATE['num_users_to_test']
MAX_WORKERS = _PROMPT_TEMPLATE['max_workers']
_BASE_SYSTEM_TEMPLATE = _PROMPT_TEMPLATE['system_base_template']
USER_CONTENT_WITH_ERRORS = _PROMPT_TEMPLATE['user_content_with_errors']
USER_CONTENT_NO_ERRORS = _PROMPT_TEMPLATE['user_content_no_errors']

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
# 加载用户错误数据
# ========================================
def load_user_errors(error_file: str) -> dict:
    """加载用户错误画像，返回 {uid: [{original, corrected, error_type}, ...]}"""
    with open(error_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_errors = {}
    for user in data.get('user_results', []):
        uid = user['user_id']
        if user['total_errors'] == 0 or not user.get('detailed_results'):
            continue
        # 使用 set 去重 (original, corrected) 组合
        seen = set()
        patterns = []
        for detail in user['detailed_results']:
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
        if patterns:
            user_errors[uid] = patterns

    return user_errors


# ========================================
# 构建一次性返回4个ccomp版本的Prompt
# ========================================
def build_user_prompt(persona: dict, error_patterns: list = None) -> tuple:
    """为用户构建请求4个ccomp版本的prompt
    Returns: (system_base, user_content)

    逻辑：
    - 如果用户有错误：所有级别返回单版本（含正确词），错误注入在后处理完成
    - 如果用户无错误：所有级别返回普通单版本
    """
    # System base - 根据类别填充
    system_base = get_system_base(CATEGORY)

    has_errors = error_patterns is not None
    target_length = persona.get('target_length', 20)

    if has_errors:
        # 有错误用户：构建正确词列表
        correct_words_str = ""
        for ep in error_patterns[:10]:
            corr = ep.get("corrected", "")
            correct_words_str += f"- \"{corr}\"\n"

        user_content = USER_CONTENT_WITH_ERRORS.format(
            target_length=target_length,
            correct_words=correct_words_str if correct_words_str else "(none)"
        )
    else:
        # 无错误用户：简单4版本
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
        from llm_client import MiniMaxAnthropicClient
        _minimax_client = MiniMaxAnthropicClient()
        print(" MiniMax API 客户端初始化完成")


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
        # 使用缓存版本的 API
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
# 解析结果（无错误用户）
# ========================================
def parse_query(text_content: str) -> dict:
    """解析JSON查询"""
    try:
        if not text_content:
            log(f"    [DEBUG] JSON解析失败: 空回复")
            return None
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        query = data.get('query', '').strip()
        if query:
            return {
                'query': query,
                'word_count': count_words(query),
            }
    except Exception as e:
        log(f"    [DEBUG] JSON解析失败: {e}, 内容: {text_content if text_content else '空'}")
    return None


# ========================================
# 解析结果（有错误用户 - 双版本）
# ========================================
def parse_query_with_errors(text_content: str) -> dict:
    """解析包含correct_query和error_query的JSON"""
    try:
        if not text_content:
            log(f"    [DEBUG] JSON解析失败: 空回复")
            return None
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        correct_q = data.get('correct_query', '').strip()
        error_q = data.get('error_query', '').strip()
        error_words = data.get('error_words', [])

        if correct_q and error_q:
            return {
                'correct_query': correct_q,
                'error_query': error_q,
                'error_words': error_words,
                'word_count': count_words(correct_q),
            }
    except Exception as e:
        log(f"    [DEBUG] JSON解析失败: {e}, 内容: {text_content if text_content else '空'}")
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


# ========================================
# 主函数
# ========================================
def main():
    # 加载用户画像
    log(f"加载用户画像 from {USER_PROFILES_FILE}...")
    with open(USER_PROFILES_FILE, 'r') as f:
        all_user_profiles = json.load(f)
    log(f"加载了 {len(all_user_profiles)} 个用户画像")

    # 加载 attr_density 用户画像，获取 words_per_attribute
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

    # 统计同时存在于两个文件的用户数
    ccomp_user_ids = set(p['user_id'] for p in all_user_profiles)
    common_user_ids = ccomp_user_ids & set(user_wpa_map.keys())
    log(f"同时存在于ccomp和attr_density的用户数: {len(common_user_ids)}")

    # 加载用户错误画像（ccomp_error.json可能不存在，跳过即可）
    log(f"加载用户错误画像 from {USER_ERROR_FILE}...")
    user_errors = {}
    if os.path.exists(USER_ERROR_FILE):
        user_errors = load_user_errors(USER_ERROR_FILE)
        log(f"加载了 {len(user_errors)} 个有错误的用户")
    else:
        log(f"错误文件不存在，跳过错误处理（所有用户按无错误处理）")

    # 构建用户画像 map
    profile_map = {p['user_id']: p for p in all_user_profiles}

    # 筛选目标用户（优先选有错误的用户，但必须在两个画像中都存在）
    error_uids = [uid for uid in user_errors if uid in profile_map and uid in user_wpa_map]
    normal_uids = [uid for uid in profile_map if uid not in user_errors and uid in user_wpa_map]
    target_uids = error_uids[:NUM_USERS_TO_TEST]
    remaining = NUM_USERS_TO_TEST - len(target_uids)
    if remaining > 0:
        target_uids += normal_uids[:remaining]
    has_error_count = sum(1 for uid in target_uids if uid in user_errors)
    log(f"目标用户: {len(target_uids)} 个（其中 {has_error_count} 个有错误）")

    # 构建 tasks
    tasks = []
    for uid in target_uids:
        profile = profile_map[uid]
        products = profile.get('products', [])
        if not products:
            continue
        prod = products[0]
        asin = prod.get('asin', '')
        errors = user_errors.get(uid, None)

        # 获取 words_per_attribute（从 attr_density 画像）
        words_per_attribute = user_wpa_map.get(uid)
        if words_per_attribute is None:
            continue

        # 获取 words_per_ccomp
        words_per_ccomp = profile.get('words_per_ccomp')
        if words_per_ccomp is None:
            words_per_ccomp = 100.0
        else:
            words_per_ccomp = float(words_per_ccomp)

        # 计算 target_length: ceil(words_per_attribute) * 5
        target_length = math.ceil(words_per_attribute) * 5

        # 计算 ground_truth_ccomp: target_length / words_per_ccomp
        if words_per_ccomp and words_per_ccomp > 0:
            ground_truth_ccomp = int(target_length / words_per_ccomp)
            ground_truth_ccomp = max(0, min(5, ground_truth_ccomp))
        else:
            ground_truth_ccomp = 0

        # 如果 ground_truth_ccomp > 3，跳过该用户
        if ground_truth_ccomp > 3:
            continue

        persona_base = {
            'user_id': uid,
            'asin': asin,
            'ccomp_sentence_ratio': profile.get('ccomp_sentence_ratio', 0.0),
            'density_label': profile.get('density_label', 'simple'),
            'length_label': profile.get('length_label', 'medium'),
            'words_per_attribute': words_per_attribute,
            'words_per_ccomp': words_per_ccomp,
            'target_length': target_length,
            'ground_truth_ccomp': ground_truth_ccomp,
            'original_attrs': {
                'A1': prod.get('A1_product_type', ''),
                'A2': prod.get('A2_brand', ''),
                'A3': prod.get('A3_price', ''),
                'A4': prod.get('A4_appearance', ''),
                'A5': prod.get('A5_use_case', ''),
            }
        }

        for target_ccomp_count in [0, 1, 2, 3]:
            task = persona_base.copy()
            task['target_ccomp_override'] = target_ccomp_count
            task['is_target_ccomp'] = (target_ccomp_count == ground_truth_ccomp)
            task['error_patterns'] = errors
            # 三种情况：
            # 1. 有错误 + ground_truth → 双版本 (correct + noisy)
            # 2. 有错误 + 非 ground_truth → 单版本但包含正确形式
            # 3. 无错误 → 普通单版本
            if errors is not None:
                task['has_errors'] = True
                task['is_ground_truth_version'] = task['is_target_ccomp']
            else:
                task['has_errors'] = False
                task['is_ground_truth_version'] = False
            tasks.append(task)

    log(f"构建了 {len(tasks)} 个查询任务 ({len(target_uids)} 用户 × 4 ccomp版本)")
    log(f"开始处理，并发数={MAX_WORKERS}")

    def process_one(task):
        uid = task['user_id']
        target_ccomp_count = task['target_ccomp_override']
        attrs = task['original_attrs']

        if task['has_errors'] and task.get('is_ground_truth_version'):
            # 有错误 + ground_truth：生成 correct + error 双版本
            prompt = build_persona_prompt_with_errors(task, task['error_patterns'])
            text = call_llm(prompt)
            query_data = parse_query_with_errors(text)
            if query_data:
                correct_filled = fill_placeholders(query_data['correct_query'], attrs)
                error_filled = fill_placeholders(query_data['error_query'], attrs)
                # 从原始 error_patterns 回填 error_type
                error_words = query_data.get('error_words', [])
                for ew in error_words:
                    for ep in task['error_patterns']:
                        if ew.get('correct') == ep.get('corrected') and ew.get('error') == ep.get('original'):
                            ew['error_type'] = ep.get('error_type', 'unknown')
                            break
                    if 'error_type' not in ew:
                        ew['error_type'] = 'unknown'
                return {
                    'user_id': uid,
                    'asin': task['asin'],
                    'target_ccomp': target_ccomp_count,
                    'words_per_ccomp': task.get('words_per_ccomp'),
                    'ground_truth_ccomp': task.get('ground_truth_ccomp'),
                    'target_length': task.get('target_length'),
                    'has_errors': True,
                    'correct_query': correct_filled,
                    'noisy_query': error_filled,
                    'error_words': error_words,
                    'word_count': query_data['word_count'],
                    'is_ground_truth': True,
                }
        elif task['has_errors'] and not task.get('is_ground_truth_version'):
            # 有错误 + 非 ground_truth：包含正确形式的单版本
            prompt = build_persona_prompt_with_correct_words(task, task['error_patterns'])
            text = call_llm(prompt)
            query_data = parse_query(text)
            if query_data:
                filled_query = fill_placeholders(query_data['query'], attrs)
                return {
                    'user_id': uid,
                    'asin': task['asin'],
                    'target_ccomp': target_ccomp_count,
                    'words_per_ccomp': task.get('words_per_ccomp'),
                    'ground_truth_ccomp': task.get('ground_truth_ccomp'),
                    'target_length': task.get('target_length'),
                    'has_errors': True,
                    'filled_query': filled_query,
                    'word_count': query_data['word_count'],
                    'is_ground_truth': False,
                }
        else:
            # 无错误用户：旧逻辑，普通单版本
            prompt = build_persona_prompt(task)
            text = call_llm(prompt)
            query_data = parse_query(text)
            if query_data:
                filled_query = fill_placeholders(query_data['query'], attrs)
                return {
                    'user_id': uid,
                    'asin': task['asin'],
                    'target_ccomp': target_ccomp_count,
                    'words_per_ccomp': task.get('words_per_ccomp'),
                    'ground_truth_ccomp': task.get('ground_truth_ccomp'),
                    'target_length': task.get('target_length'),
                    'has_errors': False,
                    'filled_query': filled_query,
                    'word_count': query_data['word_count'],
                    'is_ground_truth': (target_ccomp_count == task.get('ground_truth_ccomp')),
                }
        return None

    results = []
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    total_start = time.time()
    total_users = len(target_uids)
    user_completed_count = {}
    printed_users = set()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, t): t for t in batch_tasks}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                    uid = r['user_id']
                    user_completed_count[uid] = user_completed_count.get(uid, 0) + 1
                    if user_completed_count[uid] == 4 and uid not in printed_users:
                        printed_users.add(uid)
                        done_users = sum(1 for c in user_completed_count.values() if c >= 4)
                        err_tag = " [error user]" if r['has_errors'] else ""
                        log(f"  [{done_users}/{total_users}] user={uid[:20]}{err_tag}")

    total_elapsed = time.time() - total_start

    # 按用户分组
    user_map = {}
    for r in results:
        uid = r['user_id']
        if uid not in user_map:
            user_map[uid] = {
                'asin': r.get('asin', ''),
                'target_length': r.get('target_length'),
                'words_per_ccomp': r.get('words_per_ccomp'),
                'ground_truth_ccomp': r.get('ground_truth_ccomp'),
                'queries': []
            }

        if r['has_errors'] and r.get('is_ground_truth'):
            user_map[uid]['queries'].append({
                'ccomp': r['target_ccomp'],
                'correct_query': r['correct_query'],
                'noisy_query': r['noisy_query'],
                'error_words': r.get('error_words', []),
                'word_count': r['word_count'],
                'is_ground_truth': True,
            })
        else:
            user_map[uid]['queries'].append({
                'ccomp': r['target_ccomp'],
                'filled_query': r['filled_query'],
                'word_count': r['word_count'],
                'is_ground_truth': r.get('is_ground_truth', False),
            })

    # 排序
    for uid in user_map:
        user_map[uid]['queries'].sort(key=lambda x: x['ccomp'])

    output_data = [{'user_id': uid, 'asin': v['asin'],
                    'target_length': v.get('target_length'),
                    'words_per_ccomp': v.get('words_per_ccomp'),
                    'ground_truth_ccomp': v.get('ground_truth_ccomp'),
                    'queries': v['queries']}
                   for uid, v in user_map.items()]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 统计
    error_users = [uid for uid in user_map if any(
        q.get('correct_query') for q in user_map[uid]['queries']
    )]
    normal_users = len(user_map) - len(error_users)

    log(f"\n{'='*60}")
    log(f"总计: {len(results)} queries, {len(user_map)} users in {total_elapsed:.1f}s")
    log(f"  有错误用户: {len(error_users)} (correct+noisy 双版本)")
    log(f"  无错误用户: {normal_users} (单版本)")
    log(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
