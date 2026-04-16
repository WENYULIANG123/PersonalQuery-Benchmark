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


def inject_errors(query: str, error_patterns: list) -> str:
    """将查询中的正确词替换为错误词"""
    if not error_patterns:
        return query
    result = query
    for ep in error_patterns[:10]:
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        if orig and corr:
            # 替换正确词为错误词
            result = result.replace(corr, orig)
    return result


# ========================================
# 解析4版本查询结果
# ========================================
def parse_4_versions_query(text_content: str, ground_truth_ccomp: int = 0) -> dict:
    """解析包含4个ccomp版本的JSON（全部是单版本字符串）

    ground_truth_ccomp: 用户的真实ccomp级别（用于标记）
    """
    try:
        json_match = re.search(r'\{[\s\S]*\}', text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)

        result = {}
        for ccomp_level in ['ccomp_0', 'ccomp_1', 'ccomp_2', 'ccomp_3']:
            if ccomp_level not in data:
                return None

            item = data[ccomp_level]
            ccomp_num = int(ccomp_level.split('_')[1])
            is_ground_truth = (ccomp_num == ground_truth_ccomp)

            # 单版本: "query string"
            query = item.strip() if isinstance(item, str) else ''
            if query:
                result[ccomp_level] = {
                    'query': query,
                    'word_count': count_words(query),
                    'is_ground_truth': is_ground_truth,
                }
            else:
                return None

        return result
    except Exception as e:
        log(f"    [DEBUG] 4版本JSON解析失败: {e}, text_content={repr(text_content[:500])}")
    return None


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

    # 第一步：计算所有用户的 ground_truth_ccomp 并过滤
    all_user_data = []
    for uid in profile_map:
        if uid not in user_wpa_map:
            continue
        profile = profile_map[uid]
        products = profile.get('products', [])
        if not products:
            continue
        prod = products[0]

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

        # 过滤 ground_truth_ccomp > 3 的用户
        if ground_truth_ccomp > 3:
            continue

        all_user_data.append({
            'uid': uid,
            'profile': profile,
            'prod': prod,
            'words_per_attribute': words_per_attribute,
            'words_per_ccomp': words_per_ccomp,
            'target_length': target_length,
            'ground_truth_ccomp': ground_truth_ccomp,
            'has_errors': uid in user_errors,
        })

    log(f"过滤后（ground_truth_ccomp <= 3）的用户数: {len(all_user_data)}")

    # 第二步：优先选有错误的用户，再选无错误用户
    error_users = [u for u in all_user_data if u['has_errors']]
    normal_users = [u for u in all_user_data if not u['has_errors']]

    target_users = error_users[:NUM_USERS_TO_TEST]
    remaining = NUM_USERS_TO_TEST - len(target_users)
    if remaining > 0:
        target_users += normal_users[:remaining]

    has_error_count = sum(1 for u in target_users if u['has_errors'])
    log(f"目标用户: {len(target_users)} 个（其中 {has_error_count} 个有错误）")

    # 构建用户任务列表（每个用户一个任务，一次性返回4个版本）
    user_tasks = []
    for u in target_users:
        uid = u['uid']
        profile = u['profile']
        prod = u['prod']
        errors = user_errors.get(uid, None)

        persona_base = {
            'user_id': uid,
            'asin': prod.get('asin', ''),
            'ccomp_sentence_ratio': profile.get('ccomp_sentence_ratio', 0.0),
            'density_label': profile.get('density_label', 'simple'),
            'length_label': profile.get('length_label', 'medium'),
            'words_per_attribute': u['words_per_attribute'],
            'words_per_ccomp': u['words_per_ccomp'],
            'target_length': u['target_length'],
            'ground_truth_ccomp': u['ground_truth_ccomp'],
            'original_attrs': {
                'A1': prod.get('A1_product_type', ''),
                'A2': prod.get('A2_brand', ''),
                'A3': prod.get('A3_price', ''),
                'A4': prod.get('A4_appearance', ''),
                'A5': prod.get('A5_use_case', ''),
            },
            'errors': errors,
        }
        user_tasks.append(persona_base)

    log(f"构建了 {len(user_tasks)} 个用户任务")
    log(f"开始处理，并发数={MAX_WORKERS}")

    def process_one_user(persona):
        """处理单个用户，一次返回4个ccomp版本"""
        uid = persona['user_id']
        attrs = persona['original_attrs']
        errors = persona['errors']
        ground_truth_ccomp = persona['ground_truth_ccomp']

        has_errors = errors is not None

        # 构建 prompt
        system_base, user_content = build_user_prompt(persona, errors)
        text = call_llm(user_content, system_base=system_base)

        # 解析4个版本
        query_data = parse_4_versions_query(text, ground_truth_ccomp)
        if not query_data:
            return None

        # 构建4个结果
        results = []
        for ccomp_level in ['ccomp_0', 'ccomp_1', 'ccomp_2', 'ccomp_3']:
            target_ccomp = int(ccomp_level.split('_')[1])
            qdata = query_data[ccomp_level]

            if qdata.get('is_ground_truth') and has_errors:
                # ground_truth 级别：使用 inject_errors 生成 noisy_query
                correct_filled = fill_placeholders(qdata['query'], attrs)
                noisy_filled = inject_errors(correct_filled, errors)
                results.append({
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
                    'error_words': [{'correct': ep['corrected'], 'error': ep['original'], 'error_type': ep.get('error_type', 'unknown')} for ep in (errors or [])[:10]],
                    'word_count': qdata['word_count'],
                    'is_ground_truth': True,
                })
            else:
                # 其他级别：单版本
                filled_query = fill_placeholders(qdata['query'], attrs)
                results.append({
                    'user_id': uid,
                    'asin': persona['asin'],
                    'target_ccomp': target_ccomp,
                    'ccomp_sentence_ratio': persona.get('ccomp_sentence_ratio', 0.0),
                    'density_label': persona.get('density_label', 'simple'),
                    'length_label': persona.get('length_label', 'medium'),
                    'words_per_ccomp': persona.get('words_per_ccomp'),
                    'ground_truth_ccomp': ground_truth_ccomp,
                    'target_length': persona.get('target_length'),
                    'has_errors': has_errors,
                    'filled_query': filled_query,
                    'word_count': qdata['word_count'],
                    'is_ground_truth': qdata.get('is_ground_truth', False),
                })

        return results

    results = []
    total_start = time.time()
    total_users = len(user_tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_user, t): t for t in user_tasks}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.extend(r)
                done_users = len(set(x['user_id'] for x in results))
                err_tag = " [error user]" if r[0]['has_errors'] else ""
                log(f"  [{done_users}/{total_users}] user={r[0]['user_id'][:20]}{err_tag}")

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
