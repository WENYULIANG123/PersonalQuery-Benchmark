#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（acl占位符版本）
=================================
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
from llm_client import MiniMaxAnthropicClient


# ========================================
# 硬编码参数
# ========================================
NUM_USERS_TO_TEST = 500       # 测试用：只处理前N个用户
MAX_WORKERS = 10
BATCH_SIZE = 100
USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/acl_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/Pet_Supplies/attr_density_user_profiles.json'
USER_ERROR_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/Pet_Supplies/acl_error.json'
OUTPUT_FILE = '/fs04/ar57/wenyu/result/personal_query/06_query/Pet_Supplies/acl_query.json'


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
        patterns = []
        for detail in user['detailed_results']:
            for err in detail.get('errors', []):
                patterns.append({
                    'original': err.get('original', ''),
                    'corrected': err.get('corrected', ''),
                    'error_type': err.get('error_type', 'unknown'),
                })
        if patterns:
            user_errors[uid] = patterns

    return user_errors


# ========================================
# 根据画像构建Prompt（无错误用户 - 旧逻辑）
# ========================================
def build_persona_prompt(persona: dict) -> str:
    """根据用户画像构建查询生成prompt，使用A1-A5占位符"""

    acl_ratio = persona.get('acl_sentence_ratio', 0.0)
    density_label = persona.get('density_label', 'simple')
    length_label = persona.get('length_label', 'medium')
    acl_type_dist = persona.get('acl_type_distribution', None)

    # target_length 已在main中计算好，直接使用
    target_length = persona.get('target_length')
    if target_length is not None:
        length_instruction = f"- Target sentence length: approximately {target_length} words (tolerance: ±5 words)"
    else:
        length_instruction = ""

    if 'target_acl_override' in persona:
        target_acl_count = persona['target_acl_override']
    else:
        words_per_acl = persona.get('words_per_acl', 20.0)
        if words_per_acl > 35:
            target_acl_count = 0
        elif words_per_acl >= 18:
            target_acl_count = 1
        elif words_per_acl >= 12:
            target_acl_count = 2
        else:
            target_acl_count = 3

    if acl_type_dist and target_acl_count > 0:
        clauses = [("which", "complete clause") for _ in range(target_acl_count)]

        def format_acl_clause(i, word, clause_type):
            return f"- Clause {i+1}: use '{word}' followed by a {clause_type}, describing a feature of the main noun. Example: 'A1 {word} cost A3'"

        clauses_str = "\n".join([format_acl_clause(i, w, c) for i, (w, c) in enumerate(clauses)])

        acl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_acl_count} adjectival clause(s) (relative clause).
{clauses_str}
- Each clause must describe/modify a noun in the main sentence (e.g., the product).
- FORBIDDEN: Do NOT use 'who' (only use 'which' for this task).
- FORBIDDEN: Do NOT use 'to + verb' patterns (e.g., "A1 intended to buy").
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use 'that' as a clause marker in this task.
- The adjectival clause must modify/describe a noun, not complete the meaning of a verb."""
    elif target_acl_count == 0:
        acl_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO adjectival clauses (relative clauses).
- FORBIDDEN: Do NOT use 'which', 'who', 'whom', 'whose', 'that' to introduce any clause.
- FORBIDDEN: Do NOT use 'to + verb' patterns.
- MANDATORY: ALL 5 placeholders (A1, A2, A3, A4, A5) MUST appear exactly once.

STRUCTURE RULES:
- Use ONLY simple phrases connected by 'and' or 'or'
- Example: "A1 by A2 priced at A3 with A4 finish for A5."
- GOOD: A1 by A2 at A3 with A4 color for A5."""
    else:
        acl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_acl_count} adjectival clause(s) (relative clause).
- Each clause MUST use 'which' followed by a complete clause that describes the noun.
- Example: "A1 which are priced at A3" or "A1 which come from A2"
- FORBIDDEN: Do NOT use 'who' (only use 'which' for this task).
- FORBIDDEN: Do NOT use 'to + verb' patterns (e.g., "A1 intended to find", "A1 made to use").
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use 'that' as a clause marker.
- The 'which' clause must describe/modify a noun, not complete the meaning of a verb."""

    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Adjectival clause usage: {acl_ratio*100:.0f}% of sentences contain adjectival clauses
- Writing style: {density_label} with {length_label} sentences
{length_instruction}

Product attributes - MUST use these placeholders:
- A1 = product type (e.g., markers, fabric, ribbon)
- A2 = brand name (e.g., Darice, Fiskars)
- A3 = price (e.g., $6.63, $10.00)
- A4 = appearance/feature (e.g., red color, smooth finish, vibrant hues)
- A5 = use case/purpose (e.g., for calligraphy, for painting, for crafting)

IMPORTANT: Replace ALL specific product attributes with their placeholders:
- Replace actual product type with "A1"
- Replace actual brand with "A2"
- Replace actual price with "A3"
- Replace actual appearance/feature with "A4"
- Replace actual use case with "A5"

Example transformation:
- Real: "I want markers which are priced at $6.63 and are perfect for calligraphy."
- Placeholder: "I want A1 which are A3 and are A4 for A5."

{acl_instruction}

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses
- NO complement clauses with 'that' (e.g., "I think that it is good")
- NO 'who', 'whom', 'whose' in this task
- The placeholders A1, A2, A3, A4, A5 must appear EXACTLY as written (uppercase A followed by number)

Output format: Output ONLY a valid JSON object. No text before or after.
{{"query_id": 1, "query": "your query here with A1-A5 placeholders"}}
"""
    return prompt


# ========================================
# 根据画像构建Prompt（有错误用户 - 非 ground_truth：只含正确形式，单版本）
# ========================================
def build_persona_prompt_with_correct_words(persona: dict, error_patterns: list) -> str:
    """有错误用户但非 ground_truth：要求包含错误词的正确形式，只输出单版本"""

    acl_ratio = persona.get('acl_sentence_ratio', 0.0)
    density_label = persona.get('density_label', 'simple')
    length_label = persona.get('length_label', 'medium')

    target_length = persona.get('target_length')
    if target_length is not None:
        length_instruction = f"- Target sentence length: approximately {target_length} words (tolerance: ±5 words)"
    else:
        length_instruction = ""

    if 'target_acl_override' in persona:
        target_acl_count = persona['target_acl_override']
    else:
        words_per_acl = persona.get('words_per_acl', 20.0)
        if words_per_acl > 35:
            target_acl_count = 0
        elif words_per_acl >= 18:
            target_acl_count = 1
        elif words_per_acl >= 12:
            target_acl_count = 2
        else:
            target_acl_count = 3

    # ACL 指令
    if target_acl_count == 0:
        acl_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO adjectival clauses (relative clauses).
- FORBIDDEN: Do NOT use 'which', 'who', 'whom', 'whose', 'that' to introduce any clause.
- FORBIDDEN: Do NOT use 'to + verb' patterns.
- MANDATORY: ALL 5 placeholders (A1, A2, A3, A4, A5) MUST appear exactly once.

STRUCTURE RULES:
- Use ONLY simple phrases connected by 'and' or 'or'
- Example: "A1 by A2 priced at A3 with A4 finish for A5." """
    else:
        acl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_acl_count} adjectival clause(s) (relative clause).
- Each clause MUST use 'which' followed by a complete clause that describes the noun.
- FORBIDDEN: Do NOT use 'who' (only use 'which' for this task).
- FORBIDDEN: Do NOT use 'to + verb' patterns.
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use 'that' as a clause marker.
- The 'which' clause must describe/modify a noun, not complete the meaning of a verb."""

    # 构建必须包含的正确词列表
    forced_words_str = ""
    for ep in error_patterns[:10]:
        corr = ep.get("corrected", "")
        forced_words_str += f"- \"{corr}\"\n"

    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Adjectival clause usage: {acl_ratio*100:.0f}% of sentences contain adjectival clauses
- Writing style: {density_label} with {length_label} sentences
{length_instruction}

Product attributes - MUST use these placeholders:
- A1 = product type
- A2 = brand name
- A3 = price
- A4 = appearance/feature
- A5 = use case/purpose

{acl_instruction}

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses
- NO complement clauses with 'that' (e.g., "I think that it is good")
- NO 'who', 'whom', 'whose' in this task
- The placeholders A1, A2, A3, A4, A5 must appear EXACTLY as written (uppercase A followed by number)

CRITICAL: The query MUST naturally include ALL of the following words/phrases (not as placeholders, as regular words):
{forced_words_str}
Each of these words must appear naturally in the query context.

Output format: Output ONLY a valid JSON object. No text before or after.
{{"query_id": 1, "query": "your query here with A1-A5 placeholders"}}
"""
    return prompt


# ========================================
# 根据画像构建Prompt（有错误用户 - ground_truth：双版本）
# ========================================
def build_persona_prompt_with_errors(persona: dict, error_patterns: list) -> str:
    """根据用户画像构建查询生成prompt，强制包含用户错误单词，输出correct和error两个版本"""

    acl_ratio = persona.get('acl_sentence_ratio', 0.0)
    density_label = persona.get('density_label', 'simple')
    length_label = persona.get('length_label', 'medium')

    # target_length
    target_length = persona.get('target_length')
    if target_length is not None:
        length_instruction = f"- Target sentence length: approximately {target_length} words (tolerance: ±5 words)"
    else:
        length_instruction = ""

    if 'target_acl_override' in persona:
        target_acl_count = persona['target_acl_override']
    else:
        words_per_acl = persona.get('words_per_acl', 20.0)
        if words_per_acl > 35:
            target_acl_count = 0
        elif words_per_acl >= 18:
            target_acl_count = 1
        elif words_per_acl >= 12:
            target_acl_count = 2
        else:
            target_acl_count = 3

    # ACL 指令（与旧逻辑一致）
    if target_acl_count == 0:
        acl_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO adjectival clauses (relative clauses).
- FORBIDDEN: Do NOT use 'which', 'who', 'whom', 'whose', 'that' to introduce any clause.
- FORBIDDEN: Do NOT use 'to + verb' patterns.
- MANDATORY: ALL 5 placeholders (A1, A2, A3, A4, A5) MUST appear exactly once.

STRUCTURE RULES:
- Use ONLY simple phrases connected by 'and' or 'or'
- Example: "A1 by A2 priced at A3 with A4 finish for A5." """
    else:
        acl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_acl_count} adjectival clause(s) (relative clause).
- Each clause MUST use 'which' followed by a complete clause that describes the noun.
- FORBIDDEN: Do NOT use 'who' (only use 'which' for this task).
- FORBIDDEN: Do NOT use 'to + verb' patterns.
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use 'that' as a clause marker.
- The 'which' clause must describe/modify a noun, not complete the meaning of a verb."""

    # 构建错误模式列表
    error_list_str = ""
    forced_words = []
    for i, ep in enumerate(error_patterns[:10]):
        orig = ep.get("original", "")
        corr = ep.get("corrected", "")
        etype = ep.get("error_type", "unknown")
        error_list_str += f"{i+1}. [{etype}] user writes: \"{orig}\" but correct is: \"{corr}\"\n"
        forced_words.append({"correct": corr, "error": orig})

    forced_words_str = ""
    for fw in forced_words:
        forced_words_str += f"- correct: \"{fw['correct']}\", error: \"{fw['error']}\"\n"

    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Adjectival clause usage: {acl_ratio*100:.0f}% of sentences contain adjectival clauses
- Writing style: {density_label} with {length_label} sentences
{length_instruction}

Product attributes - MUST use these placeholders:
- A1 = product type
- A2 = brand name
- A3 = price
- A4 = appearance/feature
- A5 = use case/purpose

{acl_instruction}

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses
- NO complement clauses with 'that' (e.g., "I think that it is good")
- NO 'who', 'whom', 'whose' in this task
- The placeholders A1, A2, A3, A4, A5 must appear EXACTLY as written (uppercase A followed by number)

USER ERROR PATTERNS (this user makes these spelling mistakes):
{error_list_str}

CRITICAL: The query MUST naturally include ALL of the following correct words/phrases in the sentence (not as placeholders, as regular words):
{forced_words_str}
Each of these words must appear naturally in the query context.

You MUST output TWO versions:
1. "correct_query": The query with ALL words in their CORRECT form (from the "correct" column above)
2. "error_query": Same query but with EVERY correct word replaced by its corresponding ERROR form (from the "error" column above)

Output format: Output ONLY a valid JSON object. No text before or after.
{{
  "query_id": 1,
  "correct_query": "query with A1-A5 placeholders and all correct words",
  "error_query": "same query but with error forms replacing correct words",
  "error_words": [
    {{"correct": "...", "error": "..."}},
    ...
  ]
}}
"""
    return prompt


# ========================================
# LLM 调用
# ========================================
def call_llm(prompt: str) -> str:
    """调用LLM"""
    client = MiniMaxAnthropicClient(model='MiniMax-M2.5')
    thinking, text = client.call_with_thinking(prompt, max_tokens=16384, temperature=0.3)
    return text


# ========================================
# 解析结果（无错误用户）
# ========================================
def parse_query(text_content: str) -> dict:
    """解析JSON查询"""
    try:
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
        log(f"    [DEBUG] JSON解析失败: {e}")
    return None


# ========================================
# 解析结果（有错误用户 - 双版本）
# ========================================
def parse_query_with_errors(text_content: str) -> dict:
    """解析包含correct_query和error_query的JSON"""
    try:
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
        log(f"    [DEBUG] JSON解析失败: {e}")
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
    acl_user_ids = set(p['user_id'] for p in all_user_profiles)
    common_user_ids = acl_user_ids & set(user_wpa_map.keys())
    log(f"同时存在于acl和attr_density的用户数: {len(common_user_ids)}")

    # 加载用户错误画像
    log(f"加载用户错误画像 from {USER_ERROR_FILE}...")
    user_errors = load_user_errors(USER_ERROR_FILE)
    log(f"加载了 {len(user_errors)} 个有错误的用户")

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

        # 获取 words_per_acl
        words_per_acl = profile.get('words_per_acl')
        if words_per_acl is None:
            words_per_acl = 100.0
        else:
            words_per_acl = float(words_per_acl)

        # 计算 target_length: ceil(words_per_attribute) * 5
        target_length = math.ceil(words_per_attribute) * 5

        # 计算 ground_truth_acl: target_length / words_per_acl
        if words_per_acl and words_per_acl > 0:
            ground_truth_acl = int(target_length / words_per_acl)
            ground_truth_acl = max(0, min(5, ground_truth_acl))
        else:
            ground_truth_acl = 0

        # 如果 ground_truth_acl > 3，跳过该用户
        if ground_truth_acl > 3:
            continue

        persona_base = {
            'user_id': uid,
            'asin': asin,
            'acl_sentence_ratio': profile.get('acl_sentence_ratio', 0.0),
            'density_label': profile.get('density_label', 'simple'),
            'length_label': profile.get('length_label', 'medium'),
            'words_per_attribute': words_per_attribute,
            'words_per_acl': words_per_acl,
            'target_length': target_length,
            'ground_truth_acl': ground_truth_acl,
            'original_attrs': {
                'A1': prod.get('A1_product_type', ''),
                'A2': prod.get('A2_brand', ''),
                'A3': prod.get('A3_price', ''),
                'A4': prod.get('A4_appearance', ''),
                'A5': prod.get('A5_use_case', ''),
            }
        }

        for target_acl_count in [0, 1, 2, 3]:
            task = persona_base.copy()
            task['target_acl_override'] = target_acl_count
            task['is_target_acl'] = (target_acl_count == ground_truth_acl)
            task['error_patterns'] = errors
            # 三种情况：
            # 1. 有错误 + ground_truth → 双版本 (correct + noisy)
            # 2. 有错误 + 非 ground_truth → 单版本但包含正确形式
            # 3. 无错误 → 普通单版本
            if errors is not None:
                task['has_errors'] = True
                task['is_ground_truth_version'] = task['is_target_acl']
            else:
                task['has_errors'] = False
                task['is_ground_truth_version'] = False
            tasks.append(task)

    log(f"构建了 {len(tasks)} 个查询任务 ({len(target_uids)} 用户 × 4 acl版本)")
    log(f"开始处理，并发数={MAX_WORKERS}")

    def process_one(task):
        uid = task['user_id']
        target_acl_count = task['target_acl_override']
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
                    'target_acl': target_acl_count,
                    'words_per_acl': task.get('words_per_acl'),
                    'ground_truth_acl': task.get('ground_truth_acl'),
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
                    'target_acl': target_acl_count,
                    'words_per_acl': task.get('words_per_acl'),
                    'ground_truth_acl': task.get('ground_truth_acl'),
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
                    'target_acl': target_acl_count,
                    'words_per_acl': task.get('words_per_acl'),
                    'ground_truth_acl': task.get('ground_truth_acl'),
                    'target_length': task.get('target_length'),
                    'has_errors': False,
                    'filled_query': filled_query,
                    'word_count': query_data['word_count'],
                    'is_ground_truth': (target_acl_count == task.get('ground_truth_acl')),
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
                'words_per_acl': r.get('words_per_acl'),
                'ground_truth_acl': r.get('ground_truth_acl'),
                'queries': []
            }

        if r['has_errors'] and r.get('is_ground_truth'):
            user_map[uid]['queries'].append({
                'acl': r['target_acl'],
                'correct_query': r['correct_query'],
                'noisy_query': r['noisy_query'],
                'error_words': r.get('error_words', []),
                'word_count': r['word_count'],
                'is_ground_truth': True,
            })
        else:
            user_map[uid]['queries'].append({
                'acl': r['target_acl'],
                'filled_query': r['filled_query'],
                'word_count': r['word_count'],
                'is_ground_truth': r.get('is_ground_truth', False),
            })

    # 排序
    for uid in user_map:
        user_map[uid]['queries'].sort(key=lambda x: x['acl'])

    output_data = [{'user_id': uid, 'asin': v['asin'],
                    'target_length': v.get('target_length'),
                    'words_per_acl': v.get('words_per_acl'),
                    'ground_truth_acl': v.get('ground_truth_acl'),
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
