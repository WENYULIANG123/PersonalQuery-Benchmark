#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句（ccomp占位符版本）
=================================
将属性词替换为 A1-A5 占位符，用于测试语言结构
参数硬编码，勿改动
"""

import sys
import json
import time
import re
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
from llm_client import MiniMaxAnthropicClient


# ========================================
# 硬编码参数
# ========================================
NUM_USERS_TO_TEST = 10  # 测试用：只处理前N个用户
MAX_WORKERS = 10
BATCH_SIZE = 100
USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/ccomp_user_profiles.json'
ATTR_DENSITY_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/attr_density_user_profiles.json'
OUTPUT_FILE = None          # None时自动生成


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
# 根据画像构建Prompt（ccomp占位符版本）
# ========================================
def build_persona_prompt(persona: dict) -> str:
    """根据用户画像构建查询生成prompt，使用A1-A5占位符"""

    # 解析画像指标
    ccomp_ratio = persona.get('ccomp_sentence_ratio', 0.0)
    ccomp_per_sentence = persona.get('ccomp_per_sentence', 0.0)
    avg_length = persona.get('avg_sentence_length', 20.0)
    density_label = persona.get('density_label', 'simple')
    length_label = persona.get('length_label', 'medium')
    ccomp_type_dist = persona.get('ccomp_type_distribution', None)

    # target_length 已经在main中计算好，直接使用
    target_length = persona.get('target_length')
    if target_length is not None:
        length_instruction = f"- Target sentence length: approximately {target_length} words (tolerance: ±5 words)"
    else:
        length_instruction = ""

    # 确定需要多少个ccomp（优先使用 override）
    if 'target_ccomp_override' in persona:
        target_ccomp_count = persona['target_ccomp_override']
    else:
        # words_per_ccomp > 35 → ccomp = 0
        # 18 <= words_per_ccomp <= 35 → ccomp = 1
        # 12 <= words_per_ccomp < 18 → ccomp = 2
        # words_per_ccomp < 12 → ccomp = 3
        words_per_ccomp = persona.get('words_per_ccomp', 20.0)
        if words_per_ccomp > 35:
            target_ccomp_count = 0
        elif words_per_ccomp >= 18:
            target_ccomp_count = 1
        elif words_per_ccomp >= 12:
            target_ccomp_count = 2
        else:
            target_ccomp_count = 3

    # 根据 ccomp_type_dist 构建指令
    if ccomp_type_dist and target_ccomp_count > 0:
        # 统一使用 'that' 从句，忽略用户的其他 ccomp 类型分布
        clauses = [("that", "complete clause") for _ in range(target_ccomp_count)]

        def format_ccomp_clause(i, word, clause_type):
            return f"- Clause {i+1}: use '{word}' followed by a {clause_type}, completing the meaning of the main verb. Example: 'I think {word} it works well'"

        clauses_str = "\n".join([format_ccomp_clause(i, w, c) for i, (w, c) in enumerate(clauses)])

        ccomp_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_ccomp_count} complement clause(s).
{clauses_str}
- FORBIDDEN: Do NOT use 'to + verb' patterns (e.g., "I intend to buy", "I want to find").
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use repetitive patterns like "I think that... I believe that..."
- Keep sentences natural and varied, not mechanical.
- Each clause must complete the meaning of a verb in the main sentence."""
    elif target_ccomp_count == 0:
        ccomp_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO complement clauses.
- FORBIDDEN words: 'think', 'believe', 'know', 'feel', 'want', 'hope', 'expect', 'suppose'
- FORBIDDEN: Do NOT use 'that' to introduce any clause.
- FORBIDDEN: Do NOT use 'to + verb' patterns (e.g., "I intend to buy", "I want to find").
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- Use simple, direct phrasing without any embedded clauses."""
    else:
        ccomp_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_ccomp_count} complement clause(s).
- Each clause MUST use 'that' followed by a complete clause (e.g., "I think that it works well", "I believe that it is good").
- Keep the total sentence length close to the target ({target_length} words, ±5 tolerance).
- FORBIDDEN: Do NOT use 'to + verb' patterns (e.g., "I intend to buy", "I want to find", "I hope to confirm"). These are NOT complement clauses.
- FORBIDDEN: Do NOT use 'whether', 'if', 'what', 'whatever' to introduce clauses.
- FORBIDDEN: Do NOT use repetitive patterns like "I think that... I believe that... I know that..."
- Keep sentences natural and varied, not mechanical.
- The 'that' clause must complete the meaning of a verb like "think", "believe", "want", "know", etc."""

    # 使用占位符 A1-A5 替代实际属性值
    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Complement clause usage: {ccomp_ratio*100:.0f}% of sentences contain complement clauses
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
- Real: "I want markers that are priced at $6.63 and are perfect for calligraphy."
- Placeholder: "I want A1 that are A3 and are A4 for A5."

{ccomp_instruction}

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses
- NO relative clauses with 'which', 'who', 'whom', 'whose'
- The placeholders A1, A2, A3, A4, A5 must appear EXACTLY as written (uppercase A followed by number)

Output format: Output ONLY a valid JSON object. No text before or after.
{{"query_id": 1, "query": "your query here with A1-A5 placeholders"}}
"""
    return prompt


# ========================================
# LLM 调用
# ========================================
def call_llm(prompt: str) -> str:
    """调用LLM，遇到速率限制时自动重试"""
    client = MiniMaxAnthropicClient(model='MiniMax-M2.7-highspeed')
    wait_seconds = 5
    max_retries = 20
    for attempt in range(max_retries):
        try:
            thinking, text = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.3)
            return text
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str or 'too many request' in error_str
            if not is_rate_limit:
                raise
            if attempt < max_retries - 1:
                log(f"    [WARN] 速率限制，等待 {wait_seconds}s 后重试 (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                wait_seconds += 5  # 递增等待时间
            else:
                raise


# ========================================
# 解析结果
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
    # A1 = product type (可能是列表)
    a1_val = attrs.get('A1', '')
    if isinstance(a1_val, list):
        a1_val = ', '.join(str(v) for v in a1_val)
    result = result.replace('A1', str(a1_val))

    # A2 = brand
    a2_val = attrs.get('A2', '')
    if isinstance(a2_val, list):
        a2_val = ', '.join(str(v) for v in a2_val)
    result = result.replace('A2', str(a2_val))

    # A3 = price
    a3_val = attrs.get('A3', '')
    if isinstance(a3_val, list):
        a3_val = ', '.join(str(v) for v in a3_val)
    result = result.replace('A3', str(a3_val))

    # A4 = appearance
    a4_val = attrs.get('A4', '')
    if isinstance(a4_val, list):
        a4_val = ', '.join(str(v) for v in a4_val)
    result = result.replace('A4', str(a4_val))

    # A5 = use case
    a5_val = attrs.get('A5', '')
    if isinstance(a5_val, list):
        a5_val = ', '.join(str(v) for v in a5_val)
    result = result.replace('A5', str(a5_val))

    return result


# ========================================
# 主函数
# ========================================
def main():
    # 加载用户画像（含商品属性）
    log(f"加载用户画像 from {USER_PROFILES_FILE}...")
    with open(USER_PROFILES_FILE, 'r') as f:
        all_user_profiles = json.load(f)
    log(f"加载了 {len(all_user_profiles)} 个用户画像")

    # 构建 personas - 每个用户只取第一个商品，生成4个版本(ccomp=0,1,2,3)
    personas = []

    # 加载attr_density用户画像，获取words_per_attribute
    log(f"加载attr_density用户画像 from {ATTR_DENSITY_PROFILES_FILE}...")
    with open(ATTR_DENSITY_PROFILES_FILE, 'r') as f:
        attr_density_profiles = json.load(f)
    # 建立 user_id -> words_per_attribute 映射
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

    for profile in all_user_profiles:
        if len(personas) // 4 >= NUM_USERS_TO_TEST:
            break
        uid = profile['user_id']
        products = profile.get('products', [])
        if not products:
            continue
        prod = products[0]  # 只用第一个商品
        asin = prod.get('asin', '')

        words_per_ccomp = profile.get('words_per_ccomp')
        if words_per_ccomp is None:
            words_per_ccomp = 100.0
        else:
            words_per_ccomp = float(words_per_ccomp)

        # 获取words_per_attribute（从attr_density画像）
        words_per_attribute = user_wpa_map.get(uid)

        # 如果用户不在attr_density中，跳过
        if words_per_attribute is None:
            continue

        # 计算target_length：向上取整 * 5
        target_length = math.ceil(words_per_attribute) * 5

        # 计算 ground_truth_ccomp: target_length / words_per_ccomp
        if words_per_ccomp and words_per_ccomp > 0:
            ground_truth_ccomp = round(target_length / words_per_ccomp)
            ground_truth_ccomp = max(0, min(5, ground_truth_ccomp))  # 限制在 0-5
        else:
            ground_truth_ccomp = 0

        # 如果 ground_truth_ccomp > 3，跳过该用户（只支持 0-3 版本）
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

        # 每个用户生成6个版本(ccomp=0,1,2,3,4,5)
        for target_ccomp_count in [0, 1, 2, 3, 4, 5]:
            persona = persona_base.copy()
            persona['target_ccomp_override'] = target_ccomp_count
            personas.append(persona)

    log(f"构建了 {len(personas)} 个查询任务 ({len(personas)//6} 用户 × 6 ccomp版本)")

    log(f"开始处理 {len(personas)} 个查询任务 ({len(personas)//6} 用户 × 6 ccomp版本)，并发数={MAX_WORKERS}")

    def process_one(persona):
        start = time.time()

        # 使用 override（main 中已设置），否则从 words_per_ccomp 推导
        if 'target_ccomp_override' in persona:
            target_ccomp_count = persona['target_ccomp_override']
        else:
            words_per_ccomp = persona.get('words_per_ccomp', 20.0)
            if words_per_ccomp > 35:
                target_ccomp_count = 0
            elif words_per_ccomp >= 18:
                target_ccomp_count = 1
            elif words_per_ccomp >= 12:
                target_ccomp_count = 2
            elif words_per_ccomp >= 8:
                target_ccomp_count = 3
            elif words_per_ccomp >= 5:
                target_ccomp_count = 4
            else:
                target_ccomp_count = 5

        prompt = build_persona_prompt(persona)
        text = call_llm(prompt)
        query_data = parse_query(text)
        if query_data:
            placeholders = validate_placeholders(query_data['query'])
            filled_query = fill_placeholders(query_data['query'], persona['original_attrs'])

            result = {
                'user_id': persona['user_id'],
                'asin': persona['asin'],
                'target_ccomp': target_ccomp_count,
                'words_per_ccomp': persona.get('words_per_ccomp'),
                'ground_truth_ccomp': persona.get('ground_truth_ccomp'),
                'target_length': persona.get('target_length'),
                'filled_query': filled_query,
                'word_count': query_data['word_count'],
                'is_ground_truth': (target_ccomp_count == persona.get('ground_truth_ccomp')),
            }
            return result
        else:
            return None

    results = []
    total_batches = (len(personas) + BATCH_SIZE - 1) // BATCH_SIZE
    total_start = time.time()

    total_users = len(personas) // 6
    user_completed_count = {}  # uid -> completed query count (persistent across batches)
    printed_users = set()  # 避免重复打印 (persistent across batches)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(personas))
        batch_personas = personas[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, p): p for p in batch_personas}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                    uid = r['user_id']
                    user_completed_count[uid] = user_completed_count.get(uid, 0) + 1
                    if user_completed_count[uid] == 6 and uid not in printed_users:
                        printed_users.add(uid)
                        word_counts = {rr['target_ccomp']: rr['word_count'] for rr in [r2 for r2 in results if r2['user_id'] == uid]}
                        done_users = sum(1 for c in user_completed_count.values() if c >= 6)
                        log(f"  [{done_users}/{total_users}] user={uid[:20]}, ccomp0={word_counts.get(0,0)}w ccomp1={word_counts.get(1,0)}w ccomp2={word_counts.get(2,0)}w ccomp3={word_counts.get(3,0)}w ccomp4={word_counts.get(4,0)}w ccomp5={word_counts.get(5,0)}w")

    total_elapsed = time.time() - total_start

    # 按用户分组
    user_map = {}
    for r in results:
        uid = r['user_id']
        if uid not in user_map:
            user_map[uid] = {'asin': r.get('asin', ''), 'target_length': r.get('target_length'),
                              'words_per_ccomp': r.get('words_per_ccomp'),
                              'ground_truth_ccomp': r.get('ground_truth_ccomp'), 'queries': []}
        user_map[uid]['queries'].append({
            'ccomp': r['target_ccomp'],
            'filled_query': r['filled_query'],
            'word_count': r['word_count'],
            'is_ground_truth': r.get('is_ground_truth', False),
        })

    # 排序（ccomp 0,1,2,3）
    for uid in user_map:
        user_map[uid]['queries'].sort(key=lambda x: x['ccomp'])

    output_data = [{'user_id': uid, 'asin': v['asin'], 'target_length': v.get('target_length'),
                    'words_per_ccomp': v.get('words_per_ccomp'),
                    'ground_truth_ccomp': v.get('ground_truth_ccomp'), 'queries': v['queries']}
                   for uid, v in user_map.items()]

    # 保存结果
    output_file = '/fs04/ar57/wenyu/result/personal_query/06_query/ccomp_query.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"总计: {len(results)} queries generated in {total_elapsed:.1f}s")
    log(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
