#!/usr/bin/env python3
"""
根据用户画像生成个性化查询语句
=================================
参数硬编码，勿改动
"""

import sys
import json
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
from llm_client import MiniMaxAnthropicClient

nlp_trf = None  # 已废弃，不再使用


# ========================================
# 硬编码参数
# ========================================
NUM_USERS = 100
MAX_WORKERS = 50
BATCH_SIZE = 50
USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/acl_user_profiles.json'
OUTPUT_FILE = None          # None时自动生成


# ========================================
# 日志
# ========================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================================
# 词数统计
# ========================================
def count_words(text: str) -> int:
    """简单分词统计词数"""
    return len(text.split())


# ========================================
# 根据画像构建Prompt
# ========================================
def build_persona_prompt(persona: dict, product_attrs: dict) -> str:
    """根据用户画像构建查询生成prompt"""

    # 解析画像指标
    acl_ratio = persona.get('acl_sentence_ratio', 0.0)  # 0.0-1.0, 多少句子需要acl
    acl_per_sentence = persona.get('acl_per_sentence', 0.0)  # 每个句子平均多少个acl
    avg_length = persona.get('avg_sentence_length', 20.0)    # 平均句子长度
    density_label = persona.get('density_label', 'simple')    # simple/complex
    length_label = persona.get('length_label', 'medium')      # short/medium/long
    acl_type_dist = persona.get('acl_type_distribution', None)  # 新格式，如 {"that_comp": 1}

    # 商品属性
    cat = product_attrs.get('A1', '')
    brand = product_attrs.get('A2', '')
    price = product_attrs.get('A3', '')
    appearance = product_attrs.get('A4', '')
    material = product_attrs.get('A5', '')

    # 确定需要多少个acl（基于 words_per_acl 规则）
    # 如果 words_per_acl > 35 → acl = 0
    # 如果 words_per_acl 在 18-35 → acl = 1
    # 如果 words_per_acl 在 0-18 → acl = 2
    # 如果 words_per_acl < 10 → acl = 3
    words_per_acl = persona.get('words_per_acl', 20.0)
    if words_per_acl > 35:
        target_acl_count = 0
    elif words_per_acl >= 18:
        target_acl_count = 1
    elif words_per_acl >= 10:
        target_acl_count = 2
    else:
        target_acl_count = 3

    # 句子长度固定在15-35词范围内
    target_length = min(max(avg_length, 15), 35)

    # 根据密度标签调整acl复杂度
    if target_acl_count == 0:
        acl_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO adjectival clauses (relative clauses).
- FORBIDDEN: Do NOT use 'that', 'which', 'who', 'whom', 'whose', or any reduced clause forms like 'priced at', 'featuring', 'designed for', 'made from', 'used for', 'covering', 'measuring', etc.
- OUTPUT FORMAT: You can ONLY write EITHER:
  1) A simple phrase list: "Fiskars punches, $39.95, modern design, durable steel, crafting tool"
  2) A simple sentence with ONLY basic verb: "Fiskars punches cost $39.95 and are durable." (must use 'cost/are/is' as main verb, NOT other verb forms)
- NO other formats allowed."""
    elif target_acl_count > 0:
        acl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_acl_count} adjectival clause(s).
- An adjectival clause modifies a noun. Use 'that', 'which', or 'who' to connect, OR use a reduced clause without them.
- Examples of valid adjectival clauses:
  - "the book that costs $5" (full with 'that')
  - "the book (that) costs $5" (reduced, omitting 'that')
  - "the wire priced at $7" (reduced past participle)
  - "the tool made for beginners" (reduced past participle)
- Write 1-2 sentences containing exactly {target_acl_count} adjectival clause(s)."""

    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Adjectival clause usage: {acl_ratio*100:.0f}% of sentences contain adjectival clauses
- Average sentence length: {avg_length:.0f} words
- Writing style: {density_label} with {length_label} sentences
- Target adjectival clauses per sentence: {acl_per_sentence:.1f}

Product attributes (MUST include all 5 in the query):
- Product: {cat}
- Brand: {brand}
- Price: {price}
- Appearance: {appearance}
- Material: {material}

{acl_instruction}

Word count requirement: The query MUST contain EXACTLY {target_length:.0f} words. Count each word carefully before outputting.

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses

Output format: Output ONLY a valid JSON object. No text before or after.
{{"query_id": 1, "query": "your query here"}}
"""
    return prompt


# ========================================
# LLM 调用
# ========================================
def call_llm(prompt: str) -> str:
    """调用LLM"""
    client = MiniMaxAnthropicClient(model='MiniMax-M2.7-highspeed')
    thinking, text = client.call_with_thinking(prompt, max_tokens=8192, temperature=0.3)
    return text


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
# 主函数
# ========================================
def main():
    # 加载用户画像（含商品属性）
    log(f"加载用户画像 from {USER_PROFILES_FILE}...")
    with open(USER_PROFILES_FILE, 'r') as f:
        all_user_profiles = json.load(f)
    log(f"加载了 {len(all_user_profiles)} 个用户画像")

    # 构建 personas
    personas = []
    for profile in all_user_profiles:
        uid = profile['user_id']
        products = profile.get('products', [])
        if not products:
            continue

        # 只取第一个商品
        prod = products[0]
        asin = prod.get('asin', '')

        # 转换字段名：A1_product_type → A1, A2_brand → A2, etc.
        product_attrs = {
            'A1': prod.get('A1_product_type', ''),
            'A2': prod.get('A2_brand', ''),
            'A3': prod.get('A3_price', ''),
            'A4': prod.get('A4_appearance', ''),
            'A5': prod.get('A5_use_case', ''),
        }

        words_per_acl = profile.get('words_per_acl')
        if words_per_acl is None:
            words_per_acl = 100.0

        persona = {
            'user_id': uid,
            'asin': asin,
            'acl_sentence_ratio': profile.get('acl_sentence_ratio', 0.0),
            'acl_per_sentence': profile.get('acl_per_sentence', 0.0),
            'avg_sentence_length': profile.get('avg_sentence_length', 20.0),
            'words_per_acl': words_per_acl,
            'freq_label': profile.get('freq_label', 'medium'),
            'density_label': profile.get('density_label', 'simple'),
            'length_label': profile.get('length_label', 'medium'),
            'total_sentences': profile.get('total_sentences', 1),
            'total_acl_count': profile.get('total_acl_count', 0),
            'acl_type_distribution': profile.get('acl_type_distribution', {}),
            'product_attrs': product_attrs,
        }
        personas.append(persona)
        if len(personas) >= NUM_USERS:
            break

    log(f"找到 {len(personas)} 个有完整数据的用户")

    log(f"开始处理 {len(personas)} 个用户，并发数={MAX_WORKERS}")

    def process_one(persona):
        start = time.time()
        product = f"{persona['product_attrs'].get('A1')} by {persona['product_attrs'].get('A2')}"

        # 计算 target_acl_count
        words_per_acl = persona.get('words_per_acl', 20.0)
        if words_per_acl > 35:
            target_acl_count = 0
        elif words_per_acl >= 18:
            target_acl_count = 1
        elif words_per_acl >= 10:
            target_acl_count = 2
        else:
            target_acl_count = 3

        prompt = build_persona_prompt(persona, persona['product_attrs'])
        text = call_llm(prompt)
        query_data = parse_query(text)
        if query_data:
            # 获取实际使用的 acl_types
            acl_type_dist = persona.get('acl_type_distribution', {})
            used_acl_types = list(acl_type_dist.keys())[:target_acl_count] if target_acl_count > 0 else []

            result = {
                'user_id': persona['user_id'],
                'asin': persona['asin'],
                'product': product,
                'target_acl': target_acl_count,
                'used_acl_types': used_acl_types,
                'persona': {
                    'acl_sentence_ratio': persona.get('acl_sentence_ratio', 0.0),
                    'acl_per_sentence': persona.get('acl_per_sentence', 0.0),
                    'avg_sentence_length': persona['avg_sentence_length'],
                    'density_label': persona['density_label'],
                    'length_label': persona['length_label'],
                },
                'product_attrs': persona['product_attrs'],
                'generated_query': query_data['query'],
                'word_count': query_data['word_count'],
                'elapsed': time.time() - start,
            }
            log(f"  [target={target_acl_count}, used={used_acl_types}] OK: words={query_data['word_count']}")
            return result
        else:
            log(f"  [target={target_acl_count}] FAILED")
            return None

    results = []
    total_batches = (len(personas) + BATCH_SIZE - 1) // BATCH_SIZE
    total_start = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(personas))
        batch_personas = personas[batch_start:batch_end]

        log(f"\n--- Batch {batch_idx + 1}/{total_batches}: 用户 {batch_start + 1}-{batch_end} ---")
        batch_start_time = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, p): p for p in batch_personas}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        batch_elapsed = time.time() - batch_start_time
        log(f"  Batch {batch_idx + 1} 完成: {len(batch_personas)} 个, 耗时 {batch_elapsed:.1f}s, 进度 {batch_end}/{len(personas)}")

    total_elapsed = time.time() - total_start

    # 保存结果
    if OUTPUT_FILE:
        output_file = OUTPUT_FILE
    else:
        output_file = f'/fs04/ar57/wenyu/result/personal_query/06_query/persona_generated_acl_{len(personas)}users.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"总计: {len(results)} queries generated in {total_elapsed:.1f}s")
    log(f"Saved to {output_file}")


if __name__ == '__main__':
    main()