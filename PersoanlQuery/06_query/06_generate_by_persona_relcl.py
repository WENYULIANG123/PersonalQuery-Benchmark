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

# ========================================
# 硬编码参数
# ========================================
NUM_USERS = 5000
MAX_WORKERS = 50
BATCH_SIZE = 50
USER_PROFILES_FILE = '/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis/relcl_user_profiles.json'
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
# 根据画像构建Prompt
# ========================================
def build_persona_prompt(persona: dict, product_attrs: dict) -> str:
    """根据用户画像构建查询生成prompt"""

    # 解析画像指标
    relcl_ratio = persona.get('relcl_sentence_ratio', 0.0)  # 0.0-1.0, 多少句子需要relcl
    relcl_per_sentence = persona.get('relcl_per_sentence', 0.0)  # 每个句子平均多少个relcl
    avg_length = persona.get('avg_sentence_length', 20.0)    # 平均句子长度
    density_label = persona.get('density_label', 'simple')    # simple/complex
    length_label = persona.get('length_label', 'medium')      # short/medium/long
    relcl_pattern = persona.get('relcl_pattern', None)        # 旧格式，如 "that...which"
    relcl_type_dist = persona.get('relcl_type_distribution', None)  # 新格式，如 {"which_conj": 1}

    # 商品属性
    cat = product_attrs.get('A1', '')
    brand = product_attrs.get('A2', '')
    price = product_attrs.get('A3', '')
    appearance = product_attrs.get('A4', '')
    material = product_attrs.get('A5', '')

    # 确定需要多少个relcl（基于 words_per_relcl 规则）
    # 如果 words_per_relcl > 35 → relcl = 0
    # 如果 words_per_relcl 在 18-35 → relcl = 1
    # 如果 words_per_relcl 在 0-18 → relcl = 2
    # 如果 words_per_relcl < 10 → relcl = 3
    words_per_relcl = persona.get('words_per_relcl', 20.0)
    if words_per_relcl > 35:
        target_relcl_count = 0
    elif words_per_relcl >= 18:
        target_relcl_count = 1
    elif words_per_relcl >= 10:
        target_relcl_count = 2
    else:
        target_relcl_count = 3

    # 句子长度固定在15-35词范围内
    target_length = min(max(avg_length, 15), 35)

    # 根据密度标签调整relcl复杂度
    if relcl_type_dist:
        # 使用 relcl_type_distribution 格式，但只取 target_relcl_count 个
        if target_relcl_count == 0:
            relcl_instruction = """CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY ZERO relative clauses.
- FORBIDDEN WORDS: 'that', 'which', 'who', 'whom', 'whose' - DO NOT USE THEM AT ALL."""
        else:
            # 使用 relcl_type_distribution 格式，但只取 target_relcl_count 个
            all_clauses = []
            for relcl_type, count in relcl_type_dist.items():
                for _ in range(count):
                    if relcl_type == "which_conj":
                        all_clauses.append(("which", "active verb"))
                    elif relcl_type == "that_relcl":
                        all_clauses.append(("that", "active verb"))
                    elif relcl_type == "who_relcl":
                        all_clauses.append(("who", "active verb"))
                    elif relcl_type == "null_nsubj_ellipsis":
                        all_clauses.append((" that (ellipsis)", "verb with implicit subject"))
                    elif relcl_type.startswith("unknown_"):
                        all_clauses.append(("that", "active verb"))
                    elif relcl_type.startswith("that_"):
                        all_clauses.append(("that", "active verb"))
                    elif relcl_type.startswith("which_"):
                        all_clauses.append(("which", "active verb"))
                    else:
                        all_clauses.append(("that", "active verb"))

            # 只取前 target_relcl_count 个
            clauses = all_clauses[:target_relcl_count]

            def format_clause(i, word, verb):
                if word == " that (ellipsis)":
                    return f"- Clause {i+1}: OMIT the subject after '{word.strip()}' - use '{word.strip()}' + INTRANSITIVE/PASSIVE verb directly, modifying a NOUN. Example: 'the book that costs $5' (NOT 'the book that I costs $5')"
                return f"- Clause {i+1}: use '{word}' followed by {verb}, modifying a NOUN"

            clauses_str = "\n".join([format_clause(i, w, v) for i, (w, v) in enumerate(clauses)])
            forbidden_words = []
            used_words = set(w for w, _ in clauses)
            if "which" not in used_words:
                forbidden_words.append("'which'")
            if "that" not in used_words:
                forbidden_words.append("'that'")
            if "who" not in used_words:
                forbidden_words.append("'who'")
            forbidden_str = f"FORBIDDEN: Do NOT use {' or '.join(forbidden_words)} in your relative clauses." if forbidden_words else "No additional restrictions."

            relcl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_relcl_count} relative clause(s).
{clauses_str}
{forbidden_str}
- Each clause must have the required structure and modify a NOUN in the sentence."""
    elif relcl_pattern:
        # 指定relcl模式
        parts = relcl_pattern.split('...')
        if len(parts) == 2 and target_relcl_count == 2:
            first_word, second_word = parts[0].strip(), parts[1].strip()
            relcl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_relcl_count} relative clauses.
- FIRST relative clause: use '{first_word}' followed by an active verb, modifying a NOUN.
- SECOND relative clause: use '{second_word}' followed by an active verb, modifying a DIFFERENT NOUN.
- Structure example: "X {first_word} VERB... {second_word} VERB..."
- Example: "Dritz seam ripper {first_word} removes stitches easily {second_word} costs $2.99"
- The relative clauses MUST appear in this exact order: {first_word} clause first, {second_word} clause second."""
        else:
            relcl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_relcl_count} relative clause(s).
- Use 'that', 'which', or 'who' followed by an active verb."""
    elif density_label == 'complex':
        # 使用更多样化的relcl类型
        relcl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_relcl_count} relative clause(s).
- Each relative clause must use 'that', 'which', or 'who' followed by an active verb.
- The relative clause(s) must modify NOUNS in the main sentence.
- Use varied relative clause structures (not all starting with the same word)."""
    else:
        # 简单的relcl
        relcl_instruction = f"""CRITICAL REQUIREMENTS:
- You MUST generate EXACTLY {target_relcl_count} relative clause(s).
- Use 'that' or 'which' followed by an active verb.
- The relative clause modifies a NOUN in the sentence."""

    prompt = f"""Generate a product search query based on the user's linguistic profile.

USER PROFILE:
- Relative clause usage: {relcl_ratio*100:.0f}% of sentences contain relative clauses
- Average sentence length: {avg_length:.0f} words
- Writing style: {density_label} with {length_label} sentences
- Target relative clauses per sentence: {relcl_per_sentence:.1f}

Product attributes (MUST include all 5 in the query):
- Product: {cat}
- Brand: {brand}
- Price: {price}
- Appearance: {appearance}
- Material: {material}

{relcl_instruction}

Word count requirement: The query MUST contain EXACTLY {target_length:.0f} words. Count each word carefully before outputting.

FORBIDDEN patterns:
- NO "because/when/if/although/since" clauses
- NO conjunction-separated clauses like "X and that Y"

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

        words_per_relcl = profile.get('words_per_relcl')
        if words_per_relcl is None:
            words_per_relcl = 100.0

        persona = {
            'user_id': uid,
            'asin': asin,
            'relcl_sentence_ratio': profile.get('relcl_sentence_ratio', 0.0),
            'relcl_per_sentence': profile.get('relcl_per_sentence', 0.0),
            'avg_sentence_length': profile.get('avg_sentence_length', 20.0),
            'words_per_relcl': words_per_relcl,
            'freq_label': profile.get('freq_label', 'medium'),
            'density_label': profile.get('density_label', 'simple'),
            'length_label': profile.get('length_label', 'medium'),
            'total_sentences': profile.get('total_sentences', 1),
            'total_relcl_count': profile.get('total_relcl_count', 0),
            'relcl_type_distribution': profile.get('relcl_type_distribution', {}),
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

        # 计算 target_relcl_count
        words_per_relcl = persona.get('words_per_relcl', 20.0)
        if words_per_relcl > 35:
            target_relcl_count = 0
        elif words_per_relcl >= 18:
            target_relcl_count = 1
        elif words_per_relcl >= 10:
            target_relcl_count = 2
        else:
            target_relcl_count = 3

        prompt = build_persona_prompt(persona, persona['product_attrs'])
        text = call_llm(prompt)
        query_data = parse_query(text)
        if query_data:
            # 获取实际使用的 relcl_types
            relcl_type_dist = persona.get('relcl_type_distribution', {})
            used_relcl_types = list(relcl_type_dist.keys())[:target_relcl_count] if target_relcl_count > 0 else []

            result = {
                'user_id': persona['user_id'],
                'asin': persona['asin'],
                'product': product,
                'target_relcl': target_relcl_count,
                'used_relcl_types': used_relcl_types,
                'persona': {
                    'relcl_sentence_ratio': persona['relcl_sentence_ratio'],
                    'relcl_per_sentence': persona['relcl_per_sentence'],
                    'avg_sentence_length': persona['avg_sentence_length'],
                    'density_label': persona['density_label'],
                    'length_label': persona['length_label'],
                    'relcl_pattern': persona.get('relcl_pattern', None),
                },
                'product_attrs': persona['product_attrs'],
                'generated_query': query_data['query'],
                'word_count': query_data['word_count'],
                'elapsed': time.time() - start,
            }
            log(f"  [target={target_relcl_count}, used={used_relcl_types}] OK: words={query_data['word_count']}")
            return result
        else:
            log(f"  [target={target_relcl_count}] FAILED")
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
        output_file = f'/fs04/ar57/wenyu/result/personal_query/06_query/persona_generated_queries_{len(personas)}users.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"总计: {len(results)} queries generated in {total_elapsed:.1f}s")
    log(f"Saved to {output_file}")


if __name__ == '__main__':
    main()