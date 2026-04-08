#!/usr/bin/env python3
"""
Stage 6 v2: 基于语言学框架的复杂度可控查询生成

核心思路：
  1. 规则骨架生成（5维度 × 5属性 的 linguistically-motivated 映射）
  2. LLM 润色（保持结构 + 保持属性 + 提升流畅度）
  3. 本地spaCy评估（提取14D特征 → 算距离 → 排序选最佳）

相比 v1 的改动：
  - 移除 LingConv API（改用本地spaCy）
  - 新增 linguistically-motivated 骨架生成
  - 新增 LLM 润色步骤
"""
import glob
import json
import os
import sys
import time
import re
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# ============================================================
# 强制刷新 stdout
# ============================================================
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        self.stream.write(msg)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

print(f"[{datetime.now()}] [START] Stage 6 v2 启动", flush=True)

# ============================================================
# 路径配置
# ============================================================
STAGE0_REVIEWS_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
STAGE1_ATTRIBUTES_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/attributes_Arts_Crafts_and_Sewing.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"

ATTRIBUTE_SLOTS = ["A1", "A2", "A3", "A4", "A5"]
MIN_WORDS_FOR_FEATURES = 10

# ============================================================
# LLM 配置
# ============================================================
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import MiniMaxClient

llm_client = MiniMaxClient(model="MiniMax-M2.7-highspeed")

# ============================================================
# 本地 spaCy 14D 特征提取
# ============================================================
import spacy

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("下载 en_core_web_sm 模型...", flush=True)
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load('en_core_web_sm')


def extract_14d_features(text: str) -> Dict[str, float]:
    """从文本提取14维复杂度特征（本地spaCy计算）"""
    doc = nlp(text)
    sentences = list(doc.sents)

    if not sentences:
        return {
            'clause_depth': 0.0,
            'dep_distance': 0.0,
            'modifier_density': 0.0,
            'coord_chain': 0.0,
            'negation_scope': 0.0,
            'voice_ratio': 0.0,
            'branching_direction': 0.5,
            'relative_clause_freq': 0.0,
            'advcl_freq': 0.0,
            'comp_clause_freq': 0.0,
            'fanout': 0.0,
            'parataxis_freq': 0.0,
            'prep_density': 0.0,
            'appos_freq': 0.0,
            'word_count': 0
        }

    all_features = []
    for sent in sentences:
        tokens = [t for t in sent if not t.is_space and not t.is_punct]
        n = len(tokens)
        if n == 0:
            continue

        # 1. 从句深度（不含advcl/ccomp/xcomp）
        SUBORDINATE_DEPS = {'acl', 'acl:relcl', 'relcl', 'csubj', 'csubjpass'}
        clause_depth = sum(1 for t in tokens if t.dep_ in SUBORDINATE_DEPS) / n

        # 2. 依存距离
        total_dist = sum(abs(t.i - t.head.i) for t in tokens if t.head != t and t.head.i < len(tokens))
        max_possible = n * (n - 1) / 2 if n > 1 else 1
        dep_distance = total_dist / max_possible if max_possible > 0 else 0.0

        # 3. 修饰语密度
        MODIFIER_DEPS = {'amod', 'advmod', 'compound', 'nn', 'nummod', 'poss'}
        modifier_density = sum(1 for t in tokens if t.dep_ in MODIFIER_DEPS) / n

        # 4. 并列链
        coord_chain = min(sum(1 for t in tokens if t.dep_ == 'conj'), 5) / 5.0

        # 5. 否定作用域
        neg_scope = 0
        for t in tokens:
            if t.dep_ == 'neg':
                subtree = len(list(t.subtree))
                neg_scope = max(neg_scope, subtree)
        negation_scope = min(neg_scope / n, 1.0) if n > 0 else 0.0

        # 6. Voice Ratio (主动/被动比例)
        nsubj_count = sum(1 for t in tokens if t.dep_ == 'nsubj')
        nsubjpass_count = sum(1 for t in tokens if t.dep_ == 'nsubjpass')
        total_voice = nsubj_count + nsubjpass_count
        voice_ratio = nsubjpass_count / total_voice if total_voice > 0 else 0.0

        # 7. Branching Direction (左右分支比例)
        left_children = sum(1 for t in tokens if len(list(t.lefts)) > 0)
        right_children = sum(1 for t in tokens if len(list(t.rights)) > 0)
        total_branches = left_children + right_children
        branching_direction = left_children / total_branches if total_branches > 0 else 0.5

        # 8. Relative Clause Frequency (关系从句频率)
        RELCL_DEPS = {'relcl', 'acl:relcl'}
        relcl_count = sum(1 for t in tokens if t.dep_ in RELCL_DEPS)
        relative_clause_freq = relcl_count / n if n > 0 else 0.0

        # 9. Adverbial Clause Frequency (状语从句频率) - 高优先级
        advcl_count = sum(1 for t in tokens if t.dep_ == 'advcl')
        advcl_freq = advcl_count / n if n > 0 else 0.0

        # 10. Complement Clause Frequency (补语从句频率) - 高优先级
        COMP_DEPS = {'ccomp', 'xcomp'}
        comp_count = sum(1 for t in tokens if t.dep_ in COMP_DEPS)
        comp_clause_freq = comp_count / n if n > 0 else 0.0

        # 11. Dependency Tree Fan-out (依存树扇出度) - 高优先级
        from collections import Counter
        children_count = Counter(t.head.i for t in tokens if t.dep_ != 'ROOT')
        fanout = sum(children_count.values()) / len(children_count) if children_count else 0.0

        # 12. Parataxis Frequency (并置/插入语频率) - 中优先级
        para_count = sum(1 for t in tokens if t.dep_ == 'parataxis')
        parataxis_freq = para_count / n if n > 0 else 0.0

        # 13. Prepositional Phrase Density (介词短语密度) - 中优先级
        prep_count = sum(1 for t in tokens if t.dep_ == 'prep')
        prep_density = prep_count / n if n > 0 else 0.0

        # 14. Apposition Frequency (同位语频率) - 中优先级
        appos_count = sum(1 for t in tokens if t.dep_ == 'appos')
        appos_freq = appos_count / n if n > 0 else 0.0

        all_features.append({
            'clause_depth': clause_depth,
            'dep_distance': dep_distance,
            'modifier_density': modifier_density,
            'coord_chain': coord_chain,
            'negation_scope': negation_scope,
            'voice_ratio': voice_ratio,
            'branching_direction': branching_direction,
            'relative_clause_freq': relative_clause_freq,
            'advcl_freq': advcl_freq,
            'comp_clause_freq': comp_clause_freq,
            'fanout': fanout,
            'parataxis_freq': parataxis_freq,
            'prep_density': prep_density,
            'appos_freq': appos_freq,
            'word_count': n,
        })

    if not all_features:
        return {
            'clause_depth': 0.0, 'dep_distance': 0.0,
            'modifier_density': 0.0, 'coord_chain': 0.0,
            'negation_scope': 0.0, 'voice_ratio': 0.0,
            'branching_direction': 0.5, 'relative_clause_freq': 0.0,
            'advcl_freq': 0.0, 'comp_clause_freq': 0.0,
            'fanout': 0.0, 'parataxis_freq': 0.0,
            'prep_density': 0.0, 'appos_freq': 0.0,
            'word_count': 0
        }

    n = len(all_features)
    total_words = sum(f.get('word_count', 0) for f in all_features)
    return {
        'subordinate_clause_freq': round(sum((f.get('clause_depth', 0) + f.get('relative_clause_freq', 0)) / 2 for f in all_features) / n, 6),
        'dep_distance': round(sum(f.get('dep_distance', 0) for f in all_features) / n, 6),
        'modifier_density': round(sum(f.get('modifier_density', 0) for f in all_features) / n, 6),
        'coord_chain': round(sum(f.get('coord_chain', 0) for f in all_features) / n, 6),
        'negation_scope': round(sum(f.get('negation_scope', 0) for f in all_features) / n, 6),
        'voice_ratio': round(sum(f.get('voice_ratio', 0) for f in all_features) / n, 6),
        'branching_direction': round(sum(f.get('branching_direction', 0.5) for f in all_features) / n, 6),
        'advcl_freq': round(sum(f.get('advcl_freq', 0) for f in all_features) / n, 6),
        'comp_clause_freq': round(sum(f.get('comp_clause_freq', 0) for f in all_features) / n, 6),
        'fanout': round(sum(f.get('fanout', 0) for f in all_features) / n, 6),
        'parataxis_freq': round(sum(f.get('parataxis_freq', 0) for f in all_features) / n, 6),
        'prep_density': round(sum(f.get('prep_density', 0) for f in all_features) / n, 6),
        'appos_freq': round(sum(f.get('appos_freq', 0) for f in all_features) / n, 6),
        'word_count': total_words,
        'avg_sentence_length': round(total_words / n, 1),
    }


def complexity_distance(c: Dict[str, float], target: Dict[str, float]) -> float:
    """计算候选与目标的复杂度距离（归一化）"""
    # 每维的标准差（需要在数据集上预先计算，这里用经验值）
    stds = {
        "clause_depth": 0.3,
        "dep_distance": 0.15,
        "modifier_density": 0.25,
        "coord_chain": 0.2,
        "negation_scope": 0.2,
    }
    dist = 0.0
    for dim in stds:
        diff = abs(c.get(dim, 0) - target.get(dim, 0))
        dist += (diff / stds[dim]) ** 2
    return round(np.sqrt(dist), 4)


# ============================================================
# 属性匹配（沿用 v1 的宽松匹配）
# ============================================================
def check_attributes_present(candidate: str, attrs: Dict[str, str]) -> Tuple[bool, List[str]]:
    missing = []
    cand_lower = candidate.lower()
    for key in ['A1', 'A2', 'A3', 'A4', 'A5']:
        val = attrs.get(key, '')
        if not val:
            missing.append(key)
            continue
        if not is_present(val, cand_lower, key):
            missing.append(key)
    return len(missing) == 0, missing


def is_present(val: str, cand_lower: str, attr_key: str) -> bool:
    val_lower = val.lower().strip()
    if val_lower in cand_lower:
        return True
    if is_numeric(val_lower):
        return check_numeric(val_lower, cand_lower)
    if attr_key == 'A2':
        val_no_space = re.sub(r'[\s\-_]', '', val_lower)
        cand_no_space = re.sub(r'[\s\-_]', '', cand_lower)
        if val_no_space in cand_no_space:
            return True
    return check_fuzzy(val_lower, cand_lower)


def is_numeric(val: str) -> bool:
    cleaned = re.sub(r'[$€£¥,]', '', val)
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def check_numeric(val: str, cand: str) -> bool:
    target = float(re.sub(r'[$€£¥,]', '', val))
    numbers = re.findall(r'[\d]+\.?\d*', cand)
    for n in numbers:
        if abs(float(n) - target) < 0.01:
            return True
    return False


def check_fuzzy(val_lower: str, cand: str) -> bool:
    from difflib import SequenceMatcher
    val_compact = re.sub(r'[\s\-_]', '', val_lower)
    cand_compact = re.sub(r'[\s\-_]', '', cand)
    if val_compact in cand_compact:
        return True
    val_tokens = val_lower.split()
    if len(val_tokens) > 1 and all(t in cand for t in val_tokens):
        return True
    val_len = len(val_lower)
    for i in range(len(cand) - val_len + 1):
        window = cand[i:i + val_len + 2]
        if SequenceMatcher(None, val_lower, window).ratio() > 0.85:
            return True
    return False


# ============================================================
# 核心改动 1: Linguistically-Motivated 骨架生成
# ============================================================

def profile_to_description(profile: Dict[str, float]) -> str:
    """
    将 14D 数值 profile 翻译成 LLM 能理解的自然语言风格描述
    """
    parts = []

    cd = profile.get('clause_depth', 0)
    if cd > 0.4:
        parts.append("use subordinate clauses (e.g., 'which...', 'that...', 'because...')")
    elif cd > 0.2:
        parts.append("occasionally use a relative clause")
    else:
        parts.append("use simple, flat sentence structure without subordinate clauses")

    cc = profile.get('coord_chain', 0)
    if cc > 0.3:
        parts.append("connect multiple ideas with 'and', 'but', 'or' (e.g., 'for X, Y, and Z')")
    elif cc > 0.15:
        parts.append("use one coordination (e.g., 'X and Y')")
    else:
        parts.append("avoid coordination, keep each idea separate")

    md = profile.get('modifier_density', 0)
    if md > 0.4:
        parts.append("add many descriptive adjectives and adverbs")
    elif md > 0.2:
        parts.append("use some adjectives to describe the product")
    else:
        parts.append("be concise, avoid extra modifiers")

    ns = profile.get('negation_scope', 0)
    if ns > 0.3:
        parts.append("include negation (e.g., 'not expensive', 'never disappoints')")
    else:
        parts.append("use positive/affirmative phrasing")

    dd = profile.get('dep_distance', 0)
    if dd > 0.4:
        parts.append("write longer sentences where subject and verb are far apart, with inserted phrases in between")
    elif dd > 0.2:
        parts.append("write medium-length sentences with some inserted phrases")
    else:
        parts.append("keep subject and verb close together, write short sentences")

    asl = profile.get('avg_sentence_length', 20)
    parts.append(f"target approximately {int(asl)} words per sentence")

    return ";\n- ".join(parts)


def generate_structured_skeleton(
    attrs: Dict[str, str],
    profile: Dict[str, float],
) -> List[str]:
    """
    根据 14D profile 生成多个骨架变体

    核心映射（linguistically motivated）：
      modifier_density → A4（材质）: 修饰语是名词的天然展开方式
      clause_depth     → A2（品牌）: 品牌信息天然用从句补充说明
      coord_chain      → A5（用途）: 用途天然可枚举并列
      negation_scope   → A3（价格）: 价格天然是肯定/否定评价的对象
      dep_distance     → A1（产品名）: 产品名做主语，控制主谓间距
    """
    A1 = attrs.get('A1', 'Product')
    A2 = attrs.get('A2', 'Brand')
    A3 = attrs.get('A3', '0')
    A4 = attrs.get('A4', '')
    A5 = attrs.get('A5', 'general use')

    md = profile.get('modifier_density', 0.3)
    cd = profile.get('clause_depth', 0.3)
    cc = profile.get('coord_chain', 0.2)
    ns = profile.get('negation_scope', 0.1)
    dd = profile.get('dep_distance', 0.3)

    # === modifier_density → A4（材质）展开程度 ===
    if md < 0.2:
        a4 = f"{A4} {A1}"
    elif md < 0.4:
        a4 = f"beautiful {A4} {A1}"
    elif md < 0.6:
        a4 = f"high-quality, beautiful {A4} {A1}"
    else:
        a4 = f"the exceptionally fine, premium {A4} {A1}"

    # === clause_depth → A2（品牌）引入方式 ===
    if cd < 0.2:
        a2 = f"by {A2}"
    elif cd < 0.4:
        a2 = f"which are made by {A2}"
    else:
        a2 = f"which are made by {A2}, who is known for quality products"

    # === coord_chain → A5（用途）并列展开 ===
    extra_uses = _infer_related_uses(A5)
    if cc < 0.15:
        a5 = f"for {A5}"
    elif cc < 0.3:
        a5 = f"for {A5} and {extra_uses[0]}"
    elif cc < 0.5:
        a5 = f"for {A5}, {extra_uses[0]}, and {extra_uses[1]}"
    else:
        a5 = f"for {A5}, {extra_uses[0]}, {extra_uses[1]}, and {extra_uses[2]}"

    # === negation_scope → A3（价格）评价框架 ===
    if ns < 0.2:
        a3 = f"priced at {A3}"
    elif ns < 0.4:
        a3 = f"not expensive at {A3}"
    else:
        a3 = f"neither overpriced at {A3} nor lacking in value"

    # === dep_distance → 组装方式（主谓间距）===
    if dd < 0.2:
        skeletons = [f"{a4} {a2}, {a3}, {a5}"]
    elif dd < 0.4:
        skeletons = [f"{a4}, {a2}, {a3}, are good {a5}"]
    else:
        skeletons = [f"{a4}, {a2}, {a3}, are highly recommended {a5}"]

    return skeletons


def _infer_related_uses(primary_use: str) -> List[str]:
    """推断与主用途相关的扩展用途（用于 coord_chain 展开）"""
    use_lower = primary_use.lower()

    # 常见用途映射
    use_families = {
        "embroidery": ["crafting", "decoration", "DIY projects", "sewing"],
        "sewing": ["quilting", "crafting", "embroidery", "tailoring"],
        "crafting": ["decoration", "DIY projects", "gift-making", "scrapbooking"],
        "painting": ["drawing", "art projects", "illustration", "sketching"],
        "knitting": ["crocheting", "weaving", "textile crafts", "yarn work"],
        "jewelry": ["accessories", "gift-making", "fashion", "beading"],
        "scrapbooking": ["card-making", "paper crafts", "journaling", "decoration"],
    }

    for key, related in use_families.items():
        if key in use_lower:
            return related

    # 默认扩展
    return ["crafting", "decoration", "DIY projects", "creative work"]


# ============================================================
# 核心改动 2: LLM 润色
# ============================================================

def call_llm(prompt: str, max_retries: int = 3, temperature: float = 0.7) -> str:
    """调用 MiniMax LLM"""
    for attempt in range(max_retries):
        try:
            response = llm_client.call(prompt, max_tokens=2000, temperature=temperature)
            if response:
                return response
            print(f"[{datetime.now()}] [LLM] Attempt {attempt+1} returned empty", flush=True)
        except Exception as e:
            print(f"[{datetime.now()}] [LLM] Attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""


def llm_polish_skeletons(
    skeletons: List[str],
    attrs: Dict[str, str],
    profile: Dict[str, float],
    num_outputs: int = 20,
) -> List[str]:
    """
    用 LLM 润色骨架句，生成更自然的查询
    """
    cd = profile.get('clause_depth', 0.3)
    ns = profile.get('negation_scope', 0.1)
    md = profile.get('modifier_density', 0.3)
    asl = profile.get('avg_sentence_length', 20)

    a1 = attrs.get('A1', '')
    a2 = attrs.get('A2', '')
    a3 = attrs.get('A3', '')
    a4 = attrs.get('A4', '')
    a5 = attrs.get('A5', '')

    prompt = f"""Write {num_outputs} product search queries about {a1}.

Example: "Beautiful {a4} {a1} which are made by {a2}, not expensive at {a3}, perfect for {a5}"

Now write {num_outputs} different queries about {a1}, each as a complete sentence.

Requirements:
- Include all: {a1}, {a2}, {a3}, {a4}, {a5}
- Length: ~{int(asl)} words
- Style: {"use which/that" if cd > 0.2 else "simple"}, {"negation" if ns > 0.2 else "positive"}, {"adjectives" if md > 0.3 else "concise"}

Queries:"""

    print(f"[{datetime.now()}] [LLM] 发送润色请求...", flush=True)
    response = call_llm(prompt, temperature=0.5)

    if not response:
        print(f"[{datetime.now()}] [LLM] 未收到响应，返回原始骨架", flush=True)
        return skeletons

    # 解析候选行
    candidates = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # 跳过meta-commentary行（包含多个指令性关键词）
        line_lower = line.lower()
        skip_keywords = ['generate', 'query', 'here', 'output', 'must', 'rules',
                        'task', 'important', 'required', 'example', 'template',
                        'follow', 'we need', 'avoid', 'coordination', 'subject',
                        'negation', 'approximately', 'words', 'sentence',
                        'count', 'think', 'need to', 'should', 'could', 'will',
                        'product', 'brand', 'price', 'material', 'use', 'item',
                        'clause', 'relative', 'subordinate', 'example:',
                        "let's", 'potential:', 'now write', 'final:',
                        'now ensure', 'ensure each', 'all tokens', 'each query']
        kw_count = sum(1 for kw in skip_keywords if kw in line_lower)
        if kw_count >= 2:
            continue
        # 移除可能的编号前缀
        cleaned = re.sub(r'^[\d\.\)\-\*\•]+\s*', '', line)
        # 移除 "let's write:" "Potential:" 等前缀
        cleaned = re.sub(r'^(let\'s\s*write[:\s]*|potential[:\s]*|now\s*write[:\s]*|final[:\s]*)', '', cleaned, flags=re.IGNORECASE)
        # 移除首尾引号
        cleaned = cleaned.strip('"\'')
        if cleaned and len(cleaned) > 10 and not cleaned.startswith('//'):
            candidates.append(cleaned)

    print(f"[{datetime.now()}] [LLM] 解析出 {len(candidates)} 个候选", flush=True)
    return candidates


# ============================================================
# 质量过滤
# ============================================================

def quality_filter(candidate: str) -> bool:
    """过滤低质量候选（重复、过长、乱码、meta-commentary等）"""
    words = candidate.lower().split()

    # 太短或太长
    if len(words) < 5 or len(words) > 80:
        return False

    # n-gram 重复检测
    if len(words) > 8:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        if len(ngrams) > 0 and len(set(ngrams)) / len(ngrams) < 0.7:
            return False

    # 包含明显乱码
    if any(w in candidate.lower() for w in ['emoji', 'ringtone', 'earphone', 'maternity']):
        return False

    # 过滤meta-commentary类型的响应
    cand_lower = candidate.lower()
    meta_patterns = [
        'we need to', 'each query must', 'here are', 'generate',
        'output', 'exact value', 'must include', 'rules:',
        'task:', 'important:', 'please provide', 'the query should'
    ]
    # 如果候选包含多个meta关键词，很可能是说明而不是查询
    meta_count = sum(1 for p in meta_patterns if p in cand_lower)
    if meta_count >= 2:
        return False

    # 包含括号编号（如 "Shimmering(1) Crystal(2)..."）
    if re.search(r'\w+\s*\(\d+\)', candidate):
        return False

    return True


# ============================================================
# 数据加载（沿用 v1）
# ============================================================

_stage1_data_cache = None

def _load_stage1_attributes() -> Dict:
    global _stage1_data_cache
    if _stage1_data_cache is None:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            _stage1_data_cache = json.load(f)
    return _stage1_data_cache


def _get_user_attributes(user_id: str) -> Dict[str, str]:
    data = _load_stage1_attributes()
    for p in data.get('products', []):
        if p.get('user_id') == user_id:
            attrs = {}
            key_map = {1: 'A1_product_type', 2: 'A2_brand', 3: 'A3_price',
                       4: 'A4_appearance', 5: 'A5_use_case'}
            for i, slot in enumerate(ATTRIBUTE_SLOTS, 1):
                val = p.get(key_map[i])
                if val:
                    attrs[slot] = val if isinstance(val, str) else ', '.join(val) if isinstance(val, list) else str(val)
            return attrs
    return {}


def load_user_reviews(user_id: str) -> List[str]:
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if not os.path.exists(review_file):
        return []
    with open(review_file, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    user_reviews = []
    for item in review_data.get("results", []):
        for review in item.get("target_reviews", []) + item.get("other_reviews", []):
            text = review if isinstance(review, str) else review.get("review_text", "") if isinstance(review, dict) else ""
            if text:
                user_reviews.append(text)
    return user_reviews


def simple_split_sentences(text: str) -> List[str]:
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_long_sentences(reviews: List[str], min_words: int = 25) -> List[str]:
    long_sentences = []
    for review in reviews:
        for sent in simple_split_sentences(review):
            if len(sent.split()) >= min_words:
                long_sentences.append(sent)
    return long_sentences


def compute_avg_14d_features(sentences: List[str]) -> Optional[Dict[str, float]]:
    """使用本地spaCy计算平均14D特征"""
    if not sentences:
        return None
    all_features = []
    for sent in sentences:
        try:
            features = extract_14d_features(sent)
            all_features.append(features)  # extract_14d_features 返回的是单个 averaged dict
        except Exception as e:
            print(f"[{datetime.now()}] [EVAL] 特征提取失败: {e}", flush=True)
            continue
    if not all_features:
        return None
    n = len(all_features)
    return {
        'subordinate_clause_freq': round(sum(f.get('subordinate_clause_freq', 0) for f in all_features) / n, 6),
        'dep_distance': round(sum(f.get('dep_distance', 0) for f in all_features) / n, 6),
        'modifier_density': round(sum(f.get('modifier_density', 0) for f in all_features) / n, 6),
        'coord_chain': round(sum(f.get('coord_chain', 0) for f in all_features) / n, 6),
        'negation_scope': round(sum(f.get('negation_scope', 0) for f in all_features) / n, 6),
        'voice_ratio': round(sum(f.get('voice_ratio', 0) for f in all_features) / n, 6),
        'branching_direction': round(sum(f.get('branching_direction', 0.5) for f in all_features) / n, 6),
        'advcl_freq': round(sum(f.get('advcl_freq', 0) for f in all_features) / n, 6),
        'comp_clause_freq': round(sum(f.get('comp_clause_freq', 0) for f in all_features) / n, 6),
        'fanout': round(sum(f.get('fanout', 0) for f in all_features) / n, 6),
        'parataxis_freq': round(sum(f.get('parataxis_freq', 0) for f in all_features) / n, 6),
        'prep_density': round(sum(f.get('prep_density', 0) for f in all_features) / n, 6),
        'appos_freq': round(sum(f.get('appos_freq', 0) for f in all_features) / n, 6),
        'avg_sentence_length': round(sum(f.get('word_count', 0) for f in all_features) / n, 1),
    }


# ============================================================
# 主处理函数（简化版：无LLM润色）
# ============================================================

def process_user(user_id: str) -> Optional[Dict]:
    print(f"\n{'='*80}", flush=True)
    print(f"[{datetime.now()}] [{user_id}] 开始处理...", flush=True)

    # ===== 1. 加载数据 =====
    all_reviews = load_user_reviews(user_id)
    if not all_reviews:
        print(f"[{datetime.now()}] [{user_id}] 没有评论，跳过", flush=True)
        return None
    print(f"[{datetime.now()}] [{user_id}] 加载 {len(all_reviews)} 条评论", flush=True)

    user_attrs = _get_user_attributes(user_id)
    if not user_attrs:
        print(f"[{datetime.now()}] [{user_id}] 没有属性，跳过", flush=True)
        return None
    print(f"[{datetime.now()}] [{user_id}] 属性: {user_attrs}", flush=True)

    # ===== 2. 提取用户风格 profile（本地spaCy计算）=====
    long_sentences = extract_long_sentences(all_reviews, min_words=MIN_WORDS_FOR_FEATURES)
    print(f"[{datetime.now()}] [{user_id}] 提取到 {len(long_sentences)} 个长句", flush=True)

    avg_features = compute_avg_14d_features(long_sentences)

    if avg_features:
        target = avg_features
    else:
        # 使用默认值
        target = {
            "clause_depth": 0.3, "dep_distance": 0.1,
            "modifier_density": 0.3, "coord_chain": 0.15,
            "negation_scope": 0.1, "avg_sentence_length": 20.0,
        }
    print(f"[{datetime.now()}] [{user_id}] 目标 14D profile: {target}", flush=True)

    # ===== 3. 保存结果 =====
    return {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline": "local_spacy_14d_only",
        "num_reviews_total": len(all_reviews),
        "num_long_sentences": len(long_sentences),
        "target_complexity": target,
        "attributes": user_attrs,
        "style_description": profile_to_description(target),
    }


# ============================================================
# 配置参数
# ============================================================
MODE = "batch"
TARGET_USER = "A13OFOB1394G31"
BATCH_LIMIT = 0  # 0表示不限制，处理所有用户


# ============================================================
# 主入口
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 清理之前的输出结果
    print(f"[{datetime.now()}] [MAIN] 清理旧输出文件...", flush=True)
    for pattern in [f"queries_*.json", "all_queries_summary_v2.json"]:
        for fp in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
            os.remove(fp)
            print(f"  删除: {fp}", flush=True)
    print(f"[{datetime.now()}] [MAIN] 清理完成", flush=True)

    if MODE == "single":
        print(f"[{datetime.now()}] [MAIN] 单用户模式: {TARGET_USER}", flush=True)
        result = process_user(TARGET_USER)
        if result:
            out_fp = os.path.join(OUTPUT_DIR, f"queries_{TARGET_USER}.json")
            with open(out_fp, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            # print(f"[{datetime.now()}] [MAIN] 结果已保存: {out_fp}", flush=True)
            # print(f"\n===== 结果摘要 =====", flush=True)
            # print(f"目标 14D:  {result['target_complexity']}", flush=True)
    else:
        print(f"[{datetime.now()}] [MAIN] 批量处理模式", flush=True)
        data = _load_stage1_attributes()
        users_set = {p['user_id'] for p in data.get('products', []) if p.get('user_id')}
        user_ids = sorted(list(users_set))
        if BATCH_LIMIT > 0:
            user_ids = user_ids[:BATCH_LIMIT]
        print(f"[{datetime.now()}] [MAIN] 共 {len(user_ids)} 个用户", flush=True)

        results = []
        success_count = 0
        fail_count = 0

        for i, uid in enumerate(user_ids):
            result = process_user(uid)
            if result:
                results.append(result)
                success_count += 1
                out_fp = os.path.join(OUTPUT_DIR, f"queries_{uid}.json")
                with open(out_fp, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                fail_count += 1
            print(f"[{datetime.now()}] [PROGRESS] {i+1}/{len(user_ids)} 用户已完成: {uid}", flush=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "skeleton + llm_polish + local_spacy_eval",
            "total_users": len(user_ids),
            "success_count": success_count,
            "fail_count": fail_count,
            "results": results,
        }
        summary_fp = os.path.join(OUTPUT_DIR, "all_queries_summary_v2.json")
        with open(summary_fp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[{datetime.now()}] [MAIN] 完成! 成功: {success_count}, 失败: {fail_count}", flush=True)


if __name__ == '__main__':
    main()
