#!/usr/bin/env python3
"""
Stage 6: 句法结构提取 - 基于 LLM 的依存句法分析

功能：
1. 对用户评论逐句进行依存句法分析
2. 保留功能词（连词、介词、关系代词、冠词等）和依存拓扑
3. 将实义词替换为类型化槽位（如 {NOUN}、{ADJ}、{VERB}）
4. 计算五个复杂度维度指标
5. 筛选后保留高质量骨架，构建立体化的句法结构池
"""

import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import ChatGLMClient

# ============================================================
# 路径配置
# ============================================================
STAGE0_REVIEWS_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
STAGE1_ATTRIBUTES_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/attributes_Arts_Crafts_and_Sewing.json"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"

# 属性槽位名称（5个）
ATTRIBUTE_SLOTS = ["A1", "A2", "A3", "A4", "A5"]

# ============================================================
# 骨架筛选参数（TBD - 可调整）
# ============================================================
MIN_WORD_COUNT = 6      # 词数下限
MAX_WORD_COUNT = 25     # 词数上限
MIN_SLOT_COUNT = 3      # 最少槽位数
MAX_SKELETONS_PER_USER = 10  # 每用户保留的最大骨架数

# ============================================================
# Stage1 属性加载（缓存）
# ============================================================
_stage1_data_cache = None


def _load_stage1_attributes() -> Dict:
    """加载Stage1属性数据"""
    global _stage1_data_cache
    if _stage1_data_cache is None:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            _stage1_data_cache = json.load(f)
    return _stage1_data_cache


def _get_user_attributes(user_id: str) -> Dict[str, str]:
    """获取用户的5个属性值"""
    data = _load_stage1_attributes()
    products = data.get('products', [])

    # 找该用户的第一个产品
    user_attrs = {}
    for p in products:
        if p.get('user_id') == user_id:
            # 提取5个属性
            for i, slot in enumerate(ATTRIBUTE_SLOTS, 1):
                key = f'A{i}_product_type' if i == 1 else f'A{i}_brand' if i == 2 else f'A{i}_price' if i == 3 else f'A{i}_appearance' if i == 4 else f'A{i}_use_case'
                val = p.get(key)
                if val:
                    user_attrs[slot] = val if isinstance(val, str) else ', '.join(val) if isinstance(val, list) else str(val)
            break
    return user_attrs


def _fill_skeleton_with_attributes(skeleton: str, attributes: Dict[str, str]) -> str:
    """
    将骨架中的槽位替换为属性槽位，基于语义角色映射
    例如: "[NOUN:SUBJECT] [VERB] a [ADJ:MODIFIER] [NOUN:OBJECT]"
    映射: SUBJECT→A1, OBJECT→A3, MODIFIER→A4
    填充后: "{A1} [VERB] a [ADJ:MODIFIER] {A3}"
    """
    import re

    ROLE_TO_ATTR = {
        'SUBJECT': 'A1',
        'OBJECT': 'A3',
        'ADJUNCT': 'A5',
    }
    MODIFIER_ATTRS = ['A4', 'A2', 'A1']

    attr_used = set()

    def replace_match(match):
        full_tag = match.group(1)
        pos = full_tag.split(':')[0] if ':' in full_tag else full_tag
        role = full_tag.split(':')[1] if ':' in full_tag else None

        if role and role in ROLE_TO_ATTR:
            attr = ROLE_TO_ATTR[role]
            if attr in attributes and attr not in attr_used:
                attr_used.add(attr)
                return f'{{{attr}}}'
        elif role == 'MODIFIER':
            for a in MODIFIER_ATTRS:
                if a in attributes and a not in attr_used:
                    attr_used.add(a)
                    return f'{{{a}}}'

        return match.group(0)

    pattern = r'\[([A-Za-z_]+(?::[A-Za-z_]+)?)\]'
    filled_skeleton = re.sub(pattern, replace_match, skeleton)

    return filled_skeleton


# ============================================================
# LLM Client (ChatGLM)
# ============================================================
_client = ChatGLMClient(
    model="glm-5",
    api_key="db2682f8a0024278a672f762ce36d7cd.RC8PtxIy5xdlh8Uj"
)


def _rule_based_question_formation(skeleton: str) -> str:
    """
    基于依存树的疑问句改写
    核心：只在主句的主动词层面操作，其他结构（从句、并列、修饰语）不变
    """
    import re
    import spacy

    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        return skeleton.rstrip('?') + '?'

    TAG_TO_WORD = {
        'NOUN:SUBJECT': 'someone', 'PRON:SUBJECT': 'someone',
        'NOUN:OBJECT': 'something', 'PRON:OBJECT': 'it',
        'NOUN:MODIFIER': 'item', 'ADJ:MODIFIER': 'nice',
        'NOUN:ADJUNCT': 'place',
        'NOUN': 'item', 'VERB': 'use', 'ADJ': 'good', 'ADV': 'well',
        'DET': 'the', 'PRON': 'it', 'AUX': 'do', 'ADP': 'for',
        'NUM': 'one', 'SCONJ': 'that', 'CCONJ': 'and', 'CONJ': 'and',
    }

    def skeleton_to_text(text):
        result = text
        for tag, word in TAG_TO_WORD.items():
            result = re.sub(r'\[' + re.escape(tag) + r'\]', word, result, flags=re.IGNORECASE)
        result = re.sub(r'\[[A-Za-z_:]+\]', 'item', result)
        result = re.sub(r'\{A[1-5]\}', 'item', result)
        result = result.replace(',', ' ,').replace('?', ' ?')
        return result

    text = skeleton_to_text(skeleton)
    doc = nlp(text)

    root = None
    for token in doc:
        if token.head == token:
            root = token
            break

    if root is None:
        return text.rstrip('?') + '?'

    tokens_list = [token.text for token in doc]

    def find_child_by_dep(dep):
        for token in doc:
            if token.head == root and token.dep_ == dep:
                return token
        return None

    def find_subject():
        for token in doc:
            if token.head == root and token.dep_ == 'nsubj':
                return token
        for child in root.children:
            if child.dep_ == 'nsubj':
                return child
        return None

    aux = find_child_by_dep('aux')
    neg = find_child_by_dep('neg')
    subject = find_subject()

    if aux and subject:
        subj_idx = list(doc).index(subject)
        aux_idx = list(doc).index(aux)
        tokens_list.pop(aux_idx)
        tokens_list.insert(0, 'Do')
    elif subject:
        subj_idx = list(doc).index(subject)
        if subj_idx > 0:
            tokens_list.insert(0, 'Do')

    question = ' '.join(tokens_list)
    if not question.endswith('?'):
        question += '?'

    return question


def _rewrite_to_first_person_question(skeleton: str, user_attrs: Dict[str, str]) -> str:
    """
    规则化 + LLM 混合疑问句改写
    优先使用规则化改写保持复杂度，复杂情况 fallback 到 LLM
    """
    skeleton_filled = skeleton
    for attr, value in user_attrs.items():
        skeleton_filled = skeleton_filled.replace(f'{{{attr}}}', str(value))

    try:
        rule_result = _rule_based_question_formation(skeleton_filled)
        if rule_result and '?' in rule_result:
            return rule_result
    except Exception:
        pass

    prompt = f"""你是一个句型改写专家。请将下面的句法骨架改写为第一人称疑问句。

规则：
1. 只替换词性槽位，不要添加额外的词或短语
2. 使用第一人称 "I" 作为主语
3. 必须构成疑问句（以问号结尾）
4. 尽量保持骨架中的功能词和标点位置不变
5. 替换时只使用最简单、最常见的词语

句法骨架：
{skeleton_filled}

请直接输出改写后的疑问句，不要解释。"""

    try:
        response = _client.call_json(prompt, max_tokens=2048, temperature=0.0)
        result = response.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                for key in ["rewritten_sentence", "sentence", "text", "query"]:
                    if key in data:
                        return data[key]
        except json.JSONDecodeError:
            pass
        return result
    except Exception as e:
        _log(f"LLM 改写失败: {str(e)}", "SYSTEM")
        attr_list = [str(v) for v in user_attrs.values() if v]
        if len(attr_list) >= 3:
            return f"Can I find {attr_list[0]} with {attr_list[1]} for {attr_list[2]}?"
        elif attr_list:
            return f"Can I find {attr_list[0]}?"
        return "Can I find something?"


def _compute_query_complexity(text: str) -> Dict[str, float]:
    """
    使用 LLM 计算文本的五个复杂度维度
    """
    prompt = f'''对以下疑问句进行复杂度分析，只返回纯JSON：

文本：{text}

输出格式（只需要第一个句子的分析）：
{{"subordinate_depth":0-3的整数,"dependency_distance":0-1的小数,"modifier_density":0-1的小数,"conj_chain_depth":0-3的整数,"negation_scope":0-10的整数}}

注意：只返回JSON，不要任何解释。'''

    try:
        result = _client.call_json(prompt, max_tokens=512, temperature=0.0)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()
        data = json.loads(result)
        return {
            "subordinate_depth": int(data.get("subordinate_depth", 0)),
            "dependency_distance": float(data.get("dependency_distance", 0)),
            "modifier_density": float(data.get("modifier_density", 0)),
            "conj_chain_depth": int(data.get("conj_chain_depth", 0)),
            "negation_scope": int(data.get("negation_scope", 0))
        }
    except Exception as e:
        _log(f"计算复杂度失败: {str(e)}", "SYSTEM")
        return {"subordinate_depth": 0, "dependency_distance": 0, "modifier_density": 0, "conj_chain_depth": 0, "negation_scope": 0}


# ============================================================
# 五个复杂度维度计算函数（Python端计算）
# ============================================================

def _build_dependency_tree(dep_tree: List[Dict]) -> List[Dict]:
    """
    从 LLM 返回的 dependency_tree 构建内部树结构
    dep_tree: [{"token": "I", "pos": "PRON", "dep": "nsubj", "head": 1}, ...]
    返回: 添加了 children 列表的 tokens
    """
    n = len(dep_tree)
    tokens = []
    for i, t in enumerate(dep_tree):
        tokens.append({
            "index": i,
            "token": t.get("token", ""),
            "pos": t.get("pos", ""),
            "dep": t.get("dep", ""),
            "head": t.get("head", -1),
            "children": []
        })

    # 构建 children 关系
    for i, t in enumerate(tokens):
        head = t["head"]
        if head >= 0 and head < n and head != i:
            tokens[head]["children"].append(i)

    return tokens


def _calculate_subordinate_depth(tokens: List[Dict], idx: int, depth: int = 0) -> int:
    """
    递归计算从句嵌套深度
    从句关系包括: acl, acl:relcl, relcl, advcl, ccomp, xcomp, csubj, csubjpass
    """
    SUBORDINATE_DEPS = {"acl", "acl:relcl", "relcl", "advcl", "ccomp", "xcomp", "csubj", "csubjpass"}

    token = tokens[idx]
    max_depth = depth

    for child_idx in token["children"]:
        child_token = tokens[child_idx]
        if child_token["dep"] in SUBORDINATE_DEPS:
            child_depth = _calculate_subordinate_depth(tokens, child_idx, depth + 1)
            max_depth = max(max_depth, child_depth)

    return max_depth


def _compute_five_dimensions(dep_tree: List[Dict]) -> Dict[str, float]:
    """
    从依存关系树计算五个复杂度维度
    """
    if not dep_tree or len(dep_tree) == 0:
        return {
            "subordinate_depth": 0,
            "dependency_distance": 0.0,
            "modifier_density": 0.0,
            "conj_chain_depth": 0,
            "negation_scope": 0
        }

    n = len(dep_tree)
    tokens = _build_dependency_tree(dep_tree)

    # 1. 从句嵌套深度
    SUBORDINATE_DEPS = {"acl", "acl:relcl", "relcl", "advcl", "ccomp", "xcomp", "csubj", "csubjpass"}
    subordinate_depth = 0
    for i, t in enumerate(tokens):
        if t["dep"] in SUBORDINATE_DEPS:
            depth = _calculate_subordinate_depth(tokens, i, 1)
            subordinate_depth = max(subordinate_depth, depth)

    # 2. 依存距离（归一化）
    total_distance = 0
    for i, t in enumerate(tokens):
        head = t["head"]
        if head >= 0 and head != i:
            total_distance += abs(i - head)
    dependency_distance = total_distance / (n * (n + 1) / 2) if n > 0 else 0.0

    # 3. 修饰语密度
    MODIFIER_DEPS = {"amod", "advmod", "compound", "nn", "nummod"}
    modifier_count = sum(1 for t in tokens if t["dep"] in MODIFIER_DEPS)
    modifier_density = modifier_count / n if n > 0 else 0.0

    # 4. 并列链深度 (Union-Find)
    conj_pairs = []
    for i, t in enumerate(tokens):
        if t["dep"] == "conj":
            head = t["head"]
            if head >= 0 and head != i:
                conj_pairs.append((min(i, head), max(i, head)))

    if conj_pairs:
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in conj_pairs:
            union(i, j)

        component_size = defaultdict(int)
        for i in range(n):
            component_size[find(i)] += 1
        conj_chain_depth = max(component_size.values()) if component_size else 0
    else:
        conj_chain_depth = 0

    # 5. 否定作用域 (BFS)
    negation_scope = 0
    for i, t in enumerate(tokens):
        if t["dep"] == "neg":
            visited = set()
            queue = [i]
            scope_size = 0
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                scope_size += 1
                for child in tokens[curr]["children"]:
                    if child not in visited:
                        queue.append(child)
            negation_scope = max(negation_scope, scope_size)

    return {
        "subordinate_depth": subordinate_depth,
        "dependency_distance": round(dependency_distance, 4),
        "modifier_density": round(modifier_density, 4),
        "conj_chain_depth": conj_chain_depth,
        "negation_scope": negation_scope
    }


def _count_subordinate_depth(doc_tokens: List[Dict], token_idx: int, depth: int = 0) -> int:
    """
    递归计算从句嵌套深度
    从句关系包括: acl, acl:relcl, relcl, advcl, ccomp, xcomp, csubj, csubjpass
    """
    SUBORDINATE_DEPS = {"acl", "acl:relcl", "relcl", "advcl", "ccomp", "xcomp", "csubj", "csubjpass"}

    token = doc_tokens[token_idx]
    current_depth = depth
    max_child_depth = depth

    for child_idx in token.get("children", []):
        if child_idx < len(doc_tokens):
            child_token = doc_tokens[child_idx]
            if child_token["dep"] in SUBORDINATE_DEPS:
                child_depth = _count_subordinate_depth(doc_tokens, child_idx, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth


def _calculate_dependency_distance(doc_tokens: List[Dict]) -> float:
    """
    计算依存距离：所有 head-dependent 词对的线性距离之和
    """
    total_distance = 0
    for i, token in enumerate(doc_tokens):
        head_idx = token.get("head", i)
        if head_idx != i:  # 非根节点
            distance = abs(i - head_idx)
            total_distance += distance
    return total_distance


def _calculate_modifier_density(doc_tokens: List[Dict]) -> float:
    """
    计算修饰语密度：修饰性依存关系数与总词数之比
    修饰性关系包括: amod, advmod, compound, nn, nummod, nsubj, nsubjpass
    """
    MODIFIER_DEPS = {"amod", "advmod", "compound", "nn", "nummod", "nsubj", "nsubjpass"}
    modifier_count = sum(1 for t in doc_tokens if t.get("dep") in MODIFIER_DEPS)
    total_tokens = len(doc_tokens)
    return modifier_count / max(total_tokens, 1)


def _calculate_conj_chain_depth(doc_tokens: List[Dict]) -> int:
    """
    计算并列链深度：conj 关系连接的最长链长度
    """
    # 构建 conj 关系图
    conj_pairs = []
    for i, token in enumerate(doc_tokens):
        if token.get("dep") == "conj":
            head_idx = token.get("head", i)
            if head_idx != i:
                conj_pairs.append((min(i, head_idx), max(i, head_idx)))

    if not conj_pairs:
        return 0

    # 找最长链 - 使用 Union-Find
    parent = list(range(len(doc_tokens)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in conj_pairs:
        union(i, j)

    # 统计每个连通分量的大小
    component_size = defaultdict(int)
    for i in range(len(doc_tokens)):
        component_size[find(i)] += 1

    return max(component_size.values()) if component_size else 0


def _calculate_negation_scope(doc_tokens: List[Dict], token_idx: int) -> int:
    """
    计算否定作用域：否定词所辖依存子树的节点数
    """
    if token_idx >= len(doc_tokens):
        return 0

    token = doc_tokens[token_idx]
    if token.get("dep") != "neg":
        return 0

    # BFS 计算子树大小
    visited = set()
    queue = [token_idx]
    scope_size = 0

    while queue:
        current_idx = queue.pop(0)
        if current_idx in visited:
            continue
        visited.add(current_idx)
        scope_size += 1

        # 添加所有子节点
        for child_idx in doc_tokens[current_idx].get("children", []):
            if child_idx not in visited:
                queue.append(child_idx)

    return scope_size


# ============================================================
# 骨架提取核心（基于 LLM 的依存句法分析）
# ============================================================
def _extract_dependency_analysis_for_single_review(review: str) -> Optional[Dict]:
    """
    对单条评论进行 LLM 依存句法分析，提取骨架并计算五个复杂度维度。
    """
    text = review.strip() if isinstance(review, str) else str(review)
    if len(text) > 400:
        text = text[:400] + "..."

    prompt = '''对评论进行依存句法分析，提取骨架并计算复杂度指标。

## 任务
对每句话进行以下处理：

### 1. 骨架提取（带语义角色）
- 功能词保留原词（the, a, an, in, on, at, for, with, to, from, by, and, but, or, that, which, who, what, where, when, how, why, because, since, while, although, if, unless, be, is, are, was, were, have, has, had, do, does, did, will, would, could, should, may, might, must, can, i, you, he, she, it, we, they, me, him, her, us, them, my, your, his, its, our, their, this, that, these, those, not, very, really, quite, too, also, just）
- 实义词替换为{POS}，格式为 [POS:ROLE]，可用类型包括[NOUN:SUBJECT], [NOUN:OBJECT], [NOUN:MODIFIER], [NOUN:ADJUNCT], [VERB], [ADJ:MODIFIER], [ADV], [PRON], [DET], [AUX], [ADP], [NUM]
- 语义角色：SUBJECT=主语, OBJECT=宾语, MODIFIER=修饰语, ADJUNCT=状语
- 保留标点

### 2. 五个复杂度维度（必须计算）
- subordinate_depth: 从句嵌套深度（acl/relcl/advcl/ccomp/xcomp等从句关系最大嵌套层数，无则填0）
- dependency_distance: 归一化依存距离（|token_i - head_i|之和/(n*(n+1)/2)，0到1之间）
- modifier_density: 修饰语密度（amod/advmod/compound数除以总词数，0到1之间）
- conj_chain_depth: 并列链深度（conj连接最长链节点数，无conj则填0）
- negation_scope: 否定作用域（not节点的依存子树大小，无not则填0）

### 3. 句法类型（仅分析用）
- 简单陈述: 无从句无并列
- 并列约束: 含and/but/or
- 从句嵌套: 含that/which/who/because/if/when
- 介词链: 含多个介词
- 混合复杂: 多种结构混合

### 4. 输出格式
只返回纯JSON，不要任何解释文字：
{"sentences":[{"original":"原句","skeleton":"[PRON] [VERB] a [ADJ:MODIFIER] [NOUN:SUBJECT] that [VERB] [ADV] for [NOUN:OBJECT] in [ADJ:MODIFIER] [NOUN]","word_count":12,"slot_count":9,"slot_roles":["PRON","VERB","DET","ADJ:MODIFIER","NOUN:SUBJECT","VERB","ADV","ADP","NOUN:OBJECT","ADP","ADJ:MODIFIER","NOUN"],"has_subject_predicate":true,"subordinate_depth":1,"dependency_distance":0.45,"modifier_density":0.25,"conj_chain_depth":0,"negation_scope":0,"syntax_type":"从句嵌套"}]}

评论文本：''' + f'\n{text}\n'

    try:
        result = _client.call_json(prompt, max_tokens=2048, temperature=0.0)
        result = result.strip() if result else ""

        if not result:
            return None

        # 清理可能的markdown代码块
        if result.startswith('```'):
            lines = result.split('\n')
            result = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

        data = json.loads(result)
        sentences = data.get("sentences", [])
        if sentences:
            return {"sentences": sentences}
        return None
    except json.JSONDecodeError as e:
        _log(f"Parse failed: {e}", "LLM JSON Error")
        _log(f"result[:500]: {result[:500] if result else 'empty'}", "LLM Raw")
        return None
    except Exception as e:
        _log(f"Error: {e}", "LLM Dependency Analysis")
        _log(f"result[:500]: {result[:500] if result else 'empty'}", "LLM Raw")
        return None


def _extract_skeleton_with_llm(reviews: List[str]) -> List[Dict]:
    """使用 LLM 从用户评论中逐句提取依存骨架和复杂度指标。"""
    if not reviews:
        return []

    results = []
    for i, review in enumerate(reviews[:10]):  # 最多处理10条
        _log(f"Processing review {i+1}/{min(len(reviews), 10)}...", "LLM Dependency")
        analysis = _extract_dependency_analysis_for_single_review(review)

        if analysis and "sentences" in analysis:
            sentences = analysis["sentences"]
            results.append({
                "review_index": i,
                "original_review": review[:200] + "..." if len(review) > 200 else review,
                "sentences": sentences,
            })
            _log(f"{len(sentences)} sentence(s) extracted", "LLM Dependency")
            for j, sent in enumerate(sentences[:3]):  # 最多显示前3句
                _log(f"[{j+1}] {sent.get('skeleton', '')[:80]}...", "LLM Dependency")
                _log(f"    sub_depth={sent.get('subordinate_depth', 0)}, "
                      f"dep_dist={sent.get('dependency_distance', 0):.2f}, "
                      f"mod_dens={sent.get('modifier_density', 0):.2f}, "
                      f"conj_chain={sent.get('conj_chain_depth', 0)}, "
                      f"neg_scope={sent.get('negation_scope', 0)}", "LLM Dependency")
        else:
            _log("Failed to extract skeletons", "LLM Dependency")

    return results


def _filter_and_diversify_skeletons(sentences: List[Dict]) -> List[Dict]:
    """
    筛选高质量骨架并多样化保留。

    筛选条件：
    - 词数 ∈ [MIN_WORD_COUNT, MAX_WORD_COUNT]
    - 槽位数 ≥ MIN_SLOT_COUNT
    - 必须有主谓结构

    多样化策略：
    - 按五个复杂度维度聚类
    - 每类保留结构差异最大的
    - 最多保留 MAX_SKELETONS_PER_USER 个
    """
    filtered = []

    for sent in sentences:
        word_count = sent.get("word_count", 0)
        slot_count = sent.get("slot_count", 0)
        has_subj_pred = sent.get("has_subject_predicate", False)

        # 筛选
        if (MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT and
            slot_count >= MIN_SLOT_COUNT and
            has_subj_pred):
            filtered.append(sent)

    if not filtered:
        return []

    # 按复杂度维度多样性选择
    # 计算每个骨架的复杂度向量
    def complexity_vector(sent):
        return (
            sent.get("subordinate_depth", 0),      # 从句嵌套深度
            sent.get("dependency_distance", 0),    # 依存距离
            sent.get("modifier_density", 0),       # 修饰语密度
            sent.get("conj_chain_depth", 0),       # 并列链深度
            sent.get("negation_scope", 0),         # 否定作用域
        )

    # 使用简单的多样性采样：优先选择复杂度向量差异大的
    selected = []
    remaining = filtered.copy()

    while remaining and len(selected) < MAX_SKELETONS_PER_USER:
        if not selected:
            # 第一个：选择从句深度最大的
            remaining.sort(key=lambda x: x.get("subordinate_depth", 0), reverse=True)
            selected.append(remaining.pop(0))
        else:
            # 之后：选择与已选骨架差异最大的
            best_idx = 0
            best_min_distance = -1

            for i, candidate in enumerate(remaining):
                cand_vec = complexity_vector(candidate)
                # 计算与所有已选骨架的最小距离
                min_dist = float('inf')
                for sel in selected:
                    sel_vec = complexity_vector(sel)
                    # 欧氏距离（简化版）
                    dist = sum((a - b) ** 2 for a, b in zip(cand_vec, sel_vec))
                    min_dist = min(min_dist, dist)

                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_idx = i

            selected.append(remaining.pop(best_idx))

    return selected


def _combine_skeletons(all_skeletons: List[Dict], user_attrs: Dict[str, str]) -> str:
    """
    组合多个骨架，生成包含所有语义角色的统一骨架
    """
    if not all_skeletons:
        return ""

    skeleton_contexts = []
    for i, sk in enumerate(all_skeletons[:8]):
        original = sk.get('original', '')[:100]
        skeleton = sk.get('skeleton', '')
        slot_roles = sk.get('slot_roles', [])
        skeleton_contexts.append(f"骨架{i+1}: {skeleton}\n  原句: {original}\n  角色: {slot_roles}")

    attr_str = ", ".join([f"{k}={v}" for k, v in user_attrs.items()])

    context_text = "\n\n".join(skeleton_contexts)

    prompt = f"""你是一个句型组合专家。请根据下面的多个骨架，组合成一个统一的句法骨架。

## 任务
1. 从多个骨架中提取不同语义角色的片段
2. 将它们组合成一个自然流畅的疑问句骨架
3. 必须包含所有5个属性角色：SUBJECT（主语）、OBJECT（宾语）、MODIFIER（修饰语）、ADJUNCT（状语）
4. 保持骨架的复杂度和句法多样性
5. 输出格式：只返回骨架字符串，用[POS:ROLE]标注

## 属性信息
{attr_str}

## 骨架列表
{context_text}

## 输出要求
- 只返回组合后的骨架字符串
- 格式示例：[PRON:SUBJECT] [VERB] a [ADJ:MODIFIER] [NOUN:OBJECT] for [NOUN:ADJUNCT]?
- 必须包含SUBJECT、OBJECT、MODIFIER、ADJUNCT四种角色
- 以问号结尾表示疑问句

请直接输出组合后的骨架："""

    try:
        response = _client.call_json(prompt, max_tokens=2048, temperature=0.0)
        result = response.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                for key in ["skeleton", "combined", "result", "sentence"]:
                    if key in data:
                        return data[key]
            return result
        except json.JSONDecodeError:
            return result
    except Exception as e:
        _log(f"LLM 骨架组合失败: {str(e)}", "SYSTEM")
        return ""


def _build_template_library(user_id: str, reviews: List[str]) -> Dict:
    """为单个用户构建依存骨架模板库，包含五个复杂度维度标注。"""
    _log(f"使用 LLM 提取句法骨架和复杂度维度...", user_id)
    skeleton_results = _extract_skeleton_with_llm(reviews)

    if skeleton_results:
        _log(f"成功提取 {len(skeleton_results)} 条评论的骨架", user_id)
        # 收集所有句子骨架
        all_skeletons = []
        for sr in skeleton_results:
            if "sentences" in sr:
                all_skeletons.extend(sr["sentences"])

        all_filtered = all_skeletons
        _log(f"共提取 {len(all_filtered)} 个骨架，全部保留", user_id)

        # 计算五个维度的平均值
        mean_sub_depth = sum(s.get("subordinate_depth", 0) for s in all_filtered) / max(len(all_filtered), 1)
        mean_dep_dist = sum(s.get("dependency_distance", 0) for s in all_filtered) / max(len(all_filtered), 1)
        mean_mod_dens = sum(s.get("modifier_density", 0) for s in all_filtered) / max(len(all_filtered), 1)
        mean_conj_chain = sum(s.get("conj_chain_depth", 0) for s in all_filtered) / max(len(all_filtered), 1)
        mean_neg_scope = sum(s.get("negation_scope", 0) for s in all_filtered) / max(len(all_filtered), 1)

        _log(f"五个维度平均值: 从句深度={mean_sub_depth:.2f}, 依存距离={mean_dep_dist:.2f}, "
              f"修饰密度={mean_mod_dens:.2f}, 并列链={mean_conj_chain:.2f}, 否定域={mean_neg_scope:.2f}", user_id)

        # 找到与平均值最接近的骨架
        def _skeleton_distance(sk):
            d_sub = (sk.get("subordinate_depth", 0) - mean_sub_depth) ** 2
            d_dep = (sk.get("dependency_distance", 0) - mean_dep_dist) ** 2
            d_mod = (sk.get("modifier_density", 0) - mean_mod_dens) ** 2
            d_conj = (sk.get("conj_chain_depth", 0) - mean_conj_chain) ** 2
            d_neg = (sk.get("negation_scope", 0) - mean_neg_scope) ** 2
            return (d_sub + d_dep + d_mod + d_conj + d_neg) ** 0.5

        if all_filtered:
            closest_idx = min(range(len(all_filtered)), key=lambda i: _skeleton_distance(all_filtered[i]))
            closest_skeleton = all_filtered[closest_idx]
            _log(f"最接近平均值的骨架 (idx={closest_idx}): {closest_skeleton.get('skeleton', '')[:60]}...", user_id)
            _log(f"    该骨架复杂度: 从句深度={closest_skeleton.get('subordinate_depth')}, 依存距离={closest_skeleton.get('dependency_distance', 0):.2f}, "
                  f"修饰密度={closest_skeleton.get('modifier_density', 0):.2f}, 并列链={closest_skeleton.get('conj_chain_depth')}, 否定域={closest_skeleton.get('negation_scope')}", user_id)

            # 获取用户属性
            user_attrs = _get_user_attributes(user_id)
            _log(f"用户属性: {user_attrs}", user_id)

            if user_attrs:
                # 使用骨架组合，生成包含所有角色的统一骨架
                _log(f"正在使用 LLM 组合骨架以包含所有角色...", user_id)
                combined_skeleton = _combine_skeletons(all_filtered, user_attrs)
                
                if combined_skeleton:
                    _log(f"组合骨架: {combined_skeleton}", user_id)
                    original_skeleton = combined_skeleton
                else:
                    _log(f"组合失败，使用最接近平均值的骨架", user_id)
                    original_skeleton = closest_skeleton.get('skeleton', '')
                
                filled_skeleton = _fill_skeleton_with_attributes(original_skeleton, user_attrs)
                _log(f"属性填充后骨架: {filled_skeleton}", user_id)

                # 使用 LLM 将骨架改写为第一人称疑问句
                _log(f"正在使用 LLM 改写为第一人称疑问句...", user_id)
                final_query = _rewrite_to_first_person_question(filled_skeleton, user_attrs)
                _log(f"最终生成查询语句: {final_query}", user_id)

                # 计算改写后的复杂度
                _log(f"正在计算改写后复杂度...", user_id)
                rewritten_complexity = _compute_query_complexity(final_query)
                original_complexity = closest_skeleton
                _log(f"[改写前] 骨架复杂度: 从句深度={original_complexity.get('subordinate_depth')}, 依存距离={original_complexity.get('dependency_distance', 0):.2f}, "
                      f"修饰密度={original_complexity.get('modifier_density', 0):.2f}, 并列链={original_complexity.get('conj_chain_depth')}, 否定域={original_complexity.get('negation_scope')}", user_id)
                _log(f"[改写后] 查询复杂度: 从句深度={rewritten_complexity.get('subordinate_depth')}, 依存距离={rewritten_complexity.get('dependency_distance', 0):.2f}, "
                      f"修饰密度={rewritten_complexity.get('modifier_density', 0):.2f}, 并列链={rewritten_complexity.get('conj_chain_depth')}, 否定域={rewritten_complexity.get('negation_scope')}", user_id)

                # 创建填充后的骨架副本
                filled_closest = dict(closest_skeleton)
                filled_closest['skeleton'] = filled_skeleton
                filled_closest['filled_skeleton'] = filled_skeleton
                filled_closest['attributes'] = user_attrs
                filled_closest['original_skeleton'] = original_skeleton
            else:
                filled_closest = dict(closest_skeleton)
                filled_closest['attributes'] = {}
        else:
            closest_skeleton = {}
            filled_closest = {}

        # 打印骨架详情
        _log(f"=== 骨架详情 ===", user_id)
        for i, sk in enumerate(all_filtered):
            _log(f"[{i+1}] 原句: {sk.get('original', '')[:80]}...", user_id)
            _log(f"    骨架: {sk.get('skeleton', '')}", user_id)
            _log(f"    词数={sk.get('word_count')}, 槽位数={sk.get('slot_count')}, 类型={sk.get('syntax_type')}", user_id)
            _log(f"    复杂度: 从句深度={sk.get('subordinate_depth')}, 依存距离={sk.get('dependency_distance', 0):.2f}, "
                  f"修饰密度={sk.get('modifier_density', 0):.2f}, 并列链={sk.get('conj_chain_depth')}, 否定域={sk.get('negation_scope')}", user_id)
        _log(f"=== 骨架详情结束 ===", user_id)

        return {
            "user_id": user_id,
            "num_reviews": len(reviews),
            "num_patterns_extracted": len(all_skeletons),
            "num_patterns_filtered": len(all_filtered),
            "num_clusters": 1,
            "template_clusters": [{
                "cluster_id": 0,
                "center_pattern": filled_closest if filled_closest else {},
                "all_skeletons": all_filtered,
                "frequency": len(skeleton_results),
                "examples": skeleton_results,
            }],
            "llm_extracted": True,
            "extraction_method": "llm_dependency_parsing",
            # 五个复杂度维度的统计
            "complexity_stats": {
                "subordinate_depth": {
                    "mean": mean_sub_depth,
                    "max": max((s.get("subordinate_depth", 0) for s in all_filtered), default=0),
                    "min": min((s.get("subordinate_depth", 0) for s in all_filtered), default=0),
                },
                "dependency_distance": {
                    "mean": mean_dep_dist,
                    "max": max((s.get("dependency_distance", 0) for s in all_filtered), default=0),
                    "min": min((s.get("dependency_distance", 0) for s in all_filtered), default=0),
                },
                "modifier_density": {
                    "mean": mean_mod_dens,
                    "max": max((s.get("modifier_density", 0) for s in all_filtered), default=0),
                    "min": min((s.get("modifier_density", 0) for s in all_filtered), default=0),
                },
                "conj_chain_depth": {
                    "mean": mean_conj_chain,
                    "max": max((s.get("conj_chain_depth", 0) for s in all_filtered), default=0),
                    "min": min((s.get("conj_chain_depth", 0) for s in all_filtered), default=0),
                },
                "negation_scope": {
                    "mean": mean_neg_scope,
                    "max": max((s.get("negation_scope", 0) for s in all_filtered), default=0),
                    "min": min((s.get("negation_scope", 0) for s in all_filtered), default=0),
                },
            },
            # 最接近平均值的骨架
            "closest_to_mean": closest_skeleton,
        }
    else:
        _log(f"LLM 提取失败，返回空", user_id)
        return {
            "user_id": user_id,
            "num_reviews": len(reviews),
            "num_patterns_extracted": 0,
            "num_patterns_filtered": 0,
            "num_clusters": 0,
            "template_clusters": [],
        }


# ============================================================
# 缓存
# ============================================================
_user_template_library_cache: Dict[str, Dict] = {}
_CACHE_FILE = os.path.join(OUTPUT_DIR, "template_library_cache.json")


def _load_cache_from_file() -> Dict[str, Dict]:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache_to_file(cache: Dict[str, Dict]):
    try:
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _get_user_template_library(user_id: str, reviews: List[str]) -> Dict:
    """获取用户的 phrase pattern 模板库（带缓存）。"""
    if user_id in _user_template_library_cache:
        return _user_template_library_cache[user_id]

    file_cache = _load_cache_from_file()
    if user_id in file_cache:
        _user_template_library_cache[user_id] = file_cache[user_id]
        return file_cache[user_id]

    template_lib = _build_template_library(user_id, reviews)
    _user_template_library_cache[user_id] = template_lib

    file_cache[user_id] = template_lib
    _save_cache_to_file(file_cache)

    return template_lib


# ============================================================
# 数据加载
# ============================================================
def _load_user_reviews(user_id: str) -> List[str]:
    """加载用户的所有评论。"""
    user_reviews = []
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if not os.path.exists(review_file):
        return []

    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        for item in review_data.get("results", []):
            for review in item.get("target_reviews", []):
                if isinstance(review, str):
                    text = review
                elif isinstance(review, dict):
                    text = review.get("review_text", "")
                else:
                    text = ""
                if text:
                    user_reviews.append(text)
            for review in item.get("other_reviews", []):
                if isinstance(review, str):
                    text = review
                elif isinstance(review, dict):
                    text = review.get("review_text", "")
                else:
                    text = ""
                if text:
                    user_reviews.append(text)
    except Exception as e:
        _log(f"Warning: failed to load reviews for {user_id}: {e}", "Stage 6")

    return user_reviews


# ============================================================
# 主流程
# ============================================================
def run_skeleton_extraction(user_id: str, output_dir: str) -> Dict:
    """为单个用户提取骨架并保存。"""
    _log(f"开始提取骨架...", user_id)

    user_reviews = _load_user_reviews(user_id)
    _log(f"加载了 {len(user_reviews)} 条评论", user_id)

    if not user_reviews:
        _log(f"没有评论，跳过", user_id)
        return None

    template_lib = _get_user_template_library(user_id, user_reviews)

    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "num_reviews": len(user_reviews),
        "template_library": template_lib,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, f"skeleton_{user_id}.json")

    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _log(f"骨架保存到: {out_fp}", user_id)
    return result


def find_users_from_stage1() -> List[str]:
    """从 Stage 1 属性文件中获取所有用户 ID。"""
    try:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        products = data.get('products', [])
        users_set = set()
        for product in products:
            user_id = product.get('user_id')
            if user_id:
                users_set.add(user_id)
        return sorted(list(users_set))
    except Exception as e:
        print(f"ERROR reading Stage 1 file: {e}")
        return []


def validate_users(user_ids: List[str]) -> Dict[str, Dict]:
    """验证用户，返回有效的用户-ASIN 映射。"""
    validated_users = {}
    try:
        with open(STAGE1_ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        products = data.get('products', [])

        for product in products:
            asin = product.get("asin")
            user_id = product.get("user_id")
            if asin and user_id and user_id in user_ids:
                if user_id not in validated_users:
                    validated_users[user_id] = {'user_id': user_id, 'asins': []}
                if asin not in validated_users[user_id]['asins']:
                    validated_users[user_id]['asins'].append(asin)
    except Exception as e:
        print(f"ERROR reading Stage 1 file: {e}")
    return validated_users


def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _log(msg: str, prefix: str = ""):
    """带时间戳的日志打印"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if prefix:
        print(f"[{ts}] [{prefix}] {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", flush=True)


def main():
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    log_with_timestamp("清理缓存和旧骨架文件...")
    if os.path.exists(_CACHE_FILE):
        try:
            os.remove(_CACHE_FILE)
            log_with_timestamp("已删除缓存文件")
        except Exception:
            pass
    old_files = glob.glob(os.path.join(output_dir, 'skeleton_*.json'))
    for f in old_files:
        try:
            os.remove(f)
        except Exception:
            pass
    log_with_timestamp(f"已删除 {len(old_files)} 个旧骨架文件")

    log_with_timestamp("=" * 60)
    log_with_timestamp("Stage 6: 句法结构提取 - 依存分析 + 五维复杂度标注")
    log_with_timestamp("=" * 60)

    # 获取用户列表
    user_ids = find_users_from_stage1()
    if not user_ids:
        log_with_timestamp("ERROR: 没有找到用户!")
        sys.exit(1)

    log_with_timestamp(f"找到 {len(user_ids)} 个用户")

    # 验证用户
    validated_users = validate_users(user_ids)
    log_with_timestamp(f"验证了 {len(validated_users)} 个有效用户")

    # 只处理前1个用户（调试用，可调整）
    validated_users = dict(list(validated_users.items())[:1])
    log_with_timestamp(f"DEBUG: 限制处理前 1 个用户")

    # 逐个处理
    success_count = 0
    fail_count = 0

    for i, (user_id, user_data) in enumerate(validated_users.items()):
        try:
            result = run_skeleton_extraction(user_id, output_dir)
            if result:
                success_count += 1
        except Exception as e:
            import traceback
            log_with_timestamp(f"[{user_id}] ✗ FAILED: {e}")
            fail_count += 1

        if (i + 1) % 10 == 0:
            log_with_timestamp(f"进度: {i+1}/{len(validated_users)}")

    log_with_timestamp("=" * 60)
    log_with_timestamp(f"完成! 成功: {success_count}, 失败: {fail_count}")
    log_with_timestamp("=" * 60)

    # 生成汇总
    all_results = []
    for user_id in validated_users:
        fp = os.path.join(output_dir, f"skeleton_{user_id}.json")
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                all_results.append(json.load(f))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(validated_users),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": all_results,
    }

    summary_fp = os.path.join(output_dir, "all_skeletons_summary.json")
    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"汇总保存到: {summary_fp}")


if __name__ == '__main__':
    main()
