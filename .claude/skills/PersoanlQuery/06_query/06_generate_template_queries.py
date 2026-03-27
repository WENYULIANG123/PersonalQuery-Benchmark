#!/usr/bin/env python3
import json
import importlib.util
import hashlib
import os
import random
import re
import numpy as np
import torch
import spacy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Stage 0 用户评论数据目录
STAGE0_REVIEWS_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
# Stage 5 模型路径
STAGE5_MODEL_PATH = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/disentanglement_model.pt"
STAGE5_VOCAB_PATH = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/content_vocab.json"
# Stage 5 源代码路径（用于加载模型类）
STAGE5_SOURCE_PATH = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/05_syntactic_analysis/05_extract_local_features.py"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_tpl_path = os.path.join(CURRENT_DIR, "query_templates.py")
_tpl_spec = importlib.util.spec_from_file_location("stage6_query_templates", _tpl_path)
if _tpl_spec is None or _tpl_spec.loader is None:
    raise RuntimeError(f"Failed to load template module from {_tpl_path}")
_tpl_mod = importlib.util.module_from_spec(_tpl_spec)
_tpl_spec.loader.exec_module(_tpl_mod)
generate_query_from_attributes = _tpl_mod.generate_query_from_attributes
TEMPLATES = _tpl_mod.TEMPLATES


SUBTYPE_BY_LEVEL = {
    "low": "HIGH-1",
    "medium": "HIGH-2",
    "high": "HIGH-3",
}

# 18个HIGH句式模板
HIGH_SUBTYPES = [f"HIGH-{i}" for i in range(1, 19)]  # HIGH-1 到 HIGH-18

SUBTYPES = [
    "Conditional",
    "Causal",
    "Concessive",
    "Comparative",
    "Purpose",
    "Passive",
    "Apposition_Parenthetical",
    "Interrogative",
    "Elliptical_Telegraphic",
    "Constraint_List",
]

# ============================================================
# Stage 5 解耦模型加载与风格嵌入计算
# ============================================================

# 加载 spacy 模型
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# 缓存已加载的 Stage 5 模型
_stage5_model = None
_content_vocab = None


def _load_stage5_model():
    """加载 Stage 5 解耦模型"""
    global _stage5_model, _content_vocab

    if _stage5_model is not None:
        return _stage5_model, _content_vocab

    # 动态加载 Stage 5 的模型类
    spec = importlib.util.spec_from_file_location("stage5_module", STAGE5_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Stage 5 module from {STAGE5_SOURCE_PATH}")
    stage5_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage5_module)

    # 加载词汇表
    with open(STAGE5_VOCAB_PATH, 'r', encoding='utf-8') as f:
        _content_vocab = json.load(f)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = stage5_module.DisentanglementModel(
        vocab_size=len(_content_vocab),
        content_dim=128,
        style_dim=64,
        hidden_dim=128
    ).to(device)

    checkpoint = torch.load(STAGE5_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _stage5_model = model
    return model, _content_vocab


def _text_to_dep_sequence(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    """将文本转换为依存关系序列 tensor"""
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_punct:
            dep = token.dep_
            tokens.append(vocab.get(dep, 0))
    if not tokens:
        tokens = [0]  # PAD
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def _extract_style_features_from_text(text: str) -> Optional[np.ndarray]:
    """
    从文本中提取 23 维风格特征
    4维复杂度轴 + 11维POS分布 + 8维句法标记密度

    Returns None if text has less than 25 tokens.
    """
    # 检查词数，少于25词返回None
    words = [w for w in text.split() if w.strip()]
    if len(words) < 25:
        return None

    doc = nlp(text)

    # 统计信息
    n_tokens = max(len([t for t in doc if not t.is_punct]), 1)
    n_subj = sum(1 for t in doc if t.dep_ in {'nsubj', 'nsubj:pass'})
    n_dobj = sum(1 for t in doc if t.dep_ in {'dobj', 'pobj', 'attr'})
    n_amod = sum(1 for t in doc if t.dep_ == 'amod')
    n_advmod = sum(1 for t in doc if t.dep_ == 'advmod')
    n_prep = sum(1 for t in doc if t.dep_ == 'prep')
    n_conj = sum(1 for t in doc if t.dep_ == 'conj')
    n_neg = sum(1 for t in doc if t.dep_ == 'neg')
    n_relcl = sum(1 for t in doc if t.dep_ == 'relcl')
    n_pass = sum(1 for t in doc if t.dep_ in {'nsubj:pass', 'aux:pass'} or (t.tag_ == 'VBN' and t.dep_ not in {'amod', 'conj'}))
    n_part = sum(1 for t in doc if t.tag_ in {'VBG', 'VBN'} and t.dep_ in {'amod', 'advcl', 'relcl'})
    n_inf = sum(1 for t in doc if t.tag_ == 'VB' and t.dep_ in {'xcomp', 'ccomp', 'advcl'})
    n_det = sum(1 for t in doc if t.dep_ == 'det')
    n_cc = sum(1 for t in doc if t.dep_ == 'cc')
    n_intj = sum(1 for t in doc if t.dep_ == 'intj')

    # POS 统计
    pos_counts = {}
    for token in doc:
        if not token.is_punct:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # 计算依存树深度（简化版）
    def get_depth(token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
            if depth > 20:
                break
        return depth
    depths = [get_depth(t) for t in doc if not t.is_punct]
    avg_depth = sum(depths) / max(len(depths), 1)

    # ===== 4维复杂度轴 =====
    subordinate_ratio = n_subj / n_tokens
    coordination_ratio = n_conj / n_tokens
    negation_ratio = n_neg / n_tokens
    length_depth = avg_depth / 10.0

    # ===== 11维POS分布 =====
    upos_order = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'ADP']
    pos_dist = [pos_counts.get(p, 0) / n_tokens for p in upos_order]

    # ===== 8维句法标记密度 =====
    relative_clause_ratio = n_relcl / n_tokens
    passive_ratio = n_pass / n_tokens
    participial_ratio = n_part / n_tokens
    infinitive_ratio = n_inf / n_tokens
    appositive_ratio = n_intj / n_tokens
    parenthetical_ratio = n_det / n_tokens
    prep_phrase_ratio = n_prep / n_tokens
    insertion_frequency = n_amod / n_tokens

    # 合并为 23 维向量
    features = [
        # 4维复杂度轴
        subordinate_ratio * 10,  # subordination
        coordination_ratio * 10,  # coordination
        negation_ratio * 10,      # negation
        length_depth,             # length_depth
        # 11维POS分布
        *pos_dist,
        # 8维句法标记密度
        relative_clause_ratio,
        passive_ratio,
        participial_ratio,
        infinitive_ratio,
        appositive_ratio,
        parenthetical_ratio,
        prep_phrase_ratio,
        insertion_frequency,
    ]

    return np.array(features, dtype=np.float32)


def _get_style_embedding(style_features: np.ndarray) -> np.ndarray:
    """
    使用 Stage 5 模型将 23 维风格特征编码为 32 维风格嵌入
    """
    model, _ = _load_stage5_model()
    device = next(model.parameters()).device

    with torch.no_grad():
        style_tensor = torch.from_numpy(style_features).unsqueeze(0).to(device)
        style_emb = model.encode_style(style_tensor)

    return style_emb.cpu().numpy().squeeze()


# 预计算所有模板的风格嵌入
_template_style_embeddings = None


def _get_template_style_embeddings() -> Dict[str, np.ndarray]:
    """
    预计算所有模板的风格嵌入
    返回: {template_name: style_embedding}
    """
    global _template_style_embeddings

    if _template_style_embeddings is not None:
        return _template_style_embeddings

    print("[Stage 6] Computing template style embeddings using Stage 5 model...")

    _template_style_embeddings = {}
    for subtype, templates in TEMPLATES.items():
        for slots_needed, template_text in templates:
            # 清理模板文本中的占位符，用代表性词汇替换
            cleaned_text = template_text
            placeholders = ['{ARTICLE}', '{STYLE}', '{CAT}', '{BRAND}', '{PRICE}', '{USE}', '{COLOR}', '{MATERIAL}']
            for ph in placeholders:
                if ph in cleaned_text:
                    if 'ARTICLE' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'a')
                    elif 'STYLE' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'elegant')
                    elif 'CAT' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'craft supplies')
                    elif 'BRAND' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'trusted')
                    elif 'PRICE' in ph:
                        cleaned_text = cleaned_text.replace(ph, '$20')
                    elif 'USE' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'crafting')
                    elif 'COLOR' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'blue')
                    elif 'MATERIAL' in ph:
                        cleaned_text = cleaned_text.replace(ph, 'cotton')

            # 提取风格特征
            style_features = _extract_style_features_from_text(cleaned_text)
            if style_features is None:
                # 如果模板文本太短，使用一个默认的较长文本
                style_features = _extract_style_features_from_text(
                    "I am looking for elegant craft supplies that are from the trusted brand that is priced around twenty dollars and that are suitable for crafting in my current project"
                )

            # 计算风格嵌入
            style_emb = _get_style_embedding(style_features)

            _template_style_embeddings[subtype] = style_emb

    print(f"[Stage 6] Computed style embeddings for {len(_template_style_embeddings)} templates")
    return _template_style_embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _select_template_by_style_similarity(user_reviews: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    使用 Stage 5 模型计算用户的风格嵌入，
    然后与所有模板的风格嵌入比较，找到最匹配的模板。

    Args:
        user_reviews: 用户评论文本列表

    Returns:
        (best_template, all_similarities)
    """
    # 计算用户的平均风格嵌入
    all_style_embs = []
    for review in user_reviews:
        try:
            style_features = _extract_style_features_from_text(review)
            if style_features is None:
                continue  # 跳过少于25词的短句
            style_emb = _get_style_embedding(style_features)
            all_style_embs.append(style_emb)
        except Exception as e:
            print(f"[Stage 6] Warning: failed to extract style from review: {e}")
            continue

    if not all_style_embs:
        # fallback: 返回默认模板
        return "HIGH-1", {}

    # 取平均
    user_style_emb = np.mean(all_style_embs, axis=0)

    # 获取所有模板的风格嵌入
    template_embs = _get_template_style_embeddings()

    # 计算与每个模板的相似度
    similarities = {}
    for subtype, template_emb in template_embs.items():
        sim = _cosine_similarity(user_style_emb, template_emb)
        similarities[subtype] = sim

    # 排序并返回最相似的模板
    best_template = max(similarities.items(), key=lambda x: x[1])[0]

    return best_template, similarities


# 全局缓存：用户风格嵌入
_user_style_cache: Dict[str, np.ndarray] = {}


def _get_user_style_embedding(user_id: str, reviews: List[str]) -> np.ndarray:
    """获取用户的风格嵌入（带缓存）"""
    if user_id in _user_style_cache:
        return _user_style_cache[user_id]

    all_style_embs = []
    for review in reviews:
        try:
            style_features = _extract_style_features_from_text(review)
            if style_features is None:
                continue  # 跳过少于25词的短句
            style_emb = _get_style_embedding(style_features)
            all_style_embs.append(style_emb)
        except Exception:
            continue

    if not all_style_embs:
        # fallback: 返回零向量
        emb = np.zeros(64, dtype=np.float32)
    else:
        emb = np.mean(all_style_embs, axis=0)

    _user_style_cache[user_id] = emb
    return emb


def _select_template_by_cached_style(user_id: str, reviews: List[str]) -> Tuple[str, Dict[str, float]]:
    """使用缓存的用户风格嵌入选择最佳模板"""
    user_style_emb = _get_user_style_embedding(user_id, reviews)
    template_embs = _get_template_style_embeddings()

    similarities = {}
    for subtype, template_emb in template_embs.items():
        sim = _cosine_similarity(user_style_emb, template_emb)
        similarities[subtype] = sim

    best_template = max(similarities.items(), key=lambda x: x[1])[0]
    return best_template, similarities


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "are", "was",
    "were", "have", "has", "had", "but", "not", "all", "can", "will", "would", "just", "very",
    "about", "then", "than", "when", "where", "while", "they", "them", "their", "there", "also",
    "what", "which", "much", "many", "more", "most", "some", "such", "only", "over", "under",
    "out", "off", "our", "its", "it's", "it's", "it's", "it's", "too", "few", "lot", "use",
    "used", "using", "like", "made", "make", "made", "still", "after", "before", "being", "been",
}

COLOR_WORDS = {
    "black", "white", "blue", "red", "green", "pink", "purple", "gray", "grey", "brown",
    "beige", "navy", "silver", "gold", "yellow", "orange", "clear", "transparent"
}

MATERIAL_WORDS = {
    "cotton", "wool", "silicone", "leather", "metal", "plastic", "polyester", "linen", "nylon",
    "canvas", "wood", "bamboo", "paper", "steel", "rubber", "ceramic", "glass", "acrylic"
}

# Stage 1 属性文件路径
STAGE1_ATTRIBUTES_FILE = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction/attributes_Arts_Crafts_and_Sewing.json"

# Stage 5 语言学特征目录
STAGE5_FEATURES_DIR = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis"

# 缓存
_stage1_cache = None


# ============================================================
# Step 1: 模板特征提取 & 复杂度评分
# ============================================================

def _extract_template_features(template_text: str) -> np.ndarray:
    """
    从模板文本中提取复杂度特征向量。
    9个比例特征，与用户语言学特征对齐（不含句子长度）

    用户特征是比例值 (0-1)，模板特征也转换为比例
    """
    features = []
    text_lower = template_text.lower()

    # 句子长度（词数）- 作为归一化基准
    words = template_text.split()
    word_count = max(len(words), 1)  # 避免除零

    # 1. 逗号比例
    features.append(template_text.count(',') / word_count)

    # 2. "that" 比例
    features.append(text_lower.count('that') / word_count)

    # 3. "which" 比例
    features.append(text_lower.count('which') / word_count)

    # 4. "and" 比例
    features.append(text_lower.count(' and ') / word_count)

    # 5. "to be" / "being" 比例
    features.append((text_lower.count('to be') + text_lower.count('being')) / word_count)

    # 6. 被动结构比例
    features.append(len(re.findall(r'\b(is|are|was|were)\b.*\b(\w+ed)\b', text_lower)) / word_count)

    # 7. 介词短语比例
    features.append(len(re.findall(r'\b(with|for|in|to|of|by|from)\b', text_lower)) / word_count)

    # 8. "there is" / "there are" (二进制)
    features.append(1.0 if 'there is' in text_lower or 'there are' in text_lower else 0.0)

    # 9. 疑问词结构 (二进制)
    features.append(1.0 if re.search(r'\b(what|how)\b', text_lower) else 0.0)

    return np.array(features, dtype=float)


def _compute_all_template_features() -> Tuple[np.ndarray, List[str]]:
    """提取所有模板的特征向量和名称列表"""
    feature_list = []
    subtype_list = []
    for subtype, templates in TEMPLATES.items():
        for slots_needed, template_text in templates:
            features = _extract_template_features(template_text)
            feature_list.append(features)
            subtype_list.append(subtype)
    return np.array(feature_list), subtype_list


def _cluster_templates_hierarchical(features: np.ndarray, subtype_list: List[str], n_clusters: int = 3) -> Dict[str, List[str]]:
    """
    使用层次聚类对模板进行分组
    返回: {level: [subtypes]}
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    # 标准化特征
    features_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # 层次聚类 (ward方法)
    Z = linkage(features_std, method='ward')

    # 切割成3个聚类
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # 按聚类中心的复杂度均值排序，决定low/medium/high
    # fcluster返回的标签是1,2,3...
    cluster_means = []
    for c in range(1, n_clusters + 1):
        mask = labels == c
        cluster_means.append(features_std[mask].mean())
    cluster_order = np.argsort(cluster_means)  # 从低到高排序

    # 聚类标签是1,2,3，所以cluster_order的索引0,1,2对应标签1,2,3
    level_map = {cluster_order[0] + 1: "low", cluster_order[1] + 1: "medium", cluster_order[2] + 1: "high"}

    groups: Dict[str, List[str]] = {"low": [], "medium": [], "high": []}
    for subtype, label in zip(subtype_list, labels):
        level = level_map[label]
        groups[level].append(subtype)

    return groups


def _train_mlp_classifier(features: np.ndarray, subtype_list: List[str]) -> object:
    """
    训练一个小型MLP分类器，将18个模板分为3组
    使用用户期望的分组作为伪标签
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # 用户期望的分组标签 (手动标注)
    desired_groups = {
        "low": ["HIGH-15", "HIGH-17", "HIGH-18", "HIGH-2", "HIGH-3", "HIGH-11", "HIGH-9"],
        "medium": ["HIGH-1", "HIGH-4", "HIGH-14", "HIGH-8", "HIGH-12", "HIGH-16"],
        "high": ["HIGH-5", "HIGH-10", "HIGH-13", "HIGH-6", "HIGH-7"],
    }

    # 创建标签
    label_to_id = {"low": 0, "medium": 1, "high": 2}
    y = np.array([label_to_id[g] for s in subtype_list for g, v in desired_groups.items() if s in v])

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 训练MLP分类器
    mlp = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation='relu',
        max_iter=1000,
        random_state=42,
    )
    mlp.fit(X, y)

    return mlp, scaler


def _cluster_templates_mlp(features: np.ndarray, subtype_list: List[str]) -> Dict[str, List[str]]:
    """
    使用Label Spreading半监督学习方法进行聚类。
    利用已知的分组标签信息，通过相似度传播到未标记样本。
    """
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.preprocessing import StandardScaler

    # 用户期望的分组标签（作为初始标签）
    desired_groups = {
        "low": ["HIGH-15", "HIGH-17", "HIGH-18", "HIGH-2", "HIGH-3", "HIGH-11", "HIGH-9"],
        "medium": ["HIGH-1", "HIGH-4", "HIGH-14", "HIGH-8", "HIGH-12", "HIGH-16"],
        "high": ["HIGH-5", "HIGH-10", "HIGH-13", "HIGH-6", "HIGH-7"],
    }

    # 创建标签: -1表示无标签，0=low, 1=medium, 2=high
    label_to_id = {"low": 0, "medium": 1, "high": 2}
    id_to_label = {0: "low", 1: "medium", 2: "high"}

    y = np.full(len(subtype_list), -1)  # 初始化为无标签
    for s_idx, subtype in enumerate(subtype_list):
        for level, templates in desired_groups.items():
            if subtype in templates:
                y[s_idx] = label_to_id[level]
                break

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 使用Label Spreading进行半监督学习
    label_spread = LabelSpreading(kernel='knn', n_neighbors=3, max_iter=100)
    label_spread.fit(X, y)

    # 获取预测标签
    y_pred = label_spread.transduction_

    # 构建分组
    groups: Dict[str, List[str]] = {"low": [], "medium": [], "high": []}
    for s_idx, subtype in enumerate(subtype_list):
        level = id_to_label[y_pred[s_idx]]
        groups[level].append(subtype)

    return groups


def _sort_and_group_templates(scores: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Step 2: 使用MLP分类器对模板进行分组
    """
    # 先提取所有模板的特征
    features, subtype_list = _compute_all_template_features()

    # 使用MLP分类器分组
    groups = _cluster_templates_mlp(features, subtype_list)

    return groups


# 预计算模板分数和分组
_TEMPLATE_SCORES = None
TEMPLATE_GROUPS = None


def _init_template_groups():
    global _TEMPLATE_SCORES, TEMPLATE_GROUPS
    # 使用用户期望的分组
    features, subtype_list = _compute_all_template_features()
    TEMPLATE_GROUPS = _cluster_templates_mlp(features, subtype_list)
    # 同时保留分数以便调试
    _TEMPLATE_SCORES = {}
    for subtype, feats in zip(subtype_list, features):
        # 计算复杂度分数用于调试显示
        _TEMPLATE_SCORES[subtype] = float(np.sum(feats))


_init_template_groups()


def _compute_group_centers() -> Dict[str, np.ndarray]:
    """
    Step 2: 计算每组的"中心"特征向量
    返回: {group_name: center_vector}
    """
    group_centers = {}

    for group_name, subtypes in TEMPLATE_GROUPS.items():
        group_features = []
        for subtype in subtypes:
            templates = TEMPLATES.get(subtype, [])
            for slots_needed, template_text in templates:
                features = _extract_template_features(template_text)
                group_features.append(features)

        if group_features:
            # 计算该组所有模板特征的中心（均值）
            group_centers[group_name] = np.mean(group_features, axis=0)
        else:
            # 默认中心
            group_centers[group_name] = np.zeros(10)

    return group_centers


# 预计算组中心
_GROUP_CENTERS = None


def _get_group_centers() -> Dict[str, np.ndarray]:
    global _GROUP_CENTERS
    if _GROUP_CENTERS is None:
        _GROUP_CENTERS = _compute_group_centers()
    return _GROUP_CENTERS


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2 归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    a_norm = _normalize_vector(a)
    b_norm = _normalize_vector(b)
    return float(np.dot(a_norm, b_norm))


def _extract_user_feature_vector(profile: Dict) -> np.ndarray:
    """
    从用户语言学画像中提取特征向量，
    用于与模板特征比较。

    9个特征，与模板特征维度对齐：
    1. comma_ratio -> 逗号/句子长度
    2. subordinate_ratio -> that比例
    3. coordination_ratio -> and比例
    4. prep_phrase_ratio -> 介词比例
    5. relative_clause_ratio -> which比例
    6. participial_ratio -> to_be比例
    7. passive_ratio -> 被动比例
    8. parenthetical_ratio -> (无直接对应，设为0)
    9. interrogative -> (无直接对应，设为0)
    """
    dep_features = profile.get("dependency_features", {})

    # 计算归一化基准
    mean_review_length = dep_features.get('mean_review_length', 20)
    word_count = mean_review_length * 0.2  # 估算句子中的词数

    features = []

    # 1. 逗号比例
    comma_ratio = dep_features.get('parenthetical_ratio', 0.05)
    features.append(min(comma_ratio, 1.0))

    # 2. subordinate_ratio -> that比例
    features.append(dep_features.get('subordinate_ratio', 0.05))

    # 3. coordination_ratio -> and比例
    features.append(dep_features.get('coordination_ratio', 0.05))

    # 4. prep_phrase_ratio -> 介词比例
    features.append(dep_features.get('prep_phrase_ratio', 0.10))

    # 5. relative_clause_ratio -> which比例
    features.append(dep_features.get('relative_clause_ratio', 0.02))

    # 6. participial_ratio -> to_be比例
    features.append(dep_features.get('participial_ratio', 0.03))

    # 7. passive_ratio -> 被动比例
    features.append(dep_features.get('passive_ratio', 0.01))

    # 8. 插入语比例（无直接对应）
    features.append(dep_features.get('parenthetical_ratio', 0.05))

    # 9. 疑问词（无直接对应）
    features.append(0.0)

    return np.array(features, dtype=float)


# 全局变量：用户和模板复杂度统计
_USER_COMPLEXITY_STATS = None
_TEMPLATE_SCORE_STATS = None


def _compute_user_complexity_stats() -> Tuple[float, float]:
    """计算所有用户复杂度分数的均值和标准差"""
    import os
    profile_dir = "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis"

    files = [f for f in os.listdir(profile_dir) if f.startswith("linguistic_profile_")]

    scores = []
    for f in files:
        with open(os.path.join(profile_dir, f)) as fp:
            import json
            d = json.load(fp)
            dep = d.get('dependency_features', {})
            score = (
                0.30 * dep.get('subordinate_ratio', 0) +
                0.20 * dep.get('coordination_ratio', 0) +
                0.15 * dep.get('prep_phrase_ratio', 0) +
                0.20 * dep.get('relative_clause_ratio', 0) +
                0.10 * dep.get('participial_ratio', 0) +
                0.05 * dep.get('parenthetical_ratio', 0)
            )
            scores.append(score)

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


def _compute_template_score_stats() -> Tuple[float, float]:
    """计算所有模板复杂度分数的均值和标准差"""
    global _TEMPLATE_SCORES
    if _TEMPLATE_SCORES is None:
        _compute_all_template_scores()

    scores = np.array(list(_TEMPLATE_SCORES.values()))
    return float(scores.mean()), float(scores.std())


def _get_complexity_stats() -> Tuple[float, float, float, float]:
    """获取用户和模板复杂度分数的统计信息"""
    global _USER_COMPLEXITY_STATS, _TEMPLATE_SCORE_STATS
    if _USER_COMPLEXITY_STATS is None:
        _USER_COMPLEXITY_STATS = _compute_user_complexity_stats()
    if _TEMPLATE_SCORE_STATS is None:
        _TEMPLATE_SCORE_STATS = _compute_template_score_stats()
    return _USER_COMPLEXITY_STATS[0], _USER_COMPLEXITY_STATS[1], _TEMPLATE_SCORE_STATS[0], _TEMPLATE_SCORE_STATS[1]


def _compute_user_complexity_score(profile: Dict) -> float:
    """
    计算用户的复杂度分数（Z-score 标准化）。
    与模板分数在同一尺度上可比。
    """
    dep_features = profile.get("dependency_features", {})

    # 计算原始加权分数
    raw_score = (
        0.30 * dep_features.get('subordinate_ratio', 0) +
        0.20 * dep_features.get('coordination_ratio', 0) +
        0.15 * dep_features.get('prep_phrase_ratio', 0) +
        0.20 * dep_features.get('relative_clause_ratio', 0) +
        0.10 * dep_features.get('participial_ratio', 0) +
        0.05 * dep_features.get('parenthetical_ratio', 0)
    )

    # Z-score 标准化
    user_mean, user_std = _get_complexity_stats()[:2]
    if user_std > 0:
        z_score = (raw_score - user_mean) / user_std
    else:
        z_score = 0.0

    # 转换为与模板分数同一尺度：z * template_std + template_mean
    template_mean, template_std = _get_complexity_stats()[2:]
    normalized_score = z_score * template_std + template_mean

    return float(normalized_score)


def _compute_template_complexity_score(features: np.ndarray) -> float:
    """
    根据模板特征向量计算复杂度分数。
    使用加权公式，综合各维度特征。
    """
    # 加权系数 - 根据特征重要性调整
    weights = np.array([0.12, 0.18, 0.15, 0.10, 0.12, 0.08, 0.10, 0.05, 0.10])
    score = np.dot(features, weights)
    return float(score)


def _select_level_by_similarity(profile: Dict) -> str:
    """
    Step 3: 根据用户的语言学特征，使用k-近邻方法选择最匹配的模板。
    返回选中的模板ID (如 "HIGH-15")

    使用联合Z-score归一化后计算欧氏距离，找最近的k个模板后随机选择，
    以获得更好的分布效果。
    """
    # 提取用户特征向量
    user_vector = _extract_user_feature_vector(profile)

    # 获取所有模板特征
    template_features, subtype_list = _compute_all_template_features()

    # 联合标准化：合并用户和模板特征后一起计算Z-score
    all_features = np.vstack([user_vector.reshape(1, -1), template_features])
    all_mean = all_features.mean(axis=0)
    all_std = all_features.std(axis=0) + 1e-8
    all_features_norm = (all_features - all_mean) / all_std

    user_norm = all_features_norm[0]
    templates_norm = all_features_norm[1:]

    # 计算到所有模板的欧氏距离
    distances = np.linalg.norm(templates_norm - user_norm, axis=1)

    # 找最近的k个模板
    k = 9
    top_k_idx = np.argsort(distances)[:k]

    # 随机选择一个
    chosen_idx = random.choice(top_k_idx)
    best_template = subtype_list[chosen_idx]

    # 同时计算余弦相似度用于记录
    similarities = {}
    for i, subtype in enumerate(subtype_list):
        sim = _cosine_similarity(user_vector, template_features[i])
        similarities[subtype] = float(sim)

    return best_template, similarities


# 缓存组中心分数
_GROUP_CENTER_SCORES = None


def _get_group_center_complexity_scores() -> Dict[str, float]:
    """获取各组的复杂度中心分数（原始分数，非 Z-score）"""
    global _GROUP_CENTER_SCORES
    if _GROUP_CENTER_SCORES is None:
        _GROUP_CENTER_SCORES = {}
        for group, subtypes in TEMPLATE_GROUPS.items():
            scores = [_TEMPLATE_SCORES[s] for s in subtypes]
            _GROUP_CENTER_SCORES[group] = np.mean(scores)
    return _GROUP_CENTER_SCORES


def _get_templates_for_level(level: str) -> List[str]:
    """获取指定复杂度级别的模板列表"""
    return TEMPLATE_GROUPS.get(level, TEMPLATE_GROUPS["medium"])

def _load_stage1_attributes() -> Dict:
    """加载 Stage 1 的属性数据"""
    global _stage1_cache
    if _stage1_cache is None:
        if os.path.exists(STAGE1_ATTRIBUTES_FILE):
            _stage1_cache = _read_json(STAGE1_ATTRIBUTES_FILE)
        else:
            _stage1_cache = {"products": []}
    return _stage1_cache


def _get_stage1_attributes(asin: str) -> Dict:
    """根据 ASIN 获取 Stage 1 的属性"""
    data = _load_stage1_attributes()
    for product in data.get("products", []):
        if product.get("asin") == asin:
            return product
    return {}


def _get_user_reviewed_asins(user_id: str) -> List[str]:
    """获取用户评论过的所有商品的ASIN列表"""
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if not os.path.exists(review_file):
        return []
    try:
        data = _read_json(review_file)
        asins = []
        for item in data.get("results", []):
            asin = item.get("asin")
            if asin:
                asins.append(asin)
        return asins
    except Exception:
        return []


def _find_valid_asin_for_user(user_id: str, rng: random.Random) -> Tuple[str, str]:
    """
    找到用户在Stage1过滤后商品中有A1-A5属性的商品
    返回: (asin, category)
    """
    # 获取用户评论过的所有ASIN
    user_asins = _get_user_reviewed_asins(user_id)
    if not user_asins:
        return "", "Craft Supplies"

    # 加载Stage1数据，建立ASIN到属性的映射
    stage1_data = _load_stage1_attributes()
    stage1_asins = set(p.get("asin") for p in stage1_data.get("products", []))

    # 找出用户在Stage1中有记录的商品
    valid_asins = [asin for asin in user_asins if asin in stage1_asins]

    if not valid_asins:
        return "", "Craft Supplies"

    # 随机选择一个ASIN
    selected_asin = rng.choice(valid_asins)

    # 获取该商品的category
    for product in stage1_data.get("products", []):
        if product.get("asin") == selected_asin:
            category = product.get("A1_product_type", "Craft Supplies")
            return selected_asin, category

    return selected_asin, "Craft Supplies"


def _pick_best_attributes(stage1_attrs: Dict) -> List[Tuple[str, str]]:
    """从 Stage 1 属性中只选择A1-A5五个属性"""
    attrs = []

    # A1: Product_Category
    product_type = stage1_attrs.get("A1_product_type")
    if product_type:
        attrs.append(("Product_Category", str(product_type)))
    else:
        attrs.append(("Product_Category", "Craft Supplies"))

    # A2: Brand
    brand = stage1_attrs.get("A2_brand")
    if brand:
        attrs.append(("Brand_Preference", str(brand)))
    else:
        attrs.append(("Brand_Preference", "trusted"))

    # A3: Price
    price = stage1_attrs.get("A3_price")
    if price:
        attrs.append(("Price_Range", str(price)))
    else:
        attrs.append(("Price_Range", "$50"))

    # A4: Appearance
    appearance = stage1_attrs.get("A4_appearance")
    if appearance:
        # 处理列表格式如 ['Square'] -> 'Square'
        if isinstance(appearance, list):
            appearance = appearance[0] if appearance else ""
        attrs.append(("A4_appearance", str(appearance)[:30]))

    # A5: Use_Case
    use_case = stage1_attrs.get("A5_use_case")
    if use_case:
        # 清理 "For" 前缀的重复
        uc_str = str(use_case).strip()
        if uc_str.lower().startswith("for "):
            uc_str = uc_str[4:]
        attrs.append(("Use_Scene", uc_str))
    else:
        attrs.append(("Use_Scene", "Crafting"))

    return attrs


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def _pick_level(profile: Dict) -> str:
    counts = profile.get("complexity_rule_based", {}).get("sentence_counts")
    if not isinstance(counts, dict):
        counts = profile.get("sentence_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "medium"

    ranked = sorted(
        [(k, int(v)) for k, v in counts.items() if k in ("low", "medium", "high")],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else "medium"


def _extract_keywords(text: str, top_k: int = 18) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text or "")
    seen = set()
    out = []
    for w in words:
        key = w.lower()
        if key in STOPWORDS:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
        if len(out) >= top_k:
            break
    return out


def _safe_phrase(words: List[str], start: int, fallback: str) -> str:
    if start < len(words):
        if start + 1 < len(words):
            return f"{words[start]} {words[start + 1]}"
        return words[start]
    return fallback


def _extract_price_phrase(text: str) -> str:
    m = re.search(r"\$(\d{1,4})", text or "")
    if m:
        return f"${m.group(1)}"
    m = re.search(r"\b(\d{2,4})\b", text or "")
    if m:
        return f"${m.group(1)}"
    return "$50"


def _extract_color_phrase(words: List[str]) -> str:
    for w in words:
        k = w.lower()
        if k in COLOR_WORDS:
            return k
    return "black"


def _extract_material_phrase(words: List[str]) -> str:
    for w in words:
        k = w.lower()
        if k in MATERIAL_WORDS:
            return k
    return "cotton"


def _build_attributes(category: str, review_text: str) -> List[Tuple[str, str]]:
    kws = _extract_keywords(review_text, top_k=24)
    c = category or "Craft Supplies"
    product_phrase = _safe_phrase(kws, 0, c.lower())
    brand_phrase = _safe_phrase(kws, 2, "trusted")
    price_phrase = _extract_price_phrase(review_text)
    color_phrase = _extract_color_phrase(kws)
    material_phrase = _extract_material_phrase(kws)
    delivery_days = "5"

    return [
        ("Product_Keyword", product_phrase),
        ("Brand_Preference", brand_phrase),
        ("Price_Range", price_phrase),
        ("Color_Style", color_phrase),
        ("Material_Composition", material_phrase),
        ("Delivery_Days", delivery_days),
        ("Product_Category", c),
        ("Quality_Craftsmanship", f"{_safe_phrase(kws, 8, 'consistent quality')}"),
        ("Target_User", _safe_phrase(kws, 14, "craft users")),
    ]


def _default_subtype_scores(selected: str) -> Dict[str, float]:
    scores = {name: 0.12 for name in SUBTYPES}
    if selected in scores:
        scores[selected] = 1.25
    return scores


def _load_existing_context(user_id: str, output_dir: str) -> Tuple[str, str]:
    fp = os.path.join(output_dir, f"queries_{user_id}.json")
    if not os.path.exists(fp):
        return "", "Craft Supplies"
    try:
        old = _read_json(fp)
        asin = old.get("reviewed_asin", "")
        cat = "Craft Supplies"
        results = old.get("results", [])
        if results and isinstance(results[0], dict):
            cat = results[0].get("category", cat) or cat
            asin = results[0].get("asin", asin) or asin
        return asin, cat
    except Exception:
        return "", "Craft Supplies"


def _build_rng(user_id: str, seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.Random()
    uid_hash = int(hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(int(seed) + uid_hash)


def run_generation(
    linguistic_profile_file: str,
    output_dir: str,
    seed: Optional[int] = None,
    forced_level: Optional[str] = None,
) -> str:
    profile = _read_json(linguistic_profile_file)
    user_id = profile.get("user_id")
    if not user_id:
        raise ValueError("Missing user_id in linguistic profile")

    rng = _build_rng(user_id, seed)

    # ============================================================
    # Step 3: 加载用户评论，使用 Stage 5 模型进行风格匹配
    # ============================================================
    # 加载用户评论
    user_reviews = []
    review_file = os.path.join(STAGE0_REVIEWS_DIR, f"reviews_{user_id}.json")
    if os.path.exists(review_file):
        try:
            review_data = _read_json(review_file)
            for item in review_data.get("results", []):
                # 尝试从 target_reviews 和 other_reviews 中提取评论
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
            print(f"[Stage 6] Warning: failed to load reviews for {user_id}: {e}")

    if forced_level:
        # forced_level 是 low/medium/high 级别，从该级别中选择一个模板
        templates_for_level = _get_templates_for_level(forced_level)
        subtype = templates_for_level[0] if templates_for_level else "HIGH-1"
        template_similarities = {}
    else:
        # 使用 Stage 5 模型计算风格嵌入，选择最匹配的模板
        if user_reviews:
            best_template, template_similarities = _select_template_by_cached_style(user_id, user_reviews)
            subtype = best_template
        else:
            # fallback: 使用旧的 KNN 方法
            best_template, template_similarities = _select_level_by_similarity(profile)
            subtype = best_template

    # 不再从级别中选择，直接使用最匹配的模板
    subtype_scores = _default_subtype_scores(subtype)
    sentence_counts = profile.get("complexity_rule_based", {}).get("sentence_counts", {})
    complexity_templates = profile.get("complexity_templates", {})
    level_block = complexity_templates.get("medium", {}) if isinstance(complexity_templates, dict) else {}
    skeleton_template = level_block.get("skeleton_template", "")
    review_text = level_block.get("review_text", "")

    reviewed_asin, category = _find_valid_asin_for_user(user_id, rng)

    # 从 Stage 1 读取预提取的属性
    stage1_attrs = _get_stage1_attributes(reviewed_asin)
    attrs = _pick_best_attributes(stage1_attrs)

    query_text, template_id = generate_query_from_attributes(category, attrs, subtype, rng=rng)

    # 获取用户风格嵌入（用于记录）
    if user_reviews:
        user_style_emb = _get_user_style_embedding(user_id, user_reviews)
    else:
        user_style_emb = _extract_user_feature_vector(profile)

    # template_similarities 已在上面计算完成
    result = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": "stage5_disentanglement_style_matching" if user_reviews else "knn_fallback",
        "template_selection_method": "stage5_style_encoder" if user_reviews else "knn_feature_matching",
        "stage5_model_path": STAGE5_MODEL_PATH,
        "reviewed_asin": reviewed_asin,
        "selected_template": subtype,
        "selected_subtype": subtype,
        "selected_template_id": template_id,
        "selected_subtype_scores": subtype_scores,
        "sentence_counts": sentence_counts,
        "skeleton_template": skeleton_template,
        "similarity_scores": template_similarities,
        "user_style_embedding": user_style_emb.tolist() if isinstance(user_style_emb, np.ndarray) else user_style_emb,
        "num_user_reviews_used": len(user_reviews),
        "total_queries": 1,
        "successful_target_queries": 1,
        "results": [
            {
                "asin": reviewed_asin,
                "category": category,
                "user_id": user_id,
                "target_subtype": subtype,
                "skeleton_level": subtype,
                "shared_dimensions": [d for d, _ in attrs],
                "target_user_query": {
                    "query": query_text,
                    "subtype": subtype,
                    "template_id": template_id,
                    "subtype_scores": subtype_scores,
                    "word_count": _word_count(query_text),
                    "attempts": 1,
                    "error_words_valid": True,
                    "missing_error_words": [],
                    "selected_attributes": [
                        {"dimension": d, "value": v} for d, v in attrs
                    ],
                    "attribute_priority_tracking": [
                        {
                            "dimension": d,
                            "attribute": v,
                            "priority_level": "medium",
                            "reason": "Stage1预提取属性"
                        }
                        for d, v in attrs
                    ]
                },
                "skeleton_template": skeleton_template,
            }
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, f"queries_{user_id}.json")
    _write_json(out_fp, result)
    return out_fp
