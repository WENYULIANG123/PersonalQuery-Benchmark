#!/usr/bin/env python3
"""基于 LambdaMART 的 Token 级别错误选择器（增强版）

支持：
1. 按类别训练专属模型
2. 按用户错误类型细分（phonetic/visual/spelling）
3. 按属性类型调整（品牌、颜色、口味等）
4. 用户级别的个性化特征
"""

import json
import os
import sys
import re
import random
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import lightgbm as lgb


# ========================================
# 常量
# ========================================
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'im', 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant',
    'couldnt', 'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt',
    'havent', 'hadnt', 'hes', 'shes', 'its', 'theyre', 'weve', 'ive',
    'youre', 'thats', 'whats', 'whos', 'wheres', 'whens', 'hows',
    'lets', 'thats', 'theres', 'heres', 'whys', 'cuz', 'cause', 'bc',
}

# 类别相关的属性关键词
CATEGORY_ATTR_KEYWORDS = {
    'Baby_Products': {
        'colors': {'pink', 'blue', 'white', 'black', 'green', 'purple', 'yellow', 'brown', 'gray', 'grey', 'red', 'orange'},
        'sizes': {'small', 'medium', 'large', 'mini', 'miniature', 'compact', 'tiny', 'big', 'xl', 'xxl', 'xs', 'newborn'},
        'brands': set(),  # 动态从数据中学习
        'materials': {'plastic', 'metal', 'fabric', 'cotton', 'silicon', 'bamboo', 'glass', 'stainless'},
    },
    'Grocery_and_Gourmet_Food': {
        'flavors': {'chocolate', 'vanilla', 'coffee', 'caramel', 'strawberry', 'mint', 'almond', 'cinnamon', 'spicy', 'sweet', 'salty', 'sour'},
        'roasts': {'light', 'medium', 'dark', 'blonde'},
        'forms': {'whole', 'ground', 'instant', 'whole_bean', 'powder', 'liquid', 'frozen', 'dried'},
        'brands': set(),
    },
    'Pet_Supplies': {
        'species': {'dog', 'cat', 'bird', 'fish', 'hamster', 'rabbit', 'reptile', 'small_pet'},
        'breeds': set(),  # 动态从数据中学习
        'materials': {'plastic', 'metal', 'fabric', 'wood', 'leather', 'rubber'},
        'brands': set(),
    },
}

# 错误类型分类（基于常见模式）
PHONETIC_PATTERNS = ['ei', 'ie', 'ough', 'ight', 'tle', 'ple', 'ble', 'dle', 'kel', 'cel']
VISUAL_SIMILAR_PAIRS = [
    ('m', 'n'), ('b', 'd'), ('p', 'q'), ('i', 'l'), ('o', 'e'),
    ('ae', 'e'), ('ve', 'y'), ('our', 'or'), ('ere', 'are'),
]


# ========================================
# 工具函数
# ========================================
def char_similarity(s1: str, s2: str) -> float:
    """计算字符集相似度（Jaccard）"""
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def tokenize_query(query: str) -> list:
    """将 query 分词"""
    query_fixed = re.sub(r"(\w)'(\w)", r"\1'\2", query)
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*", query_fixed)
    return tokens


def remove_stop_words(tokens: list) -> list:
    """删除停用词"""
    result = []
    for token in tokens:
        token_lower = token.lower()
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        if token_lower not in STOP_WORDS:
            result.append(token)
    return result


def classify_error_type(original: str, corrected: str) -> str:
    """分类错误类型: phonetic, visual, spelling"""
    orig_lower = original.lower()
    corr_lower = corrected.lower()

    # 检查音近模式
    for pattern in PHONETIC_PATTERNS:
        if pattern in orig_lower or pattern in corr_lower:
            return 'phonetic'

    # 检查视觉相似
    for a, b in VISUAL_SIMILAR_PAIRS:
        if (a in orig_lower and b in corr_lower) or (b in orig_lower and a in corr_lower):
            return 'visual'

    # 默认拼写错误
    return 'spelling'


def get_user_error_words(user_errors: list) -> tuple:
    """从用户错误中提取信息（增强版）"""
    corrected_words = set()
    error_words = set()
    char_freq = Counter()
    bigram_freq = Counter()
    error_types = {'phonetic': 0, 'visual': 0, 'spelling': 0}
    phonetic_errors = []
    visual_errors = []
    spelling_errors = []

    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()
        if not original or not corrected:
            continue
        if ' ' not in original.strip() and ' ' not in corrected.strip():
            corrected_words.add(corrected)
            error_words.add(original)

        # 字符频率
        for c in original:
            if c.isalpha():
                char_freq[c] += 1

        # Bigram 频率
        for i in range(len(original) - 1):
            bigram = original[i:i+2]
            bigram_freq[bigram] += 1

        # 错误类型分类
        err_type = classify_error_type(original, corrected)
        error_types[err_type] += 1

        if err_type == 'phonetic':
            phonetic_errors.append(original)
        elif err_type == 'visual':
            visual_errors.append(original)
        else:
            spelling_errors.append(original)

    return corrected_words, error_words, char_freq, bigram_freq, error_types, phonetic_errors, visual_errors, spelling_errors


def top_k_similarity(query_token: str, error_words: set, k: int = 5) -> tuple:
    """计算 top-k 相似度"""
    if not error_words:
        return 0.0, []
    similarities = [(err_word, char_similarity(query_token, err_word)) for err_word in error_words]
    similarities.sort(key=lambda x: x[1], reverse=True)
    if not similarities:
        return 0.0, []
    return similarities[0][1], similarities[:k]


def contains_error_chars(token: str, char_freq: Counter) -> float:
    """计算常错字符分数"""
    if not char_freq or not token:
        return 0.0
    total_chars = sum(char_freq.values())
    if total_chars == 0:
        return 0.0
    token_chars = Counter(token.lower())
    error_char_count = sum(token_chars.get(c, 0) for c in char_freq)
    total_token_chars = len([c for c in token if c.isalpha()])
    if total_token_chars == 0:
        return 0.0
    error_ratio = error_char_count / total_token_chars
    max_char_freq = max(char_freq.values())
    char_boost = sum(
        (char_freq[c] / max_char_freq) * (token_chars.get(c, 0) / total_token_chars)
        for c in token.lower() if c in char_freq
    )
    return min(error_ratio + char_boost * 0.5, 1.0)


def contains_error_bigrams(token: str, bigram_freq: Counter) -> float:
    """计算常错 bigram 分数"""
    if not bigram_freq or len(token) < 2:
        return 0.0
    total_bigrams = sum(bigram_freq.values())
    if total_bigrams == 0:
        return 0.0
    token_bigrams = [token[i:i+2].lower() for i in range(len(token) - 1)]
    if not token_bigrams:
        return 0.0
    max_bigram_freq = max(bigram_freq.values())
    error_bigram_score = 0.0
    for bg in token_bigrams:
        if bg in bigram_freq:
            freq_ratio = bigram_freq[bg] / max_bigram_freq
            error_bigram_score += freq_ratio
    return min(error_bigram_score / len(token_bigrams), 1.0)


def surface_difficulty(token: str) -> float:
    """计算表层难度"""
    if not token:
        return 0.0
    score = 0.0
    length = len(token)
    score += min(length * 0.05, 0.3)
    if any(c.isdigit() for c in token):
        score += 0.2
    if any(c.isupper() for c in token):
        score += 0.1
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            score += 0.1
            break
    rare_chars = set('wxqz')
    if any(c.lower() in rare_chars for c in token):
        score += 0.1
    vowels = set('aeiouAEIOU')
    has_alternation = False
    for i in range(len(token) - 1):
        c1_is_vowel = token[i] in vowels
        c2_is_vowel = token[i + 1] in vowels
        if c1_is_vowel != c2_is_vowel:
            has_alternation = True
            break
    if not has_alternation and length > 3:
        score += 0.1
    return min(score, 1.0)


def attribute_type_risk(token: str, attrs_used: dict = None) -> float:
    """计算属性类型风险"""
    if not token:
        return 0.0
    score = 0.0
    if attrs_used:
        token_lower = token.lower()
        for attr_value in attrs_used.values():
            if isinstance(attr_value, str) and token_lower == attr_value.lower():
                score += 0.3
                break
            if isinstance(attr_value, str) and token_lower in attr_value.lower():
                score += 0.2
                break
    if token[0].isupper() and any(c.islower() for c in token):
        score += 0.15
    if any(c.isdigit() for c in token):
        score += 0.2
    colors = {'red', 'blue', 'green', 'white', 'black', 'pink', 'purple',
              'orange', 'yellow', 'brown', 'gray', 'grey', 'gold', 'silver'}
    if token.lower() in colors:
        score += 0.15
    sizes = {'small', 'medium', 'large', 'mini', 'miniature', 'compact',
             'tiny', 'big', 'xl', 'xxl', 'xs'}
    if token.lower() in sizes:
        score += 0.1
    return min(score, 1.0)


def compute_features_enhanced(
    token: str,
    corrected_words: set,
    error_words: set,
    char_freq: Counter,
    bigram_freq: Counter,
    error_types: dict,
    phonetic_errors: list,
    visual_errors: list,
    spelling_errors: list,
    attrs_used: dict = None,
    category: str = None,
    user_error_count: int = 0
) -> dict:
    """计算增强版 token 特征

    包含：
    1. 基础特征
    2. 错误类型细分特征
    3. 类别相关属性特征
    4. 用户级别特征
    """
    # ========== 基础特征 ==========
    max_sim, _ = top_k_similarity(token, error_words, k=5)
    char_score = contains_error_chars(token, char_freq)
    bigram_score = contains_error_bigrams(token, bigram_freq)
    diff_score = surface_difficulty(token)
    attr_score = attribute_type_risk(token, attrs_used)

    # ========== 错误类型细分特征 ==========
    # 音近错误匹配
    phonetic_sim = 0.0
    if phonetic_errors:
        phonetic_sims = [char_similarity(token, pe) for pe in phonetic_errors]
        phonetic_sim = max(phonetic_sims) if phonetic_sims else 0.0

    # 视觉相似错误匹配
    visual_sim = 0.0
    if visual_errors:
        visual_sims = [char_similarity(token, ve) for ve in visual_errors]
        visual_sim = max(visual_sims) if visual_sims else 0.0

    # 拼写错误匹配
    spelling_sim = 0.0
    if spelling_errors:
        spelling_sims = [char_similarity(token, se) for se in spelling_errors]
        spelling_sim = max(spelling_sims) if spelling_sims else 0.0

    # 错误类型分布
    total_errors = sum(error_types.values()) if error_types else 1
    phonetic_ratio = error_types.get('phonetic', 0) / total_errors
    visual_ratio = error_types.get('visual', 0) / total_errors
    spelling_ratio = error_types.get('spelling', 0) / total_errors

    # ========== 类别相关属性特征 ==========
    is_color = 0.0
    is_size = 0.0
    is_brand = 0.0
    is_flavor = 0.0
    is_material = 0.0
    is_species = 0.0

    if category and category in CATEGORY_ATTR_KEYWORDS:
        cat_attrs = CATEGORY_ATTR_KEYWORDS[category]
        token_lower = token.lower()

        if token_lower in cat_attrs.get('colors', set()):
            is_color = 1.0
        if token_lower in cat_attrs.get('sizes', set()):
            is_size = 1.0
        if token_lower in cat_attrs.get('flavors', set()):
            is_flavor = 1.0
        if token_lower in cat_attrs.get('materials', set()):
            is_material = 1.0
        if token_lower in cat_attrs.get('species', set()):
            is_species = 1.0

    # 品牌检测：大写字母开头 + 混合字母
    if token[0].isupper() and any(c.islower() for c in token) and len(token) > 2:
        is_brand = 1.0

    # ========== 用户级别特征 ==========
    # 用户错误频率（归一化）
    user_error_rate = min(user_error_count / 10.0, 1.0) if user_error_count else 0.0

    # ========== 其他基础特征 ==========
    token_len = len(token)
    has_upper = float(any(c.isupper() for c in token))
    has_digit = float(any(c.isdigit() for c in token))
    has_rare_char = float(any(c.lower() in 'wxqz' for c in token))

    # 连续相同字母
    has_double_letter = 0.0
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            has_double_letter = 1.0
            break

    # 元音辅音交替
    vowels = set('aeiouAEIOU')
    vowel_ratio = sum(1 for c in token if c.lower() in vowels) / max(len(token), 1)

    return {
        # 基础特征
        'topk_sim': max_sim,
        'char_score': char_score,
        'bigram_score': bigram_score,
        'surface_difficulty': diff_score,
        'attr_risk': attr_score,

        # 错误类型细分
        'phonetic_sim': phonetic_sim,
        'visual_sim': visual_sim,
        'spelling_sim': spelling_sim,
        'phonetic_ratio': phonetic_ratio,
        'visual_ratio': visual_ratio,
        'spelling_ratio': spelling_ratio,

        # 类别属性
        'is_color': is_color,
        'is_size': is_size,
        'is_brand': is_brand,
        'is_flavor': is_flavor,
        'is_material': is_material,
        'is_species': is_species,

        # 用户级别
        'user_error_rate': user_error_rate,

        # 其他
        'token_len': token_len,
        'has_upper': has_upper,
        'has_digit': has_digit,
        'has_rare_char': has_rare_char,
        'has_double_letter': has_double_letter,
        'vowel_ratio': vowel_ratio,
    }


def compute_features(token: str, corrected_words: set, error_words: set,
                     char_freq: Counter, bigram_freq: Counter,
                     attrs_used: dict = None) -> dict:
    """兼容旧版：计算基础特征"""
    max_sim, _ = top_k_similarity(token, error_words, k=5)
    char_score = contains_error_chars(token, char_freq)
    bigram_score = contains_error_bigrams(token, bigram_freq)
    diff_score = surface_difficulty(token)
    attr_score = attribute_type_risk(token, attrs_used)

    return {
        'topk_sim': max_sim,
        'char_score': char_score,
        'bigram_score': bigram_score,
        'surface_difficulty': diff_score,
        'attr_risk': attr_score,
        'token_len': len(token),
        'has_upper': float(any(c.isupper() for c in token)),
        'has_digit': float(any(c.isdigit() for c in token)),
        'has_rare_char': float(any(c.lower() in 'wxqz' for c in token)),
    }


def extract_training_data_enhanced(
    output_file: str,
    user_errors_file: str,
    query_file: str,
    category: str = None
) -> tuple:
    """从现有输出中提取增强版训练数据"""
    # 加载用户错误
    with open(user_errors_file, 'r', encoding='utf-8') as f:
        errors_data = json.load(f)
    user_errors_dict = {}
    for user in errors_data:
        uid = user['user_id']
        error_details = user.get('error_details', [])
        if error_details:
            user_errors_dict[uid] = error_details

    # 加载查询
    with open(query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    if isinstance(query_data, dict):
        if 'records' in query_data:
            query_data = query_data['records']
        elif 'queries' in query_data:
            query_data = query_data['queries']
        else:
            query_data = []

    query_dict = {}
    for rec in query_data:
        if not isinstance(rec, dict):
            continue
        uid = rec.get('user_id')
        asin = rec.get('asin')
        query = rec.get('query') or ''
        if not query and isinstance(rec.get('syntax_depth_query'), dict):
            query = rec.get('syntax_depth_query', {}).get('query', '')
        if not query and isinstance(rec.get('acl_query'), dict):
            query = rec.get('acl_query', {}).get('query', '')
        if not query and isinstance(rec.get('ccomp_query'), dict):
            query = rec.get('ccomp_query', {}).get('query', '')
        if uid and asin and query:
            query_dict[(uid, asin)] = query

    # 加载输出
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    X = []
    y = []
    query_ids = []
    token_texts = []

    for rec in output_data:
        uid = rec.get('uid')
        asin = rec.get('asin')
        selected_token = rec.get('selected_token')
        errors = user_errors_dict.get(uid, [])
        clean_query = query_dict.get((uid, asin), '')

        if not errors or not clean_query:
            continue

        (
            corrected_words, error_words, char_freq, bigram_freq,
            error_types, phonetic_errors, visual_errors, spelling_errors
        ) = get_user_error_words(errors)

        tokens = tokenize_query(clean_query)
        clean_tokens = remove_stop_words(tokens)

        if not clean_tokens:
            continue

        # 为每个 token 创建样本
        for token in clean_tokens:
            features = compute_features_enhanced(
                token,
                corrected_words, error_words, char_freq, bigram_freq,
                error_types, phonetic_errors, visual_errors, spelling_errors,
                attrs_used=None,  # 如果有 attrs_used 可以传入
                category=category,
                user_error_count=len(errors)
            )
            label = 1 if token == selected_token else 0

            X.append(features)
            y.append(label)
            query_ids.append((uid, asin))
            token_texts.append(token)

    return X, y, query_ids, token_texts


def extract_training_data(output_file: str, user_errors_file: str, query_file: str) -> tuple:
    """从现有输出中提取训练数据（兼容旧版）"""
    # 加载用户错误
    with open(user_errors_file, 'r', encoding='utf-8') as f:
        errors_data = json.load(f)
    user_errors_dict = {}
    for user in errors_data:
        uid = user['user_id']
        error_details = user.get('error_details', [])
        if error_details:
            user_errors_dict[uid] = error_details

    # 加载查询
    with open(query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    if isinstance(query_data, dict):
        if 'records' in query_data:
            query_data = query_data['records']
        elif 'queries' in query_data:
            query_data = query_data['queries']
        else:
            query_data = []

    query_dict = {}
    for rec in query_data:
        if not isinstance(rec, dict):
            continue
        uid = rec.get('user_id')
        asin = rec.get('asin')
        query = rec.get('query') or ''
        if not query and isinstance(rec.get('syntax_depth_query'), dict):
            query = rec.get('syntax_depth_query', {}).get('query', '')
        if not query and isinstance(rec.get('acl_query'), dict):
            query = rec.get('acl_query', {}).get('query', '')
        if not query and isinstance(rec.get('ccomp_query'), dict):
            query = rec.get('ccomp_query', {}).get('query', '')
        if uid and asin and query:
            query_dict[(uid, asin)] = query

    # 加载输出
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    X = []
    y = []
    query_ids = []
    token_texts = []

    for rec in output_data:
        uid = rec.get('uid')
        asin = rec.get('asin')
        selected_token = rec.get('selected_token')
        errors = user_errors_dict.get(uid, [])
        clean_query = query_dict.get((uid, asin), '')

        if not errors or not clean_query:
            continue

        corrected_words, error_words, char_freq, bigram_freq = get_user_error_words(errors)
        tokens = tokenize_query(clean_query)
        clean_tokens = remove_stop_words(tokens)

        if not clean_tokens:
            continue

        # 为每个 token 创建样本
        for token in clean_tokens:
            features = compute_features(token, corrected_words, error_words, char_freq, bigram_freq)
            label = 1 if token == selected_token else 0

            X.append(features)
            y.append(label)
            query_ids.append((uid, asin))
            token_texts.append(token)

    return X, y, query_ids, token_texts


def train_lambdamart(X: list, y: list, query_ids: list) -> tuple:
    """训练 LambdaMART 模型"""
    if not X:
        raise ValueError("X is empty")

    feature_names = list(X[0].keys()) if isinstance(X[0], dict) else [f"f{i}" for i in range(len(X[0]))]
    if isinstance(X[0], dict):
        X_matrix = np.array([[x.get(f, 0.0) for f in feature_names] for x in X])
    else:
        X_matrix = np.array(X)

    # 按 query_id 分组
    query_start = {}
    query_len = {}
    for qid in query_ids:
        if qid not in query_start:
            query_start[qid] = len(query_len)
            query_len[qid] = 0
        query_len[qid] += 1

    # 构建 group
    group = [query_len[qid] for qid in sorted(query_start.keys())]
    y_array = np.array(y, dtype=np.float32)

    # 训练模型
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3, 5],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
    }

    train_data = lgb.Dataset(
        X_matrix,
        label=y_array,
        group=group,
        feature_name=feature_names,
    )

    model = lgb.train(params, train_data, num_boost_round=100)
    return model, feature_names


def predict_with_model(model, feature_names: list,
                       tokens: list, corrected_words: set, error_words: set,
                       char_freq: Counter, bigram_freq: Counter,
                       attrs_used: dict = None) -> list:
    """使用模型预测 token 分数"""
    if not tokens:
        return []

    X = []
    for token in tokens:
        features = compute_features(token, corrected_words, error_words, char_freq, bigram_freq, attrs_used)
        X.append([features.get(f, 0.0) for f in feature_names])

    scores = model.predict(X)
    return list(zip(tokens, scores))


# ========================================
# 主流程
# ========================================
def main():
    import glob

    print("=" * 60)
    print("LambdaMART Token 错误选择器（增强版）")
    print("=" * 60)

    # 配置路径
    output_dir = '/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy'
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    categories = ['Baby_Products', 'Grocery_and_Gourmet_Food', 'Pet_Supplies']

    for category in categories:
        print(f"\n处理类别: {category}")

        output_file = os.path.join(output_dir, category, 'noisy_query_by_token.json')
        user_errors_file = f'/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/writing_error.json'
        query_file_pattern = f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category}/*train10_holdout10*.json'

        # 查找 query 文件
        query_files = glob.glob(query_file_pattern)
        if not query_files:
            query_file_pattern = f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category}/*.json'
            query_files = glob.glob(query_file_pattern)
        if not query_files:
            print(f"  跳过: 未找到 query 文件")
            continue
        query_file = query_files[0]

        # 检查文件是否存在
        if not os.path.exists(output_file):
            print(f"  跳过: 输出文件不存在")
            continue

        print(f"  输出文件: {output_file}")
        print(f"  用户错误文件: {user_errors_file}")
        print(f"  Query 文件: {query_file}")

        try:
            # 提取增强版训练数据
            print("  提取增强版训练数据...")
            X, y, query_ids, token_texts = extract_training_data_enhanced(
                output_file, user_errors_file, query_file, category=category
            )
            print(f"    样本数: {len(X)}, 正样本: {sum(y)}")

            if len(X) < 100:
                print(f"  跳过: 样本数太少")
                continue

            # 训练模型
            print("  训练 LambdaMART...")
            model, feature_names = train_lambdamart(X, y, query_ids)

            # 保存模型
            model_file = os.path.join(model_dir, f'lambdamart_{category}_enhanced.json')
            model.save_model(model_file)
            print(f"    模型保存到: {model_file}")

            # 打印特征重要性
            importance = dict(zip(feature_names, model.feature_importance()))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("    特征重要性 (Top 15):")
            for feat, imp in sorted_imp[:15]:
                print(f"      {feat}: {imp}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
