#!/usr/bin/env python3
"""基于 LambdaMART 的 Token 级别错误选择器

使用 LightGBM LambdaMART 学习 token 评分权重，对每个 query 选择最合适的 token 进行错误注入。
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

PUNCTUATION = set('.,!?;:()[]{}""\'`-—–…·''""@#$%^&*+=<>/\\|~')


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


def get_user_error_words(user_errors: list) -> tuple:
    """从用户错误中提取信息"""
    corrected_words = set()
    error_words = set()
    char_freq = Counter()
    bigram_freq = Counter()

    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()
        if not original or not corrected:
            continue
        if ' ' not in original.strip() and ' ' not in corrected.strip():
            corrected_words.add(corrected)
            error_words.add(original)
        for c in original:
            if c.isalpha():
                char_freq[c] += 1
        for i in range(len(original) - 1):
            bigram = original[i:i+2]
            bigram_freq[bigram] += 1

    return corrected_words, error_words, char_freq, bigram_freq


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


def compute_features(token: str, corrected_words: set, error_words: set,
                     char_freq: Counter, bigram_freq: Counter,
                     attrs_used: dict = None) -> dict:
    """计算 token 的特征"""
    max_sim, _ = top_k_similarity(token, error_words, k=5)
    char_score = contains_error_chars(token, char_freq)
    bigram_score = contains_error_bigrams(token, bigram_freq)
    diff_score = surface_difficulty(token)
    attr_score = attribute_type_risk(token, attrs_used)

    # 额外特征
    token_len = len(token)
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    has_rare_char = any(c.lower() in 'wxqz' for c in token)

    return {
        'topk_sim': max_sim,
        'char_score': char_score,
        'bigram_score': bigram_score,
        'surface_difficulty': diff_score,
        'attr_risk': attr_score,
        'token_len': token_len,
        'has_upper': float(has_upper),
        'has_digit': float(has_digit),
        'has_rare_char': float(has_rare_char),
    }


def extract_training_data(output_file: str, user_errors_file: str, query_file: str) -> tuple:
    """从现有输出中提取训练数据

    Returns:
        (X, y, query_ids, token_texts)
    """
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

    # 处理不同的 query 文件格式
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
        # 尝试多种可能的 query 字段
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


def train_lambdamart(X: list, y: list, query_ids: list) -> lgb.LGBMRanker:
    """训练 LambdaMART 模型"""
    if not X:
        raise ValueError("X is empty")

    feature_names = list(X[0].keys()) if isinstance(X[0], dict) else [f"f{i}" for i in range(len(X[0]))]
    if isinstance(X[0], dict):
        X_matrix = np.array([[x.get(f, 0.0) for f in feature_names] for x in X])
    else:
        X_matrix = np.array(X)

    # 按 query_id 分组
    query_groups = {}
    for qid in query_ids:
        query_groups[qid] = query_groups.get(qid, 0) + 1

    # 按 query 分组计算 label
    query_start = {}
    query_len = {}
    current_idx = 0
    for qid in query_ids:
        if qid not in query_start:
            query_start[qid] = current_idx
            query_len[qid] = 0
        query_len[qid] += 1
        current_idx += 1

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


def predict_with_model(model: lgb.LGBMRanker, feature_names: list,
                       tokens: list, corrected_words: set, error_words: set,
                       char_freq: Counter, bigram_freq: Counter,
                       attrs_used: dict = None) -> list:
    """使用模型预测 token 分数

    Returns:
        list of (token, score)
    """
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
    import re
    import glob

    print("=" * 60)
    print("LambdaMART Token 错误选择器")
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
            # 尝试备选模式
            query_file_pattern = f'/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category}/*.json'
            query_files = glob.glob(query_file_pattern)
        if not query_files:
            print(f"  跳过: 未找到 query 文件")
            continue
        # 选择第一个文件
        query_file = query_files[0]

        # 检查文件是否存在
        if not os.path.exists(output_file):
            print(f"  跳过: 输出文件不存在")
            continue

        print(f"  输出文件: {output_file}")
        print(f"  用户错误文件: {user_errors_file}")
        print(f"  Query 文件: {query_file}")

        try:
            # 提取训练数据
            print("  提取训练数据...")
            X, y, query_ids, token_texts = extract_training_data(
                output_file, user_errors_file, query_file
            )
            print(f"    样本数: {len(X)}, 正样本: {sum(y)}")

            if len(X) < 100:
                print(f"  跳过: 样本数太少")
                continue

            # 训练模型
            print("  训练 LambdaMART...")
            model, feature_names = train_lambdamart(X, y, query_ids)

            # 保存模型
            model_file = os.path.join(model_dir, f'lambdamart_{category}.json')
            model.save_model(model_file)
            print(f"    模型保存到: {model_file}")

            # 打印特征重要性
            importance = dict(zip(feature_names, model.feature_importance()))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("    特征重要性:")
            for feat, imp in sorted_imp[:10]:
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
