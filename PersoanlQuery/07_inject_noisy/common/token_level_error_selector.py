"""基于 Token 级别的用户错误注入选择器

根据用户历史错误模式，对新 query 中的 token 进行评分，
选择最合适注入错误的 token，并匹配最相似的历史错误案例。
"""

import re
import json
import os
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

# Stop words 列表
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
    'cause', 'cos', 'so', ' coz ', 'cuz ',
}

# 常见标点符号
PUNCTUATION = set('.,!?;:()[]{}""\'`-—–…·''""@#$%^&*+=<>/\\|~')


def tokenize_query(query: str) -> List[str]:
    """将 query 分词为 token 列表

    保留撇号（'）在单词内，以便正确处理 "I'm" 等缩写
    """
    # 先处理常见的缩写：将 I'm, I've, I'll, I'd, can't, don't 等还原为完整形式
    # 或至少保留撇号在单词内
    query_fixed = query
    # 保留撇号在单词中间的情况（如 I'm -> I'm）
    query_fixed = re.sub(r"(\w)'(\w)", r"\1'\2", query_fixed)
    # 按空格分割，然后提取字母数字混合的 token（保留撇号）
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*", query_fixed)
    return tokens


def remove_stop_words_and_punctuation(tokens: List[str]) -> List[str]:
    """删除 stop words 和标点后的 token 列表

    同时过滤掉：
    - 纯数字 token
    - 太短的 token（长度 < 2）
    """
    result = []
    for token in tokens:
        token_lower = token.lower()
        # 跳过纯数字
        if token.isdigit():
            continue
        # 跳过长度太短的
        if len(token) < 2:
            continue
        # 跳过 stop words
        if token_lower not in STOP_WORDS and not all(c in PUNCTUATION for c in token):
            result.append(token)
    return result


def get_user_error_words(user_errors: List[Dict]) -> Tuple[Set[str], Set[str], Counter, Counter]:
    """从用户错误记录中提取信息

    Returns:
        - corrected_words: 正确词集合（用户可能拼错的词）
        - error_words: 错误词集合（用户实际写错的词）
        - char_freq: 用户常错字符频率
        - bigram_freq: 用户常错 bigram 频率
    """
    corrected_words = set()
    error_words = set()
    char_freq = Counter()
    bigram_freq = Counter()

    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()

        if not original or not corrected:
            continue

        # 单 token 错误
        if ' ' not in original.strip() and ' ' not in corrected.strip():
            corrected_words.add(corrected)
            error_words.add(original)

        # 字符频率统计（从错误词中）
        for c in original:
            if c.isalpha():
                char_freq[c] += 1

        # Bigram 统计
        for i in range(len(original) - 1):
            bigram = original[i:i+2]
            if bigram_freq is not None:
                bigram_freq[bigram] += 1

    return corrected_words, error_words, char_freq, bigram_freq


def char_similarity(s1: str, s2: str) -> float:
    """计算两个字符串的字符集相似度（Jaccard）"""
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


def substring_edit_distance(s1: str, s2: str) -> float:
    """计算子串编辑距离比率（s1 是 query token，s2 是错误词）"""
    # 计算 s1 到 s2 的编辑距离，但允许子串匹配
    if not s1 or not s2:
        return 1.0  # 最大距离

    # 如果 s1 包含 s2 或被包含，计算包含关系
    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # 检查包含关系
    if s2_lower in s1_lower:
        # s2 是 s1 的子串，高度相似
        return 0.3 * (len(s2) / len(s1))
    if s1_lower in s2_lower:
        # s1 是 s2 的子串
        return 0.5 * (len(s1) / len(s2))

    # 计算标准编辑距离
    dist = edit_distance(s1_lower, s2_lower)
    max_len = max(len(s1), len(s2))
    return dist / max_len if max_len > 0 else 1.0


def top_k_similarity(query_token: str, error_words: Set[str], k: int = 5) -> Tuple[float, List[Tuple[str, float]]]:
    """计算 query token 与用户历史错误词的 top-k 相似度

    Returns:
        - max_sim: 最大相似度分数
        - top_k_list: top-k 相似词及其分数列表
    """
    if not error_words:
        return 0.0, []

    similarities = []
    for err_word in error_words:
        # 使用字符集相似度
        sim = char_similarity(query_token, err_word)
        similarities.append((err_word, sim))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    if not similarities:
        return 0.0, []

    return similarities[0][1], similarities[:k]


def contains_error_chars(token: str, char_freq: Counter, threshold: float = 0.3) -> float:
    """计算 token 是否包含用户常错字符

    Returns:
        - 分数：token 中包含的常错字符占比
    """
    if not char_freq or not token:
        return 0.0

    total_chars = sum(char_freq.values())
    if total_chars == 0:
        return 0.0

    # Token 中的字符频率
    token_chars = Counter(token.lower())
    error_char_count = sum(token_chars.get(c, 0) for c in char_freq)
    total_token_chars = len([c for c in token if c.isalpha()])

    if total_token_chars == 0:
        return 0.0

    # 计算错误字符占比
    error_ratio = error_char_count / total_token_chars

    # 如果用户频繁犯某个字符错误，这个字符出现时分数更高
    max_char_freq = max(char_freq.values())
    char_boost = sum(
        (char_freq[c] / max_char_freq) * (token_chars.get(c, 0) / total_token_chars)
        for c in token.lower() if c in char_freq
    )

    return min(error_ratio + char_boost * 0.5, 1.0)


def contains_error_bigrams(token: str, bigram_freq: Counter, threshold: float = 0.3) -> float:
    """计算 token 是否包含用户常错 bigram

    Returns:
        - 分数：token 中包含的常错 bigram 加权占比
    """
    if not bigram_freq or len(token) < 2:
        return 0.0

    total_bigrams = sum(bigram_freq.values())
    if total_bigrams == 0:
        return 0.0

    # Token 中的 bigrams
    token_bigrams = [token[i:i+2].lower() for i in range(len(token) - 1)]
    if not token_bigrams:
        return 0.0

    max_bigram_freq = max(bigram_freq.values())

    # 计算常错 bigram 的加权覆盖
    error_bigram_score = 0.0
    for bg in token_bigrams:
        if bg in bigram_freq:
            freq_ratio = bigram_freq[bg] / max_bigram_freq
            error_bigram_score += freq_ratio

    return min(error_bigram_score / len(token_bigrams), 1.0)


def surface_difficulty(token: str) -> float:
    """计算 token 的表层难度

    考虑因素：
    - 长度越长越容易出错
    - 包含数字更容易出错
    - 包含大写字母可能更容易出错
    - 包含连续相同字母更容易出错
    - 非典型字母组合

    Returns:
        - 难度分数 [0, 1]
    """
    if not token:
        return 0.0

    score = 0.0

    # 长度因子：长度越长，难度略微增加
    length = len(token)
    score += min(length * 0.05, 0.3)

    # 数字因子
    if any(c.isdigit() for c in token):
        score += 0.2

    # 大写字母因子
    if any(c.isupper() for c in token):
        score += 0.1

    # 连续相同字母（如 "ll", "ss"）
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            score += 0.1
            break

    # 非典型字母（包含 w, x, q, z 等较少见的字母）
    rare_chars = set('wxqz')
    if any(c.lower() in rare_chars for c in token):
        score += 0.1

    # 计算典型辅音-元音交替模式
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


def attribute_type_risk(token: str, attrs_used: Dict = None) -> float:
    """计算 token 的属性类型风险

    某些属性类型（如品牌名、数字、颜色词）更容易出现拼写错误

    Returns:
        - 风险分数 [0, 1]
    """
    if not token:
        return 0.0

    score = 0.0

    # 如果 token 是属性值的一部分，风险更高
    if attrs_used:
        token_lower = token.lower()
        for attr_value in attrs_used.values():
            if isinstance(attr_value, str) and token_lower == attr_value.lower():
                score += 0.3
                break
            # 检查是否是属性值的子串
            if isinstance(attr_value, str) and token_lower in attr_value.lower():
                score += 0.2
                break

    # 品牌名模式：大写字母开头，混合字母
    if token[0].isupper() and any(c.islower() for c in token):
        score += 0.15

    # 数字相关 token
    if any(c.isdigit() for c in token):
        score += 0.2

    # 颜色词（常见颜色）
    colors = {'red', 'blue', 'green', 'white', 'black', 'pink', 'purple',
              'orange', 'yellow', 'brown', 'gray', 'grey', 'gold', 'silver'}
    if token.lower() in colors:
        score += 0.15

    # 尺寸词
    sizes = {'small', 'medium', 'large', 'mini', 'miniature', 'compact',
             'tiny', 'big', 'xl', 'xxl', 'xs'}
    if token.lower() in sizes:
        score += 0.1

    return min(score, 1.0)


def compute_token_score(
    token: str,
    corrected_words: Set[str],
    error_words: Set[str],
    char_freq: Counter,
    bigram_freq: Counter,
    attrs_used: Dict = None,
    weights: Dict = None
) -> float:
    """计算单个 token 的综合错误注入分数

    Args:
        token: 待评分 token
        corrected_words: 用户历史正确词集合
        error_words: 用户历史错误词集合
        char_freq: 用户常错字符频率
        bigram_freq: 用户常错 bigram 频率
        attrs_used: 属性字典
        weights: 各因子权重

    Returns:
        - 综合分数 [0, 1]
    """
    if weights is None:
        weights = {
            'topk_similarity': 0.35,      # 与用户错误词的相似度
            'error_chars': 0.20,         # 常错字符包含
            'error_bigrams': 0.15,       # 常错 bigram 包含
            'surface_difficulty': 0.15,  # 表层难度
            'attr_risk': 0.15,          # 属性类型风险
        }

    # Top-k 相似度：与用户错误词（original）的相似度
    # 用户可能把 corrected 拼成 original，所以我们找与 original 相似的 token
    max_sim, _ = top_k_similarity(token, error_words, k=5)
    sim_score = max_sim

    # 常错字符
    char_score = contains_error_chars(token, char_freq)

    # 常错 bigram
    bigram_score = contains_error_bigrams(token, bigram_freq)

    # 表层难度
    diff_score = surface_difficulty(token)

    # 属性风险
    attr_score = attribute_type_risk(token, attrs_used)

    # 加权求和
    total_score = (
        weights['topk_similarity'] * sim_score +
        weights['error_chars'] * char_score +
        weights['error_bigrams'] * bigram_score +
        weights['surface_difficulty'] * diff_score +
        weights['attr_risk'] * attr_score
    )

    return total_score


def score_all_tokens(
    tokens: List[str],
    user_errors: List[Dict],
    attrs_used: Dict = None
) -> List[Tuple[str, float, Dict]]:
    """对所有 token 进行评分

    Returns:
        - List of (token, score, details_dict)
    """
    corrected_words, error_words, char_freq, bigram_freq = get_user_error_words(user_errors)

    scored_tokens = []
    for token in tokens:
        score = compute_token_score(
            token, corrected_words, error_words,
            char_freq, bigram_freq, attrs_used
        )
        details = {
            'topk_sim': top_k_similarity(token, error_words, k=5),
            'char_score': contains_error_chars(token, char_freq),
            'bigram_score': contains_error_bigrams(token, bigram_freq),
            'surface_difficulty': surface_difficulty(token),
            'attr_risk': attribute_type_risk(token, attrs_used),
        }
        scored_tokens.append((token, score, details))

    # 按分数降序排序
    scored_tokens.sort(key=lambda x: x[1], reverse=True)
    return scored_tokens


def select_token_sample(scored_tokens: List[Tuple[str, float, Dict]], top_n: int = 3, strategy: str = 'highest') -> Tuple[str, float, Dict]:
    """选择要注入错误的 token

    Args:
        scored_tokens: 已评分的 token 列表
        top_n: 从 top-n 中选择
        strategy: 'highest' - 选最高分, 'sample' - 按分数采样

    Returns:
        - (selected_token, score, details)
    """
    if not scored_tokens:
        return None, 0.0, {}

    # 只考虑前 top_n 个
    candidates = scored_tokens[:min(top_n, len(scored_tokens))]

    if strategy == 'highest':
        return candidates[0]
    elif strategy == 'sample':
        # 按分数采样：分数高的被选概率更高
        scores = [t[1] for t in candidates]
        total_score = sum(scores)
        if total_score == 0:
            # 均匀采样
            import random
            return random.choice(candidates)

        # 加权采样
        import random
        probs = [s / total_score for s in scores]
        idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[idx]
    else:
        return candidates[0]


def find_similar_error_case(
    query_token: str,
    user_errors: List[Dict],
    top_k: int = 3
) -> List[Tuple[Dict, float]]:
    """找到与 query token 最相似的历史错误案例

    Returns:
        - List of (error_pattern, similarity_score)，按相似度降序
    """
    if not user_errors:
        return []

    scored_cases = []
    for err in user_errors:
        original = err.get('original', '').lower()
        corrected = err.get('corrected', '').lower()

        if not original or not corrected:
            continue

        # 计算与 query token 的相似度
        sim = char_similarity(query_token, original)
        # 也考虑与 corrected 的相似度
        sim_to_corrected = char_similarity(query_token, corrected)

        # 综合相似度：优先与 original（错误词）的相似度
        combined_sim = sim * 0.7 + sim_to_corrected * 0.3

        scored_cases.append((err, combined_sim))

    # 排序并返回 top-k
    scored_cases.sort(key=lambda x: x[1], reverse=True)
    return scored_cases[:top_k]


def select_and_apply_error(
    query: str,
    user_errors: List[Dict],
    attrs_used: Dict = None,
    strategy: str = 'highest',
    top_n: int = 3
) -> Dict:
    """主函数：选择 token 并应用错误

    Args:
        query: 原始查询
        user_errors: 用户错误列表
        attrs_used: 属性字典
        strategy: 'highest' 或 'sample'
        top_n: 从 top-n 中选择

    Returns:
        - 包含选择结果的字典
    """
    # Step 1: Tokenize
    tokens = tokenize_query(query)

    # Step 2: 删除 stop words 和标点
    clean_tokens = remove_stop_words_and_punctuation(tokens)

    if not clean_tokens:
        return {
            'query': query,
            'selected_token': None,
            'score': 0.0,
            'applied_error': None,
            'similar_cases': [],
            'details': {},
            'reason': 'no_valid_tokens',
        }

    # Step 3-4: 对每个 token 评分
    scored_tokens = score_all_tokens(clean_tokens, user_errors, attrs_used)

    # Step 5: 选择 token
    selected_token, score, details = select_token_sample(scored_tokens, top_n=top_n, strategy=strategy)

    if selected_token is None:
        return {
            'query': query,
            'selected_token': None,
            'score': 0.0,
            'applied_error': None,
            'similar_cases': [],
            'details': {},
            'reason': 'no_token_selected',
        }

    # Step 6: 找到最相似的历史错误案例
    similar_cases = find_similar_error_case(selected_token, user_errors, top_k=3)

    # Step 7: 选择要迁移的错误类型
    applied_error = None
    if similar_cases:
        # 选择最相似的案例作为错误注入来源
        best_case, best_sim = similar_cases[0]
        applied_error = {
            'original': best_case.get('original'),      # 错误形式（用户实际写的）
            'corrected': best_case.get('corrected'),   # 正确形式
            'error_type': best_case.get('error_type', 'writing_error'),
            'similarity': best_sim,
        }

    return {
        'query': query,
        'selected_token': selected_token,
        'score': score,
        'applied_error': applied_error,
        'similar_cases': [
            {'original': c[0].get('original'), 'corrected': c[0].get('corrected'),
             'similarity': c[1]}
            for c in similar_cases
        ],
        'details': details,
        'reason': 'success',
    }


def apply_error_to_query(
    query: str,
    error_case: Dict,
    token: str
) -> str:
    """将错误类型迁移到 query 中的 token

    Args:
        query: 原始查询
        error_case: 错误案例（包含 original 和 corrected）
        token: 要被替换的 token

    Returns:
        - 注入错误后的查询
    """
    if not error_case or not token:
        return query

    corrected = error_case.get('corrected', '')
    original = error_case.get('original', '')

    if not corrected or not original:
        return query

    # 将 query 中的 token（正确形式）替换为错误形式
    # 使用单词边界匹配
    pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(token) + r'(?![a-zA-Z])', re.IGNORECASE)

    # 检查 token 是否与 corrected 匹配
    if token.lower() == corrected.lower():
        # 直接替换
        noisy_query = pattern.sub(original, query, count=1)
    else:
        # 可能需要更复杂的匹配逻辑
        # 先尝试直接替换
        noisy_query = pattern.sub(original, query, count=1)

    return noisy_query


# ============================================================================
# 批量处理接口
# ============================================================================

def process_user_query_task(
    task: Dict,
    strategy: str = 'highest',
    top_n: int = 3
) -> Dict:
    """处理单个用户查询任务

    Args:
        task: 任务字典，包含 uid, asin, clean_query, errors, attrs_used
        strategy: 'highest' 或 'sample'
        top_n: 从 top-n 中选择

    Returns:
        - 处理结果
    """
    uid = task.get('uid')
    asin = task.get('asin')
    clean_query = task.get('clean_query', '')
    user_errors = task.get('errors', [])
    attrs_used = task.get('attrs_used')

    if not clean_query or not user_errors:
        return {
            'uid': uid,
            'asin': asin,
            'clean_query': clean_query,
            'noisy_query': clean_query,
            'selected_token': None,
            'score': 0.0,
            'applied_error': None,
            'status': 'skipped',
            'reason': 'empty_query_or_errors',
        }

    # 执行选择和错误应用
    result = select_and_apply_error(
        query=clean_query,
        user_errors=user_errors,
        attrs_used=attrs_used,
        strategy=strategy,
        top_n=top_n
    )

    # 应用错误到查询
    noisy_query = clean_query
    if result.get('applied_error') and result.get('selected_token'):
        noisy_query = apply_error_to_query(
            clean_query,
            result['applied_error'],
            result['selected_token']
        )

    return {
        'uid': uid,
        'asin': asin,
        'clean_query': clean_query,
        'noisy_query': noisy_query,
        'selected_token': result.get('selected_token'),
        'score': result.get('score', 0.0),
        'applied_error': result.get('applied_error'),
        'similar_cases': result.get('similar_cases', []),
        'details': result.get('details', {}),
        'status': 'success' if result.get('applied_error') else 'no_error_applied',
        'reason': result.get('reason'),
    }


if __name__ == '__main__':
    # 简单测试
    test_query = "I need a Small Baby Stroller for my newborn"
    test_errors = [
        {'original': 'stroller', 'corrected': 'stroller', 'error_type': 'writing_error'},
        {'original': 'recieved', 'corrected': 'received', 'error_type': 'writing_error'},
        {'original': 'newborn', 'corrected': 'newborn', 'error_type': 'writing_error'},
    ]
    test_attrs = {
        'A1': 'Stroller',
        'A2': 'Baby',
        'A3': 'Small',
        'A4': 'newborn',
    }

    result = select_and_apply_error(test_query, test_errors, test_attrs)
    print(json.dumps(result, indent=2, ensure_ascii=False))