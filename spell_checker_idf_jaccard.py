#!/usr/bin/env python3
"""
基于 IDF + Jaccard 相似度的拼写错误检测脚本

核心逻辑：
1. 使用 IDF 识别罕见词（可能是拼写错误）
2. 使用 Jaccard 相似度（基于 bigram）找到相似的高频词
3. 如果找到相似的高频词，判定为拼写错误

不需要模型训练，纯数学方法。
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import math
import os


def extract_words(text: str) -> List[str]:
    """从文本中提取单词（小写，去除标点）"""
    # 使用正则表达式提取单词（包括连字符和撇号）
    words = re.findall(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)?\b", text.lower())
    return words


def get_bigrams(word: str) -> Set[str]:
    """获取单词的 bigram 集合"""
    if len(word) < 2:
        return set()
    return set(word[i:i+2] for i in range(len(word) - 1))


def jaccard_similarity(word1: str, word2: str) -> float:
    """计算两个单词的 Jaccard 相似度（基于 bigram）"""
    bigrams1 = get_bigrams(word1)
    bigrams2 = get_bigrams(word2)
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def edit_distance(word1: str, word2: str) -> int:
    """计算两个单词的编辑距离（Levenshtein distance）"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def calculate_idf(word: str, word_doc_freq: Dict[str, int], total_docs: int) -> float:
    """计算单词的 IDF 值"""
    doc_freq = word_doc_freq.get(word, 0)
    if doc_freq == 0:
        return float('inf')  # 从未出现的词，IDF 为无穷大
    return math.log(total_docs / doc_freq)


def load_external_dictionary() -> Optional[List[str]]:
    """
    加载外部英语词典作为高频词表
    
    支持多种方法（按优先级）：
    1. pyspellchecker 库
    2. 系统词典文件 (/usr/share/dict/words)
    3. nltk 的英语词典
    4. 下载的常见英语词典文件
    
    Returns:
        英语词典单词列表（小写），如果无法加载则返回 None
    """
    dictionary_words = set()
    
    # 方法 1: 尝试使用 pyspellchecker
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language='en')
        # 获取所有已知单词
        dictionary_words.update(spell.word_frequency.words())
        print(f"  ✓ 使用 pyspellchecker 加载了 {len(dictionary_words)} 个单词")
        return sorted(list(dictionary_words))
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠ pyspellchecker 加载失败: {e}")
    
    # 方法 2: 尝试使用系统词典文件
    system_dict_paths = [
        '/usr/share/dict/words',
        '/usr/share/dict/american-english',
        '/usr/share/dict/british-english',
        '/usr/dict/words'
    ]
    
    for dict_path in system_dict_paths:
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        # 只保留字母单词，过滤专有名词和复合词
                        if word and word.isalpha() and len(word) >= 2:
                            dictionary_words.add(word)
                if dictionary_words:
                    print(f"  ✓ 从 {dict_path} 加载了 {len(dictionary_words)} 个单词")
                    return sorted(list(dictionary_words))
            except Exception as e:
                print(f"  ⚠ 读取 {dict_path} 失败: {e}")
                continue
    
    # 方法 3: 尝试使用 nltk
    try:
        import nltk
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            print("  ℹ 正在下载 nltk words 数据...")
            nltk.download('words', quiet=True)
        
        from nltk.corpus import words
        dictionary_words.update([w.lower() for w in words.words() if w.isalpha() and len(w) >= 2])
        if dictionary_words:
            print(f"  ✓ 使用 nltk 加载了 {len(dictionary_words)} 个单词")
            return sorted(list(dictionary_words))
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠ nltk 加载失败: {e}")
    
    # 方法 4: 尝试下载并使用常见英语词典
    # 这里可以添加下载逻辑，但为了简单起见，我们返回 None
    print("  ⚠ 无法加载外部词典，将使用数据集中的高频词")
    return None


def find_similar_words(
    target_word: str,
    candidate_words: List[str],
    min_similarity: float = 0.5,
    max_edit_distance_ratio: float = 0.3,  # 编辑距离不能超过单词长度的30%
    top_k: int = 5
) -> List[Tuple[str, float, int]]:
    """找到与目标词最相似的高频词"""
    similarities = []
    max_edit_dist = int(len(target_word) * max_edit_distance_ratio)
    
    for candidate in candidate_words:
        if candidate == target_word:
            continue
        
        # 长度差异不能太大
        if abs(len(candidate) - len(target_word)) > max_edit_dist:
            continue
        
        sim = jaccard_similarity(target_word, candidate)
        if sim >= min_similarity:
            edit_dist = edit_distance(target_word, candidate)
            # 编辑距离不能太大
            if edit_dist <= max_edit_dist:
                similarities.append((candidate, sim, edit_dist))
    
    # 按相似度降序排序，相似度相同时按编辑距离升序排序
    similarities.sort(key=lambda x: (-x[1], x[2]))
    return similarities[:top_k]


def detect_spelling_errors(
    reviews: List[str],
    min_word_freq: int = 3,  # 低于此频率的词被认为是罕见词
    min_idf_threshold: float = 4.0,  # IDF 阈值，高于此值认为是罕见词（对于小数据集，降低阈值）
    min_similarity: float = 0.4,  # Jaccard 相似度阈值（稍微降低以捕获更多错误）
    top_frequent_words: int = 10000,  # 用于比较的高频词数量（仅当不使用外部词典时）
    use_external_dictionary: bool = True,  # 是否使用外部词典
    external_dictionary: Optional[List[str]] = None  # 外部词典（如果为 None 则自动加载）
) -> List[Dict]:
    """
    检测拼写错误
    
    Args:
        reviews: 评论文本列表
        min_word_freq: 最小词频，低于此频率的词被认为是罕见词
        min_idf_threshold: IDF 阈值
        min_similarity: Jaccard 相似度阈值
        top_frequent_words: 用于比较的高频词数量
    
    Returns:
        检测到的拼写错误列表
    """
    print("步骤 1: 提取所有单词并统计词频...")
    # 统计词频和文档频率
    word_freq = Counter()  # 词频（在所有文档中的总出现次数）
    word_doc_freq = defaultdict(int)  # 文档频率（包含该词的文档数量）
    all_words_in_docs = []  # 每个文档的单词列表
    
    for review in reviews:
        words = extract_words(review)
        unique_words_in_doc = set(words)
        all_words_in_docs.append(words)
        
        for word in words:
            word_freq[word] += 1
        
        for word in unique_words_in_doc:
            word_doc_freq[word] += 1
    
    total_docs = len(reviews)
    print(f"  处理了 {total_docs} 个文档")
    print(f"  发现 {len(word_freq)} 个不同的单词")
    
    print("\n步骤 2: 计算 IDF 值并识别罕见词...")
    # 计算每个词的 IDF
    word_idf = {}
    rare_words = []
    
    # 调试：显示一些统计信息
    idf_values = []
    for word, freq in word_freq.items():
        idf = calculate_idf(word, word_doc_freq, total_docs)
        word_idf[word] = idf
        if idf != float('inf'):
            idf_values.append(idf)
        
        # 识别罕见词：IDF 高 且 词频低
        if idf >= min_idf_threshold and freq < min_word_freq:
            rare_words.append((word, idf, freq))
    
    if idf_values:
        print(f"  IDF 统计: 最小={min(idf_values):.2f}, 最大={max(idf_values):.2f}, 平均={sum(idf_values)/len(idf_values):.2f}")
    
    rare_words.sort(key=lambda x: x[1], reverse=True)  # 按 IDF 降序排序
    print(f"  识别出 {len(rare_words)} 个罕见词（可能是拼写错误）")
    
    # 显示前 10 个罕见词作为示例
    if rare_words:
        print("  前 10 个罕见词示例:")
        for word, idf, freq in rare_words[:10]:
            print(f"    {word}: IDF={idf:.2f}, 频率={freq}")
    
    print("\n步骤 3: 构建高频词表...")
    # 优先使用外部词典，如果没有则使用数据集中的高频词
    if use_external_dictionary:
        if external_dictionary is None:
            external_dictionary = load_external_dictionary()
        
        if external_dictionary:
            # 使用外部词典作为高频词表
            # 过滤：只保留长度在合理范围内的词（2-20个字符）
            frequent_words = [
                word for word in external_dictionary
                if 2 <= len(word) <= 20 and word.isalpha()
            ]
            print(f"  使用外部词典，包含 {len(frequent_words)} 个词")
        else:
            # 外部词典加载失败，回退到数据集高频词
            print("  ⚠ 外部词典加载失败，使用数据集中的高频词")
            frequent_words = [
                word for word, freq in word_freq.most_common(top_frequent_words)
                if freq >= min_word_freq
            ]
            print(f"  高频词表包含 {len(frequent_words)} 个词")
    else:
        # 使用数据集中的高频词
        frequent_words = [
            word for word, freq in word_freq.most_common(top_frequent_words)
            if freq >= min_word_freq  # 确保是真正的高频词
        ]
        print(f"  高频词表包含 {len(frequent_words)} 个词")
    
    print("\n步骤 4: 使用 Jaccard 相似度寻找候选纠正词...")
    detected_errors = []
    
    for rare_word, idf, freq in rare_words:
        # 跳过太短的词（可能是缩写或专有名词的一部分）
        if len(rare_word) < 4:  # 至少4个字符
            continue
        
        # 跳过太长的词（可能是复合词或专有名词）
        if len(rare_word) > 20:
            continue
        
        # 寻找相似的高频词
        similar_words = find_similar_words(
            rare_word,
            frequent_words,
            min_similarity=min_similarity,
            max_edit_distance_ratio=0.3,  # 编辑距离不超过30%
            top_k=3
        )
        
        if similar_words:
            # 找到相似的高频词，判定为拼写错误
            best_match = similar_words[0]
            # 额外检查：如果最佳匹配的相似度不够高，或者编辑距离太大，跳过
            if best_match[1] < 0.5 or best_match[2] > len(rare_word) * 0.3:
                continue
            
            detected_errors.append({
                'original_word': rare_word,
                'corrected_word': best_match[0],
                'similarity': best_match[1],
                'edit_distance': best_match[2],
                'idf': idf,
                'frequency': freq,
                'all_candidates': [
                    {'word': word, 'similarity': sim, 'edit_distance': ed}
                    for word, sim, ed in similar_words
                ]
            })
    
    print(f"  检测到 {len(detected_errors)} 个拼写错误")
    
    return detected_errors


def main():
    """主函数"""
    input_file = 'spelling_analysis_combined_batch_ulhdemmplr.json'
    output_file = 'spelling_errors_detected_idf_jaccard.json'
    
    print("=" * 60)
    print("基于 IDF + Jaccard 相似度的拼写错误检测")
    print("=" * 60)
    
    # 读取 JSON 文件
    print(f"\n读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有 review 文本
    reviews = []
    if isinstance(data, list):
        # 如果是数组格式
        for item in data:
            if 'review' in item:
                reviews.append(item['review'])
    elif isinstance(data, dict):
        # 如果是字典格式
        if 'errors' in data:
            for error in data['errors']:
                if 'review' in error:
                    reviews.append(error['review'])
        elif 'users' in data:
            for user in data['users']:
                if 'error_types' in user:
                    for error_type, errors in user['error_types'].items():
                        for error in errors:
                            if 'review' in error:
                                reviews.append(error['review'])
    
    print(f"提取了 {len(reviews)} 条评论")
    
    if not reviews:
        print("错误：没有找到任何评论文本！")
        return
    
    # 检测拼写错误
    print("\n" + "=" * 60)
    print("加载外部词典...")
    print("=" * 60)
    external_dict = load_external_dictionary()
    
    print("\n" + "=" * 60)
    print("开始检测拼写错误...")
    print("=" * 60)
    
    detected_errors = detect_spelling_errors(
        reviews,
        min_word_freq=2,  # 词频低于 2 的认为是罕见词
        min_idf_threshold=5.0,  # IDF 阈值（提高阈值以减少误报）
        min_similarity=0.5,  # Jaccard 相似度阈值（提高阈值以减少误报）
        top_frequent_words=10000,  # 使用前 10000 个高频词（仅当不使用外部词典时）
        use_external_dictionary=True,  # 使用外部词典
        external_dictionary=external_dict  # 传入已加载的词典
    )
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detected_errors, f, indent=2, ensure_ascii=False)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("检测结果统计")
    print("=" * 60)
    print(f"总共检测到 {len(detected_errors)} 个拼写错误")
    
    if detected_errors:
        print("\n前 10 个检测结果示例：")
        for i, error in enumerate(detected_errors[:10], 1):
            print(f"\n{i}. {error['original_word']} -> {error['corrected_word']}")
            print(f"   相似度: {error['similarity']:.3f}, 编辑距离: {error['edit_distance']}, IDF: {error['idf']:.2f}, 频率: {error['frequency']}")
            if len(error['all_candidates']) > 1:
                candidates_str = ', '.join([f"{c['word']}(相似度:{c['similarity']:.2f}, 编辑距离:{c['edit_distance']})" for c in error['all_candidates'][1:]])
                print(f"   其他候选: {candidates_str}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()
