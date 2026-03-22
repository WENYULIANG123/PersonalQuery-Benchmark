#!/usr/bin/env python3
"""
Stage 10: Unified Noisy Query Generator

统一的噪声查询生成脚本，支持：
1. 个性化错误注入（基于用户真实错误历史）
2. 统计错误注入（基于真实错误分布）
3. 拼写错误注入
4. 语法错误注入
5. 混合模式（个性化+统计降级）

Input:
  - Stage 7 dual_queries or iterative_refinement_results.json
  - Stage 5 writing_analysis (可选，用于个性化)

Output:
  - Queries with ONE injected error per query
"""

import json
import os
import sys
import re
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

# ============================================================================
# Hardcoded Configuration
# ============================================================================

BASE_DIR = "/fs04/ar57/wenyu"

# User ID will be determined from sys.argv or environment
USER_ID = None

# Input files - will be set later once USER_ID is determined
STAGE7_RESULTS_FILE = None
WRITING_ANALYSIS_FILE = None
QUERY_SOURCE = "stage6"

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/09_targeted_noisy_query")

RANDOM_SEED = 42

# ============================================================================
# Utility Functions
# ============================================================================

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio()


FUZZY_MATCH_THRESHOLD = 0.60


# ============================================================================
# Part 1: Statistical Error Injection (基于真实错误分布)
# ============================================================================

# 高频常见拼写错误 (来源: Wikipedia, Birkbeck Corpus)
COMMON_MISSPELLINGS = {
    # 极高频词
    "the": ["teh", "hte", "tje", "tge"],
    "and": ["abd", "snd", "amd", "anbd"],
    "that": ["taht", "thta", "tjat", "thet"],
    "with": ["witth", "wiht", "witrh", "witjh"],
    "for": ["fos", "fir", "fot", "fro", "fpr"],
    "are": ["aer", "rea", "are", "ar"],
    "have": ["haev", "hvae", "hav", "hvae"],
    "from": ["fosrm", "form", "frmo", "frpm"],
    "this": ["thsi", "tihs", "ths", "thsi"],
    "they": ["tehy", "thye", "they", "teyh"],
    "was": ["saw", "wsa", "was"],
    "were": ["weer", "wer", "were"],
    "been": ["ben", "been", "beem"],

    # 产品/购物相关
    "product": ["prodcut", "protuct", "poduct", "prodict"],
    "products": ["prodcuts", "protucts", "poducts"],
    "review": ["revie", "reveiw", "reivew", "reviwe"],
    "reviews": ["revies", "reveiws", "reivews"],
    "quality": ["qualtiy", "qaulity", "qualitty", "qualty"],
    "shipping": ["shiping", "shpping", "shippng", "shiping"],
    "deliver": ["deliver", "delivry", "dlivery", "delever"],
    "delivery": ["delivry", "delever", "dlivery", "delivry"],
    "price": ["prcie", "pirce", "prcie", "priec"],
    "prices": ["prcies", "pirces", "prcies"],
    "order": ["oder", "roder", "ordr"],
    "orders": ["oders", "roders", "ordrs"],
    "available": ["availabe", "avaialble", "availble", "availabe"],
    "excellent": ["excelent", "excellant", "excelent", "excellet"],
    "satisfied": ["satisifed", "sattisfied", "satified", "satisifed"],
    "return": ["retun", "retrun", "rturn"],
    "returns": ["retuns", "retruns", "rturns"],
    "money": ["mony", "moeny", "moeny"],
    "refund": ["refund", "re fund", "refund"],

    # 复杂词
    "definitely": ["definitly", "definately", "defintely", "definatly"],
    "separate": ["seperate", "seperated", "seperately", "seperat"],
    "necessary": ["neccessary", "necesary", "nessasary", "necesery"],
    "accommodate": ["accomodate", "acommodate", "acomadate", "acomodate"],
    "occurrence": ["occurence", "occurrance", "occurence", "occurrance"],
    "embarrass": ["embarass", "embarrass", "embarass", "embaras"],
    "conscientious": ["concientious", "conscientous", "consciencious"],
    "recommend": ["recomend", "reccommend", "recommed", "reccomed"],
    "immediately": ["immediatly", "immediatley", "immediatly"],

    # Amazon 搜索相关
    "amazon": ["amazn", "amzon", "amazonn", "amazom"],
    "prime": ["prim", "prmie", "prine"],
    "rating": ["ratng", "ratign", "raing"],
    "seller": ["sller", "sseler", "seler"],
    "purchase": ["purcahse", "purhcase", "purchace"],
    "bought": ["bouth", "bouhgt", "bot"],
    "cheap": ["chep", "cheep", "cheaap"],
    "expensive": ["expesnive", "exspensive", "expensve"],

    # Crafting 品类相关
    "die": ["di", "die", "dile"],
    "cut": ["cutt", "cuz", "cot"],
    "cuts": ["cutts", "cuz", "cot"],
    "cutting": ["cuting", "cuttng", "cuttin"],
    "embossing": ["embosing", "embosing", "embossing"],
    "folder": ["fodler", "foldr", "foler"],
    "paper": ["papaer", "papre", "ppaer"],
    "card": ["cad", "crd", "card"],
    "cards": ["cads", "crds", "card"],
    "craft": ["crfat", "cract", "crft"],
    "crafting": ["crfatting", "craftin", "crfatting"],
    "scissors": ["scisor", "scisors", "scissor"],
    "glue": ["glue", "gule", "glwe"],
    "marker": ["makrer", "markr", "makre"],
    "paint": ["panit", "paitn", "painte"],
    "brush": ["brus", "brush", "burhs"],
}

# QWERTY 键盘相邻键映射
KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qeas', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'ygji', 'i': 'uhko', 'o': 'iklp', 'p': 'ol',
    'a': 'sqz', 's': 'awedxz', 'd': 'erfcx', 'f': 'rtgvc', 'g': 'tyhbv',
    'h': 'yujnb', 'j': 'uiknm', 'k': 'iojm', 'l': 'op',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk'
}

# 错误类型分布 (基于真实用户错误统计)
ERROR_TYPE_DISTRIBUTION = {
    "keyboard_typo": 0.35,       # 键盘打字错误
    "common_misspelling": 0.25,   # 常见拼写错误
    "double_letter": 0.15,       # 双字母错误
    "missing_letter": 0.10,      # 漏字母
    "vowel_error": 0.10,        # 元音错误
    "word_split": 0.05,         # 单词拆分
}


class StatisticalErrorInjector:
    """基于统计分布的错误注入器"""

    def __init__(self, user_error_profile: Optional[Dict] = None):
        """
        Args:
            user_error_profile: 可选的用户错误画像 (从 Stage 5 获取)
        """
        self.user_profile = user_error_profile or {}

    def select_error_type(self) -> str:
        """根据概率分布选择错误类型"""
        r = random.random()
        cumulative = 0.0

        for error_type, prob in ERROR_TYPE_DISTRIBUTION.items():
            cumulative += prob
            if r < cumulative:
                return error_type

        return "common_misspelling"

    def inject_single_error(self, query: str) -> Dict:
        """
        向 query 注入一个拼写错误

        Returns:
            {
                "original": str,
                "noisy": str,
                "modified": bool,
                "error_type": str,
                "original_word": str,
                "noisy_word": str,
                "reason": str
            }
        """

        # 1. 分词（保留位置信息）
        tokens = query.split()
        if len(tokens) < 2:
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": "Query too short"
            }

        # 2. 选择目标词（排除太短的词和纯数字）
        candidates = []
        for i, token in enumerate(tokens):
            clean = re.sub(r'[^a-zA-Z]', '', token)
            # 至少4个字母，且不是纯数字
            if len(clean) >= 4 and clean.isalpha():
                candidates.append((i, token, clean))

        if not candidates:
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": "No suitable target word found"
            }

        # 3. 根据用户画像加权选择目标词 (如果有)
        if self.user_profile and 'spelling' in self.user_profile:
            # 给用户常犯错的词更高权重
            weighted_candidates = []
            spelling_stats = self.user_profile.get('spelling', {})
            for idx, token, clean in candidates:
                # 简单策略：如果词在用户错误类型中，增加选中概率
                weight = 1.0
                for error_type, count in spelling_stats.items():
                    if count > 0:
                        weight += count * 0.1
                weighted_candidates.extend([(idx, token, clean)] * int(weight))

            if weighted_candidates:
                idx, word, clean_word = random.choice(weighted_candidates)
            else:
                idx, word, clean_word = random.choice(candidates)
        else:
            idx, word, clean_word = random.choice(candidates)

        # 4. 选择错误类型
        error_type = self.select_error_type()

        # 5. 应用错误
        noisy_word = self._apply_error(clean_word.lower(), error_type)

        if noisy_word == clean_word.lower():
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": f"Error type '{error_type}' produced no change"
            }

        # 6. 还原原始标点
        noisy_word = self._restore_punctuation(word, noisy_word)

        # 7. 替换目标词
        tokens[idx] = noisy_word
        noisy_query = " ".join(tokens)

        # 8. 验证修改
        if noisy_query.lower() == query.lower():
            return {
                "original": query,
                "noisy": query,
                "modified": False,
                "reason": "Verification failed: no actual change"
            }

        return {
            "original": query,
            "noisy": noisy_query,
            "modified": True,
            "error_type": error_type,
            "original_word": word,
            "noisy_word": noisy_word,
            "reason": f"Applied {error_type} to '{clean_word}'"
        }

    def _apply_error(self, word: str, error_type: str) -> str:
        """应用特定类型的错误"""

        if error_type == "common_misspelling":
            return self._common_misspelling_error(word)

        elif error_type == "keyboard_typo":
            return self._keyboard_error(word)

        elif error_type == "double_letter":
            return self._double_letter_error(word)

        elif error_type == "missing_letter":
            return self._missing_letter_error(word)

        elif error_type == "vowel_error":
            return self._vowel_error(word)

        elif error_type == "word_split":
            return self._word_split_error(word)

        return word

    def _common_misspelling_error(self, word: str) -> str:
        """从常见错误词典中选择"""
        if word in COMMON_MISSPELLINGS:
            return random.choice(COMMON_MISSPELLINGS[word])

        # 对于未收录的词，使用其他错误类型
        return self._random_known_error(word)

    def _keyboard_error(self, word: str) -> str:
        """模拟键盘相邻键错误"""
        result = list(word)
        changed = False

        for i in range(len(result)):
            char = result[i]
            if char in KEYBOARD_NEIGHBORS:
                # 30% 概率出错
                if random.random() < 0.3:
                    result[i] = random.choice(KEYBOARD_NEIGHBORS[char])
                    changed = True

        if changed:
            return "".join(result)
        return word

    def _double_letter_error(self, word: str) -> str:
        """双字母错误：多写一个或漏写一个"""

        # 找双字母
        double_pattern = re.compile(r'(.)\1')
        doubles = list(double_pattern.finditer(word))

        if doubles and random.random() < 0.5:
            # 删除一个双字母 (accommodate -> acomodate)
            match = random.choice(doubles)
            char = match.group(1)
            return word[:match.start()] + char + word[match.end():]
        else:
            # 添加一个双字母 (till -> till)
            for i in range(len(word) - 1):
                if word[i] == word[i+1]:
                    # 在两个字母之间插入一个
                    return word[:i+1] + word[i] + word[i+1:]

            # 随机位置添加双字母
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                return word[:i] + word[i] + word[i:]

        return word

    def _missing_letter_error(self, word: str) -> str:
        """漏写字母 (常见于辅音簇)"""
        if len(word) <= 4:
            return word

        # 辅音簇位置
        consonants = "bcdfghjklmnpqrstvwxyz"
        candidates = [i for i, c in enumerate(word) if c in consonants]

        if candidates:
            idx = random.choice(candidates)
            return word[:idx] + word[idx+1:]

        return word

    def _vowel_error(self, word: str) -> str:
        """元音替换错误 (最常见的拼写错误类型)"""
        vowels = 'aeiou'
        result = list(word)
        changed = False

        for i in range(len(result)):
            if result[i] in vowels:
                # 20% 概率出错
                if random.random() < 0.2:
                    result[i] = random.choice(vowels.replace(result[i], ''))
                    changed = True

        if changed:
            return "".join(result)
        return word

    def _word_split_error(self, word: str) -> str:
        """单词拆分错误 (常见于复合词)"""
        # 常见拆分模式
        split_pairs = [
            ("card", "card"), ("stock", "stock"),
            ("making", "making"), ("paper", "paper"),
            ("cutting", "cutting"), ("craft", "craft")
        ]

        for short, full in split_pairs:
            if word.endswith(full) and len(word) > len(full) + 2:
                prefix = word[:-len(full)]
                # 在中间插入空格
                if len(prefix) > 2:
                    split_point = len(prefix) - 1
                    return prefix[:split_point] + " " + prefix[split_point:] + full

        # 随机拆分
        if len(word) > 5:
            i = random.randint(2, len(word) - 3)
            return word[:i] + " " + word[i:]

        return word

    def _random_known_error(self, word: str) -> str:
        """对未知词应用随机已知错误类型"""
        error_types = ["double_letter", "missing_letter", "vowel_error"]
        chosen = random.choice(error_types)

        # 调用对应的错误方法
        if chosen == "double_letter":
            result = self._double_letter_error(word)
        elif chosen == "missing_letter":
            result = self._missing_letter_error(word)
        elif chosen == "vowel_error":
            result = self._vowel_error(word)
        else:
            result = word

        if result != word:
            return result
        return word

    def _restore_punctuation(self, original: str, clean: str) -> str:
        """还原原始标点"""
        # 提取首尾标点
        leading_match = re.match(r'^([^\w]*)', original)
        trailing_match = re.search(r'([^\w]*)$', original)

        leading = leading_match.group(1) if leading_match else ""
        trailing = trailing_match.group(1) if trailing_match else ""

        # 保持原始大小写风格
        if original and original[0].isupper():
            clean = clean.capitalize()

        return leading + clean + trailing


# ============================================================================
# Part 2: Personalized Spelling Error Injection (基于用户历史)
# ============================================================================

class PersonalizedSpellingErrorExtractor:
    """从Stage 5的writing analysis提取用户拼写错误模式"""

    def __init__(self, writing_analysis_file: str):
        """
        Args:
            writing_analysis_file: Stage 5输出的writing_analysis_*.json
        """
        with open(writing_analysis_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # corrected → [original variants] 映射
        self.user_error_dict: Dict[str, Set[str]] = defaultdict(set)

        # 错误类型统计
        self.error_type_stats: Dict[str, int] = defaultdict(int)

        # 用户真实错误示例
        self.error_examples: List[Dict] = []

        self._extract_user_errors()

    def _extract_user_errors(self):
        log_with_timestamp("Extracting user spelling errors...")

        if 'detailed_errors' in self.data:
            self._extract_from_p3_format()
        elif 'detailed_character_errors' in self.data:
            self._extract_from_new_format()
        elif 'results' in self.data:
            self._extract_from_old_format()
        else:
            log_with_timestamp("  Warning: No recognized error format found")

        log_with_timestamp(f"  Extracted {len(self.user_error_dict)} unique error patterns")
        log_with_timestamp(f"  Total error instances: {sum(len(v) for v in self.user_error_dict.values())}")

    def _extract_from_p3_format(self):
        """Extract errors from P3 optimal template format (Stage 04 output)"""
        detailed_errors = self.data.get('detailed_errors', [])
        log_with_timestamp(f"  Reading P3 optimal template format: {len(detailed_errors)} detailed error reviews")
        
        total_errors_processed = 0
        anomaly_filtered = 0
        
        for review_data in detailed_errors:
            errors_list = review_data.get('errors', [])
            
            for error in errors_list:
                error_type = error.get('type', 'unknown')
                details = error.get('details', {})
                
                original = details.get('original', '').lower().strip()
                corrected = details.get('corrected', '').lower().strip()
                
                if not original or not corrected:
                    continue
                if original == corrected:
                    continue
                
                if self._is_anomalous_error_pair(original, corrected):
                    anomaly_filtered += 1
                    continue
                
                similarity = calculate_similarity(original, corrected)
                
                self.user_error_dict[corrected].add(original)
                self.error_type_stats[error_type] += 1
                total_errors_processed += 1
                
                self.error_examples.append({
                    'original': original,
                    'corrected': corrected,
                    'error_type': error_type,
                    'similarity': similarity,
                    'asin': review_data.get('asin', ''),
                    'review_idx': review_data.get('review_idx', -1)
                })
        
        if len(original.split()) == 1 and len(corrected.split()) > 1:
            return True

        log_with_timestamp(f"  Processed {total_errors_processed} total errors, filtered {anomaly_filtered} anomalies")
    
    def _is_anomalous_error_pair(self, original: str, corrected: str) -> bool:
        """检测异常的错误对"""
        import re
        
        orig_len = len(original)
        corr_len = len(corrected)
        
        if orig_len < 2 or corr_len < 2:
            return True
        
        if corr_len > orig_len * 3:
            return True
        
        if orig_len > corr_len * 3:
            return True
        
        if '&nbsp;' in corrected or '&nbsp;' in original:
            return True
        
        if '"' in corrected or '"' in original or "'" in corrected or "'" in original:
            return True
        
        letter_only_orig = re.sub(r'[^a-z]', '', original.lower())
        letter_only_corr = re.sub(r'[^a-z]', '', corrected.lower())
        
        if not letter_only_corr or not letter_only_orig:
            return True
        
        if letter_only_orig == letter_only_corr:
            return True
        
        special_count_orig = len(re.findall(r'[^a-z]', original))
        special_count_corr = len(re.findall(r'[^a-z]', corrected))
        
        if special_count_corr > len(letter_only_corr):
            return True
        
        if special_count_orig > len(letter_only_orig):
            return True
        
        if re.match(r'^[\W_]+$', original) or re.match(r'^[\W_]+$', corrected):
            return True
        
        if re.match(r'^\d+$', original) or re.match(r'^\d+$', corrected):
            return True
        
        if len(original.split()) == 1 and len(corrected.split()) > 1:
            return True
        
        return False

    def _extract_from_new_format(self):
        errors = self.data.get('detailed_character_errors', [])
        log_with_timestamp(f"  Reading new format (NLTK validated): {len(errors)} errors")

        for error in errors:
            original = error.get('original', '').lower().strip()
            corrected = error.get('corrected', '').lower().strip()
            error_type = error.get('error_type', 'unknown')

            if not original or not corrected:
                continue
            if original == corrected:
                continue
            if len(original) < 3:
                continue

            similarity = calculate_similarity(original, corrected)
            if similarity < 0.7:
                log_with_timestamp(f"  Filtered word-level error: {original} → {corrected} (similarity: {similarity:.2f})")
                continue

            self.user_error_dict[corrected].add(original)
            self.error_type_stats[error_type] += 1

            self.error_examples.append({
                'original': original,
                'corrected': corrected,
                'error_type': error_type,
                'similarity': similarity,
                'asin': error.get('asin', ''),
                'review_idx': error.get('review_idx', -1)
            })

    def _extract_from_old_format(self):
        results = self.data.get('results', [])
        log_with_timestamp(f"  Reading old format: {len(results)} results")

        for result in results:
            spelling_errors = result.get('spelling_errors', {})

            for error_type, errors in spelling_errors.items():
                self.error_type_stats[error_type] += len(errors)

                for error in errors:
                    original = error.get('original', '').lower().strip()
                    corrected = error.get('corrected', '').lower().strip()

                    if not original or not corrected:
                        continue
                    if original == corrected:
                        continue
                    if len(original) < 3:
                        continue

                    similarity = calculate_similarity(original, corrected)
                    if similarity < 0.7:
                        log_with_timestamp(f"  Filtered word-level error: {original} → {corrected} (similarity: {similarity:.2f})")
                        continue
                    
                    if error_type == 'Character_unknown':
                        log_with_timestamp(f"  Filtered Character_unknown: {original} → {corrected}")
                        continue

                    self.user_error_dict[corrected].add(original)

                    self.error_examples.append({
                        'original': original,
                        'corrected': corrected,
                        'error_type': error_type,
                        'similarity': similarity,
                        'asin': result.get('asin'),
                        'review_idx': result.get('review_idx')
                    })

    def get_user_error_for_word(self, word: str) -> Optional[str]:
        """
        获取用户对这个词的历史错误拼写

        Args:
            word: 正确拼写

        Returns:
            用户的错误拼写，如果没有历史则返回None
        """
        word_lower = word.lower()
        if word_lower in self.user_error_dict:
            # 随机选择一个历史错误（用户可能有多种错误方式）
            return random.choice(list(self.user_error_dict[word_lower]))
        return None

    def has_error_for_word(self, word: str) -> bool:
        return word.lower() in self.user_error_dict

    def find_fuzzy_matches(self, word: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> List[Tuple[str, float]]:
        word_lower = word.lower()
        if not word_lower:
            return []
        
        matches = []
        for correct_word in self.user_error_dict.keys():
            if correct_word == word_lower:
                continue
            
            similarity = calculate_similarity(word_lower, correct_word)
            if similarity >= threshold and similarity < 1.0:
                matches.append((correct_word, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_fuzzy_error_for_word(self, word: str) -> Optional[Tuple[str, str, float]]:
        matches = self.find_fuzzy_matches(word)
        if matches:
            best_match, similarity = matches[0]
            user_error = self.get_user_error_for_word(best_match)
            if user_error:
                return (best_match, user_error, similarity)
        return None

    def get_fuzzy_error_for_correct_word(self, word: str, threshold: float = 0.75) -> Optional[Tuple[str, str, float]]:
        """
        查询单词与"正确修正"词进行模糊匹配
        
        Args:
            word: 查询中的单词（正确的）
            threshold: 相似度阈值
            
        Returns:
            (正确修正词, 用户错误拼写, 相似度) 或 None
        """
        word_lower = word.lower()
        if not word_lower or len(word_lower) < 3:
            return None
        
        best_match = None
        best_similarity = 0.0
        best_user_error = None
        
        for correct_word in self.user_error_dict.keys():
            if correct_word == word_lower:
                continue
            
            # 关键修复：避免与超短单词的模糊匹配导致误注入
            # 例如：'making' 不应仅因为与 'ing' 相似就被注入 'ed' 错误
            # 对于长度 < 4 的"正确修正"词，使用更高的阈值 (>= 0.85)
            effective_threshold = threshold
            if len(correct_word) < 4:
                effective_threshold = max(threshold, 0.85)
            
            similarity = calculate_similarity(word_lower, correct_word)
            if similarity >= effective_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = correct_word
                # 获取用户的错误拼写
                user_errors = self.user_error_dict[correct_word]
                if user_errors:
                    best_user_error = random.choice(list(user_errors))
        
        if best_match and best_user_error:
            return (best_match, best_user_error, best_similarity)
        return None

    def get_error_type_distribution(self) -> Dict[str, int]:
        """获取用户错误类型分布"""
        return dict(self.error_type_stats)

    def get_summary(self) -> Dict:
        """获取用户错误画像摘要"""
        return {
            'total_unique_errors': len(self.user_error_dict),
            'total_error_instances': sum(len(v) for v in self.user_error_dict.values()),
            'error_types': dict(self.error_type_stats),
            'sample_errors': self.error_examples[:10] if self.error_examples else []
        }

    def is_error_form(self, word: str) -> bool:
        """检查一个词是否已经是某个错误形式"""
        word_lower = word.lower()
        for error_set in self.user_error_dict.values():
            if word_lower in error_set:
                return True
        return False

    def get_correct_for_error_form(self, word: str) -> Optional[str]:
        """获取错误形式对应的正确词"""
        word_lower = word.lower()
        for correct_word, error_set in self.user_error_dict.items():
            if word_lower in error_set:
                return correct_word
        return None


class PersonalizedSpellingInjector:
    """个性化拼写错误注入器 - 优先使用用户历史错误"""

    def __init__(self, error_extractor: PersonalizedSpellingErrorExtractor):
        """
        Args:
            error_extractor: 用户拼写错误提取器
        """
        self.error_extractor = error_extractor

    def inject_error(self, query: str) -> Dict:
        """
        注入多个拼写错误（移除了"一个查询一个错误"的限制）
        
        新逻辑：
        1. 查询中的正确单词，如果与错误模式中的"正确修正"相似度 ≥ 0.75
        2. 则将该单词转换为用户的错误拼写
        3. 对所有匹配的单词都应用错误（不再限制为一个）
        
        例如：错误模式是 recieve(错误) → receive(正确)
        - 查询词 "received" 与 "receive" 相似度 0.83 ≥ 0.75
        - 转换为 "recieve"
        """
        tokens = query.split()
        if len(tokens) < 2:
            return self._unchanged(query, "Query too short")

        user_error_targets = []
        
        for i, token in enumerate(tokens):
            clean = self._clean_word(token)
            if not clean:
                continue
            
            # 优先检查：词本身是否已经是用户的错误形式
            if self.error_extractor.is_error_form(clean):
                correct_word = self.error_extractor.get_correct_for_error_form(clean)
                if correct_word and correct_word.lower() != clean.lower():
                    if len(correct_word.split()) == 1 and not self._is_invalid_substitution(clean, correct_word):
                        user_error_targets.append((i, token, clean, correct_word, "error_form"))
            else:
                # 其次检查精确匹配
                if self.error_extractor.has_error_for_word(clean):
                    user_error = self.error_extractor.get_user_error_for_word(clean)
                    if user_error and user_error.lower() != clean.lower():
                        if len(user_error.split()) == 1 and not self._is_invalid_substitution(clean, user_error):
                            user_error_targets.append((i, token, clean, user_error, "exact"))
                        continue
                
                # 最后进行模糊匹配：与"正确修正"词相似度 ≥ 0.66
                fuzzy_result = self.error_extractor.get_fuzzy_error_for_correct_word(clean, threshold=0.66)
                if fuzzy_result:
                    matched_correct, user_error, similarity = fuzzy_result
                    if len(user_error.split()) == 1 and not self._is_invalid_substitution(clean, user_error):
                        user_error_targets.append((i, token, clean, user_error, f"fuzzy_{similarity:.2f}"))

        if not user_error_targets:
            return self._unchanged(query, "No matching user error patterns (exact or fuzzy)")

        # 对所有匹配的目标应用错误（移除了随机选择一个的限制）
        injected_errors = []
        for idx, original_token, clean_word, user_error, match_type in user_error_targets:
            noisy_word = self._preserve_format(original_token, user_error)
            tokens[idx] = noisy_word
            injected_errors.append({
                "original_word": original_token,
                "noisy_word": noisy_word,
                "clean_word": clean_word,
                "user_error": user_error,
                "match_method": match_type
            })
        
        noisy_query = " ".join(tokens)

        # 构建返回结果（保持向后兼容，同时添加多错误信息）
        return {
            "original": query,
            "noisy": noisy_query,
            "modified": True,
            "error_type": "user_historical_spelling_error",
            "injected_errors": injected_errors,  # 新增：包含所有注入的错误
            "num_errors_injected": len(injected_errors),  # 新增：注入的错误数量
            # 保持向后兼容的字段（使用第一个错误的信息）
            "original_word": injected_errors[0]["original_word"],
            "noisy_word": injected_errors[0]["noisy_word"],
            "match_method": injected_errors[0]["match_method"],
            "reason": f"Applied {len(injected_errors)} user spelling error(s): " + 
                     ", ".join([f"{e['clean_word']} → {e['user_error']} [{e['match_method']}]" 
                               for e in injected_errors])
        }

    def _clean_word(self, token: str) -> str:
        """清理词：去除标点，保留字母"""
        clean = re.sub(r'[^a-zA-Z]', '', token)
        return clean if clean.isalpha() else ""

    def _preserve_format(self, original: str, new_word: str) -> str:
        """保持原始大小写和标点"""
        # 提取标点
        leading_match = re.match(r'^([^\w]*)', original)
        trailing_match = re.search(r'([^\w]*)$', original)

        leading = leading_match.group(1) if leading_match else ""
        trailing = trailing_match.group(1) if trailing_match else ""

        # 保持大小写
        processed = new_word
        if original and original[0].isupper():
            processed = new_word.capitalize()

        return leading + processed + trailing

    def _is_invalid_substitution(self, orig: str, noisy: str) -> bool:
        """检查是否为无效替换（会被过滤掉）"""
        # 1. 完全相同（仅标点/大小写差异）
        if orig.lower() == noisy.lower():
            return True
        
        # 2. HTML引号（异常格式）
        if '"' in noisy or "'" in noisy:
            return True
        
        # 3. 多词短语（已通过 len(user_error.split()) == 1 检查，此处冗余但保留）
        if len(noisy.split()) > 1:
            return True
        
        # 4. 包含非字母字符（特殊字符异常）
        if not noisy.replace('-', '').isalpha():  # 允许连字符
            return True
        
        return False

    def _unchanged(self, query: str, reason: str) -> Dict:
        return {
            "original": query,
            "noisy": query,
            "modified": False,
            "reason": reason
        }


# ============================================================================
# Part 3: Personalized Grammar Error Injection (基于用户历史)
# ============================================================================

class PersonalizedGrammarErrorExtractor:
    """从Stage 5的writing analysis提取用户语法错误模式"""

    def __init__(self, writing_analysis_file: str):
        with open(writing_analysis_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 语法错误模式提取
        self.grammar_patterns = defaultdict(list)

        self._extract_grammar_patterns()

    def _extract_grammar_patterns(self):
        """提取用户语法错误模式"""

        log_with_timestamp("Extracting user grammar patterns...")

        pattern_categories = {
            'Hyphenation': 'compound_adjective',      # good size → good-sized
            'Collocation': 'word_choice',            # 词搭配错误
            'Agreement': 'subject_verb_agreement',   # 主谓一致
            'Pronoun': 'pronoun_usage',              # 代词使用
            'Preposition': 'preposition_usage',      # 介词使用
            'Suffix': 'suffix_error',                # 后缀错误
        }

        for result in self.data.get('results', []):
            grammar_errors = result.get('grammar_errors', {})

            for error_type, errors in grammar_errors.items():
                category = pattern_categories.get(error_type, error_type.lower())

                for error in errors:
                    original = error.get('original', '')
                    corrected = error.get('corrected', '')
                    reason = error.get('reason', '')

                    if original and corrected and original != corrected:
                        self.grammar_patterns[category].append({
                            'original': original,
                            'corrected': corrected,
                            'error_type': error_type,
                            'reason': reason,
                            'context': error.get('fragment', '')
                        })

        for cat, patterns in self.grammar_patterns.items():
            log_with_timestamp(f"  {cat}: {len(patterns)} patterns")

    def get_patterns_by_category(self, category: str) -> List[Dict]:
        """获取特定类别的语法错误模式"""
        return self.grammar_patterns.get(category, [])

    def get_all_patterns(self) -> Dict[str, List[Dict]]:
        """获取所有语法错误模式"""
        return dict(self.grammar_patterns)

    def has_grammar_errors(self) -> bool:
        """用户是否有语法错误历史"""
        return len(self.grammar_patterns) > 0


class PersonalizedGrammarInjector:
    """个性化语法错误注入器"""

    def __init__(self, grammar_extractor: PersonalizedGrammarErrorExtractor):
        """
        Args:
            grammar_extractor: 用户语法错误提取器
        """
        self.grammar_extractor = grammar_extractor

    def inject_error(self, query: str) -> Dict:
        """
        向查询注入一个语法错误

        策略：
        1. 分析query结构
        2. 查找匹配的用户语法错误模式
        3. 应用模式转换

        Returns:
            {
                "original": str,
                "noisy": str,
                "modified": bool,
                "error_type": str,
                "original_phrase": str,
                "noisy_phrase": str,
                "reason": str
            }
        """

        # 获取用户所有语法错误模式
        all_patterns = self.grammar_extractor.get_all_patterns()

        if not all_patterns:
            return self._unchanged(query, "No grammar error patterns found for user")

        # 尝试每种模式
        for category, patterns in all_patterns.items():
            if not patterns:
                continue

            # 随机选择该类别的一个模式
            pattern = random.choice(patterns)
            corrected = pattern['corrected']
            original_error = pattern['original']

            # 检查query中是否包含正确的短语
            # 按长度排序，优先匹配长短语（更精确）
            if corrected.lower() in query.lower():
                # 应用错误
                noisy_query = self._apply_pattern(query, corrected, original_error)

                if noisy_query != query:
                    return {
                        "original": query,
                        "noisy": noisy_query,
                        "modified": True,
                        "error_type": f"user_historical_grammar_error:{category}",
                        "original_phrase": corrected,
                        "noisy_phrase": original_error,
                        "reason": f"Applied user's grammar error pattern: {corrected} → {original_error}"
                    }

        return self._unchanged(query, "No matching grammar patterns in query")

    def _apply_pattern(self, query: str, correct: str, error: str) -> str:
        """应用语法错误模式"""
        # 使用正则表达式进行替换（保持大小写）
        pattern = re.compile(re.escape(correct), re.IGNORECASE)

        def replacer(match):
            original = match.group(0)
            # 保持原始大小写
            if original and original[0].isupper():
                return error.capitalize()
            return error

        result = pattern.sub(replacer, query)
        return result

    def _unchanged(self, query: str, reason: str) -> Dict:
        return {
            "original": query,
            "noisy": query,
            "modified": False,
            "reason": reason
        }


# ============================================================================
# Part 4: Unified Personalized Error Injector (仅拼写)
# ============================================================================

class UnifiedPersonalizedErrorInjector:
    """
    统一的个性化错误注入器（仅拼写错误）

    策略：
    1. 仅使用用户的拼写错误
    2. 如果失败，返回未修改
    """

    def __init__(self, writing_analysis_file: str):
        """
        Args:
            writing_analysis_file: Stage 5的writing analysis文件
        """
        log_with_timestamp("Initializing personalized spelling error injector...")

        # 只使用拼写错误提取器和注入器
        self.spelling_extractor = PersonalizedSpellingErrorExtractor(writing_analysis_file)
        self.spelling_injector = PersonalizedSpellingInjector(self.spelling_extractor)

        # 打印摘要
        spelling_summary = self.spelling_extractor.get_summary()
        log_with_timestamp(f"✓ Spelling errors: {spelling_summary['total_unique_errors']} unique, "
                          f"{spelling_summary['total_error_instances']} total")
        log_with_timestamp("  Using spelling errors only (grammar errors disabled)")

    def inject_error(self, query: str) -> Dict:
        """
        注入一个拼写错误

        Args:
            query: 原始查询

        Returns:
            注入结果字典
        """

        # 只尝试拼写错误
        result = self.spelling_injector.inject_error(query)
        return result

    def get_user_profile_summary(self) -> Dict:
        """获取用户错误画像摘要"""
        return {
            'spelling': self.spelling_extractor.get_summary()
        }


# ============================================================================
# Part 5: Hybrid Error Injector (纯个性化拼写)
# ============================================================================

class HybridErrorInjector:
    """
    纯个性化拼写错误注入器

    策略：
    1. 仅使用个性化拼写错误注入（基于用户真实历史）
    2. 如果无法匹配，保持查询不变（不强制注入）
    """

    def __init__(self, writing_analysis_file: str):
        """
        Args:
            writing_analysis_file: Stage 5的writing analysis文件（必需）
        """
        self.has_personalization = False
        self.personalized_injector = None

        if writing_analysis_file and os.path.exists(writing_analysis_file):
            try:
                self.personalized_injector = UnifiedPersonalizedErrorInjector(writing_analysis_file)
                self.has_personalization = True
                log_with_timestamp("✓ Personalized spelling error injection enabled")
                log_with_timestamp("  Only injecting user's real spelling errors (grammar errors disabled)")
            except Exception as e:
                log_with_timestamp(f"⚠ Failed to load personalized injector: {e}")
                log_with_timestamp("  Error: Writing analysis is required for personalized injection")
                raise ValueError("Writing analysis file is required")
        else:
            log_with_timestamp("❌ Error: No writing analysis file provided")
            log_with_timestamp("  Personalized injection requires user error history")
            raise ValueError("Writing analysis file is required for personalized injection")

    def inject_single_error(self, query: str) -> Dict:
        """
        向query注入一个错误（仅使用用户真实历史）

        Returns:
            {
                "original": str,
                "noisy": str,
                "modified": bool,
                "error_type": str,
                "method": str,  # "personalized" or "unmodified"
                ...
            }
        """

        # 仅使用个性化注入
        if self.has_personalization:
            result = self.personalized_injector.inject_error(query)

            if result['modified']:
                result['method'] = 'personalized'
                return result

            # 无匹配：保持查询不变
            log_with_timestamp("  No matching user error patterns, query kept unchanged")
            result['method'] = 'unmodified'
            return result

        # 无用户画像：返回未修改
        return {
            "original": query,
            "noisy": query,
            "modified": False,
            "reason": "No user profile available",
            "method": "no_profile"
        }

    def get_user_profile_summary(self) -> Dict:
        """获取用户画像摘要"""
        if self.has_personalization:
            return self.personalized_injector.get_user_profile_summary()
        return {}


# ============================================================================
# Part 6: Main Processing Logic
# ============================================================================

def process_user(user_id: str,
                 aligned_queries_file: str,
                 injector: HybridErrorInjector,
                 output_dir: str,
                 query_source: str):
    """处理单个用户的查询"""

    log_with_timestamp(f"Processing user {user_id}...")

    # 加载对齐查询
    with open(aligned_queries_file, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    results = []
    personalized_count = 0
    unmodified_count = 0
    total_count = 0
    modified_count = 0

    # 处理不同的文件格式
    if isinstance(aligned_data, list):
        queries = aligned_data
    elif 'queries' in aligned_data:
        queries = aligned_data['queries']
    elif 'results' in aligned_data:
        queries = aligned_data['results']
    else:
        queries = []

    log_with_timestamp(f"  Processing {len(queries)} queries...")

    for query_data in queries:
        asin = query_data.get('asin')

        # 尝试不同的字段名
        aligned_query = (
            query_data.get('final_query') or
            query_data.get('aligned_query') or
            query_data.get('final_aligned_query') or
            query_data.get('personalized_query') or
            query_data.get('query', '')
        )

        # 尝试从嵌套结构获取
        if not aligned_query:
            target_query = query_data.get('target_user_query', {})
            mass_query = query_data.get('mass_market_query', {})
            aligned_query = target_query.get('query') or mass_query.get('query')

        if not aligned_query:
            log_with_timestamp(f"  No aligned query found for ASIN {asin}")
            continue

        total_count += 1

        # 注入错误
        result = injector.inject_single_error(aligned_query)

        if result['modified']:
            modified_count += 1
            method = result.get('method', 'unknown')

            if method == 'personalized':
                personalized_count += 1

            log_with_timestamp(f"  ✓ {asin}: {result.get('reason', 'Success')} [{method}]")
        else:
            unmodified_count += 1
            log_with_timestamp(f"  ✗ {asin}: {result.get('reason', 'Unknown reason')}")

        # 构建结果
        query_result = {
            "asin": asin,
            "personalized_query": {
                "original": result['original'],
                "noisy": result['noisy'],
                "modified": result['modified'],
                "method": result.get('method', 'statistical'),
                "injection_version": "hybrid_v1"
            }
        }

        # 添加详细信息
        if result['modified']:
            if 'error_type' in result:
                query_result["personalized_query"]["error_type"] = result['error_type']
            if 'original_word' in result:
                query_result["personalized_query"]["original_word"] = result['original_word']
            if 'noisy_word' in result:
                query_result["personalized_query"]["noisy_word"] = result['noisy_word']
            if 'original_phrase' in result:
                query_result["personalized_query"]["original_phrase"] = result['original_phrase']
            if 'noisy_phrase' in result:
                query_result["personalized_query"]["noisy_phrase"] = result['noisy_phrase']
            if 'injected_errors' in result:
                query_result["personalized_query"]["injected_errors"] = result['injected_errors']
            if 'num_errors_injected' in result:
                query_result["personalized_query"]["num_errors_injected"] = result['num_errors_injected']
            query_result["personalized_query"]["reason"] = result.get('reason', '')
        else:
            query_result["personalized_query"]["reason"] = result.get('reason', 'Unknown reason')

        results.append(query_result)

    # 保存结果
    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "method": "personalized_only_v1",
        "total_queries": total_count,
        "modified_queries": modified_count,
        "unmodified_queries": unmodified_count,
        "modification_rate": round(modified_count / total_count * 100, 1) if total_count > 0 else 0,
        "personalized_injections": personalized_count,
        "user_profile_summary": injector.get_user_profile_summary(),
        "queries": results
    }

    if query_source == 'stage7':
        output_file = os.path.join(output_dir, f"iterative_noisy_query_{user_id}.json")
    else:
        output_file = os.path.join(output_dir, f"noisy_queries_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    if total_count > 0:
        log_with_timestamp(f"  Modified {modified_count}/{total_count} queries ({modified_count/total_count*100:.1f}%)")
        log_with_timestamp(f"  Unmodified {unmodified_count}/{total_count} queries ({unmodified_count/total_count*100:.1f}%)")
        if modified_count > 0:
            log_with_timestamp(f"    - All personalized injections (no statistical fallback)")
    log_with_timestamp(f"  Saved to {output_file}")

    return output_data


def main():
    global USER_ID, STAGE7_RESULTS_FILE, WRITING_ANALYSIS_FILE, OUTPUT_DIR, QUERY_SOURCE
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate noisy queries with personalized error injection')
    parser.add_argument('--user', type=str, help='User ID to process (if not set, will be loaded from batch script)')
    parser.add_argument('--source', type=str, default='stage6', choices=['stage6', 'stage7'])
    parser.add_argument('--output-dir', type=str, default=None)
    args, unknown_args = parser.parse_known_args()
    
    # If user argument provided, use it
    if args.user:
        USER_ID = args.user
    elif not USER_ID:
        log_with_timestamp("❌ Error: No user ID provided via --user argument or configuration")
        return 1
    
    QUERY_SOURCE = args.source

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/09_targeted_noisy_query")

    # Set file paths based on USER_ID
    if QUERY_SOURCE == 'stage7':
        STAGE7_RESULTS_FILE = os.path.join(BASE_DIR, f"result/personal_query/07_iterative_refinement/{USER_ID}_interative_query.json")
    else:
        STAGE7_RESULTS_FILE = os.path.join(BASE_DIR, f"result/personal_query/06_query/queries_{USER_ID}.json")
    WRITING_ANALYSIS_FILE = os.path.join(BASE_DIR, f"result/personal_query/04_writing_analysis/writing_analysis_{USER_ID}.json")
    
    log_with_timestamp("=" * 60)
    log_with_timestamp("Stage 10: Personalized Noisy Query Generator")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"User ID: {USER_ID}")
    log_with_timestamp(f"Query source: {QUERY_SOURCE}")
    log_with_timestamp(f"Input query file: {STAGE7_RESULTS_FILE}")
    log_with_timestamp(f"Writing analysis: {WRITING_ANALYSIS_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        injector = HybridErrorInjector(WRITING_ANALYSIS_FILE)
    except ValueError as e:
        log_with_timestamp(f"Error: {e}")
        return 1

    log_with_timestamp(f"Loading Stage 7 results from {STAGE7_RESULTS_FILE}...")
    with open(STAGE7_RESULTS_FILE, 'r', encoding='utf-8') as f:
        stage7_data = json.load(f)

    # 提取用户查询
    if 'results' in stage7_data:
        all_queries = stage7_data['results']
    else:
        all_queries = stage7_data if isinstance(stage7_data, list) else []

    user_queries = defaultdict(list)
    for q in all_queries:
        uid = q.get('user_id')
        if uid:
            user_queries[uid].append(q)

    user_ids = [USER_ID] if USER_ID in user_queries else list(user_queries.keys())

    log_with_timestamp(f"Found {len(user_ids)} users to process")

    summary = []
    for user_id in user_ids:
        temp_file = f"/tmp/queries_{user_id}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(user_queries[user_id], f, indent=2)

        result = process_user(
            user_id=user_id,
            aligned_queries_file=temp_file,
            injector=injector,
            output_dir=OUTPUT_DIR,
            query_source=QUERY_SOURCE
        )

        summary.append({
            "user_id": user_id,
            "total_queries": result['total_queries'],
            "modified_queries": result['modified_queries'],
            "unmodified_queries": result['unmodified_queries'],
            "modification_rate": result['modification_rate'],
            "personalized_injections": result.get('personalized_injections', 0)
        })

        os.remove(temp_file)

    if QUERY_SOURCE == 'stage7':
        summary_file = os.path.join(OUTPUT_DIR, "iterative_noisy_query_summary.json")
    else:
        summary_file = os.path.join(OUTPUT_DIR, "noisy_queries_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "method": "personalized_spelling_only_v1",
            "error_type": "spelling",
            "random_seed": RANDOM_SEED,
            "total_users": len(summary),
            "users": summary
        }, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nSummary saved to {summary_file}")
    log_with_timestamp("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
