#!/usr/bin/env python3
"""
Stage 6: Generate Target User Queries Only

使用改进的优先级选择方法生成目标用户查询：
1. 优先选择包含用户拼写错误词的属性
2. 确保选择的词在目标产品中实际存在
3. 为每个产品生成目标用户查询

NOTE: Mass market query generation has been DISABLED - only generating target user queries

Input: result/personal_query/02_processing/query_{user_id}.json
Output: result/personal_query/06_query/dual_queries_{user_id}.json
"""

import re
import json
import gzip
import random
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')
from llm_client import LLMClient

# Import BM25 tokenizer for vocabulary validation alignment
try:
    import bm25s
except ImportError:
    bm25s = None

# Import build_document_text for consistent text source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '12_retrieval', 'utils'))
from utils import build_document_text


def log_with_timestamp(message: str):
    """带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def calculate_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度（使用 SequenceMatcher）
    
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
    
    Returns:
        相似度分数（0.0-1.0）
    """
    if not s1 or not s2:
        return 0.0
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def load_user_spelling_errors(writing_analysis_file: str) -> Dict[str, List[str]]:
    """
    加载用户拼写错误历史（支持新旧格式）

    Returns:
        字典：{正确词: [错误词1, 错误词2, ...]}
    """
    try:
        with open(writing_analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        user_error_patterns = defaultdict(set)

        # 新格式：detailed_character_errors
        if 'detailed_character_errors' in data:
            for error in data.get('detailed_character_errors', []):
                corrected = error.get('corrected', '').strip()
                original = error.get('original', '').strip()
                
                if corrected and original and corrected != original:
                    # 过滤相似度太低的错误
                    similarity = calculate_similarity(original.lower(), corrected.lower())
                    if similarity >= 0.7:
                        user_error_patterns[clean_word(corrected)].add(clean_word(original))
        
        # 旧格式：results
        else:
            for result in data.get('results', []):
                if 'spelling_errors' in result:
                    spelling = result['spelling_errors']

                    if isinstance(spelling, dict):
                        for error_type, errors in spelling.items():
                            if isinstance(errors, list):
                                for error in errors:
                                    if isinstance(error, dict):
                                        corrected = error.get('corrected', '')
                                        original = error.get('original', '')

                                        if corrected and original and corrected != original:
                                            user_error_patterns[clean_word(corrected)].add(clean_word(original))

        # Convert sets to lists for type consistency
        return {k: list(v) for k, v in user_error_patterns.items()}

    except Exception as e:
        log_with_timestamp(f"⚠️  Warning: Could not load user spelling errors: {e}")
        return {}


def clean_word(word: str) -> str:
    """清理单词：去除标点和多余空格"""
    return re.sub(r'[^\w\s-]', '', word).strip().lower()


def extract_words_from_text(text: str, min_length: int = 3) -> Set[str]:
    """
    从文本中提取所有单词

    Args:
        text: 输入文本
        min_length: 最小词长

    Returns:
        单词集合（小写，去重）
    """
    if not text:
        return set()

    # 分词并清理
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # 过滤短词
    return {w for w in words if len(w) >= min_length}


def contains_user_error_word(text: str,
                             user_error_patterns: Dict[str, List[str]],
                             fuzzy_threshold: float = 0.75) -> List[Tuple[str, str, str]]:
    """
    检查文本是否包含用户拼写错误相关的词汇（支持模糊匹配）

    Args:
        text: 要检查的文本
        user_error_patterns: 用户错误模式字典 {正确词: [错误词1, ...]}
        fuzzy_threshold: 模糊匹配阈值（默认0.75）

    Returns:
        匹配的(正确词, 匹配词, 匹配类型)列表
        匹配类型: "exact" 或 "fuzzy_0.XX"
    """
    if not text or not user_error_patterns:
        return []

    text_lower = text.lower()
    words = extract_words_from_text(text_lower)

    matches = []
    for word in words:
        clean = clean_word(word)
        if not clean or len(clean) < 3:
            continue

        # 检查是否是用户的正确词
        for correct_word in user_error_patterns.keys():
            error_words = user_error_patterns[correct_word]

            # 精确匹配：文本包含正确词
            if clean == correct_word.lower():
                matches.append((correct_word, correct_word, "exact"))
                continue

            # 精确匹配：包含任何错误词
            if clean in [e.lower() for e in error_words]:
                matches.append((correct_word, clean, "exact"))
                continue
            
            # 模糊匹配：与正确词相似度 >= 阈值
            similarity = calculate_similarity(clean, correct_word.lower())
            if similarity >= fuzzy_threshold:
                matches.append((correct_word, clean, f"fuzzy_{similarity:.2f}"))

    return matches


# ============================================================================
# 产品词汇库（预加载）
# ============================================================================

class ProductVocabulary:
    """产品词汇库 - 用于检查词是否在产品中"""

    def __init__(self, meta_file: str, asins: List[str]):
        """
        初始化产品词汇库

        Args:
            meta_file: 元数据文件路径
            asins: 需要加载的产品ASIN列表
        """
        self.meta_file = meta_file
        self.target_asins = set(asins)
        self.product_vocabularies = {}  # {asin: set(words)}
        self.asin_documents = {}  # {asin: document_text}

    def load_vocabulary_for_asin(self, asin: str) -> bool:
        """
        加载单个产品的词汇，使用与BM25一致的tokenization

        Returns:
            是否成功加载
        """
        if asin in self.product_vocabularies:
            return True

        try:
            with gzip.open(self.meta_file, 'rt') as f:
                for line in f:
                    try:
                        product = json.loads(line)
                        if product.get('asin') == asin:
                            # 提取文档文本（使用核心字段，与原有方式相同，避免性能问题）
                            title = product.get('title', '')
                            brand = product.get('brand', '')
                            feature = product.get('feature', [])
                            description = product.get('description', [])

                            document_parts = [title, brand] + feature + description
                            document_text = ' '.join(document_parts)

                            # 使用bm25s.tokenize进行分词，与BM25索引的tokenization保持一致
                            # 这确保了验证使用与BM25相同的stopwords过滤和分词逻辑
                            if bm25s is not None:
                                tokenized = bm25s.tokenize(
                                    [document_text],
                                    lower=True,
                                    stopwords="english",
                                    show_progress=False
                                )
                                # bm25s.tokenize返回Tokenized对象，vocab包含实际的词
                                words = set(tokenized.vocab)
                            else:
                                # 回退方案：如果bm25s不可用，使用原有的提取方式
                                words = extract_words_from_text(document_text, min_length=4)
                            
                            self.product_vocabularies[asin] = words
                            self.asin_documents[asin] = document_text
                            return True
                    except:
                        continue
        except Exception as e:
            log_with_timestamp(f"  ⚠️  加载产品 {asin} 失败: {e}")

        return False

    def word_exists_in_product(self, word: str, asin: str) -> bool:
        """
        检查词是否在指定产品中

        Args:
            word: 要检查的词
            asin: 产品ASIN

        Returns:
            词是否在产品中
        """
        # 确保词汇已加载
        if asin not in self.product_vocabularies:
            self.load_vocabulary_for_asin(asin)

        if asin in self.product_vocabularies:
            return word.lower() in self.product_vocabularies[asin]
        return False
    
    def word_percentage_exists_in_product(self, phrase: str, asin: str) -> float:
        """
        计算属性值（短语）中有多少百分比的词在产品词汇中存在
        
        方案A改进：允许属性值只需部分词在商品中（而非所有词）
        
        Args:
            phrase: 属性值短语，如 "floral pattern"
            asin: 产品ASIN
        
        Returns:
            0-1之间的匹配比例。1.0表示100%词都在，0.5表示50%词在，等等
        """
        # 确保词汇已加载
        if asin not in self.product_vocabularies:
            self.load_vocabulary_for_asin(asin)
        
        if asin not in self.product_vocabularies:
            return 0.0
        
        # 提取短语中的词
        words = extract_words_from_text(phrase, min_length=2)
        if not words:
            return 0.0
        
        # 计算有多少词在产品中
        product_vocab = self.product_vocabularies[asin]
        words_in_product = sum(1 for word in words if word.lower() in product_vocab)
        
        return words_in_product / len(words) if words else 0.0


# ============================================================================
# 改进的优先级选择
# ============================================================================

def select_dimensions_with_improved_priority(
    attributes_by_dimension: Dict[str, List[Dict]],
    user_error_patterns: Dict[str, List[str]],
    product_vocabulary: ProductVocabulary,
    target_asin: str,
    num_dimensions: int = 5,
    product_metadata: Optional[Dict[str, Dict]] = None
) -> Tuple[List[Tuple[str, str]], Dict]:
    """
    改进的优先级选择：要求最终的属性值必须100%在商品文本中
    并优先选择包含商品品牌或标题词的属性

    优先级分类（按选择顺序）：
    1. 超高优先级：包含品牌名称或产品标题词的属性（100%词汇验证）
    2. 最高优先级：包含用户错误词 AND 词在目标产品中 AND 100%词汇验证通过
    3. 中优先级：100%词在产品中 + 不包含错误词
    4. 备选优先级：50%-99%词汇匹配

    最终过滤：选定的5个属性值必须全部通过100%验证（所有词都在商品中）

    Args:
        attributes_by_dimension: 维度到属性列表的映射
        user_error_patterns: 用户拼写错误模式
        product_vocabulary: 产品词汇库
        target_asin: 目标产品ASIN
        num_dimensions: 需要选择的维度数量
        product_metadata: 可选的产品元数据字典 {asin: {brand, title, ...}}

    Returns:
        (选择的属性列表, 统计信息)
    """

    # 提取目标产品的品牌和标题词
    brand_lower = ''
    title_words = set()
    
    if product_metadata and target_asin in product_metadata:
        product_info = product_metadata[target_asin]
        brand_lower = product_info.get('brand', '').lower()
        title = product_info.get('title', '').lower()
        title_words = set(w for w in title.split()[:15] if len(w) > 2)
    
    # 分类属性 - 超高优先级用于包含品牌/标题的属性
    brand_title_priority = []  # 包含品牌/标题 + 100%词汇验证通过
    highest_priority = []      # 100%词在产品中 + 包含错误词
    medium_priority = []       # 100%词在产品中 + 不包含错误词
    fallback_priority = []     # 50%+词在产品中 + 包含错误词
    low_priority = []          # 50%+词在产品中 + 不包含错误词

    stats = {
        'total_attrs': 0,
        'brand_title_priority': 0,
        'highest_priority': 0,
        'medium_priority': 0,
        'fallback_priority': 0,
        'low_priority': 0
    }

    for dim, attrs in attributes_by_dimension.items():
        if not attrs:
            continue

        for attr in attrs:
            attr_value = attr.get('attribute', '')
            stats['total_attrs'] += 1

            # 检查属性值的词汇匹配比例
            match_ratio = product_vocabulary.word_percentage_exists_in_product(attr_value, target_asin)
            
            # 检查是否包含品牌名称或产品标题词
            attr_value_lower = attr_value.lower()
            has_brand = brand_lower and brand_lower in attr_value_lower
            has_title = any(word in attr_value_lower for word in title_words)
            
            # 如果属性值包含品牌或标题词且通过100%验证，放入超高优先级
            if (has_brand or has_title) and match_ratio >= 1.0:
                brand_title_priority.append((dim, attr_value))
                stats['brand_title_priority'] += 1
            # 否则，按照常规优先级分类
            else:
                # 检查是否包含用户错误词（支持模糊匹配）
                error_matches = contains_user_error_word(attr_value, user_error_patterns)

                if error_matches:
                    # 找到包含的用户错误词
                    for correct_word, matched_word, match_type in error_matches:
                        # 优先选择100%词汇匹配的
                        if match_ratio >= 1.0:
                            highest_priority.append((dim, attr_value, correct_word, matched_word))
                            stats['highest_priority'] += 1
                        # 次优：至少50%词汇匹配
                        elif match_ratio >= 0.5:
                            fallback_priority.append((dim, attr_value, correct_word, matched_word))
                            stats['fallback_priority'] += 1
                        else:
                            low_priority.append((dim, attr_value, correct_word, matched_word))
                            stats['low_priority'] += 1
                else:
                    # 不包含用户错误词的属性
                    if match_ratio >= 1.0:
                        medium_priority.append((dim, attr_value))
                        stats['medium_priority'] += 1
                    elif match_ratio >= 0.5:
                        fallback_priority.append((dim, attr_value))
                        stats['fallback_priority'] += 1

    # 按优先级选择（严格执行100%词汇验证）
    selected = []
    selection_record = []

    # 第零优先级（超高）：包含品牌名称或产品标题词的属性（MUST选中）
    if brand_title_priority:
        num_brand_title = min(num_dimensions, len(brand_title_priority))
        chosen = random.sample(brand_title_priority, num_brand_title)
        for dim, attr in chosen:
            selected.append((dim, attr))
            selection_record.append({
                'dimension': dim,
                'attribute': attr,
                'priority_level': 'brand_title',
                'reason': '包含品牌名称或产品标题词 + 100%词汇验证通过',
                'word_match_ratio': 1.0
            })

    # 第一优先级：100%词在产品中 + 包含错误词
    if len(selected) < num_dimensions and highest_priority:
        num_highest = min(num_dimensions - len(selected), len(highest_priority))
        chosen = random.sample(highest_priority, num_highest)
        for dim, attr, _, _ in chosen:
            selected.append((dim, attr))
            selection_record.append({
                'dimension': dim,
                'attribute': attr,
                'priority_level': 'highest',
                'reason': '100%词汇验证通过 + 包含用户错误词',
                'word_match_ratio': 1.0
            })

    # 第二优先级：100%词在产品中 + 不包含错误词（但满足关键要求）
    if len(selected) < num_dimensions and medium_priority:
        num_medium = min(num_dimensions - len(selected), len(medium_priority))
        chosen = random.sample(medium_priority, num_medium)
        for dim, attr in chosen:
            match_ratio = product_vocabulary.word_percentage_exists_in_product(attr, target_asin)
            selected.append((dim, attr))
            selection_record.append({
                'dimension': dim,
                'attribute': attr,
                'priority_level': 'medium',
                'reason': '100%词汇验证通过 + 不包含用户错误词',
                'word_match_ratio': match_ratio
            })

    # 第三优先级（备选）：50%-99%词汇匹配 + 包含错误词
    # 仅在无法凑够5个维度时使用
    if len(selected) < num_dimensions and fallback_priority:
        num_fallback = min(num_dimensions - len(selected), len(fallback_priority))
        chosen = random.sample(fallback_priority, num_fallback)
        for item in chosen:
            if len(item) == 4:  # 包含错误词的情况
                dim, attr, _, _ = item
            else:  # 不包含错误词的情况
                dim, attr = item
            match_ratio = product_vocabulary.word_percentage_exists_in_product(attr, target_asin)
            selected.append((dim, attr))
            selection_record.append({
                'dimension': dim,
                'attribute': attr,
                'priority_level': 'fallback',
                'reason': f'{match_ratio*100:.0f}%词汇验证通过 (低于100%)',
                'word_match_ratio': match_ratio
            })

    stats['selection_record'] = selection_record
    stats['final_selected_count'] = len(selected)
    stats['target_count'] = num_dimensions
    return selected, stats


class ImprovedPrioritySelector:
    """改进的优先级选择器（批量处理优化版）"""

    def __init__(self, meta_file: str, writing_analysis_file: str):
        """
        Args:
            meta_file: 产品元数据文件
            writing_analysis_file: 用户写作分析文件
        """
        self.meta_file = meta_file
        self.writing_analysis_file = writing_analysis_file
        self.user_error_patterns = load_user_spelling_errors(writing_analysis_file)

        log_with_timestamp(f"用户错误模式: {len(self.user_error_patterns)} 个正确词")

        # 缓存已加载的产品词汇和元数据
        self.loaded_asins = set()
        self.product_vocabularies = {}
        self.product_metadata_cache = {}

    def _load_product_metadata(self, asins: List[str]) -> Dict[str, Dict]:
        """加载指定ASIN的产品元数据（品牌和标题）"""
        result = {}
        
        for asin in asins:
            if asin in self.product_metadata_cache:
                result[asin] = self.product_metadata_cache[asin]
                continue
            
            try:
                with gzip.open(self.meta_file, 'rt') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get('asin') == asin:
                                metadata = {
                                    'brand': data.get('brand', ''),
                                    'title': data.get('title', '')
                                }
                                self.product_metadata_cache[asin] = metadata
                                result[asin] = metadata
                                break
                        except:
                            continue
            except Exception as e:
                log_with_timestamp(f"  ⚠️  加载产品元数据失败 {asin}: {e}")
        
        return result

    def select_for_product(self,
                          attributes_by_dimension: Dict[str, List[Dict]],
                          target_asin: str,
                          num_dimensions: int = 5) -> Tuple[List[Tuple[str, str]], Dict]:
        """
        为单个产品选择属性

        Args:
            attributes_by_dimension: 维度到属性列表的映射
            target_asin: 目标产品ASIN
            num_dimensions: 需要选择的维度数量

        Returns:
            (选择的属性列表, 统计信息)
        """
        # 创建临时的词汇库对象
        vocab = ProductVocabulary(self.meta_file, [target_asin])
        vocab.load_vocabulary_for_asin(target_asin)

        # 加载产品元数据（品牌和标题）
        product_metadata = self._load_product_metadata([target_asin])

        # 使用改进的选择逻辑
        return select_dimensions_with_improved_priority(
            attributes_by_dimension,
            self.user_error_patterns,
            vocab,
            target_asin,
            num_dimensions,
            product_metadata=product_metadata
        )


# ============================================================================
# 查询生成
# ============================================================================

def extract_attributes_by_dimension(attributes: List[Dict]) -> Dict[str, List[Dict]]:
    """Group attributes by their dimension."""
    by_dimension = defaultdict(list)
    for attr in attributes:
        dimension = attr.get('dimension', 'Unknown')
        if dimension and attr.get('attribute'):
            by_dimension[dimension].append(attr)
    return dict(by_dimension)


def generate_target_user_query_prompt(selected_attrs: List[Tuple[str, str]],
                                       category: str,
                                       prev_word_count: Optional[int] = None,
                                       missing_words: Optional[List[str]] = None) -> str:
    """Generate prompt for target user query."""
    attr_list = '\n'.join([
        f"  - {dim}: {attr}" for dim, attr in selected_attrs
    ])
    
    # Give targeted feedback based on whether previous response was too long or too short
    if prev_word_count is not None:
        if prev_word_count < 20:
            word_emphasis = f"""
CRITICAL: Your previous response was TOO SHORT ({prev_word_count} words). You MUST EXPAND to 20-35 words.
Add more detail about what you need. Describe the product features more fully.
Count your words carefully before responding."""
        else:  # prev_word_count > 35
            word_emphasis = f"""
CRITICAL: Your previous response was TOO LONG ({prev_word_count} words). You MUST shorten to 20-35 words.
Be concise. Do NOT expand on each attribute - just mention them briefly.
Count your words carefully before responding."""
    else:
        word_emphasis = ""
    
    # Add specific word requirements if there are missing words
    if missing_words:
        # Extract the actual words from the format "word (from: attribute)"
        required_words = []
        for missing in missing_words:
            if ' (from: ' in missing:
                word = missing.split(' (from: ')[0]
                required_words.append(word)
            else:
                required_words.append(missing)
        
        words_list = ', '.join([f'"{w}"' for w in required_words])
        word_requirement = f"""
CRITICAL WORD REQUIREMENT:
Your previous attempt was missing these REQUIRED words: {words_list}
You MUST include these exact words in your response. These words come from the attribute values above.
Make sure to use them naturally in the query - do not just list them.
"""
    else:
        word_requirement = ""
    
    prompt = f"""Generate a first-person search query for an Amazon shopper looking for products in the "{category}" category.

SELECTED USER PREFERENCES (5 dimensions, 5 values):
{attr_list}

REQUIREMENTS:
1. Write in FIRST PERSON ("I am looking for...", "I need...", "I want...")
2. Incorporate ALL FIVE attribute values naturally into the query
3. Word count: 20-35 words (STRICTLY ENFORCED - count carefully!)
4. Make it sound like a natural search query a real person would type
5. Do NOT mention the dimension names - just use the values
6. Do NOT add quotes, markdown, or explanations - just the query text
{word_requirement}{word_emphasis}

EXAMPLE FORMAT:
"I am looking for a high-quality embossing folder that creates raised patterns on my cardstock for greeting card making. It should work well with my Cuttlebug machine." (28 words)

OUTPUT (just the query, 20-35 words):"""

    return prompt


def generate_mass_market_query_prompt(selected_attrs: List[Tuple[str, str]],
                                       category: str,
                                       prev_word_count: Optional[int] = None,
                                       missing_words: Optional[List[str]] = None) -> str:
    """Generate prompt for mass market query."""
    attr_list = '\n'.join([
        f"  - {dim}: {attr}" for dim, attr in selected_attrs
    ])
    
    # Give targeted feedback based on whether previous response was too long or too short
    if prev_word_count is not None:
        if prev_word_count < 20:
            word_emphasis = f"""
CRITICAL: Your previous response was TOO SHORT ({prev_word_count} words). You MUST EXPAND to 20-35 words.
Add more detail about what you need. Describe the product features more fully.
Count your words carefully before responding."""
        else:  # prev_word_count > 35
            word_emphasis = f"""
CRITICAL: Your previous response was TOO LONG ({prev_word_count} words). You MUST shorten to 20-35 words.
Be concise. Do NOT expand on each attribute - just mention them briefly.
Count your words carefully before responding."""
    else:
        word_emphasis = ""
    
    # Add specific word requirements if there are missing words
    if missing_words:
        # Extract the actual words from the format "word (from: attribute)"
        required_words = []
        for missing in missing_words:
            if ' (from: ' in missing:
                word = missing.split(' (from: ')[0]
                required_words.append(word)
            else:
                required_words.append(missing)
        
        words_list = ', '.join([f'"{w}"' for w in required_words])
        word_requirement = f"""
CRITICAL WORD REQUIREMENT:
Your previous attempt was missing these REQUIRED words: {words_list}
You MUST include these exact words in your response. These words come from the attribute values above.
Make sure to use them naturally in the query - do not just list them.
"""
    else:
        word_requirement = ""
    
    prompt = f"""Generate a first-person search query for a typical Amazon shopper looking for products in the "{category}" category.

SELECTED PRODUCT ATTRIBUTES (5 dimensions, 5 values):
{attr_list}

REQUIREMENTS:
1. Write in FIRST PERSON ("I am looking for...", "I need...", "I want...")
2. Incorporate ALL FIVE attribute values naturally into the query
3. Word count: 20-35 words (STRICTLY ENFORCED - count carefully!)
4. Make it sound like a natural search query a typical shopper would type
5. Focus on general product features, not specific brand preferences
6. Do NOT mention the dimension names - just use the values
7. Do NOT add quotes, markdown, or explanations - just the query text
{word_requirement}{word_emphasis}

EXAMPLE FORMAT:
"I need a reliable glitter glue that doesn't clog and has a fine-point nozzle for precise application on my craft projects and greeting cards." (27 words)

OUTPUT (just the query, 20-35 words):"""

    return prompt


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def validate_error_words_in_query(query: str, 
                                   selected_attrs: List[Tuple[str, str]],
                                   user_error_patterns: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """
    验证包含用户拼写错误模式的单词是否原封不动地出现在查询中
    
    只检查那些属性值中包含用户错误词的单词，不检查所有属性值。
    
    Args:
        query: 生成的查询文本
        selected_attrs: 选中的属性列表 [(dimension, value), ...]
        user_error_patterns: 用户错误模式 {正确词: [错误词1, ...]}
    
    Returns:
        (是否全部出现, 缺失的错误词列表)
    """
    if not query or not selected_attrs or not user_error_patterns:
        return True, []  # 没有错误词需要验证时，返回通过
    
    query_lower = query.lower()
    missing_words = []
    
    for dim, attr_value in selected_attrs:
        # 检查该属性值是否包含用户错误词
        error_matches = contains_user_error_word(attr_value, user_error_patterns)
        
        if error_matches:
            # 该属性包含错误词，需要验证这些词是否在查询中
            for correct_word, matched_word, match_type in error_matches:
                # 检查匹配的词是否在查询中（只用精确匹配，保持性能）
                if matched_word.lower() not in query_lower:
                    missing_words.append(f"{matched_word} (from: {attr_value})")
    
    all_present = len(missing_words) == 0
    return all_present, missing_words


def generate_query_with_retry(llm_client: LLMClient,
                               selected_attrs: List[Tuple[str, str]],
                               category: str,
                               query_type: str,
                               user_error_patterns: Dict[str, List[str]],
                               max_retries: int = 5) -> Tuple[str, int, int, bool, List[str]]:
    """
    Generate query with word count validation, error word presence validation, and retry.
    
    Returns:
        (query, word_count, attempts, all_error_words_present, missing_words)
    """
    prev_word_count = None
    prev_missing_words = None
    query = ""
    word_count = 0
    all_error_words_present = False
    missing_words = []

    for attempt in range(max_retries):
        if query_type == 'target_user':
            prompt = generate_target_user_query_prompt(selected_attrs, category, prev_word_count, prev_missing_words)
        else:
            prompt = generate_mass_market_query_prompt(selected_attrs, category, prev_word_count, prev_missing_words)

        response = llm_client.call(prompt, max_tokens=256, temperature=0.7)
        if not response:
            continue
        
        query = response.strip()

        # Extract query from <think> tags if present
        think_match = re.search(r'<think>(.*?)</think>', query, re.DOTALL)
        if think_match:
            think_content = think_match.group(1)
            quoted_matches = re.findall(r'"([^"]+)"', think_content)
            if quoted_matches:
                for quoted in quoted_matches:
                    if len(quoted.split()) >= 15:
                        query = quoted
                        break
        
        # Remove remaining <think> tags
        query = re.sub(r'<think>.*?</think>', '', query, flags=re.DOTALL).strip()
        
        # Remove surrounding quotes
        query = re.sub(r'^["\']|["\']$', '', query)
        
        # Remove common prefixes
        for prefix in ['output:', 'query:', 'answer:', 'the query is:']:
            if query.lower().startswith(prefix):
                query = query.split(':', 1)[1].strip()
        
        # Remove word count annotations
        query = re.sub(r'\s*\(\d+\s*words?\)\s*$', '', query, flags=re.IGNORECASE).strip()

        word_count = count_words(query)
        
        # Validate error words are present in query
        all_error_words_present, missing_words = validate_error_words_in_query(
            query, selected_attrs, user_error_patterns
        )

        # Check if both word count and error word presence are satisfied
        if 20 <= word_count <= 35 and all_error_words_present:
            return query, word_count, attempt + 1, True, []
        
        # Log validation failure for debugging
        if not all_error_words_present:
            log_with_timestamp(f"    Attempt {attempt + 1}: Missing error words in query: {missing_words}")

        # Update previous values for next iteration
        prev_word_count = word_count
        prev_missing_words = missing_words if not all_error_words_present else None

    # If all retries failed, return the last attempt with validation status
    return query, word_count, max_retries, all_error_words_present, missing_words


def process_query_set(query_set: Dict,
                       llm_client: LLMClient,
                       user_id: str,
                       improved_selector: ImprovedPrioritySelector) -> Dict:
    """Process a single query set using improved priority selection."""
    asin = query_set.get('asin', 'unknown')
    category = query_set.get('category', 'general')
    target_user_attrs = query_set.get('target_attributes', {}).get('selected_attributes', [])
    # MASS MARKET QUERY DISABLED - Only generating target user queries
    # mass_market_attrs = query_set.get('public_attributes', {}).get('selected_attributes', [])

    # Extract attributes by dimension
    target_user_by_dim = extract_attributes_by_dimension(target_user_attrs)
    # MASS MARKET QUERY DISABLED
    # mass_market_by_dim = extract_attributes_by_dimension(mass_market_attrs)

    log_with_timestamp(f"  Processing ASIN: {asin}")

    # Select attributes using improved priority selection
    target_user_selected, tu_stats = improved_selector.select_for_product(
        target_user_by_dim, asin, num_dimensions=5
    )

    # MASS MARKET QUERY DISABLED
    # mass_market_selected, mm_stats = improved_selector.select_for_product(
    #     mass_market_by_dim, asin, num_dimensions=5
    # )

    # Get user error patterns from selector
    user_error_patterns = improved_selector.user_error_patterns

    # Generate queries with error word validation
    (target_user_query, tu_word_count, tu_attempts,
     tu_error_words_valid, tu_missing_words) = generate_query_with_retry(
        llm_client, target_user_selected, category, 'target_user', user_error_patterns
    )

    # MASS MARKET QUERY DISABLED
    # (mass_market_query, mm_word_count, mm_attempts,
    #  mm_error_words_valid, mm_missing_words) = generate_query_with_retry(
    #     llm_client, mass_market_selected, category, 'mass_market', user_error_patterns
    # )

    # Build result with validation status and priority tracking
    result = {
        'asin': asin,
        'category': category,
        'user_id': user_id,
        'shared_dimensions': [dim for dim, _ in target_user_selected],
        'target_user_query': {
            'query': target_user_query,
            'word_count': tu_word_count,
            'attempts': tu_attempts,
            'error_words_valid': tu_error_words_valid,
            'missing_error_words': tu_missing_words if not tu_error_words_valid else [],
            'selected_attributes': [
                {'dimension': dim, 'value': attr}
                for dim, attr in target_user_selected
            ],
            'attribute_priority_tracking': tu_stats.get('selection_record', [])
        },
        # MASS MARKET QUERY DISABLED
        # 'mass_market_query': {
        #     'query': mass_market_query,
        #     'word_count': mm_word_count,
        #     'attempts': mm_attempts,
        #     'error_words_valid': mm_error_words_valid,
        #     'missing_error_words': mm_missing_words if not mm_error_words_valid else [],
        #     'selected_attributes': [
        #         {'dimension': dim, 'value': attr}
        #         for dim, attr in mass_market_selected
        #     ]
        # }
    }

    # Log with validation status
    tu_status = "✓" if tu_error_words_valid else "✗"
    # MASS MARKET QUERY DISABLED
    # mm_status = "✓" if mm_error_words_valid else "✗"
    log_with_timestamp(
        f"  ASIN {asin}: TU={tu_word_count}w {tu_status}"
    )
    if not tu_error_words_valid:
        log_with_timestamp(f"    TU missing error words: {tu_missing_words}")
    # MASS MARKET QUERY DISABLED
    # if not mm_error_words_valid:
    #     log_with_timestamp(f"    MM missing error words: {mm_missing_words}")

    return result


def process_query_result_wrapper(args: Tuple) -> Tuple[int, Dict]:
    """Wrapper for parallel processing."""
    index, query_set, llm_client, user_id, improved_selector = args
    result = process_query_set(query_set, llm_client, user_id, improved_selector)
    return index, result


def main():
    parser = argparse.ArgumentParser(
        description="Generate target user queries only (mass market generation disabled)"
    )
    parser.add_argument(
        "--input-file",
        default="/fs04/ar57/wenyu/result/personal_query/03_processing/query_A13OFOB1394G31.json",
        help="Input JSON file from Stage 3"
    )
    parser.add_argument(
        "--output-dir",
        default="/fs04/ar57/wenyu/result/personal_query/07_query",
        help="Output directory for generated queries"
    )
    parser.add_argument(
        "--writing-analysis-file",
        default="/fs04/ar57/wenyu/result/personal_query/05_writing_analysis/results/writing_analysis_A13OFOB1394G31.json",
        help="User writing analysis file from Stage 5"
    )
    parser.add_argument(
        "--meta-file",
        default="/fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz",
        help="Product metadata file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)"
    )
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        log_with_timestamp(f"Random seed set to: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input file
    log_with_timestamp(f"Loading input file: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    user_id = data.get('user_id', 'Unknown')
    query_results = data.get('query_results', [])
    
    log_with_timestamp(f"User ID: {user_id}")
    log_with_timestamp(f"Total query results: {len(query_results)}")
    log_with_timestamp(f"Using {args.workers} concurrent workers")
    
    # Initialize improved priority selector
    log_with_timestamp("Initializing improved priority selector...")
    improved_selector = ImprovedPrioritySelector(args.meta_file, args.writing_analysis_file)
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Process query results concurrently
    results_dict = {}
    success_target = 0
    # MASS MARKET QUERY DISABLED
    # success_mass_market = 0
    valid_target_error_words = 0
    # MASS MARKET QUERY DISABLED
    # valid_mass_market_error_words = 0
    
    # Prepare arguments for parallel processing
    tasks = [
        (i, query_set, llm_client, user_id, improved_selector) 
        for i, query_set in enumerate(query_results)
    ]
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_query_result_wrapper, task): task[0] 
            for task in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            try:
                index, result = future.result()
                results_dict[index] = result

                if result.get('target_user_query', {}).get('query'):
                    success_target += 1
                    if result.get('target_user_query', {}).get('error_words_valid', False):
                        valid_target_error_words += 1

                # MASS MARKET QUERY DISABLED
                # if result.get('mass_market_query', {}).get('query'):
                #     success_mass_market += 1
                #     if result.get('mass_market_query', {}).get('error_words_valid', False):
                #         valid_mass_market_error_words += 1

            except Exception as e:
                index = future_to_index[future]
                log_with_timestamp(f"  Error processing item {index}: {e}")
                results_dict[index] = {
                    "asin": "Error",
                    "error": str(e)
                }
    
    # Convert dict to list in order
    results = [results_dict[i] for i in range(len(query_results))]
    
    # Build output data with validation statistics
    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "input_file": args.input_file,
        "total_queries": len(results),
        "successful_target_queries": success_target,
        # MASS MARKET QUERY DISABLED
        # "successful_mass_market_queries": success_mass_market,
        "valid_target_error_words": valid_target_error_words,
        # MASS MARKET QUERY DISABLED
        # "valid_mass_market_error_words": valid_mass_market_error_words,
        "results": results
    }

    # Save output
    output_file = os.path.join(args.output_dir, f"dual_queries_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Calculate validation rates
    target_valid_rate = (valid_target_error_words / success_target * 100) if success_target > 0 else 0
    # MASS MARKET QUERY DISABLED
    # mass_market_valid_rate = (valid_mass_market_error_words / success_mass_market * 100) if success_mass_market > 0 else 0

    log_with_timestamp(f"\n{'='*60}")
    log_with_timestamp(f"Summary:")
    log_with_timestamp(f"  Total queries processed: {len(results)}")
    log_with_timestamp(f"  Successful target user queries: {success_target}")
    # MASS MARKET QUERY DISABLED
    # log_with_timestamp(f"  Successful mass market queries: {success_mass_market}")
    log_with_timestamp(f"")
    log_with_timestamp(f"  Error Word Validation:")
    log_with_timestamp(f"    Target user queries with all error words: {valid_target_error_words}/{success_target} ({target_valid_rate:.1f}%)")
    # MASS MARKET QUERY DISABLED
    # log_with_timestamp(f"    Mass market queries with all error words: {valid_mass_market_error_words}/{success_mass_market} ({mass_market_valid_rate:.1f}%)")
    log_with_timestamp(f"")
    log_with_timestamp(f"  Output saved to: {output_file}")
    log_with_timestamp(f"{'='*60}")


if __name__ == "__main__":
    main()
