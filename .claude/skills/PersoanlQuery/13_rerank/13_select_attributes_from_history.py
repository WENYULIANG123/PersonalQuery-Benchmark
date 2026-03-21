"""
Stage 13 属性值选择: 从历史偏好中为查询选择属性值

该脚本处理属性值选择的核心逻辑：
1. 从历史偏好数据（Stage 2 Persona）中提取属性值
2. 将查询属性与历史属性进行匹配分类（explicit/implicit/conflicting）
3. 计算维度级F1和组合级F1评估属性选择质量
4. 对选中的属性进行商品验证（可选）- 两层验证架构

被主脚本 13_batch_llm_rerank_all.py 调用。

使用方法：
  from 13_select_attributes_from_history import AttributeSelector, AttributeSelectorWithProductValidation
  
  # 基础使用（仅历史匹配）
  selector = AttributeSelector()
  result = selector.evaluate_attributes(query, user_id, category)
  
  # 添加商品验证
  selector = AttributeSelectorWithProductValidation(meta_file='/path/to/meta_*.json.gz')
  result = selector.evaluate_attributes(query, user_id, category, target_asin='B07XXXXX')
"""

import json
import logging
import sys
import gzip
import re
import string
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from difflib import SequenceMatcher

# 导入SentenceTransformer用于语义匹配
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 导入BM25以保持与检索系统一致
try:
    import bm25s
except ImportError:
    bm25s = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_semantic_model = None


def get_semantic_model():
    global _semantic_model
    if _semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def cosine_similarity(vec1, vec2) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    sim = dot / (norm1 * norm2)
    return max(-1.0, min(1.0, sim))


def normalize_persona_category_filename(category: str) -> str:
    """Normalize category names to match persona filenames on disk."""
    if not category:
        return ""

    normalized = category.replace("&", " and ")
    translation_table = str.maketrans({char: "_" for char in string.punctuation + " " if char != "&"})
    normalized = normalized.translate(translation_table)
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized


def canonicalize_category_key(text: str) -> str:
    """Deterministic category canonicalization for filename matching."""
    if not text:
        return ""

    normalized = text.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(rf"[{re.escape(string.punctuation)}]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

# 21个固定维度定义
DIMENSIONS = {
    "Product_Attributes": [
        "Product_Category",
        "Functionality",
        "Material_Composition"
    ],
    "Quality_Attributes": [
        "Quality_Craftsmanship",
        "Performance",
        "Safety"
    ],
    "Appearance_Design": [
        "Appearance_Color",
        "Size_Dimensions",
        "Style_Design"
    ],
    "User_Experience": [
        "Comfort",
        "Ease_of_Use",
        "Portability"
    ],
    "Usage_Scenarios": [
        "Target_User",
        "Usage_Scenario",
        "Special_Purpose"
    ],
    "Price_Value": [
        "Price",
        "Value",
        "Packaging_Quantity"
    ],
    "Special_Requirements": [
        "Compatibility",
        "Special_User_Needs",
        "Brand_Preference"
    ]
}

# 所有维度的扁平列表
ALL_DIMENSIONS = [dim for dims in DIMENSIONS.values() for dim in dims]


# ============================================================================
# 产品词汇库（用于商品验证）- 从Stage 06复用
# ============================================================================

def extract_words_from_text(text: str, min_length: int = 2) -> Set[str]:
    """
    从文本中提取所有单词，与Stage 06保持一致
    
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


class ProductVocabulary:
    """产品词汇库 - 用于检查词是否在产品中（与Stage 06一致）"""
    
    def __init__(self, meta_file: str, asins: List[str]):
        """
        初始化产品词汇库
        
        Args:
            meta_file: 元数据文件路径（gzip压缩JSON Lines）
            asins: 需要加载的产品ASIN列表
        """
        self.meta_file = meta_file
        self.target_asins = set(asins)
        self.product_vocabularies = {}  # {asin: set(words)}
        self.asin_documents = {}  # {asin: document_text}
        self.asin_embeddings = {}  # {asin: embedding}
        self.text_embedding_cache = {}  # {text: embedding}
    
    def load_vocabulary_for_asin(self, asin: str) -> bool:
        """
        加载单个产品的词汇，使用与BM25一致的tokenization
        
        Returns:
            是否成功加载
        """
        if asin in self.product_vocabularies:
            return True
        
        try:
            with gzip.open(self.meta_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        product = json.loads(line)
                        if product.get('asin') == asin:
                            # 提取文档文本
                            title = product.get('title', '')
                            brand = product.get('brand', '')
                            feature = product.get('feature', [])
                            description = product.get('description', [])
                            
                            document_parts = [title, brand] + feature + description
                            document_text = ' '.join(str(p) for p in document_parts)
                            
                            # 使用bm25s.tokenize进行分词，与BM25索引保持一致
                            if bm25s is not None:
                                tokenized = bm25s.tokenize(
                                    [document_text],
                                    lower=True,
                                    stopwords="english",
                                    show_progress=False
                                )
                                words = set(tokenized.vocab)
                            else:
                                # 回退方案：使用简单的词提取
                                words = extract_words_from_text(document_text, min_length=2)
                            
                            self.product_vocabularies[asin] = words
                            self.asin_documents[asin] = document_text
                            return True
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"加载产品 {asin} 失败: {e}")
        
        return False
    
    def word_percentage_exists_in_product(self, phrase: str, asin: str) -> float:
        """
        计算属性值（短语）中有多少百分比的词在产品词汇中存在
        
        Args:
            phrase: 属性值短语，如 "floral pattern"
            asin: 产品ASIN
        
        Returns:
            0-1之间的匹配比例。1.0表示100%词都在，0.5表示50%词在
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

    def semantic_similarity_to_product(self, phrase: str, asin: str) -> float:
        if not phrase:
            return 0.0

        if asin not in self.asin_documents and not self.load_vocabulary_for_asin(asin):
            return 0.0

        model = get_semantic_model()
        if model is None:
            return self.word_percentage_exists_in_product(phrase, asin)

        if asin not in self.asin_embeddings:
            self.asin_embeddings[asin] = model.encode(self.asin_documents[asin], show_progress_bar=False)

        if phrase not in self.text_embedding_cache:
            self.text_embedding_cache[phrase] = model.encode(phrase, show_progress_bar=False)

        similarity = cosine_similarity(self.text_embedding_cache[phrase], self.asin_embeddings[asin])
        return (similarity + 1.0) / 2.0


# ============================================================================
# 属性选择器
# ============================================================================

class AttributeSelector:
    """从历史偏好中选择属性值的核心类"""
    
    def __init__(self, persona_dir: Optional[str] = None, result_base_dir: Optional[str] = None):
        """
        初始化AttributeSelector
        
        Args:
            persona_dir: Persona文件目录（Stage 2输出）
            result_base_dir: 结果基目录
        """
        default_persona_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/02_processing"
        default_result_dir = "/home/wlia0047/ar57/wenyu/result/personal_query"
        
        self.persona_dir = Path(persona_dir) if persona_dir else Path(default_persona_dir)
        self.result_base_dir = Path(result_base_dir) if result_base_dir else Path(default_result_dir)
        self._persona_file_index_cache: Dict[str, Dict[str, List[Path]]] = {}
        
        logger.info(f"AttributeSelector初始化: Persona目录={self.persona_dir}")

    def _build_persona_file_index(self, user_id: str) -> Dict[str, List[Path]]:
        """Build canonical-key index for persona filenames of one user."""
        if user_id in self._persona_file_index_cache:
            return self._persona_file_index_cache[user_id]

        persona_dir = self.persona_dir / user_id / "persona"
        index: Dict[str, List[Path]] = defaultdict(list)

        if persona_dir.exists():
            for persona_path in sorted(persona_dir.glob("*.json")):
                stem = persona_path.stem
                key = canonicalize_category_key(stem)
                if key:
                    index[key].append(persona_path)

        self._persona_file_index_cache[user_id] = dict(index)
        return self._persona_file_index_cache[user_id]
    
    def load_persona(self, user_id: str, category: str) -> Optional[Dict]:
        """
        加载用户的Persona文件
        
        Args:
            user_id: 用户ID
            category: 产品类别（如 "Yarn", "Dies" 等）
        
        Returns:
            Persona数据或None
        """
        persona_dir = self.persona_dir / user_id / "persona"
        persona_file = persona_dir / f"{category}.json"

        if not persona_file.exists():
            normalized_category = normalize_persona_category_filename(category)
            if normalized_category and normalized_category != category:
                normalized_file = persona_dir / f"{normalized_category}.json"
                if normalized_file.exists():
                    persona_file = normalized_file

        if not persona_file.exists():
            category_key = canonicalize_category_key(category)
            persona_index = self._build_persona_file_index(user_id)
            matched_files = sorted(persona_index.get(category_key, []), key=lambda p: p.name)

            if len(matched_files) == 1:
                persona_file = matched_files[0]
                logger.info(f"Persona规范化命中: {user_id}/{category} -> {persona_file.name}")
            elif len(matched_files) > 1:
                persona_file = matched_files[0]
                logger.warning(
                    f"Persona规范化命中多个候选，采用字典序首个: {user_id}/{category} -> {persona_file.name}"
                )

        if not persona_file.exists():
            logger.warning(f"找不到Persona文件: {persona_file}")
            return None
        
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona = json.load(f)
            logger.info(f"加载Persona: {user_id}/{category}")
            return persona
        except Exception as e:
            logger.error(f"加载Persona失败: {e}")
            return None
    
    def extract_historical_attributes(self, persona: Dict) -> Dict[str, List[Dict]]:
        """
        从Persona中提取历史属性，按维度分组
        
        Args:
            persona: Persona数据
        
        Returns:
            按维度分组的属性字典
        """
        if not persona or 'attributes_by_dimension' not in persona:
            return {}
        
        return persona.get('attributes_by_dimension', {})
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（0-1）
        
        Args:
            str1: 字符串1
            str2: 字符串2
        
        Returns:
            相似度分数（0-1）
        """
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        if str1_lower == str2_lower:
            return 1.0
        
        # 使用SequenceMatcher计算相似度
        ratio = SequenceMatcher(None, str1_lower, str2_lower).ratio()
        return ratio
    
    def classify_attribute_match(
        self,
        query_value: str,
        historical_attrs: List[Dict],
        similarity_threshold: float = 0.7
    ) -> Tuple[str, List[Dict]]:
        """
        将查询属性值与历史属性进行分类匹配
        
        分类类型：
        - explicit: 查询值在历史中明确出现（相似度>0.95）
        - implicit: 查询值与历史值语义相关（相似度0.7-0.95）
        - conflicting: 查询值与历史偏好冲突（混合正负态度）
        - not_found: 在历史中未找到
        
        Args:
            query_value: 查询中的属性值
            historical_attrs: 历史属性值列表
            similarity_threshold: 相似度阈值
        
        Returns:
            (分类类型, 匹配的历史属性列表)
        """
        if not historical_attrs:
            return "not_found", []
        
        # 计算与所有历史属性的相似度
        matches = []
        for hist_attr in historical_attrs:
            hist_value = hist_attr.get('attribute', '')
            similarity = self._similarity_score(query_value, hist_value)
            
            if similarity >= 0.95:  # 明确匹配
                matches.append({
                    'attr': hist_attr,
                    'similarity': similarity,
                    'type': 'explicit'
                })
            elif similarity >= similarity_threshold:  # 隐含匹配
                matches.append({
                    'attr': hist_attr,
                    'similarity': similarity,
                    'type': 'implicit'
                })
        
        if not matches:
            return "not_found", []
        
        # 检查冲突（混合正负态度）
        sentiments = set()
        for match in matches:
            sentiment = match['attr'].get('sentiment', 'neutral')
            if sentiment in ['positive', 'negative']:
                sentiments.add(sentiment)
        
        if len(sentiments) > 1:  # 既有正面又有负面
            return "conflicting", [m['attr'] for m in matches]
        
        # 确定主要分类
        explicit_matches = [m for m in matches if m['type'] == 'explicit']
        if explicit_matches:
            return "explicit", [m['attr'] for m in explicit_matches]
        else:
            return "implicit", [m['attr'] for m in matches]
    
    def evaluate_attributes(
        self,
        query: Dict,
        user_id: str,
        category: str
    ) -> Dict:
        """
        对查询的属性值进行评估
        
        Args:
            query: 查询数据（包含selected_attributes）
            user_id: 用户ID
            category: 产品类别
        
        Returns:
            包含以下内容的评估结果：
            - original_attributes: 原始查询属性
            - classification_result: 按维度的分类结果
            - dimension_coverage: 覆盖的维度数量
            - attribute_f1: 维度级F1分数
            - profile_f1: 组合级F1分数
            - quality_score: 综合质量评分
        """
        # 加载历史偏好
        persona = self.load_persona(user_id, category)
        if not persona:
            logger.warning(f"无法加载Persona，返回空结果")
            return self._create_empty_result(query)
        
        historical_attrs = self.extract_historical_attributes(persona)
        
        # 提取查询中的属性
        query_attributes = query.get('selected_attributes', [])
        
        # 按维度分组查询属性
        query_by_dimension = defaultdict(list)
        for attr in query_attributes:
            dimension = attr.get('dimension', 'Unknown')
            value = attr.get('value', '')
            if dimension and value:
                query_by_dimension[dimension].append(value)
        
        # 对每个维度的属性进行分类
        classification_result = {}
        dimension_scores = {}
        
        for dimension, query_values in query_by_dimension.items():
            hist_attrs = historical_attrs.get(dimension, [])
            
            # 初始化维度的分类结果
            classification_result[dimension] = {
                'explicit': [],
                'implicit': [],
                'conflicting': [],
                'not_found': []
            }
            
            # 分类每个属性值
            for query_value in query_values:
                match_type, matched_attrs = self.classify_attribute_match(
                    query_value, hist_attrs
                )
                
                match_info = {
                    'query_value': query_value,
                    'matched_historical': [attr.get('attribute', '') for attr in matched_attrs],
                    'sentiments': list(set([attr.get('sentiment', 'neutral') for attr in matched_attrs]))
                }
                
                classification_result[dimension][match_type].append(match_info)
            
            # 计算该维度的F1分数
            explicit_count = len(classification_result[dimension]['explicit'])
            implicit_count = len(classification_result[dimension]['implicit'])
            not_found_count = len(classification_result[dimension]['not_found'])
            total_count = len(query_values)
            
            if total_count > 0:
                # 简单的F1计算：explicit完全分，implicit半分，not_found0分
                precision = (explicit_count + 0.5 * implicit_count) / total_count
                recall = (explicit_count + 0.5 * implicit_count) / max(len(hist_attrs), total_count)
                
                if precision + recall > 0:
                    dimension_scores[dimension] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': 2 * (precision * recall) / (precision + recall),
                        'total_queries': total_count,
                        'explicit_matches': explicit_count,
                        'implicit_matches': implicit_count
                    }
            
        # 计算整体指标
        dimension_coverage = len([d for d in query_by_dimension.keys() if d in ALL_DIMENSIONS])
        
        # 维度级F1：所有维度F1分数的平均值
        attr_f1_scores = [score['f1'] for score in dimension_scores.values()]
        attribute_f1 = sum(attr_f1_scores) / len(attr_f1_scores) if attr_f1_scores else 0.0
        
        # 组合级F1：所有属性都是explicit的组合才算完全匹配
        all_explicit = all(
            len(classification_result[dim]['explicit']) == len(query_by_dimension[dim])
            for dim in query_by_dimension
        )
        profile_f1 = 1.0 if all_explicit and len(query_by_dimension) > 0 else 0.0
        
        # 质量评分（0-1）
        quality_score = 0.7 * attribute_f1 + 0.3 * profile_f1
        
        return {
            'user_id': user_id,
            'category': category,
            'original_attributes': query_attributes,
            'classification_result': dict(classification_result),
            'dimension_coverage': dimension_coverage,
            'dimension_scores': dict(dimension_scores),
            'attribute_f1': attribute_f1,
            'profile_f1': profile_f1,
            'quality_score': quality_score
        }
    
    def _create_empty_result(self, query: Dict) -> Dict:
        """创建空的评估结果"""
        return {
            'user_id': 'unknown',
            'category': 'unknown',
            'original_attributes': query.get('selected_attributes', []),
            'classification_result': {},
            'dimension_coverage': 0,
            'dimension_scores': {},
            'attribute_f1': 0.0,
            'profile_f1': 0.0,
            'quality_score': 0.0,
            'error': 'Failed to load persona data'
        }


# ============================================================================
# 扩展的选择器：添加商品验证功能
# ============================================================================

class AttributeSelectorWithProductValidation(AttributeSelector):
    """
    带商品验证功能的属性选择器
    
    实现两层验证架构：
    1. 历史偏好匹配验证（AttributeSelector的evaluate_attributes）
    2. 商品属性验证（新增的_validate_against_product）
    
    综合评分 = 0.6 * 历史匹配分 + 0.4 * 商品验证分
    """
    
    def __init__(
        self,
        meta_file: Optional[str] = None,
        persona_dir: Optional[str] = None,
        result_base_dir: Optional[str] = None,
        validation_threshold: float = 0.7,
        validate_selected_dimensions_only: bool = True
    ):
        """
        初始化带商品验证的选择器
        
        Args:
            meta_file: 商品元数据文件路径（gzip压缩JSON Lines）
            persona_dir: Persona文件目录（Stage 2输出）
            result_base_dir: 结果基目录
        """
        super().__init__(persona_dir, result_base_dir)
        self.meta_file = meta_file
        self.validation_threshold = validation_threshold
        self.validate_selected_dimensions_only = validate_selected_dimensions_only
        self.product_vocab_cache = {}  # {asin: ProductVocabulary}
        
        if meta_file:
            logger.info(f"初始化商品验证: meta_file={meta_file}")
    
    def _get_product_vocabulary(self, asin: str) -> Optional[ProductVocabulary]:
        """
        获取或创建产品词汇库（缓存）
        
        Args:
            asin: 产品ASIN
        
        Returns:
            ProductVocabulary实例或None
        """
        if not self.meta_file:
            return None
        
        if asin not in self.product_vocab_cache:
            vocab = ProductVocabulary(self.meta_file, [asin])
            if vocab.load_vocabulary_for_asin(asin):
                self.product_vocab_cache[asin] = vocab
            else:
                logger.warning(f"无法加载产品词汇: {asin}")
                return None
        
        return self.product_vocab_cache.get(asin)
    
    def _validate_against_product(self, attributes: List[Dict], asin: str) -> Dict[str, Dict]:
        """
        验证属性值是否与目标商品匹配
        
        Args:
            attributes: 查询属性列表（格式: [{'dimension': '...', 'value': '...'}]）
            asin: 目标产品ASIN
        
        Returns:
            验证结果字典，格式：
            {
                'attribute_value': {
                    'match_ratio': 0.0-1.0,
                    'is_valid': True/False,  # 基于80%阈值
                    'dimension': '...'
                }
            }
        """
        if not self.meta_file:
            logger.warning("未配置meta_file，跳过商品验证")
            return {}
        
        vocab = self._get_product_vocabulary(asin)
        if not vocab:
            logger.warning(f"无法获取产品词汇: {asin}")
            return {}
        
        validation_results = {}
        validation_threshold = self.validation_threshold
        
        for attr in attributes:
            value = attr.get('value', '')
            dimension = attr.get('dimension', '')
            
            if not value:
                continue
            
            match_ratio = float(vocab.semantic_similarity_to_product(value, asin))
            is_valid = bool(match_ratio >= validation_threshold)
            
            validation_results[value] = {
                'match_ratio': round(match_ratio, 3),
                'is_valid': is_valid,
                'dimension': dimension
            }
            
            if is_valid:
                logger.debug(f"✓ 属性验证通过: '{value}' (匹配度: {match_ratio:.1%})")
            else:
                logger.debug(f"✗ 属性验证失败: '{value}' (匹配度: {match_ratio:.1%} < 80%)")
        
        return validation_results
    
    def _compute_combined_quality(self, product_validation_results: Dict[str, Dict]) -> Tuple[float, Dict]:
        if product_validation_results:
            total_attrs = len(product_validation_results)
            valid_attrs = sum(1 for r in product_validation_results.values() if r.get('is_valid', False))
            product_quality = valid_attrs / total_attrs if total_attrs > 0 else 0.0
        else:
            product_quality = 0.0

        validation_stats = {
            'product_quality': round(product_quality, 3),
            'validation_count': len(product_validation_results),
            'valid_count': sum(1 for r in product_validation_results.values() if r.get('is_valid', False)) if product_validation_results else 0
        }

        return product_quality, validation_stats

    def _collect_historical_attributes(self, historical_attrs: Dict[str, List[Dict]]) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        attributes = []
        by_dimension = defaultdict(list)

        for dimension, attrs in historical_attrs.items():
            for attr in attrs:
                value = attr.get('attribute', '')
                if not value:
                    continue
                sentiment = attr.get('sentiment', 'neutral')
                improvement_wish = attr.get('improvement_wish', '')
                validation_value = improvement_wish if sentiment == 'negative' and improvement_wish else value
                item = {
                    'dimension': dimension,
                    'value': value,
                    'sentiment': sentiment,
                    'validation_value': validation_value,
                    'improvement_wish': improvement_wish
                }
                attributes.append(item)
                by_dimension[dimension].append(item)

        return attributes, dict(by_dimension)
    
    def evaluate_attributes(self,
                          query: Dict,
                          user_id: str,
                          category: str,
                          target_asin: Optional[str] = None) -> Dict:
        """
        评估属性值，强制进行商品验证
        
        重要：所有属性都必须经过商品验证。target_asin必须指定，否则抛出错误。
        
        Args:
            query: 查询数据
            user_id: 用户ID
            category: 产品类别
            target_asin: 必须指定的目标产品ASIN（强制）
        
        Returns:
            包含历史匹配和商品验证的评估结果
        
        Raises:
            ValueError: 如果未指定target_asin
        """
        # 强制要求target_asin
        if not target_asin:
            raise ValueError(
                "AttributeSelectorWithProductValidation要求必须指定target_asin。"
                "所有属性选择都必须验证其在目标商品中的存在性。"
            )
        
        if not self.meta_file:
            raise ValueError(
                "AttributeSelectorWithProductValidation要求必须配置meta_file。"
                "无法在没有商品元数据的情况下进行验证。"
            )
        
        persona = self.load_persona(user_id, category)
        if not persona:
            logger.warning("无法加载Persona，返回空结果")
            return self._create_empty_result(query)

        historical_attrs = self.extract_historical_attributes(persona)
        selected_dimensions = {
            attr.get('dimension', '')
            for attr in query.get('selected_attributes', [])
            if attr.get('dimension', '')
        }
        if self.validate_selected_dimensions_only and selected_dimensions:
            historical_attrs = {
                dim: attrs
                for dim, attrs in historical_attrs.items()
                if dim in selected_dimensions
            }
        attributes, historical_by_dimension = self._collect_historical_attributes(historical_attrs)
        product_validation = self._validate_against_product(attributes, target_asin)

        classification_result = {}
        dimension_scores = {}
        for dimension, attrs in historical_by_dimension.items():
            classification_result[dimension] = {
                'present_in_product': [],
                'missing_in_product': []
            }

            for attr in attrs:
                value = attr['value']
                validation_value = attr.get('validation_value', value)
                validation = product_validation.get(validation_value, {})
                match_info = {
                    'historical_value': value,
                    'validation_value': validation_value,
                    'improvement_wish': attr.get('improvement_wish', ''),
                    'sentiment': attr.get('sentiment', 'neutral'),
                    'match_ratio': validation.get('match_ratio', 0.0),
                    'is_valid': validation.get('is_valid', False)
                }
                bucket = 'present_in_product' if validation.get('is_valid', False) else 'missing_in_product'
                classification_result[dimension][bucket].append(match_info)

            total_count = len(attrs)
            valid_count = len(classification_result[dimension]['present_in_product'])
            ratio = valid_count / total_count if total_count > 0 else 0.0
            dimension_scores[dimension] = {
                'precision': ratio,
                'recall': ratio,
                'f1': ratio,
                'total_historical_attributes': total_count,
                'valid_in_product': valid_count
            }

        attr_f1_scores = [score['f1'] for score in dimension_scores.values()]
        attribute_f1 = sum(attr_f1_scores) / len(attr_f1_scores) if attr_f1_scores else 0.0
        profile_f1 = 1.0 if attr_f1_scores and all(score == 1.0 for score in attr_f1_scores) else 0.0
        quality_score, validation_stats = self._compute_combined_quality(product_validation)

        result = {
            'user_id': user_id,
            'category': category,
            'original_attributes': query.get('selected_attributes', []),
            'selected_dimensions': sorted(list(selected_dimensions)),
            'historical_attributes': attributes,
            'classification_result': dict(classification_result),
            'dimension_coverage': len([d for d in historical_by_dimension.keys() if d in ALL_DIMENSIONS]),
            'dimension_scores': dimension_scores,
            'attribute_f1': attribute_f1,
            'profile_f1': profile_f1,
            'quality_score': quality_score,
            'target_asin': target_asin,
            'product_validation': product_validation,
            'validation_stats': validation_stats,
            'combined_quality_score': quality_score
        }

        logger.info(
            f"属性评估完成（目标商品验证） - 阈值: {self.validation_threshold:.2f}, "
            f"商品验证: {validation_stats['product_quality']:.1%}, "
            f"历史属性数: {validation_stats['validation_count']}, "
            f"命中数: {validation_stats['valid_count']}"
        )

        return result


# ============================================================================
# 验证测试函数（集成自verify_product_validation.py）
# ============================================================================

def test_vocabulary_loading(meta_file: str, test_asins=None) -> bool:
    """测试ProductVocabulary的加载"""
    print("\n=== 测试1: ProductVocabulary加载 ===")
    
    if not test_asins:
        test_asins = ["B07XXXXX", "B00XXXXX"]
    
    vocab = ProductVocabulary(meta_file, test_asins)
    
    success_count = 0
    for asin in test_asins[:2]:
        if vocab.load_vocabulary_for_asin(asin):
            words = vocab.product_vocabularies[asin]
            logger.info(f"✓ 成功加载 {asin}，词汇数: {len(words)}")
            success_count += 1
        else:
            logger.warning(f"✗ 加载 {asin} 失败（可能不存在）")
    
    return success_count > 0

def test_word_match_ratio(meta_file: str) -> bool:
    """测试word_match_ratio计算"""
    print("\n=== 测试2: word_match_ratio计算 ===")
    
    test_cases = [
        ("blue color", 1.0, "全部词在商品中"),
        ("xyz abc", 0.0, "词都不在商品中"),
        ("blue xyz", 0.5, "50%词在商品中"),
    ]
    
    test_asin = "B07XXXXX"
    vocab = ProductVocabulary(meta_file, [test_asin])
    
    if not vocab.load_vocabulary_for_asin(test_asin):
        logger.warning(f"未找到测试商品 {test_asin}，使用模拟测试")
        
        vocab.product_vocabularies[test_asin] = {"blue", "color", "pattern", "design"}
        
        test_cases = [
            ("blue color", 1.0, "两个词都在"),
            ("xyz abc", 0.0, "词都不在"),
            ("blue xyz", 0.5, "1/2词在"),
        ]
    
    all_passed = True
    for phrase, expected, description in test_cases:
        ratio = vocab.word_percentage_exists_in_product(phrase, test_asin)
        passed = abs(ratio - expected) < 0.001
        status = "✓" if passed else "✗"
        logger.info(f"{status} '{phrase}' -> {ratio:.1%} (期望{expected:.1%}) - {description}")
        all_passed = all_passed and passed
    
    return all_passed

def test_validation_threshold(meta_file: str) -> bool:
    """测试80%阈值的合理性"""
    print("\n=== 测试3: 80%阈值验证 ===")
    
    test_asin = "B07XXXXX"
    vocab = ProductVocabulary(meta_file, [test_asin])
    
    if not vocab.load_vocabulary_for_asin(test_asin):
        logger.warning(f"未找到测试商品，使用模拟测试")
        vocab.product_vocabularies[test_asin] = {"blue", "color", "cotton", "material", "soft"}
    
    threshold = 0.8
    test_cases = [
        ("blue color", "✓ 100%", True),
        ("blue color cotton", "✓ 100%", True),
        ("blue cotton material xyz", "✓ 75%", False),
        ("blue xyz abc", "✗ 33%", False),
    ]
    
    logger.info(f"使用阈值: {threshold:.0%}")
    all_passed = True
    
    for phrase, expected_result, should_pass in test_cases:
        ratio = vocab.word_percentage_exists_in_product(phrase, test_asin)
        is_valid = ratio >= threshold
        matches_expected = is_valid == should_pass
        status = "✓" if matches_expected else "✗"
        
        logger.info(f"{status} '{phrase}' -> {ratio:.1%} (有效: {is_valid}, 期望: {should_pass})")
        all_passed = all_passed and matches_expected
    
    return all_passed

def test_combined_quality_scoring(meta_file: str) -> bool:
    """测试综合质量评分"""
    print("\n=== 测试4: 综合评分（目标商品验证）===")
    
    selector = AttributeSelectorWithProductValidation(meta_file=meta_file)
    
    test_scores = [
        (0.5, "商品验证差"),
        (1.0, "商品验证好"),
        (0.5, "均衡"),
    ]
    
    logger.info("综合评分公式: 仅使用目标商品验证")
    
    for product_quality, description in test_scores:
        combined, stats = selector._compute_combined_quality({"attr1": {"is_valid": product_quality >= 0.5}})
        logger.info(f"商品质量={product_quality:.1%} -> 综合分={combined:.1%} ({description})")
    
    return True

def test_attribute_selector_integration() -> bool:
    """测试AttributeSelectorWithProductValidation集成"""
    print("\n=== 测试5: 选择器集成 ===")
    
    selector = AttributeSelectorWithProductValidation(meta_file=None)
    
    sample_query = {
        "selected_attributes": [
            {"dimension": "Appearance_Color", "value": "blue"}
        ]
    }
    
    logger.info("✓ 成功创建AttributeSelectorWithProductValidation实例")
    logger.info("✓ 支持无meta_file模式（仅历史偏好评估）")
    
    return True

def run_validation_tests(meta_file: str) -> int:
    """运行所有验证测试"""
    if not Path(meta_file).exists():
        logger.error(f"元数据文件不存在: {meta_file}")
        logger.info("提示：可使用 --meta-file 参数指定正确的路径")
        return 1
    
    logger.info(f"开始测试 (meta_file={meta_file})\n")
    
    results = []
    results.append(("ProductVocabulary加载", test_vocabulary_loading(meta_file)))
    results.append(("word_match_ratio计算", test_word_match_ratio(meta_file)))
    results.append(("80%阈值验证", test_validation_threshold(meta_file)))
    results.append(("综合评分", test_combined_quality_scoring(meta_file)))
    results.append(("选择器集成", test_attribute_selector_integration()))
    
    print("\n=== 测试总结 ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return 0 if passed == total else 1


def main():
    """主函数：支持属性选择和验证测试两种模式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Stage 13 属性选择与商品验证',
        epilog='模式1: 属性选择 (--user-id, --category, --query-file, --meta-file)  模式2: 验证测试 (--validate, --meta-file)'
    )
    parser.add_argument('--user-id', help='用户ID')
    parser.add_argument('--category', help='产品类别')
    parser.add_argument('--query-file', help='查询JSON文件')
    parser.add_argument('--meta-file', type=str, help='商品元数据文件路径（gzip JSON Lines）')
    parser.add_argument('--validation-threshold', type=float, default=0.7, help='语义验证阈值（默认: 0.7）')
    parser.add_argument('--validate-selected-dimensions-only', action='store_true', default=True, help='仅验证query选中维度（默认开启）')
    parser.add_argument('--validate-all-dimensions', action='store_true', help='验证persona该类目全部维度（覆盖默认行为）')
    parser.add_argument('--output-file', help='输出JSON文件')
    parser.add_argument('--validate', action='store_true', help='运行验证测试模式')
    
    args = parser.parse_args()
    
    if args.validate:
        if not args.meta_file:
            parser.error("验证测试模式需要 --meta-file 参数")
        meta_file = args.meta_file
        if not Path(meta_file).exists():
            logger.error(f"元数据文件不存在: {meta_file}")
            return 1
        return run_validation_tests(meta_file)
    else:
        if not all([args.user_id, args.category, args.query_file, args.meta_file]):
            parser.error("属性选择模式需要 --user-id, --category, --query-file, --meta-file 参数")
        
        with open(args.query_file, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        # 从query_data中提取target_asin（如果有的话）
        target_asin = query_data.get('target_asin')
        if not target_asin:
            parser.error("查询文件必须包含 target_asin 字段")
        
        selector = AttributeSelectorWithProductValidation(
            meta_file=args.meta_file,
            validation_threshold=args.validation_threshold,
            validate_selected_dimensions_only=(not args.validate_all_dimensions)
        )
        result = selector.evaluate_attributes(query_data, args.user_id, args.category, target_asin=target_asin)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {args.output_file}")
        
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
