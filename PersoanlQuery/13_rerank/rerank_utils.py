#!/usr/bin/env python3
"""
LLM重排序公共工具模块
提供通用的数据加载和处理函数，避免代码重复
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime


def load_top10_results(filepath: str) -> Dict[str, Any]:
    """加载Top10检索结果"""
    if not os.path.exists(filepath):
        print(f"⚠️ 文件不存在: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def load_queries(filepath: str) -> Dict[str, Any]:
    """加载查询元数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_product_metadata_cache(metadata_file: str, required_asins: Set[str]) -> Dict[str, Dict]:
    """
    从元数据文件加载指定的产品信息（流式加载以节省内存）

    Args:
        metadata_file: 元数据文件路径
        required_asins: 需要加载的ASIN集合

    Returns:
        产品字典，key为ASIN，value为产品信息
    """
    print(f"加载产品元数据（{len(required_asins)}个产品）...")
    products = {}

    with open(metadata_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                product = json.loads(line)
                asin = product.get('asin')

                if asin in required_asins:
                    products[asin] = product

                    if len(products) % 500 == 0:
                        print(f"  已加载: {len(products)}/{len(required_asins)}")

                    if len(products) == len(required_asins):
                        break
            except:
                continue

    print(f"✓ 成功加载 {len(products)} 个产品")
    return products


def get_product_document(asin: str, product: Optional[Dict], include_reviews: bool = False) -> str:
    """
    将产品信息格式化为文档字符串

    Args:
        asin: 产品ASIN
        product: 产品信息字典
        include_reviews: 是否包含评论信息
    """
    if not product:
        return f"[ASIN: {asin} - 无法获取产品信息]"

    doc = f"[ASIN: {asin}]\n"

    # 标题
    title = product.get('title', 'Unknown')
    doc += f"Title: {title}\n"

    # 品牌
    brand = product.get('brand', 'Unknown')
    doc += f"Brand: {brand}\n"

    # 类别
    categories = product.get('category', [])
    if categories:
        doc += f"Categories: {' > '.join(categories)}\n"

    # 特征
    features = product.get('features', [])
    if features:
        doc += "Features:\n"
        for feature in features[:5]:  # 只取前5个特征
            doc += f"  - {feature}\n"

    # 描述
    description = product.get('description', '')
    if description:
        doc += f"Description: {description[:500]}...\n"

    # 价格
    price = product.get('price', '')
    if price:
        doc += f"Price: ${price}\n"

    # 评论信息
    if include_reviews:
        reviews = product.get('reviews', [])
        if reviews:
            avg_rating = reviews.get('average_rating', 0)
            total_reviews = reviews.get('total_reviews', 0)
            doc += f"Rating: {avg_rating}/5 ({total_reviews} reviews)\n"

            # 取前几条评论
            reviews_list = reviews.get('reviews', [])
            if reviews_list:
                doc += "Recent Reviews:\n"
                for review in reviews_list[:3]:
                    rating = review.get('rating', 0)
                    title = review.get('summary', '')
                    text = review.get('text', '')[:200]
                    doc += f"  {rating}★: {title}\n  {text}...\n"

    return doc


def extract_keywords(query_text: str, min_length: int = 3, stop_words: Optional[Set[str]] = None) -> Set[str]:
    """
    从查询文本中提取关键词

    Args:
        query_text: 查询文本
        min_length: 最小词长度
        stop_words: 停用词集合

    Returns:
        关键词集合
    """
    if stop_words is None:
        stop_words = {
            'i', 'am', 'is', 'are', 'was', 'were', 'looking', 'for', 'a', 'an', 'the', 'to',
            'in', 'that', 'with', 'and', 'or', 'my', 'on', 'of', 'but', 'have', 'has',
            'had', 'been', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'these', 'those', 'they', 'them', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'can', 'be', 'at', 'by', 'from', 'as', 'it',
            'its', 'if', 'you', 'your', 'he', 'she', 'we', 'our', 'his', 'her', 'me',
            'him', 'us', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }

    # 转换为小写并分割
    words = re.findall(r'\b[a-z]+\b', query_text.lower())

    # 过滤停用词和短词
    keywords = set(w for w in words if w not in stop_words and len(w) >= min_length)

    return keywords


def create_basic_rerank_prompt(query_text: str, top10_asins: List[Tuple[str, float]]) -> str:
    """创建基础重排序提示"""
    products_str = "\n".join(
        f"{i+1}. ASIN: {asin} (原始得分: {score:.2f})"
        for i, (asin, score) in enumerate(top10_asins)
    )

    prompt = f"""你是一个电商产品搜索排序专家。给定用户查询和10个产品ASIN列表，请根据查询文本重新排序这些产品。

用户查询: {query_text}

原始排序的产品列表:
{products_str}

请根据以下标准重新排序：
1. 产品与用户查询的相关性
2. 用户查询中关键词的匹配度
3. 产品特性与用户需求的契合度

请按重要性从高到低，只输出ASIN列表，一个ASIN一行，不要其他信息。例如:
B010S4AT1W
B01DXYRWCI
...
"""
    return prompt


def create_advanced_rerank_prompt(query_text: str, top10_asins: List[Tuple[str, float]],
                                keywords: Set[str], use_product_docs: bool = False,
                                product_docs: Optional[Dict[str, str]] = None) -> str:
    """创建高级重排序提示"""
    products_str = "\n".join(
        f"{i+1}. ASIN: {asin}"
        for i, (asin, score) in enumerate(top10_asins)
    )

    keywords_str = ", ".join(sorted(keywords)[:10])  # 限制显示的关键词数量

    doc_info = ""
    if use_product_docs and product_docs:
        # 示例产品文档
        example_doc = next(iter(product_docs.values()))[:300] if product_docs else ""
        doc_info = f"""
产品信息示例:
{example_doc}

"""

    prompt = f"""你是一个专业的电商产品推荐专家。你的任务是重新排序以下产品，使其更符合用户查询的需求。

用户查询: {query_text}
查询关键词: {keywords_str}

{doc_info}待重新排序的产品列表:
{products_str}

请严格按照以下标准重新排序：
1. **相关性**：产品与用户查询的直接匹配程度
2. **关键词覆盖度**：产品包含多少用户查询的关键词
3. **功能匹配**：产品功能是否满足用户需求
4. **质量评分**：基于产品描述和特征的合理性

请按最佳匹配度从高到低输出，格式为：
1. ASIN: 最佳匹配的产品ASIN
2. ASIN: 第二匹配的产品ASIN
...
10. ASIN: 最匹配的产品ASIN

只输出序号和ASIN，不要其他解释。
"""
    return prompt


def create_document_based_rerank_prompt(query_text: str, top10_with_docs: List[Tuple[str, float, str]]) -> str:
    """基于产品文档的重排序提示"""
    products_str = "\n\n".join(
        f"产品 {i+1}:\n{doc}"
        for i, (asin, score, doc) in enumerate(top10_with_docs)
    )

    prompt = f"""你是一个专业的电商产品推荐专家。基于用户查询和以下10个产品的详细信息，请对这些产品进行重新排序。

用户查询: {query_text}

产品详细信息:
{products_str}

请根据以下标准重新排序：
1. **相关性**：产品与用户查询的直接匹配程度
2. **功能匹配**：产品功能是否完全满足用户需求
3. **质量描述**：产品描述的详细程度和质量
4. **用户体验**：产品是否能解决用户的问题或满足期望

请按推荐优先级从高到低输出，格式为：
1. ASIN: 最佳匹配的产品ASIN
2. ASIN: 第二匹配的产品ASIN
...
10. ASIN: 最匹配的产品ASIN

只输出序号和ASIN，不要其他解释。
"""
    return prompt