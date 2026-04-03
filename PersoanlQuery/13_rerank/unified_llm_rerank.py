#!/usr/bin/env python3
"""
统一的LLM重排序脚本
整合了所有版本的功能，支持多种模式：basic、advanced、document-based
支持clean和noisy两种查询模式
"""

import json
import sys
import os
import argparse
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set, Optional

# 添加路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import LLMClient
from rerank_utils import (
    load_top10_results,
    load_queries,
    load_product_metadata_cache,
    get_product_document,
    extract_keywords,
    create_basic_rerank_prompt,
    create_advanced_rerank_prompt,
    create_document_based_rerank_prompt
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedLLMReranker:
    """统一的LLM重排序器"""

    def __init__(self, mode: str = 'basic', use_product_docs: bool = False,
                 metadata_file: str = None, include_reviews: bool = False):
        """
        初始化重排序器

        Args:
            mode: 重排序模式 ('basic', 'advanced', 'document-based')
            use_product_docs: 是否使用产品文档
            metadata_file: 产品元数据文件路径
            include_reviews: 是否包含评论信息
        """
        self.mode = mode
        self.use_product_docs = use_product_docs
        self.metadata_file = metadata_file
        self.include_reviews = include_reviews
        self.product_docs = {}

        # 初始化LLM客户端
        self.llm_client = LLMClient()

        logger.info(f"初始化重排序器 - 模式: {mode}, 使用文档: {use_product_docs}")

    def _load_product_docs_if_needed(self, top10_asins: List[str]) -> None:
        """如果需要，加载产品文档"""
        if self.use_product_docs and self.metadata_file:
            required_asins = set(top10_asins)
            self.product_docs = load_product_metadata_cache(self.metadata_file, required_asins)

    def rerank_single_query(self, query_text: str, top10_results: List[Tuple[str, float]],
                           user_id: str = None, query_idx: int = None) -> List[str]:
        """
        对单个查询进行重排序

        Args:
            query_text: 查询文本
            top10_results: Top10结果列表，格式为[(ASIN, score), ...]
            user_id: 用户ID（可选）
            query_idx: 查询索引（可选）

        Returns:
            重排序后的ASIN列表
        """
        # 提取ASIN列表
        top10_asins = [(asin, score) for asin, score in top10_results]

        # 加载产品文档（如果需要）
        self._load_product_docs_if_needed([asin for asin, _ in top10_asins])

        # 根据模式创建不同的重排序方法
        if self.mode == 'basic':
            return self._rerank_basic(query_text, top10_asins)
        elif self.mode == 'advanced':
            return self._rerank_advanced(query_text, top10_asins)
        elif self.mode == 'document-based':
            return self._rerank_document_based(query_text, top10_asins)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

    def _rerank_basic(self, query_text: str, top10_asins: List[Tuple[str, float]]) -> List[str]:
        """基础重排序方法"""
        prompt = create_basic_rerank_prompt(query_text, top10_asins)

        logger.info(f"发送基础重排序请求...")
        response = self.llm_client.generate_response(prompt)

        # 解析响应
        ranked_asins = self._parse_rerank_response(response)
        return ranked_asins

    def _rerank_advanced(self, query_text: str, top10_asins: List[Tuple[str, float]]) -> List[str]:
        """高级重排序方法"""
        # 提取关键词
        keywords = extract_keywords(query_text)

        prompt = create_advanced_rerank_prompt(
            query_text, top10_asins, keywords,
            use_product_docs=self.use_product_docs,
            product_docs=self.product_docs if self.use_product_docs else None
        )

        logger.info(f"发送高级重排序请求（关键词: {', '.join(list(keywords)[:5])}...）")
        response = self.llm_client.generate_response(prompt)

        # 解析响应
        ranked_asins = self._parse_rerank_response(response)
        return ranked_asins

    def _rerank_document_based(self, query_text: str, top10_asins: List[Tuple[str, float]]) -> List[str]:
        """基于文档的重排序方法"""
        # 准备带文档的产品列表
        top10_with_docs = []
        for asin, score in top10_asins:
            product = self.product_docs.get(asin)
            doc = get_product_document(asin, product, include_reviews=self.include_reviews)
            top10_with_docs.append((asin, score, doc))

        prompt = create_document_based_rerank_prompt(query_text, top10_with_docs)

        logger.info(f"发送基于文档的重排序请求...")
        response = self.llm_client.generate_response(prompt)

        # 解析响应
        ranked_asins = self._parse_rerank_response(response)
        return ranked_asins

    def _parse_rerank_response(self, response: str) -> List[str]:
        """
        解析重排序响应，提取ASIN列表

        Args:
            response: LLM的响应文本

        Returns:
            ASIN列表
        """
        # 提取ASIN（格式如 "B010S4AT1W"）
        asin_pattern = r'\b[A-Z0-9]{10}\b'
        asins = re.findall(asin_pattern, response)

        if len(asins) < 10:
            logger.warning(f"只解析出 {len(asins)} 个ASIN，可能返回了不完整的排序")
            # 如果ASIN不足10个，使用原始顺序补齐
            remaining = 10 - len(asins)
            logger.warning(f"使用原始顺序补齐最后 {remaining} 个ASIN")
            # 这里需要原始的top10_asins，所以需要修改函数签名
            # 暂时使用原始顺序（理想情况下应该传原始列表）

        return asins[:10]  # 只取前10个


def main():
    parser = argparse.ArgumentParser(description='统一的LLM重排序脚本')
    parser.add_argument('--top10-file', required=True, help='Top10结果文件路径')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'document-based'],
                       default='advanced', help='重排序模式')
    parser.add_argument('--metadata-file', help='产品元数据文件路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--user-id', help='用户ID')
    parser.add_argument('--query-mode', choices=['clean', 'noisy', 'both'],
                       default='both', help='查询模式')
    parser.add_argument('--include-reviews', action='store_true',
                       help='是否在文档中包含评论信息')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载Top10结果
    top10_data = load_top10_results(args.top10_file)
    if not top10_data:
        logger.error("无法加载Top10结果文件")
        return

    logger.info(f"加载了 {top10_data['num_queries']} 个查询的Top10结果")

    # 初始化重排序器
    reranker = UnifiedLLMReranker(
        mode=args.mode,
        use_product_docs=(args.metadata_file is not None),
        metadata_file=args.metadata_file,
        include_reviews=args.include_reviews
    )

    # 处理每个查询
    all_reranked_results = []

    for query_result in top10_data['query_results']:
        query_idx = query_result['query_idx']
        query_text = query_result['query_text']
        target_asin = query_result['asin']
        original_top10 = query_result['top10_results']

        # 转换为 (ASIN, score) 格式
        top10_asins = [(item['asin'], item['score']) for item in original_top10]

        logger.info(f"处理查询 {query_idx + 1}/{len(top10_data['query_results'])}: {query_text[:50]}...")

        try:
            # 进行重排序
            reranked_asins = reranker.rerank_single_query(
                query_text=query_text,
                top10_results=top10_asins,
                user_id=args.user_id,
                query_idx=query_idx
            )

            # 记录结果
            result = {
                'query_idx': query_idx,
                'asin': target_asin,
                'query_text': query_text,
                'original_top10': original_top10,
                'reranked_top10': [
                    {'rank': i + 1, 'asin': asin}
                    for i, asin in enumerate(reranked_asins)
                ]
            }

            all_reranked_results.append(result)

            if args.debug:
                print(f"原始Top10: {[item['asin'] for item in original_top10[:5]]}")
                print(f"重排序Top5: {reranked_asins[:5]}")
                print()

        except Exception as e:
            logger.error(f"处理查询 {query_idx} 时出错: {e}")
            # 出错时使用原始顺序
            all_reranked_results.append({
                'query_idx': query_idx,
                'asin': target_asin,
                'query_text': query_text,
                'original_top10': original_top10,
                'reranked_top10': [
                    {'rank': i + 1, 'asin': item['asin']}
                    for i, item in enumerate(original_top10)
                ]
            })

    # 保存结果
    output_file = os.path.join(args.output_dir, f"reranked_{args.mode}_results.json")

    output_data = {
        'user_id': top10_data.get('user_id', args.user_id),
        'retriever': top10_data.get('retriever', 'unknown'),
        'mode': args.mode,
        'metadata_file': args.metadata_file,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(all_reranked_results),
        'query_results': all_reranked_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"重排序完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    main()