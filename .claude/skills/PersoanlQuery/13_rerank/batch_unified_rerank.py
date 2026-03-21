#!/usr/bin/env python3
"""
统一的批量LLM重排序脚本
替代所有版本的批量处理脚本，支持多种配置和模式
"""

import json
import os
import sys
import argparse
import logging
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from unified_llm_rerank import UnifiedLLMReranker
from rerank_utils import load_top10_results

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchUnifiedReranker:
    """统一的批量重排序器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化批量重排序器

        Args:
            config: 配置字典
        """
        self.config = config
        self.metadata_file = config.get('metadata_file')
        self.include_reviews = config.get('include_reviews', False)
        self.mode = config.get('mode', 'advanced')
        self.query_mode = config.get('query_mode', 'both')

        logger.info(f"初始化批量重排序器 - 模式: {self.mode}, 查询模式: {self.query_mode}")

    def process_all_users(self, user_ids: List[str], output_base_dir: str) -> None:
        """
        处理所有用户的重排序

        Args:
            user_ids: 用户ID列表
            output_base_dir: 输出基础目录
        """
        for user_id in user_ids:
            self.process_single_user(user_id, output_base_dir)

    def process_single_user(self, user_id: str, output_base_dir: str) -> None:
        """
        处理单个用户的重排序

        Args:
            user_id: 用户ID
            output_base_dir: 输出基础目录
        """
        user_output_dir = os.path.join(output_base_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)

        logger.info(f"开始处理用户: {user_id}")

        # 加载该用户的Top10结果
        top10_files = self._get_user_top10_files(user_id)
        if not top10_files:
            logger.warning(f"用户 {user_id} 没有找到Top10结果文件")
            return

        for retriever_name, file_path in top10_files.items():
            logger.info(f"处理检索器: {retriever_name}")

            # 根据查询模式处理
            if self.query_mode == 'both':
                # 处理clean和noisy两种模式
                self._process_retriever_mode(user_id, retriever_name, file_path,
                                           user_output_dir, 'clean')
                self._process_retriever_mode(user_id, retriever_name, file_path,
                                           user_output_dir, 'noisy')
            else:
                # 处理指定模式
                self._process_retriever_mode(user_id, retriever_name, file_path,
                                           user_output_dir, self.query_mode)

    def _get_user_top10_files(self, user_id: str) -> Dict[str, str]:
        """
        获取用户的Top10结果文件

        Args:
            user_id: 用户ID

        Returns:
            检索器名到文件路径的映射
        """
        top10_files = {}
        base_path = f"/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/{user_id}"

        if not os.path.exists(base_path):
            return {}

        # 获取所有Top10结果文件
        for file in os.listdir(base_path):
            if file.endswith('_top10_results.json'):
                # 提取检索器名
                retriever_name = file.replace('_top10_results.json', '')
                # 根据查询模式筛选文件
                if self.query_mode == 'both' or (self.query_mode in file):
                    file_path = os.path.join(base_path, file)
                    top10_files[retriever_name] = file_path

        return top10_files

    def _process_retriever_mode(self, user_id: str, retriever_name: str,
                              file_path: str, output_dir: str, mode: str) -> None:
        """
        处理单个检索器的单个模式

        Args:
            user_id: 用户ID
            retriever_name: 检索器名称
            file_path: 输入文件路径
            output_dir: 输出目录
            mode: 查询模式 ('clean' 或 'noisy')
        """
        logger.info(f"处理 {retriever_name} 的 {mode} 模式")

        # 创建重排序器
        reranker = UnifiedLLMReranker(
            mode=self.mode,
            use_product_docs=(self.metadata_file is not None),
            metadata_file=self.metadata_file,
            include_reviews=self.include_reviews
        )

        # 加载Top10结果
        top10_data = load_top10_results(file_path)
        if not top10_data:
            logger.error(f"无法加载文件: {file_path}")
            return

        # 处理每个查询
        all_reranked_results = []

        for query_result in top10_data['query_results']:
            query_idx = query_result['query_idx']
            query_text = query_result['query_text']
            target_asin = query_result['asin']
            original_top10 = query_result['top10_results']

            # 转换为 (ASIN, score) 格式
            top10_asins = [(item['asin'], item['score']) for item in original_top10]

            logger.info(f"处理查询 {query_idx + 1}/{len(top10_data['query_results'])}")

            try:
                # 进行重排序
                reranked_asins = reranker.rerank_single_query(
                    query_text=query_text,
                    top10_results=top10_asins,
                    user_id=user_id,
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
        output_file = os.path.join(output_dir,
                                  f"reranked_{retriever_name}_{mode}_{self.mode}_results.json")

        output_data = {
            'user_id': user_id,
            'retriever': retriever_name,
            'query_mode': mode,
            'rerank_mode': self.mode,
            'metadata_file': self.metadata_file,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(all_reranked_results),
            'query_results': all_reranked_results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存结果: {output_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_all_user_ids() -> List[str]:
    """获取所有用户ID列表"""
    base_path = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"
    if not os.path.exists(base_path):
        return []

    user_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # 过滤掉非用户目录（如 __pycache__）
    user_ids = [d for d in user_dirs if not d.startswith('_')]
    return sorted(user_ids)


def main():
    parser = argparse.ArgumentParser(description='统一的批量LLM重排序脚本')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--users', nargs='+', help='指定的用户ID列表（不指定则处理所有用户）')
    parser.add_argument('--output-dir', default='/tmp/rerank_results', help='输出目录')
    parser.add_argument('--metadata-file', help='产品元数据文件路径')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'document-based'],
                       default='advanced', help='重排序模式')
    parser.add_argument('--query-mode', choices=['clean', 'noisy', 'both'],
                       default='both', help='查询模式')
    parser.add_argument('--include-reviews', action='store_true',
                       help='是否在文档中包含评论信息')
    parser.add_argument('--list-users', action='store_true', help='列出所有用户ID并退出')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式，不实际处理')

    args = parser.parse_args()

    # 列出所有用户
    if args.list_users:
        user_ids = get_all_user_ids()
        print("可用的用户ID:")
        for user_id in user_ids:
            print(f"  - {user_id}")
        return

    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"已加载配置文件: {args.config}")
    else:
        # 使用命令行参数作为配置
        config = {
            'metadata_file': args.metadata_file,
            'mode': args.mode,
            'query_mode': args.query_mode,
            'include_reviews': args.include_reviews
        }

    # 合并命令行参数
    if args.metadata_file:
        config['metadata_file'] = args.metadata_file
    if args.mode:
        config['mode'] = args.mode
    if args.query_mode:
        config['query_mode'] = args.query_mode
    if args.include_reviews:
        config['include_reviews'] = args.include_reviews

    # 获取用户列表
    if args.users:
        user_ids = args.users
    else:
        user_ids = get_all_user_ids()

    if not user_ids:
        logger.error("没有找到任何用户")
        return

    logger.info(f"将处理 {len(user_ids)} 个用户: {user_ids}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建批量重排序器
    batch_reranker = BatchUnifiedReranker(config)

    if args.dry_run:
        logger.info("试运行模式 - 将要处理的文件:")
        for user_id in user_ids:
            top10_files = batch_reranker._get_user_top10_files(user_id)
            for retriever_name, file_path in top10_files.items():
                logger.info(f"  {user_id}/{retriever_name}: {file_path}")
        return

    # 处理所有用户
    start_time = datetime.now()
    batch_reranker.process_all_users(user_ids, args.output_dir)
    end_time = datetime.now()

    logger.info(f"批量处理完成，耗时: {end_time - start_time}")
    logger.info(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()