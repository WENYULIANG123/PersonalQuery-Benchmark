"""
Stage 13 Rerank: 批量LLM重排序评估脚本

该脚本用于批量处理所有查询的三路LLM评估。
对每个查询调用 UserQueryEvaluator 进行全量评估。

使用方法：
  python3 13_batch_llm_rerank_all.py --config 15_config.json
  python3 13_batch_llm_rerank_all.py --config 15_config.json --meta-file products.json
  python3 13_batch_llm_rerank_all.py --config 15_config.json --debug

功能：
- 批量处理所有用户的查询
- 对每个查询执行三路LLM评估
- 支持中间结果保存和断点续训
- 生成详细的评估汇总报告
"""

import json
import os
import sys
import logging
import argparse
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def _load_user_query_evaluator() -> Any:
    module_path = Path(__file__).parent / "13_llm_rerank.py"
    if not module_path.exists():
        raise ImportError(f"无法找到评估器文件: {module_path}")

    spec = importlib.util.spec_from_file_location("stage13_llm_rerank", module_path)
    if not spec or not spec.loader:
        raise ImportError("无法创建stage13_llm_rerank模块的加载器")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "UserQueryEvaluator"):
        raise ImportError("stage13_llm_rerank模块缺少UserQueryEvaluator类")

    return module.UserQueryEvaluator


UserQueryEvaluator = _load_user_query_evaluator()

logger = logging.getLogger(__name__)


def _extract_query_entries(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _extract_query_id(entry: Dict[str, Any]) -> str:
    return str(
        entry.get("query_id")
        or entry.get("target_asin")
        or entry.get("asin")
        or ""
    )


def setup_logging(log_dir: Path, timestamp: str) -> None:
    """设置日志系统"""
    log_file = log_dir / f'batch_rerank_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)


class BatchEvaluator:
    """批量评估管理器"""

    # 配置常量
    QUERY_FILENAME_PATTERN = "dual_queries_*.json"
    QUERY_ID_KEY = "query_id"
    GROUPBY_COLUMNS = ['sentiment', 'noise_level', 'retriever', 'llm']
    TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
    SAVE_INTERVAL = 50  # 每50次保存一次
    TIME_SAVE_THRESHOLD = 300  # 5分钟

    def __init__(self, config_path: str = "15_config.json", meta_file: Optional[str] = None, user_id: Optional[str] = None):
        if not meta_file:
            raise ValueError(
                "meta_file是必须的。Stage 13要求所有属性都必须验证其在目标商品中的存在性。"
                "请提供 --meta-file 参数。"
            )

        self.config = self._load_config(config_path)
        self._timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
        self.user_id = user_id

        try:
            self.output_dir = Path(self.config["output_paths"]["results_dir"])
            self.logs_dir = Path(self.config["output_paths"]["logs_dir"])
        except KeyError as e:
            raise ValueError(f"配置文件缺少必要的键: {e}")

        # 确保输出目录存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"无法创建输出目录: {e}")
            raise

        setup_logging(self.logs_dir, self._timestamp)

        # 直接初始化UserQueryEvaluator
        self.evaluator = UserQueryEvaluator(config_path=config_path, meta_file=meta_file)

        self.all_results = []
        self.failed_queries = []
        self._start_time = time.time()
        logger.info(f"初始化BatchEvaluator完成 (timestamp={self._timestamp})")
        if self.user_id:
            logger.info(f"已启用单用户模式: {self.user_id}")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"配置文件格式错误: {config_path}")
            raise

    def load_query_ids(self) -> List[str]:
        """加载所有查询ID"""
        logger.info("加载所有查询ID...")

        try:
            queries_dir = Path(self.config["input_paths"]["queries_dir"])
        except KeyError:
            logger.error("配置文件缺少 input_paths.queries_dir")
            return []

        if not queries_dir.exists():
            logger.error(f"查询目录不存在: {queries_dir}")
            return []

        query_pattern = f"dual_queries_{self.user_id}.json" if self.user_id else self.QUERY_FILENAME_PATTERN
        query_files = sorted(queries_dir.glob(query_pattern))
        if not query_files:
            logger.warning(f"未在 {queries_dir} 中找到匹配模式的文件: {query_pattern}")
            return []

        query_ids = set()
        total_queries = 0
        for query_file in query_files:
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    queries = _extract_query_entries(json.load(f))
                    for query in queries:
                        if self.user_id and str(query.get("user_id", "")) != self.user_id:
                            continue
                        total_queries += 1
                        query_id = _extract_query_id(query)
                        if query_id:
                            query_ids.add(query_id)
            except FileNotFoundError as e:
                logger.error(f"查询文件不存在 {query_file}: {e}")
                continue
            except json.JSONDecodeError as e:
                logger.error(f"查询文件JSON格式错误 {query_file}: {e}")
                continue

        query_ids = sorted(query_ids)
        if self.user_id:
            logger.info(f"用户 {self.user_id}: 从 {len(query_files)} 个文件中加载 {len(query_ids)} 个唯一查询ID (共 {total_queries} 个查询)")
        else:
            logger.info(f"从 {len(query_files)} 个文件中加载 {len(query_ids)} 个唯一查询ID (共 {total_queries} 个查询)")
        return query_ids

    def evaluate_query_direct(self, query_id: str) -> bool:
        """直接导入模块评估单个查询"""
        logger.info(f"评估查询: {query_id}")

        try:
            # 直接使用已初始化的UserQueryEvaluator避免重复初始化
            query_results = self.evaluator.evaluate_query(query_id)

            if query_results:
                self.all_results.extend(query_results)
                return True
            else:
                logger.warning(f"查询 {query_id} 未返回结果")
                self.failed_queries.append(query_id)
                return False

        except Exception as e:
            logger.error(f"查询 {query_id} 评估异常: {e}")
            self.failed_queries.append(query_id)
            return False

    def run_batch_evaluation(self):
        """执行批量评估"""
        logger.info("=" * 80)
        logger.info("开始批量评分式rerank")
        logger.info("=" * 80)

        # 加载所有查询ID
        query_ids = self.load_query_ids()
        if not query_ids:
            logger.error("没有找到任何查询ID，退出评估")
            return

        if self.user_id:
            self.evaluator.preload_user_candidate_documents(self.user_id)

        # 评估每个查询
        successful = 0
        failed = 0
        last_save_time = time.time()

        for idx, query_id in enumerate(query_ids, 1):
            start_time = time.time()
            logger.info(f"[{idx}/{len(query_ids)}] 处理查询 {query_id}")

            if self.evaluate_query_direct(query_id):
                successful += 1
            else:
                failed += 1

            # 定期保存中间结果或超过5分钟
            current_time = time.time()
            if idx % 50 == 0 or current_time - last_save_time > 300:  # 每50次或5分钟保存
                self.save_intermediate_results()
                last_save_time = current_time

            # 显示进度信息
            duration = current_time - start_time
            avg_duration = current_time - self._start_time
            eta = (avg_duration / idx) * (len(query_ids) - idx)
            logger.info(f"进度: {idx}/{len(query_ids)} ({idx/len(query_ids)*100:.1f}%) | "
                       f"成功: {successful} | 失败: {failed} | "
                       f"当前耗时: {duration:.2f}s | 预计剩余: {eta/60:.1f}分钟")

        logger.info("=" * 80)
        logger.info(f"批量评估完成: {successful} 成功, {failed} 失败")
        logger.info(f"总耗时: {(time.time() - self._start_time)/60:.1f}分钟")
        logger.info("=" * 80)

        # 保存最终结果
        self.save_final_results()

    def save_intermediate_results(self):
        """保存中间结果"""
        if not self.all_results:
            return

        intermediate_file = Path(self.output_dir) / f"intermediate_results_{len(self.all_results)}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"中间结果已保存: {intermediate_file} ({len(self.all_results)} 条)")

    def save_final_results(self):
        """保存最终结果"""
        logger.info(f"保存 {len(self.all_results)} 条最终评估结果...")

        # 保存完整结果
        output_file = Path(self.output_dir) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"完整结果已保存: {output_file}")

        # 生成汇总统计
        if self.all_results:
            df = pd.DataFrame(self.all_results)

            summary = df.groupby(['sentiment', 'noise_level', 'retriever', 'llm']).agg({
                'query_id': 'count',
                'total_gap': ['mean', 'std', 'min', 'max'],
                'rerank_metrics': lambda x: x.apply(lambda y: y['token_recall']).mean()
            }).round(4)

            summary_file = Path(self.output_dir) / f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary.to_json(summary_file)
            logger.info(f"汇总统计已保存: {summary_file}")

        # 保存失败记录
        if self.failed_queries:
            failed_file = Path(self.logs_dir) / f"failed_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_queries, f, ensure_ascii=False, indent=2)
            logger.info(f"失败记录已保存: {failed_file}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Stage 13: 批量评分式rerank')
    parser.add_argument('--config', type=str, default='15_config.json', help='配置文件路径')
    parser.add_argument('--meta-file', type=str, help='商品元数据文件路径（用于属性商品验证，可选）')
    parser.add_argument('--user-id', type=str, help='仅对指定用户执行rerank（可选）')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    evaluator = BatchEvaluator(config_path=args.config, meta_file=args.meta_file, user_id=args.user_id)
    evaluator.run_batch_evaluation()


if __name__ == '__main__':
    main()
