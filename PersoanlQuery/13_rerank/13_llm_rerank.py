"""
Stage 13 Rerank: 单个查询的三路LLM评估脚本

该脚本处理单个查询的全量评估，包括所有sentiment、noise_level、retriever和llm的组合。
被主脚本 13_batch_llm_rerank_all.py 调用。

使用方法：
  python3 13_llm_rerank.py --query-id Q001 --config 15_config.json
"""

import json
import os
import sys
import logging
import argparse
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from datetime import datetime
import re

# 添加llm_client路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import LLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载属性选择模块
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "attribute_selector",
        Path(__file__).parent / "13_select_attributes_from_history.py"
    )
    if spec and spec.loader:
        attribute_selector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(attribute_selector_module)
        AttributeSelector = attribute_selector_module.AttributeSelector
        AttributeSelectorWithProductValidation = attribute_selector_module.AttributeSelectorWithProductValidation
        ProductVocabulary = attribute_selector_module.ProductVocabulary
    else:
        AttributeSelector = None
        AttributeSelectorWithProductValidation = None
        ProductVocabulary = None
except Exception as e:
    logger.warning(f"无法加载AttributeSelector: {e}")
    AttributeSelector = None
    AttributeSelectorWithProductValidation = None
    ProductVocabulary = None


class UserQueryEvaluator:
    """单个查询的三路评估器"""
    
    def __init__(self, config_path: str = "15_config.json", meta_file: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
            meta_file: 商品元数据文件路径（用于属性商品验证）
        """
        self.config = self._load_config(config_path)
        self.output_dir = self.config["output_paths"]["results_dir"]
        self.cache_dir = self.config["output_paths"]["cache_dir"]
        self.meta_file = meta_file
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化LLMClient（支持多模型）
        self.llm_clients = {}
        for llm_name in self.config["evaluation_config"]["llms"]:
            self.llm_clients[llm_name] = LLMClient(model=llm_name)
            logger.info(f"初始化LLM: {llm_name}")
        
        # 初始化属性选择器（优先使用带验证的版本）
        self.attribute_selector = None
        if self.meta_file and AttributeSelectorWithProductValidation:
            try:
                self.attribute_selector = AttributeSelectorWithProductValidation(meta_file=self.meta_file)
                logger.info(f"初始化AttributeSelectorWithProductValidation完成 (meta_file={self.meta_file})")
            except Exception as e:
                logger.warning(f"初始化AttributeSelectorWithProductValidation失败: {e}")
                # 回退到基础选择器
                if AttributeSelector:
                    try:
                        self.attribute_selector = AttributeSelector()
                        logger.info("回退到基础AttributeSelector")
                    except Exception as e2:
                        logger.warning(f"初始化AttributeSelector失败: {e2}")
        elif AttributeSelector:
            try:
                self.attribute_selector = AttributeSelector()
                logger.info("初始化基础AttributeSelector完成")
            except Exception as e:
                logger.warning(f"初始化AttributeSelector失败: {e}")
        
        self.cache = {}
        self.ground_truth = None
        self.query_index = None
        self.retrieval_cache = {}
        self.product_document_cache = {}
        self.sim_weight = self.config.get("evaluation_config", {}).get("sim_weight", 0.1)
        self.scoring_max_workers = self.config.get("evaluation_config", {}).get("scoring_max_workers", 3)
        logger.info("初始化UserQueryEvaluator完成")

    @staticmethod
    def _extract_query_entries(payload) -> List[Dict]:
        if isinstance(payload, dict):
            results = payload.get('results')
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    @staticmethod
    def _normalize_query_data(query: Dict) -> Dict:
        target_query = query.get('target_user_query')
        if not isinstance(target_query, dict):
            target_query = {}
        query_text = query.get('query_text') or target_query.get('query', '')
        query_id = str(query.get('query_id') or query.get('target_asin') or query.get('asin') or '')

        normalized = dict(query)
        normalized['query_id'] = query_id
        normalized['target_asin'] = query.get('target_asin') or query.get('asin') or query_id
        normalized['query_text'] = query_text
        normalized['query_preference'] = query.get('query_preference') or query_text
        return normalized

    def _build_query_index(self) -> Dict[str, Dict]:
        if self.query_index is not None:
            return self.query_index

        queries_dir = Path(self.config["input_paths"]["queries_dir"])
        query_files = sorted(queries_dir.glob("queries_*.json"))
        query_index = {}

        for query_file in query_files:
            with open(query_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            for query in self._extract_query_entries(payload):
                normalized = self._normalize_query_data(query)
                query_id = normalized.get('query_id')
                if query_id:
                    query_index[query_id] = normalized

        self.query_index = query_index
        return self.query_index
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_ground_truth(self) -> Dict:
        """加载GT映射"""
        if self.ground_truth is not None:
            return self.ground_truth
            
        logger.info("加载Ground Truth映射...")
        gt_dir = Path(self.config["input_paths"]["ground_truth_dir"])
        gt_file = gt_dir / "gt_mapping.json"
        
        if not gt_file.exists():
            logger.warning(f"找不到GT文件：{gt_file}")
            query_index = self._build_query_index()
            self.ground_truth = {
                query_id: {'products': [query['target_asin']]}
                for query_id, query in query_index.items()
                if query.get('target_asin')
            }
        else:
            with open(gt_file, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
        
        return self.ground_truth
    
    def load_retrieval_results(self, retriever: str, noise_level: str, user_id: str) -> Dict:
        """加载检索结果"""
        logger.debug(f"加载检索结果: {retriever}, noise={noise_level}")

        cache_key = (user_id, retriever, noise_level)
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]

        retrieval_dir = Path(self.config["input_paths"]["retrieval_results_dir"])
        result_file = retrieval_dir / user_id / f"retrieval_{retriever}_{noise_level}_top10_results.json"

        if not result_file.exists():
            logger.warning(f"找不到检索结果：{result_file}")
            return {}

        with open(result_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        query_results = payload.get('query_results', []) if isinstance(payload, dict) else []
        retrieval_map = {}
        for item in query_results:
            if not isinstance(item, dict):
                continue
            query_id = str(item.get('query_id') or item.get('target_asin') or item.get('asin') or '')
            top10_results = item.get('top10_results', [])
            retrieval_map[query_id] = {
                'retrieved_products': [result.get('asin') for result in top10_results if isinstance(result, dict) and result.get('asin')],
                'candidate_entries': [
                    {
                        'asin': result.get('asin'),
                        'dense_score': float(result.get('score', 0.0)),
                        'original_rank': int(result.get('rank', idx + 1))
                    }
                    for idx, result in enumerate(top10_results)
                    if isinstance(result, dict) and result.get('asin')
                ]
            }

        self.retrieval_cache[cache_key] = retrieval_map
        return retrieval_map

    def preload_user_candidate_documents(self, user_id: str) -> None:
        if not user_id:
            return

        evaluation_config = self.config.get("evaluation_config", {})
        retrievers = evaluation_config.get("retrievers", ["dense"])
        noise_levels = evaluation_config.get("noise_levels", ["noisy"])

        candidate_asins = set()
        for retriever in retrievers:
            for noise_level in noise_levels:
                retrieval_results = self.load_retrieval_results(retriever, noise_level, user_id)
                for item in retrieval_results.values():
                    for candidate in item.get('candidate_entries', []):
                        asin = str(candidate.get('asin', '')).strip()
                        if asin:
                            candidate_asins.add(asin)

        if candidate_asins:
            logger.info(f"预加载用户 {user_id} 的候选商品文档: {len(candidate_asins)} 个ASIN")
            self._preload_product_documents(sorted(candidate_asins))
    
    def load_query_data(self, query_id: str):
        """加载单个查询数据"""
        logger.info(f"加载查询 {query_id}")

        query_index = self._build_query_index()
        if str(query_id) in query_index:
            return query_index[str(query_id)]
        
        logger.warning(f"找不到查询 {query_id}")
        return {}
    
    def extract_tokens(self, text: str) -> Set[str]:
        """提取文本中的tokens（简单分词）"""
        if not text:
            return set()
        tokens = re.findall(r'\w+', text.lower())
        return set(tokens)
    
    def compute_token_metrics(self, predicted_tokens: Set[str], gt_tokens: Set[str]) -> Dict:
        """计算token级别的指标"""
        
        if not gt_tokens:
            return {
                'token_recall': 0.0,
                'token_precision': 0.0,
                'token_f1': 0.0,
                'predicted_tokens': len(predicted_tokens),
                'gt_tokens': 0,
                'matched_tokens': 0
            }
        
        # 交集
        intersection = predicted_tokens & gt_tokens
        
        # Recall: 预测中包含的GT token数
        recall = len(intersection) / len(gt_tokens)
        
        # Precision: 预测token中有多少是GT的
        precision = len(intersection) / len(predicted_tokens) if predicted_tokens else 0.0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'token_recall': float(recall),
            'token_precision': float(precision),
            'token_f1': float(f1),
            'predicted_tokens': len(predicted_tokens),
            'gt_tokens': len(gt_tokens),
            'matched_tokens': len(intersection)
        }
    
    def _build_candidate_prompt(self, preference: str, asin: str, product_text: str) -> str:
        prompt = (
            f'You are a helpful assistant that examines if a product '
            f'satisfies the requirements in a given query and assigns a score from 0.0 to 1.0. '
            f'If the product does not satisfy any requirement in the query, the score should be 0.0. '
            f'If there is explicit and strong evidence supporting that product '
            f'satisfies all aspects mentioned by the query, the score should be 1.0. If partial or weak '
            f'evidence exists, the score should be between 0.0 and 1.0.\n'
            f'Here is the query:\n"{preference}"\n'
            f'Here is the information about the product:\n{product_text}\n\n'
            f'Please score the product based on how well it satisfies the query. '
            f'ONLY output the floating point score WITHOUT anything else. '
            f'Output: The numeric score of this product is: '
        )
        return prompt

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_float_score(self, text: str) -> float:
        if not text:
            return 0.0

        matches = re.findall(r'0\.\d+|1\.0', text)
        if len(matches) != 1:
            return 0.0
        return max(0.0, min(1.0, round(self._coerce_score(matches[0]), 4)))

    @staticmethod
    def _best_rank(products: List[str], gt_products: List[str]) -> int:
        gt_set = set(gt_products)
        for index, asin in enumerate(products, 1):
            if asin in gt_set:
                return index
        return 0
    
    def call_llm(self, cache_key: str, prompt: str, llm: str) -> str:
        """调用真实LLM API"""
        if cache_key in self.cache:
            logger.debug(f"缓存命中: {llm}")
            return self.cache[cache_key]

        try:
            logger.debug(f"调用LLM: {llm}, cache_key={cache_key}")

            if llm not in self.llm_clients:
                logger.warning(f"LLM {llm} 未初始化，使用默认LLM")
                llm_name = list(self.llm_clients.keys())[0]
            else:
                llm_name = llm
            
            client = self.llm_clients[llm_name]
            response = client.call(
                prompt=prompt,
                max_tokens=5,
                temperature=0.0,
                max_retries=3
            )

            self.cache[cache_key] = response

            return response
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""
    
    def _get_cache_key(self, products: List[str], preference: str, llm: str) -> str:
        """生成缓存key"""
        import hashlib
        content = f"{','.join(sorted(products))}_{preference}_{llm}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_product_document(self, asin: str) -> str:
        if asin in self.product_document_cache:
            return self.product_document_cache[asin]

        document_text = asin
        if self.meta_file and ProductVocabulary:
            try:
                vocab = ProductVocabulary(self.meta_file, [asin])
                if vocab.load_vocabulary_for_asin(asin):
                    document_text = vocab.asin_documents.get(asin, asin)
            except Exception as e:
                logger.warning(f"加载商品文档失败 {asin}: {e}")

        self.product_document_cache[asin] = document_text
        return document_text

    def _preload_product_documents(self, asins: List[str]) -> None:
        missing_asins = [asin for asin in asins if asin and asin not in self.product_document_cache]
        if not missing_asins:
            return

        fallback_docs = {asin: asin for asin in missing_asins}
        if not self.meta_file:
            self.product_document_cache.update(fallback_docs)
            return

        remaining_asins = set(missing_asins)
        try:
            with gzip.open(self.meta_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not remaining_asins:
                        break

                    try:
                        product = json.loads(line)
                    except Exception:
                        continue

                    asin = str(product.get('asin', ''))
                    if asin not in remaining_asins:
                        continue

                    title = product.get('title', '')
                    brand = product.get('brand', '')
                    feature = product.get('feature', [])
                    description = product.get('description', [])
                    document_parts = [title, brand] + feature + description
                    self.product_document_cache[asin] = ' '.join(str(part) for part in document_parts)
                    remaining_asins.remove(asin)
        except Exception as e:
            logger.warning(f"批量预加载商品文档失败: {e}")

        for asin in remaining_asins:
            self.product_document_cache[asin] = fallback_docs[asin]

    def _score_candidates(self, candidates: List[Dict[str, Any]], preference: str, llm: str) -> List[Dict[str, Any]]:
        self._preload_product_documents([
            str(candidate['asin']) for candidate in candidates if candidate.get('asin')
        ])

        prepared_candidates = []
        for candidate in candidates:
            asin = candidate.get('asin', '')
            if not asin:
                continue

            product_text = self._get_product_document(asin)
            prepared_candidate = dict(candidate)
            prepared_candidate['product_text'] = product_text
            prepared_candidates.append(prepared_candidate)

        def score_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
            asin = str(candidate['asin'])
            product_text = str(candidate.get('product_text', asin))
            cache_key = self._get_cache_key([asin], preference, llm)
            prompt = self._build_candidate_prompt(preference, asin, product_text)
            llm_output = self.call_llm(cache_key, prompt, llm)
            llm_score = self._extract_float_score(llm_output)
            original_rank = int(candidate.get('original_rank', 1))
            cand_len = max(1, len(prepared_candidates))
            sim_score = (cand_len - original_rank + 1) / cand_len
            fused_score = llm_score + self.sim_weight * sim_score

            scored_candidate = dict(candidate)
            scored_candidate.pop('product_text', None)
            scored_candidate['llm_score'] = llm_score
            scored_candidate['sim_score'] = sim_score
            scored_candidate['fused_score'] = fused_score
            scored_candidate['llm_output'] = llm_output

            return scored_candidate

        if not prepared_candidates:
            return []

        scored_candidates = []
        max_workers = max(1, min(self.scoring_max_workers, len(prepared_candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(score_one, candidate) for candidate in prepared_candidates]
            for future in as_completed(futures):
                scored_candidates.append(future.result())

        scored_candidates.sort(key=lambda item: (-item['fused_score'], item.get('original_rank', 9999)))
        for index, item in enumerate(scored_candidates, 1):
            item['rerank_rank'] = index

        return scored_candidates
    
    def evaluate_single_query(
        self,
        query_id: str,
        query_text: str,
        query_preference: str,
        candidate_entries: List[Dict[str, Any]],
        gt_products: List[str],
        sentiment: str,
        noise_level: str,
        retriever: str,
        llm: str
    ) -> Dict:
        
        # 计算检索recall@K
        retrieved_topk = [str(candidate['asin']) for candidate in candidate_entries if candidate.get('asin')]
        gt_in_topk = [p for p in gt_products if p in retrieved_topk]
        retrieval_recall_at_k = len(gt_in_topk) / len(gt_products) if gt_products else 0.0

        ranked_candidates = self._score_candidates(candidate_entries, query_preference, llm)
        reranked_asins = [str(item['asin']) for item in ranked_candidates if item.get('asin')]

        retrieval_best_rank = self._best_rank(retrieved_topk, gt_products)
        rerank_best_rank = self._best_rank(reranked_asins, gt_products)

        retrieval_mrr = 1.0 / retrieval_best_rank if retrieval_best_rank else 0.0
        rerank_mrr = 1.0 / rerank_best_rank if rerank_best_rank else 0.0
        total_gap = rerank_mrr - retrieval_mrr

        rerank_metrics = {
            'token_recall': float(rerank_mrr),
            'retrieval_mrr': float(retrieval_mrr),
            'rerank_mrr': float(rerank_mrr),
            'retrieval_best_rank': retrieval_best_rank,
            'rerank_best_rank': rerank_best_rank,
            'top1_hit': float(rerank_best_rank == 1),
            'top3_hit': float(0 < rerank_best_rank <= 3),
            'top5_hit': float(0 < rerank_best_rank <= 5),
            'scored_candidates': len(ranked_candidates)
        }
        
        # 构建结果
        result = {
            'query_id': query_id,
            'query_text': query_text,
            'sentiment': sentiment,
            'noise_level': noise_level,
            'retriever': retriever,
            'llm': llm,
            'retrieval_recall_at_k': float(retrieval_recall_at_k),
            'rerank_metrics': rerank_metrics,
            'rerank_output': json.dumps(ranked_candidates, ensure_ascii=False),
            'reranked_asins': reranked_asins,
            'candidate_scores': ranked_candidates,
            'total_gap': float(total_gap),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def evaluate_query(self, query_id: str) -> List[Dict]:
        """评估单个查询的全量组合"""
        logger.info(f"开始评估查询: {query_id}")
        
        # 加载查询数据
        query_data = self.load_query_data(query_id)
        if not query_data:
            logger.error(f"无法加载查询 {query_id}")
            return []
        
        evaluation_config = self.config.get("evaluation_config", {})
        use_persona = evaluation_config.get("use_persona", True)

        # 进行属性选择评估（强制商品验证）
        attribute_selection_result = None
        if use_persona and self.attribute_selector and hasattr(query_data, '__getitem__'):
            try:
                user_id = query_data.get('user_id', '')
                category = query_data.get('category', '')
                target_asin = query_data.get('target_asin') or query_data.get('asin') or query_id
                
                if not target_asin:
                    logger.error("查询缺少target_asin，属性验证需要指定目标商品ASIN")
                    raise ValueError("target_asin is required for attribute validation")
                
                if user_id and category:
                    # 强制使用带验证的选择器
                    attribute_selection_result = self.attribute_selector.evaluate_attributes(
                        query_data, user_id, category, target_asin=target_asin
                    )
                    logger.info(f"属性选择评估完成（强制验证）: combined_quality={attribute_selection_result.get('combined_quality_score', 'N/A')}")
            except Exception as e:
                logger.error(f"属性选择评估失败: {e}")
        elif not use_persona:
            logger.info("已禁用用户画像，跳过Persona和属性选择评估")
        
        # 加载GT和sentiment
        ground_truth = self.load_ground_truth()
        gt_products = ground_truth.get(query_id, {}).get('products', [])
        query_sentiment = query_data.get('sentiment')
        
        if not gt_products:
            logger.warning(f"查询 {query_id} 没有GT")
            return []
        
        query_text = query_data.get('query_text', '')
        query_preference = query_data.get('query_preference', '') if use_persona else query_text
        
        results = []

        sentiment = query_sentiment or "unknown"
        noise_levels = evaluation_config.get("noise_levels", ["noisy"])
        retrievers = evaluation_config.get("retrievers", ["dense"])
        llms = evaluation_config.get("llms", ["GLM-4.7"])

        noise_level = noise_levels[0] if noise_levels else "noisy"
        retriever = retrievers[0] if retrievers else "dense"
        llm = llms[0] if llms else "GLM-4.7"

        retrieval_results = self.load_retrieval_results(retriever, noise_level, query_data.get('user_id', ''))
        candidate_entries = retrieval_results.get(query_id, {}).get('candidate_entries', [])

        try:
            result = self.evaluate_single_query(
                query_id=query_id,
                query_text=query_text,
                query_preference=query_preference,
                candidate_entries=candidate_entries,
                gt_products=gt_products,
                sentiment=sentiment,
                noise_level=noise_level,
                retriever=retriever,
                llm=llm
            )

            if attribute_selection_result:
                result['attribute_selection'] = {
                    'quality_score': attribute_selection_result.get('quality_score', 0.0),
                    'attribute_f1': attribute_selection_result.get('attribute_f1', 0.0),
                    'profile_f1': attribute_selection_result.get('profile_f1', 0.0),
                    'dimension_coverage': attribute_selection_result.get('dimension_coverage', 0),
                    'combined_quality_score': attribute_selection_result.get('combined_quality_score', 0.0),
                    'product_validation': attribute_selection_result.get('product_validation', {}),
                    'validation_stats': attribute_selection_result.get('validation_stats', {})
                }

            results.append(result)
        except Exception as e:
            logger.error(f"评估失败 {query_id}: {e}")
        
        logger.info(f"查询 {query_id} 完成，共 {len(results)} 条结果")
        return results


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Stage 13: 单个查询的三路评估')
    parser.add_argument('--query-id', type=str, required=True, help='查询ID')
    parser.add_argument('--config', type=str, default='15_config.json', help='配置文件路径')
    parser.add_argument('--output-file', type=str, help='输出文件路径（可选）')
    parser.add_argument('--meta-file', type=str, help='商品元数据文件路径（用于属性商品验证，可选）')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = UserQueryEvaluator(config_path=args.config, meta_file=args.meta_file)
    results = evaluator.evaluate_query(args.query_id)
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {args.output_file}")
    else:
        # 标准输出
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    return results


if __name__ == '__main__':
    main()
