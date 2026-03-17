#!/usr/bin/env python3
"""
Unified LOOCV Experiment: Personalized vs Cross-User Error Patterns

实验 A：个性化用户错误模式
- 对于每个测试用户，仅使用该用户自己的错误模式（clean-noisy 查询对）微调模型
- 学习该特定用户的拼写习惯

实验 B：跨用户混合错误模式  
- 使用其他用户的错误模式混合微调，不包含目标用户的数据
- 学习通用的噪声模式，而不是用户特有的模式

流程：
1. prepare_loocv_data() - 为每个用户准备训练/验证/测试数据集
2. train_experiment_a() - 实验 A：使用个人数据微调模型
3. train_experiment_b() - 实验 B：使用跨用户数据微调模型
4. evaluate_and_compare() - 评估并对比两个实验的结果
"""

import json
import os
import sys
import torch
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import random
from collections import defaultdict

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LOOCVExperiment:
    def __init__(self, data_dir: str = '.', output_dir: str = './loocv_results', 
                 doc_cache_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loocv_data_dir = self.output_dir / 'loocv_data'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        
        self.loocv_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.doc_cache_dir = Path(doc_cache_dir) if doc_cache_dir else None
        self.all_product_asins = None
        self.product_metadata = None
        self._load_product_library()
    
    def prepare_loocv_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        logger.info("=" * 80)
        logger.info("準備 LOOCV 數據分割")
        logger.info("=" * 80)
        
        all_data = self._load_all_training_data()
        user_data = self._group_data_by_user(all_data)
        
        logger.info(f"總用戶數: {len(user_data)}")
        logger.info(f"總數據點: {len(all_data)}")
        
        loocv_splits = {}
        for target_user in user_data.keys():
            user_dir = self.loocv_data_dir / f'user_{target_user}'
            user_dir.mkdir(parents=True, exist_ok=True)
            
            global_train, personal_train, holdout = self._split_loocv_for_user(
                target_user, user_data
            )
            
            self._save_json(user_dir / 'global_train.json', {
                'pairs': global_train,
                'description': f'Other users training data (for experiment B)'
            })
            self._save_json(user_dir / 'personal_train.json', {
                'pairs': personal_train,
                'description': f'User {target_user} personal training data (for experiment A)'
            })
            self._save_json(user_dir / 'holdout.json', {
                'pairs': holdout,
                'description': f'User {target_user} validation data'
            })
            
            loocv_splits[target_user] = {
                'global_train': global_train,
                'personal_train': personal_train,
                'holdout': holdout
            }
            
            logger.info(f"✓ {target_user}: global={len(global_train)}, personal={len(personal_train)}, holdout={len(holdout)}")
        
        return loocv_splits
    
    def _precompute_embeddings(self, model: SentenceTransformer, model_name: str) -> Dict[str, np.ndarray]:
        logger.info(f"預計算 {len(self.all_product_asins)} 個商品的embeddings...")
        product_embeddings = {}
        
        batch_size = 32
        for i in range(0, len(self.all_product_asins), batch_size):
            batch_asins = self.all_product_asins[i:i+batch_size]
            batch_titles = []
            
            for asin in batch_asins:
                if asin in self.product_metadata:
                    title = self.product_metadata[asin].get('title', '')
                    batch_titles.append(title)
                else:
                    batch_titles.append('')
            
            if batch_titles:
                embeddings = model.encode(batch_titles, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
                for asin, emb in zip(batch_asins, embeddings):
                    product_embeddings[asin] = emb
            
            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"  已處理: {min(i + batch_size, len(self.all_product_asins))}/{len(self.all_product_asins)}")
        
        logger.info(f"✓ 預計算完成，共 {len(product_embeddings)} 個商品embeddings")
        return product_embeddings
    
    def train_experiment_a(self, loocv_splits: Dict) -> Dict[str, str]:
        logger.info("\n" + "=" * 80)
        logger.info("實驗 A：個性化用戶錯誤模式微調 (Dense 模型)")
        logger.info("=" * 80)
        
        base_model = "intfloat/e5-base-v2"
        checkpoints = {}
        
        for user_id, splits in loocv_splits.items():
            logger.info(f"\n微調用戶 {user_id} 的模型...")
            
            model = SentenceTransformer(base_model)
            train_examples = self._prepare_training_examples(splits['personal_train'])
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(model)
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,
                warmup_steps=100,
                show_progress_bar=True
            )
            
            checkpoint_path = self.checkpoints_dir / f'experiment_a_user_{user_id}'
            model.save(str(checkpoint_path))
            checkpoints[f'experiment_a_{user_id}'] = str(checkpoint_path)
            
            logger.info(f"✓ 模型已保存到 {checkpoint_path}")
            
            product_embeddings = self._precompute_embeddings(model, f'experiment_a_{user_id}')
            embedding_cache_path = self.checkpoints_dir / f'embeddings_experiment_a_{user_id}.pkl'
            with open(embedding_cache_path, 'wb') as f:
                pickle.dump(product_embeddings, f)
            logger.info(f"✓ Embeddings已保存到 {embedding_cache_path}")
        
        return checkpoints
    
    def train_experiment_b(self, loocv_splits: Dict) -> Dict[str, str]:
        logger.info("\n" + "=" * 80)
        logger.info("實驗 B：跨用戶混合錯誤模式微調 (Dense 模型)")
        logger.info("=" * 80)
        
        base_model = "intfloat/e5-base-v2"
        checkpoints = {}
        
        for target_user, splits in loocv_splits.items():
            logger.info(f"\n使用其他用戶數據微調 {target_user} 的模型...")
            
            model = SentenceTransformer(base_model)
            train_examples = self._prepare_training_examples(splits['global_train'])
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(model)
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,
                warmup_steps=100,
                show_progress_bar=True
            )
            
            checkpoint_path = self.checkpoints_dir / f'experiment_b_user_{target_user}'
            model.save(str(checkpoint_path))
            checkpoints[f'experiment_b_{target_user}'] = str(checkpoint_path)
            
            logger.info(f"✓ 模型已保存到 {checkpoint_path}")
            
            product_embeddings = self._precompute_embeddings(model, f'experiment_b_{target_user}')
            embedding_cache_path = self.checkpoints_dir / f'embeddings_experiment_b_{target_user}.pkl'
            with open(embedding_cache_path, 'wb') as f:
                pickle.dump(product_embeddings, f)
            logger.info(f"✓ Embeddings已保存到 {embedding_cache_path}")
        
        return checkpoints
    
    def evaluate_and_compare(self, loocv_splits: Dict, checkpoints: Dict) -> Dict:
        logger.info("\n" + "=" * 80)
        logger.info("評估並對比結果")
        logger.info("=" * 80)
        
        results = defaultdict(dict)
        
        for user_id, splits in loocv_splits.items():
            holdout_data = splits['holdout']
            if not holdout_data:
                continue
            
            model_a_path = checkpoints.get(f'experiment_a_{user_id}')
            model_b_path = checkpoints.get(f'experiment_b_{user_id}')
            
            if not (model_a_path and model_b_path):
                logger.warning(f"跳過 {user_id}：缺少檢查點")
                continue
            
            logger.info(f"\n評估用戶 {user_id}...")
            
            embedding_a_path = self.checkpoints_dir / f'embeddings_experiment_a_{user_id}.pkl'
            embedding_b_path = self.checkpoints_dir / f'embeddings_experiment_b_{user_id}.pkl'
            
            with open(embedding_a_path, 'rb') as f:
                product_embeddings_a = pickle.load(f)
            logger.info(f"  ✓ 加載實驗A embeddings: {len(product_embeddings_a)} 商品")
            
            with open(embedding_b_path, 'rb') as f:
                product_embeddings_b = pickle.load(f)
            logger.info(f"  ✓ 加載實驗B embeddings: {len(product_embeddings_b)} 商品")
            
            model_a = SentenceTransformer(model_a_path)
            model_b = SentenceTransformer(model_b_path)
            
            metrics_a = self._evaluate_model(model_a, holdout_data, product_embeddings_a)
            metrics_b = self._evaluate_model(model_b, holdout_data, product_embeddings_b)
            
            mrr_a = metrics_a.get('mrr@10', 0.0)
            mrr_b = metrics_b.get('mrr@10', 0.0)
            
            results[user_id] = {
                'experiment_a': metrics_a,
                'experiment_b': metrics_b,
                'improvement': mrr_a - mrr_b
            }
            
            logger.info(f"\n用戶 {user_id}:")
            logger.info(f"  實驗 A (個性化)：MRR@10 = {mrr_a:.4f}, NDCG@10 = {metrics_a.get('ndcg@10', 0.0):.4f}")
            logger.info(f"  實驗 B (跨用戶)：MRR@10 = {mrr_b:.4f}, NDCG@10 = {metrics_b.get('ndcg@10', 0.0):.4f}")
            logger.info(f"  改進：{results[user_id]['improvement']:+.4f}")
        
        self._save_json(self.results_dir / 'comparison_results.json', dict(results))
        self._print_summary(results)
        
        return results
    
    def _load_all_training_data(self) -> List[Dict]:
        all_data = []
        
        for data_file in ['training_data_v4_stratified.json', 'holdout_data_v4_stratified.json', 'test_data_v4_stratified.json']:
            file_path = self.data_dir / data_file
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    all_data.extend(data.get('pairs', []))
                    logger.info(f"  載入 {data_file}: {len(data.get('pairs', []))} 對")
        
        return all_data
    
    def _group_data_by_user(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        user_data = defaultdict(list)
        for pair in data:
            user_id = pair.get('user_id')
            if user_id:
                user_data[user_id].append(pair)
        return user_data
    
    def _split_loocv_for_user(self, target_user: str, user_data: Dict) -> Tuple[List, List, List]:
        global_train = []
        for user_id, pairs in user_data.items():
            if user_id != target_user:
                global_train.extend(pairs)
        
        target_pairs = user_data[target_user]
        random.shuffle(target_pairs)
        
        split_idx = int(len(target_pairs) * 0.8)
        personal_train = target_pairs[:split_idx]
        holdout = target_pairs[split_idx:]
        
        return global_train, personal_train, holdout
    
    def _load_product_library(self):
        if not self.doc_cache_dir:
            self.doc_cache_dir = Path("/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache")
        
        metadata_file = self.doc_cache_dir / "Arts_Crafts_and_Sewing_metadata.pkl"
        if metadata_file.exists():
            logger.info(f"加載全量產品元數據 (302k商品)...")
            with open(metadata_file, 'rb') as f:
                self.product_metadata = pickle.load(f)
            self.all_product_asins = list(self.product_metadata.keys())
            logger.info(f"✓ 已加載 {len(self.all_product_asins)} 個產品 ASINs")
        else:
            logger.warning(f"找不到產品元數據文件: {metadata_file}")
            self.product_metadata = {}
            self.all_product_asins = []
    
    def _prepare_training_examples(self, pairs: List[Dict]) -> List[InputExample]:
        examples = []
        for pair in pairs:
            query = pair.get('query', '')
            positive = pair.get('positive', '')
            
            if query and positive:
                examples.append(InputExample(
                    texts=[query, positive],
                    label=1.0
                ))
        
        return examples
    
    def _compute_metrics(self, ranked_asins: List[str], relevant_asin: str, k: int = 10) -> Dict:
        ranked_k = ranked_asins[:k]
        
        num_relevant = 1 if relevant_asin in ranked_k else 0
        precision = num_relevant / k if k > 0 else 0
        recall = num_relevant
        
        ap = 0
        if relevant_asin in ranked_k:
            position = ranked_k.index(relevant_asin)
            ap = 1.0 / (position + 1)
        
        dcg = 0
        if relevant_asin in ranked_k:
            position = ranked_k.index(relevant_asin)
            dcg = 1.0 / np.log2(position + 2)
        
        idcg = 1.0 / np.log2(2)
        ndcg = dcg / idcg if idcg > 0 else 0
        
        mrr = 0
        if relevant_asin in ranked_k:
            position = ranked_k.index(relevant_asin)
            mrr = 1.0 / (position + 1)
        
        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'ap': ap,
            'ndcg': ndcg,
            'mrr': mrr
        }
    
    def _evaluate_model(self, model: SentenceTransformer, test_data: List[Dict], 
                       product_embeddings: Dict[str, np.ndarray], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        if not test_data or not product_embeddings:
            return {f'mrr@{k}': 0.0 for k in k_values}
        
        all_metrics = {k: [] for k in k_values}
        
        for pair in test_data:
            # ✅ 修复：用NOISY query评估模型的纠正能力
            # 之前：query = pair.get('query', '')  # CLEAN - 无法测试纠正
            # 现在：query = pair.get('positive', '')  # NOISY - 测试纠正能力
            query = pair.get('positive', '')
            asin = pair.get('asin', '')
            
            if not (query and asin):
                continue
            
            try:
                query_embedding = model.encode(query, convert_to_numpy=True)
                
                similarities = []
                for product_asin, product_emb in product_embeddings.items():
                    if product_emb is not None:
                        sim = np.dot(query_embedding, product_emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(product_emb) + 1e-8
                        )
                        similarities.append((product_asin, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                ranked_asins = [asin_sim[0] for asin_sim in similarities]
                
                for k in k_values:
                    metrics = self._compute_metrics(ranked_asins, asin, k)
                    all_metrics[k].append(metrics)
            
            except Exception as e:
                logger.warning(f"評估查詢時出錯: {query[:50]}... - {str(e)}")
                continue
        
        aggregated = {}
        for k in k_values:
            if all_metrics[k]:
                avg_mrr = np.mean([m['mrr'] for m in all_metrics[k]])
                avg_ndcg = np.mean([m['ndcg'] for m in all_metrics[k]])
                avg_precision = np.mean([m['precision_at_k'] for m in all_metrics[k]])
                avg_recall = np.mean([m['recall_at_k'] for m in all_metrics[k]])
                avg_ap = np.mean([m['ap'] for m in all_metrics[k]])
                
                aggregated[f'mrr@{k}'] = round(avg_mrr, 4)
                aggregated[f'ndcg@{k}'] = round(avg_ndcg, 4)
                aggregated[f'p@{k}'] = round(avg_precision, 4)
                aggregated[f'r@{k}'] = round(avg_recall, 4)
                aggregated[f'ap@{k}'] = round(avg_ap, 4)
            else:
                aggregated[f'mrr@{k}'] = 0.0
                aggregated[f'ndcg@{k}'] = 0.0
        
        return aggregated
    
    def _save_json(self, path: Path, data: Any):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _print_summary(self, results: Dict):
        logger.info("\n" + "=" * 80)
        logger.info("最終對比結果")
        logger.info("=" * 80)
        
        improvements = []
        for user_id, metrics in results.items():
            improvement = metrics['improvement']
            improvements.append(improvement)
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            logger.info(f"平均改進：{avg_improvement:+.4f}")
            logger.info(f"最大改進：{max(improvements):+.4f}")
            logger.info(f"最小改進：{min(improvements):+.4f}")


def main():
    script_dir = Path(__file__).parent.absolute()
    doc_cache_dir = "/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache"
    experiment = LOOCVExperiment(
        data_dir=str(script_dir), 
        output_dir=str(script_dir / 'loocv_results'),
        doc_cache_dir=doc_cache_dir
    )
    
    logger.info("開始統一 LOOCV 實驗\n")
    logger.info(f"輸出目錄：{experiment.output_dir}")
    
    loocv_splits = experiment.prepare_loocv_data()
    
    checkpoints_a = experiment.train_experiment_a(loocv_splits)
    checkpoints_b = experiment.train_experiment_b(loocv_splits)
    
    checkpoints = {**checkpoints_a, **checkpoints_b}
    
    results = experiment.evaluate_and_compare(loocv_splits, checkpoints)
    
    logger.info(f"\n✓ 實驗完成！結果已保存到 {experiment.results_dir}")


if __name__ == '__main__':
    main()
