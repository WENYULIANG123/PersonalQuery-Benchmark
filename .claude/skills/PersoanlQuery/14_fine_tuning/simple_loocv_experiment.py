#!/usr/bin/env python3
"""
简化版LOOCV实验：只训练2个模型
实验 A：用10个用户的混合数据微调（跨用户）
实验 B：用1个holdout用户自己的数据微调（个性化）

只需要在holdout用户上测试，看哪个更有效
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


class SimpleLOOCVExperiment:
    def __init__(self, data_dir: str = '.', output_dir: str = './loocv_results_simple', 
                 doc_cache_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.doc_cache_dir = Path(doc_cache_dir) if doc_cache_dir else None
        self.all_product_asins = None
        self.product_metadata = None
        self._load_product_library()
    
    def prepare_simple_split(self, holdout_user: str = None) -> Tuple[List, List, List]:
        """
        简化分割：选择一个用户作为holdout，其他作为训练
        Args:
            holdout_user: holdout用户ID。如果None，自动选择数据最多的用户
        
        Returns:
            (cross_user_train, personal_train, holdout_test)
        """
        all_data = self._load_all_training_data()
        user_data = self._group_data_by_user(all_data)
        
        # 如果没指定holdout用户，选择数据最多的
        if holdout_user is None:
            holdout_user = max(user_data.keys(), key=lambda u: len(user_data[u]))
        
        logger.info("="*80)
        logger.info(f"简化LOOCV: 用户 {holdout_user} 作为holdout")
        logger.info("="*80)
        
        # 分割数据
        cross_user_train = []
        for user_id, pairs in user_data.items():
            if user_id != holdout_user:
                cross_user_train.extend(pairs)
        
        target_pairs = user_data[holdout_user]
        random.shuffle(target_pairs)
        
        split_idx = int(len(target_pairs) * 0.8)
        personal_train = target_pairs[:split_idx]
        holdout_test = target_pairs[split_idx:]
        
        logger.info(f"✓ 跨用户训练数据: {len(cross_user_train)} 对（来自其他{len(user_data)-1}个用户）")
        logger.info(f"✓ 个性化训练数据: {len(personal_train)} 对（来自{holdout_user}）")
        logger.info(f"✓ Holdout测试数据: {len(holdout_test)} 对（来自{holdout_user}）")
        
        return cross_user_train, personal_train, holdout_test, holdout_user
    
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
    
    def train_experiment_a(self, cross_user_train: List[Dict]) -> str:
        """实验A: 跨用户混合训练"""
        logger.info("\n" + "="*80)
        logger.info("實驗 A：跨用戶混合錯誤模式微調")
        logger.info("="*80)
        
        base_model = "intfloat/e5-base-v2"
        logger.info(f"\n訓練模型...")
        
        model = SentenceTransformer(base_model)
        train_examples = self._prepare_training_examples(cross_user_train)
        
        logger.info(f"  訓練樣本: {len(train_examples)}")
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        checkpoint_path = self.checkpoints_dir / 'experiment_a'
        model.save(str(checkpoint_path))
        logger.info(f"✓ 模型已保存到 {checkpoint_path}")
        
        product_embeddings = self._precompute_embeddings(model, 'experiment_a')
        embedding_cache_path = self.checkpoints_dir / 'embeddings_experiment_a.pkl'
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump(product_embeddings, f)
        logger.info(f"✓ Embeddings已保存到 {embedding_cache_path}")
        
        return str(checkpoint_path)
    
    def train_experiment_b(self, personal_train: List[Dict]) -> str:
        """实验B: 个性化训练"""
        logger.info("\n" + "="*80)
        logger.info("實驗 B：個性化用戶錯誤模式微調")
        logger.info("="*80)
        
        base_model = "intfloat/e5-base-v2"
        logger.info(f"\n訓練模型...")
        
        model = SentenceTransformer(base_model)
        train_examples = self._prepare_training_examples(personal_train)
        
        logger.info(f"  訓練樣本: {len(train_examples)}")
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        checkpoint_path = self.checkpoints_dir / 'experiment_b'
        model.save(str(checkpoint_path))
        logger.info(f"✓ 模型已保存到 {checkpoint_path}")
        
        product_embeddings = self._precompute_embeddings(model, 'experiment_b')
        embedding_cache_path = self.checkpoints_dir / 'embeddings_experiment_b.pkl'
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump(product_embeddings, f)
        logger.info(f"✓ Embeddings已保存到 {embedding_cache_path}")
        
        return str(checkpoint_path)
    
    def evaluate_and_compare(self, holdout_data: List[Dict], model_a_path: str, 
                            model_b_path: str, holdout_user: str) -> Dict:
        logger.info("\n" + "="*80)
        logger.info(f"評估並對比結果 (用戶: {holdout_user})")
        logger.info("="*80)
        
        embedding_a_path = self.checkpoints_dir / 'embeddings_experiment_a.pkl'
        embedding_b_path = self.checkpoints_dir / 'embeddings_experiment_b.pkl'
        
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
        
        results = {
            'holdout_user': holdout_user,
            'experiment_a_cross_user': metrics_a,
            'experiment_b_personalized': metrics_b,
            'improvement_a_over_b': mrr_a - mrr_b
        }
        
        logger.info(f"\n用戶 {holdout_user}:")
        logger.info(f"  實驗 A (跨用戶)：MRR@10 = {mrr_a:.4f}, NDCG@10 = {metrics_a.get('ndcg@10', 0.0):.4f}")
        logger.info(f"  實驗 B (個性化)：MRR@10 = {mrr_b:.4f}, NDCG@10 = {metrics_b.get('ndcg@10', 0.0):.4f}")
        logger.info(f"  改進 (A-B)：{results['improvement_a_over_b']:+.4f}")
        
        self._save_json(self.results_dir / 'results.json', results)
        
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
            # 用NOISY query评估模型的纠正能力
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
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    script_dir = Path(__file__).parent.absolute()
    doc_cache_dir = "/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache"
    
    experiment = SimpleLOOCVExperiment(
        data_dir=str(script_dir), 
        output_dir=str(script_dir / 'loocv_results_simple'),
        doc_cache_dir=doc_cache_dir
    )
    
    logger.info("開始簡化 LOOCV 實驗\n")
    logger.info(f"輸出目錄：{experiment.output_dir}")
    
    # 分割数据：选择数据最多的用户作为holdout
    cross_user_train, personal_train, holdout_test, holdout_user = experiment.prepare_simple_split()
    
    # 训练2个模型
    model_a_path = experiment.train_experiment_a(cross_user_train)
    model_b_path = experiment.train_experiment_b(personal_train)
    
    # 评估
    results = experiment.evaluate_and_compare(holdout_test, model_a_path, model_b_path, holdout_user)
    
    logger.info(f"\n✓ 實驗完成！結果已保存到 {experiment.results_dir}")
    logger.info("\n" + "="*80)
    logger.info("實驗結論")
    logger.info("="*80)
    logger.info(f"跨用户混合训练 vs 个性化单用户训练")
    logger.info(f"在用户 {holdout_user} 上的表现对比：")
    logger.info(f"  跨用户训练：MRR@10 = {results['experiment_a_cross_user']['mrr@10']:.4f}")
    logger.info(f"  个性化训练：MRR@10 = {results['experiment_b_personalized']['mrr@10']:.4f}")
    logger.info(f"  差异：{results['improvement_a_over_b']:+.4f}")


if __name__ == '__main__':
    main()
