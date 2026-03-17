#!/usr/bin/env python3
"""
评估预训练的e5-base-v2（不微调）的检索性能
看看微调的收益是多少
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# 使用原始LOOCV数据
holdout_file = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data/user_A3E5V5TSTAY3R9/holdout.json')
with open(holdout_file) as f:
    holdout_data = json.load(f)['pairs']

# 加载预训练模型
print("="*80)
print("加载预训练的e5-base-v2模型（不微调）")
print("="*80)
model_pretrained = SentenceTransformer("intfloat/e5-base-v2")
print("✓ 模型加载完成")

# 加载产品embeddings（用预训练模型计算）
print("\n计算产品embeddings...")
metadata_file = Path("/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl")
with open(metadata_file, 'rb') as f:
    product_metadata = pickle.load(f)

all_product_asins = list(product_metadata.keys())
product_embeddings_pretrained = {}

batch_size = 32
for i in range(0, len(all_product_asins), batch_size):
    batch_asins = all_product_asins[i:i+batch_size]
    batch_titles = []
    
    for asin in batch_asins:
        if asin in product_metadata:
            title = product_metadata[asin].get('title', '')
            batch_titles.append(title)
        else:
            batch_titles.append('')
    
    if batch_titles:
        embeddings = model_pretrained.encode(batch_titles, batch_size=batch_size, 
                                            convert_to_numpy=True, show_progress_bar=False)
        for asin, emb in zip(batch_asins, embeddings):
            product_embeddings_pretrained[asin] = emb
    
    if (i // batch_size + 1) % 100 == 0:
        print(f"  已处理: {min(i + batch_size, len(all_product_asins))}/{len(all_product_asins)}")

print(f"✓ 计算完成，共 {len(product_embeddings_pretrained)} 个产品embeddings")

# 评估函数
def compute_metrics(ranked_asins, relevant_asin, k=10):
    ranked_k = ranked_asins[:k]
    
    if relevant_asin in ranked_k:
        position = ranked_k.index(relevant_asin)
        mrr = 1.0 / (position + 1)
        dcg = 1.0 / np.log2(position + 2)
    else:
        mrr = 0.0
        dcg = 0.0
    
    idcg = 1.0 / np.log2(2)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {'mrr': mrr, 'ndcg': ndcg}

# 评估预训练模型
print("\n" + "="*80)
print("评估预训练e5-base-v2（不微调）")
print("="*80)

k_values = [1, 3, 5, 10]
all_metrics = {k: [] for k in k_values}

valid_queries = 0
for pair in holdout_data:
    # 用NOISY query评估
    noisy_query = pair.get('positive', '')
    target_asin = pair.get('asin', '')
    
    if not (noisy_query and target_asin):
        continue
    
    valid_queries += 1
    
    # 编码NOISY query
    query_emb = model_pretrained.encode(noisy_query, convert_to_numpy=True)
    
    # 计算与所有产品的相似度
    scores = []
    for asin, prod_emb in product_embeddings_pretrained.items():
        if prod_emb is not None:
            sim = np.dot(query_emb, prod_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(prod_emb) + 1e-8
            )
            scores.append((asin, sim))
    
    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_asins = [asin for asin, _ in scores]
    
    # 计算各k值的指标
    for k in k_values:
        metrics = compute_metrics(ranked_asins, target_asin, k)
        all_metrics[k].append(metrics)

# 聚合结果
print(f"\n有效查询数: {valid_queries}/{len(holdout_data)}")
print("\n预训练模型的性能指标：")
print("="*80)

results_dict = {}
for k in k_values:
    if all_metrics[k]:
        avg_mrr = np.mean([m['mrr'] for m in all_metrics[k]])
        avg_ndcg = np.mean([m['ndcg'] for m in all_metrics[k]])
        results_dict[f'mrr@{k}'] = avg_mrr
        results_dict[f'ndcg@{k}'] = avg_ndcg
        print(f"k={k:2d}  |  MRR = {avg_mrr:.4f}  |  NDCG = {avg_ndcg:.4f}")

# 对比微调版本
print("\n" + "="*80)
print("与微调版本对比")
print("="*80)

results_file = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/results/comparison_results.json')
with open(results_file) as f:
    finetuned_all = json.load(f)

# 使用同一用户A3E5V5TSTAY3R9的结果
user_results = finetuned_all['A3E5V5TSTAY3R9']

print("\n预训练 vs 个性化微调 (Exp B) - 用户A3E5V5TSTAY3R9:")
print("-" * 80)
for k in k_values:
    pretrained_mrr = results_dict.get(f'mrr@{k}', 0.0)
    finetuned_mrr = user_results['experiment_b'].get(f'mrr@{k}', 0.0)
    
    improvement = finetuned_mrr - pretrained_mrr
    improvement_pct = (improvement / pretrained_mrr * 100) if pretrained_mrr > 0 else (float('inf') if finetuned_mrr > 0 else 0)
    
    print(f"k={k:2d}  |  预训练: {pretrained_mrr:.4f}  |  微调: {finetuned_mrr:.4f}  |  改善: {improvement:+.4f}")

