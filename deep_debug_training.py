#!/usr/bin/env python3
"""
深度调试：检查Experiment B为什么失败
假设：模型在训练时学到了什么，但在评估时无法应用
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 加载模型
model_path_a = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/experiment_a_user_A13OFOB1394G31')
model_path_b = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/experiment_b_user_A13OFOB1394G31')

print("="*80)
print("加载模型")
print("="*80)

try:
    model_a = SentenceTransformer(str(model_path_a))
    print(f"✓ Experiment A 模型加载成功")
except Exception as e:
    print(f"✗ Experiment A 加载失败: {e}")
    model_a = None

try:
    model_b = SentenceTransformer(str(model_path_b))
    print(f"✓ Experiment B 模型加载成功")
except Exception as e:
    print(f"✗ Experiment B 加载失败: {e}")
    model_b = None

# 加载holdout和embeddings
holdout_file = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data/user_A13OFOB1394G31/holdout.json')
with open(holdout_file) as f:
    holdout_data = json.load(f)['pairs']

emb_a_path = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/embeddings_experiment_a_A13OFOB1394G31.pkl')
emb_b_path = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/embeddings_experiment_b_A13OFOB1394G31.pkl')

with open(emb_a_path, 'rb') as f:
    embeddings_a = pickle.load(f)
with open(emb_b_path, 'rb') as f:
    embeddings_b = pickle.load(f)

print(f"✓ Embeddings A: {len(embeddings_a)} 产品")
print(f"✓ Embeddings B: {len(embeddings_b)} 产品")

print("\n" + "="*80)
print("测试第一个holdout查询")
print("="*80)

test_pair = holdout_data[0]
query = test_pair['query']
target_asin = test_pair['asin']

print(f"\n查询: {query[:60]}...")
print(f"目标ASIN: {target_asin}")

if model_a and model_b:
    # 编码查询
    query_emb_a = model_a.encode(query, convert_to_numpy=True)
    query_emb_b = model_b.encode(query, convert_to_numpy=True)
    
    print(f"\n查询嵌入尺寸:")
    print(f"  A: {query_emb_a.shape}")
    print(f"  B: {query_emb_b.shape}")
    
    # 计算相似度
    scores_a = []
    scores_b = []
    target_rank_a = None
    target_rank_b = None
    
    for i, (asin, emb) in enumerate(embeddings_a.items()):
        if emb is None:
            continue
        sim_a = np.dot(query_emb_a, emb) / (np.linalg.norm(query_emb_a) * np.linalg.norm(emb) + 1e-8)
        sim_b = np.dot(query_emb_b, embeddings_b[asin]) / (np.linalg.norm(query_emb_b) * np.linalg.norm(embeddings_b[asin]) + 1e-8)
        
        scores_a.append((asin, sim_a))
        scores_b.append((asin, sim_b))
        
        if asin == target_asin:
            target_rank_a = len(scores_a)
            target_rank_b = len(scores_b)
    
    # 排序
    scores_a.sort(key=lambda x: x[1], reverse=True)
    scores_b.sort(key=lambda x: x[1], reverse=True)
    
    # 找到目标ASIN的排名
    for rank, (asin, sim) in enumerate(scores_a[:10], 1):
        if asin == target_asin:
            target_rank_a = rank
            print(f"\n✓ Experiment A: 目标ASIN排名 #{rank}, 相似度={sim:.4f}")
            break
    else:
        target_rank_a = None
        if target_asin in [asin for asin, _ in scores_a[:100]]:
            rank = [asin for asin, _ in scores_a].index(target_asin) + 1
            print(f"\n△ Experiment A: 目标ASIN在Top-100中，排名 #{rank}")
        else:
            print(f"\n✗ Experiment A: 目标ASIN不在Top-100中")
    
    for rank, (asin, sim) in enumerate(scores_b[:10], 1):
        if asin == target_asin:
            target_rank_b = rank
            print(f"✓ Experiment B: 目标ASIN排名 #{rank}, 相似度={sim:.4f}")
            break
    else:
        target_rank_b = None
        if target_asin in [asin for asin, _ in scores_b[:100]]:
            rank = [asin for asin, _ in scores_b].index(target_asin) + 1
            print(f"△ Experiment B: 目标ASIN在Top-100中，排名 #{rank}")
        else:
            print(f"✗ Experiment B: 目标ASIN不在Top-100中")
    
    # 显示Top-5
    print(f"\nExperiment A Top-5:")
    for rank, (asin, sim) in enumerate(scores_a[:5], 1):
        marker = "→" if asin == target_asin else " "
        print(f"  {marker} #{rank}: {asin} (sim={sim:.4f})")
    
    print(f"\nExperiment B Top-5:")
    for rank, (asin, sim) in enumerate(scores_b[:5], 1):
        marker = "→" if asin == target_asin else " "
        print(f"  {marker} #{rank}: {asin} (sim={sim:.4f})")

