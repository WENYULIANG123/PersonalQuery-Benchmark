#!/usr/bin/env python3
"""
建议：用NOISY query重新评估Experiment A和B
看看结果是否更符合"noisy query correction"的目标
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 加载模型
user_id = "A13OFOB1394G31"
model_a_path = Path(f'/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/experiment_a_user_{user_id}')
model_b_path = Path(f'/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/experiment_b_user_{user_id}')

print("加载模型...")
try:
    model_a = SentenceTransformer(str(model_a_path))
    print(f"✓ Exp A 加载成功")
except:
    print(f"✗ Exp A 加载失败")
    model_a = None

try:
    model_b = SentenceTransformer(str(model_b_path))
    print(f"✓ Exp B 加载成功")
except:
    print(f"✗ Exp B 加载失败")
    model_b = None

# 加载holdout数据
holdout_file = Path(f'/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data/user_{user_id}/holdout.json')
with open(holdout_file) as f:
    holdout = json.load(f)['pairs']

# 加载embeddings
emb_a_path = Path(f'/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/embeddings_experiment_a_{user_id}.pkl')
emb_b_path = Path(f'/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/checkpoints/embeddings_experiment_b_{user_id}.pkl')

with open(emb_a_path, 'rb') as f:
    emb_a = pickle.load(f)
with open(emb_b_path, 'rb') as f:
    emb_b = pickle.load(f)

print("\n" + "="*80)
print("假设：评估应该用NOISY query")
print("="*80)

def evaluate_with_noisy(model, embeddings, holdout_data, model_name):
    """用NOISY query评估"""
    mrr_scores = []
    
    for pair in holdout_data:
        clean_query = pair.get('query', '')
        noisy_query = pair.get('positive', '')  # ← 关键：用noisy！
        target_asin = pair.get('asin', '')
        
        if not (noisy_query and target_asin):
            continue
        
        # 编码NOISY query
        noisy_emb = model.encode(noisy_query, convert_to_numpy=True)
        
        # 计算相似度
        scores = []
        for asin, prod_emb in embeddings.items():
            if prod_emb is not None:
                sim = np.dot(noisy_emb, prod_emb) / (
                    np.linalg.norm(noisy_emb) * np.linalg.norm(prod_emb) + 1e-8
                )
                scores.append((asin, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_asins = [asin for asin, _ in scores]
        
        # 计算MRR@10
        if target_asin in ranked_asins[:10]:
            rank = ranked_asins[:10].index(target_asin) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        
        mrr_scores.append(mrr)
    
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    print(f"{model_name}: MRR@10 = {avg_mrr:.4f} (用NOISY query)")
    return avg_mrr

if model_a:
    eval_a = evaluate_with_noisy(model_a, emb_a, holdout, "Experiment A")

if model_b:
    eval_b = evaluate_with_noisy(model_b, emb_b, holdout, "Experiment B")

print("\n" + "="*80)
print("对比：CLEAN query vs NOISY query评估")
print("="*80)
print("""
当前结果（用CLEAN query评估）:
  Exp A: MRR@10 = 0.1111
  Exp B: MRR@10 = 0.0000
  
如果改为NOISY query评估:
  Exp A: MRR@10 = ?
  Exp B: MRR@10 = ?
  
如果NOISY评估后：
  - 两个都还是0，说明模型没有学到noisy→clean映射
  - 如果有差异，说明当前评估逻辑有问题
""")

