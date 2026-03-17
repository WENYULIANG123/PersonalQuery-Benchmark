#!/bin/bash
#SBATCH --job-name=diagnose_retrieval
#SBATCH --output=/home/wlia0047/ar57/wenyu/diagnose_retrieval_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

cd /home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval/evaluators

python3 << 'PYTHON'
import json
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from document_manager import get_document_manager
from retriever_manager import get_retriever_manager

STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
METADATA_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"

print("=" * 80)
print("检索结果诊断 - 检查BM25为什么P@10这么低")
print("=" * 80)

# 加载查询
with open(os.path.join(STAGE9_DIR, "noisy_queries_A13OFOB1394G31.json"), 'r') as f:
    data = json.load(f)

queries = data['queries']
print(f"\n总查询数: {len(queries)}")

# 加载metadata
with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

all_asins = list(metadata.keys())
docs = [dict(metadata[a], asin=a) for a in all_asins]

print(f"总文档数: {len(docs)}")

# 初始化检索器
rm = get_retriever_manager()
bm25 = rm.get_retriever('bm25', docs, metadata)

# 测试前10个查询
print(f"\n测试前10个查询的检索结果:")
print("-" * 80)

matches_in_top1 = 0
matches_in_top5 = 0
matches_in_top10 = 0
total_queries = 0

for idx, query_obj in enumerate(queries[:10]):
    target_asin = query_obj['asin']
    query_text = query_obj['personalized_query']['original']
    
    results = bm25.search(query_text, top_k=20)
    retrieved_asins = [r[0] for r in results]
    
    found_in_top1 = target_asin in retrieved_asins[:1]
    found_in_top5 = target_asin in retrieved_asins[:5]
    found_in_top10 = target_asin in retrieved_asins[:10]
    rank = retrieved_asins.index(target_asin) + 1 if target_asin in retrieved_asins else -1
    
    if found_in_top1:
        matches_in_top1 += 1
    if found_in_top5:
        matches_in_top5 += 1
    if found_in_top10:
        matches_in_top10 += 1
    total_queries += 1
    
    print(f"\n[查询 {idx+1}]")
    print(f"  目标ASIN: {target_asin}")
    print(f"  查询: {query_text[:60]}...")
    print(f"  Top-1/5/10中找到: {found_in_top1}/{found_in_top5}/{found_in_top10}")
    print(f"  排名: {rank if rank > 0 else '未找到'}")
    print(f"  Top-5: {retrieved_asins[:5]}")

print(f"\n" + "=" * 80)
print(f"诊断结果 (前{total_queries}个查询):")
print(f"  P@1  = {matches_in_top1/total_queries:.4f} ({matches_in_top1}/{total_queries})")
print(f"  P@5  = {matches_in_top5/total_queries:.4f} ({matches_in_top5}/{total_queries})")
print(f"  P@10 = {matches_in_top10/total_queries:.4f} ({matches_in_top10}/{total_queries})")
print("=" * 80)

if matches_in_top10 == 0:
    print("\n⚠️ WARNING: BM25没有在任何查询的Top-10中找到目标ASIN!")
    print("可能的原因:")
    print("  1. BM25 keyword matching性能确实很差")
    print("  2. 查询文本太长/太复杂")
    print("  3. 产品metadata中缺少关键词")
    print("  4. 检索索引构建有问题")

PYTHON
