#!/bin/bash
#SBATCH --job-name=test_bm25s
#SBATCH --output=/home/wlia0047/ar57/wenyu/test_bm25s_%j.log
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

from utils.retrievers import BM25
from utils.utils import log_with_timestamp

STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
METADATA_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"

print("=" * 80)
print("BM25s改进测试 - 对比新旧tokenization")
print("=" * 80)

# 加载metadata
with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

# 加载查询
with open(os.path.join(STAGE9_DIR, "noisy_queries_A13OFOB1394G31.json"), 'r') as f:
    data = json.load(f)

queries_data = data['queries'][:10]

# 构建documents列表
all_asins = list(metadata.keys())
documents = []
for asin in all_asins[:100]:  # 用前100个文档测试
    doc = dict(metadata[asin])
    doc['asin'] = asin
    documents.append(doc)

print(f"\n使用 {len(documents)} 个文档构建BM25索引...")

# 创建新BM25实例
bm25 = BM25()
bm25.fit(documents, metadata)

print(f"\n测试前10个查询的检索结果:")
print("-" * 80)

matches_in_top1 = 0
matches_in_top5 = 0
matches_in_top10 = 0

for idx, query_obj in enumerate(queries_data):
    target_asin = query_obj['asin']
    query_text = query_obj['personalized_query']['original']
    
    results = bm25.search(query_text, top_k=10)
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
    
    print(f"\n[查询 {idx+1}]")
    print(f"  目标ASIN: {target_asin}")
    print(f"  查询: {query_text[:60]}...")
    print(f"  在Top-1/5/10中: {found_in_top1}/{found_in_top5}/{found_in_top10}")
    if rank > 0:
        print(f"  排名: {rank}")
    else:
        print(f"  排名: 未找到（在前10之外）")
    print(f"  Top-5: {retrieved_asins[:5]}")

print(f"\n" + "=" * 80)
print(f"新BM25s (100个文档测试):")
print(f"  P@1  = {matches_in_top1}/10 = {matches_in_top1/10:.4f}")
print(f"  P@5  = {matches_in_top5}/10 = {matches_in_top5/10:.4f}")
print(f"  P@10 = {matches_in_top10}/10 = {matches_in_top10/10:.4f}")
print("=" * 80)

if matches_in_top10 > 0:
    print("\n✅ 新BM25s有改进！继续使用它。")
else:
    print("\n⚠️  小规模测试中仍未找到。继续用完整corpus测试。")

PYTHON
