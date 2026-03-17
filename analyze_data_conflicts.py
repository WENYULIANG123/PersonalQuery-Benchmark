#!/usr/bin/env python3
"""
分析跨用户训练数据的冲突
"""
import json
from pathlib import Path
from collections import defaultdict

loocv_data_dir = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data')

# 分析第一个用户的全局训练数据
user_id = "A13OFOB1394G31"
user_dir = loocv_data_dir / f'user_{user_id}'

with open(user_dir / 'global_train.json') as f:
    global_train = json.load(f)['pairs']

# 按原始用户分组
user_sources = defaultdict(list)
query_conflicts = defaultdict(list)

for pair in global_train:
    src_user = pair['user_id']
    user_sources[src_user].append(pair)
    
    # 记录query -> ASIN映射
    query_conflicts[pair['query']].append({
        'user': src_user,
        'asin': pair['asin'],
        'product': pair.get('positive', '')
    })

print(f"分析用户 {user_id} 的 global_train")
print("="*80)
print(f"总样本: {len(global_train)}")
print(f"来自的用户数: {len(user_sources)}")
print()

# 显示用户分布
print("来源用户分布:")
for src_user in sorted(user_sources.keys()):
    count = len(user_sources[src_user])
    print(f"  {src_user}: {count} 个样本")

# 查找冲突：同一个query对应多个ASIN
conflicts = [q for q, mappings in query_conflicts.items() if len(mappings) > 1]
print(f"\n查询冲突 (同一query对应不同ASIN)：{len(conflicts)}")

if conflicts:
    print("\n前5个冲突例子:")
    for query in conflicts[:5]:
        mappings = query_conflicts[query]
        print(f"\n  Query: {query[:60]}...")
        for m in mappings:
            print(f"    -> ASIN {m['asin']} (来自用户 {m['user']})")

# 分析数据统计
print("\n" + "="*80)
print("数据分析")
print("="*80)

# 查询语句相似性
queries = [p['query'] for p in global_train]
unique_queries = len(set(queries))
print(f"唯一查询: {unique_queries}/{len(queries)}")

# ASIN多样性
asins = [p['asin'] for p in global_train]
unique_asins = len(set(asins))
print(f"唯一ASIN: {unique_asins}/{len(asins)}")

# 样本过度表示
from collections import Counter
query_counts = Counter(queries)
asin_counts = Counter(asins)

most_common_queries = query_counts.most_common(3)
most_common_asins = asin_counts.most_common(3)

print(f"\n最常见的查询 (重复次数):")
for query, count in most_common_queries:
    print(f"  {count}x: {query[:50]}...")

print(f"\n最常见的ASIN (重复次数):")
for asin, count in most_common_asins:
    print(f"  {count}x: {asin}")

