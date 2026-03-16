#!/usr/bin/env python3
"""完整对比所有检索器的所有指标"""

import json
import os

results_dir = 'result/personal_query/13_retrieval'
retrievers = ['bm25', 'tfidf', 'dirichlet', 'bge', 'e5', 'ance', 'minilm', 'mpnet', 'star', 'dense', 'colbert']

# 所有指标
metrics = ['P@1', 'R@1', 'MAP@1', 'NDCG@1', 'MRR@1',
           'P@3', 'R@3', 'MAP@3', 'NDCG@3', 'MRR@3',
           'P@5', 'R@5', 'MAP@5', 'NDCG@5', 'MRR@5',
           'P@10', 'R@10', 'MAP@10', 'NDCG@10', 'MRR@10']

print('=' * 160)
print('Stage 13 检索评估完整指标对比 (GPU加速)')
print('=' * 160)
print()

# 为每个检索器收集数据
summary = {}

for ret in retrievers:
    clean_file = f'{results_dir}/retrieval_{ret}_clean_A13OFOB1394G31.json'
    noisy_file = f'{results_dir}/retrieval_{ret}_noisy_A13OFOB1394G31.json'

    if os.path.exists(clean_file) and os.path.exists(noisy_file):
        with open(clean_file) as f:
            clean_data = json.load(f)
        with open(noisy_file) as f:
            noisy_data = json.load(f)

        clean_metrics = clean_data['metrics']
        noisy_metrics = noisy_data['metrics']

        summary[ret] = {
            'clean': clean_metrics,
            'noisy': noisy_metrics
        }

# 打印每个k值的对比
for k in [1, 3, 5, 10]:
    print('=' * 160)
    print(f'@{k} 指标对比')
    print('=' * 160)
    print()

    # 表头
    header = f"{'Retriever':<15} {'Clean P@'+str(k):<12} {'Noisy P@'+str(k):<12} {'Diff':<10} {'Clean R@'+str(k):<12} {'Noisy R@'+str(k):<12} {'Diff':<10} {'Clean MAP@'+str(k):<12} {'Noisy MAP@'+str(k):<12} {'Diff':<10}"
    print(header)
    print('-' * 160)

    for ret in retrievers:
        if ret in summary:
            clean_p = summary[ret]['clean'][f'P@{k}']
            noisy_p = summary[ret]['noisy'][f'P@{k}']
            diff_p = noisy_p - clean_p

            clean_r = summary[ret]['clean'][f'R@{k}']
            noisy_r = summary[ret]['noisy'][f'R@{k}']
            diff_r = noisy_r - clean_r

            clean_map = summary[ret]['clean'][f'MAP@{k}']
            noisy_map = summary[ret]['noisy'][f'MAP@{k}']
            diff_map = noisy_map - clean_map

            # 标记状态
            if diff_p > 0.001:  # 提升>0.1%
                status_p = '+'
            elif diff_p < -0.001:
                status_p = '-'
            else:
                status_p = '='

            row = f"{ret:<15} {clean_p:<12.4f} {noisy_p:<12.4f} {diff_p:+<10.4f} {clean_r:<12.4f} {noisy_r:<12.4f} {diff_r:+<10.4f} {clean_map:<12.4f} {noisy_map:<12.4f} {diff_map:+<10.4f} {status_p}"
            print(row)

    print()

    # NDCG 和 MRR
    header = f"{'Retriever':<15} {'Clean NDCG@'+str(k):<12} {'Noisy NDCG@'+str(k):<12} {'Diff':<10} {'Clean MRR@'+str(k):<12} {'Noisy MRR@'+str(k):<12} {'Diff':<10}"
    print(header)
    print('-' * 160)

    for ret in retrievers:
        if ret in summary:
            clean_ndcg = summary[ret]['clean'][f'NDCG@{k}']
            noisy_ndcg = summary[ret]['noisy'][f'NDCG@{k}']
            diff_ndcg = noisy_ndcg - clean_ndcg

            clean_mrr = summary[ret]['clean'][f'MRR@{k}']
            noisy_mrr = summary[ret]['noisy'][f'MRR@{k}']
            diff_mrr = noisy_mrr - clean_mrr

            row = f"{ret:<15} {clean_ndcg:<12.4f} {noisy_ndcg:<12.4f} {diff_ndcg:+<10.4f} {clean_mrr:<12.4f} {noisy_mrr:<12.4f} {diff_mrr:+<10.4f}"
            print(row)

    print()

# 统计汇总
print('=' * 160)
print('统计汇总：P@1 差异分布')
print('=' * 160)
print()

p1_diffs = []
for ret in retrievers:
    if ret in summary:
        diff = summary[ret]['noisy']['P@1'] - summary[ret]['clean']['P@1']
        p1_diffs.append((ret, diff))

p1_diffs.sort(key=lambda x: x[1])

print(f"{'Retriever':<15} {'P@1 Diff':<12} {'Status':<10}")
print('-' * 50)

improved = []
same = []
degraded = []

for ret, diff in p1_diffs:
    if diff > 0.001:
        status = '⚠️ Noisy更好'
        improved.append((ret, diff))
    elif diff < -0.001:
        status = '✅ 正常'
        degraded.append((ret, diff))
    else:
        status = '✅ 相同'
        same.append((ret, diff))

    print(f"{ret:<15} {diff:+<12.4f} {status:<10}")

print()
print(f'总结：')
print(f'  Noisy更好（异常）: {len(improved)}/{len(retrievers)}')
for ret, diff in improved:
    print(f'    {ret}: {diff:+.4f} ({diff*100:+.2f}%)')
print(f'  Noisy相同: {len(same)}/{len(retrievers)}')
print(f'  Noisy更差（正常）: {len(degraded)}/{len(retrievers)}')
for ret, diff in degraded:
    print(f'    {ret}: {diff:+.4f} ({diff*100:+.2f}%)')
print()
