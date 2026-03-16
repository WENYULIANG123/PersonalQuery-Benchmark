#!/usr/bin/env python3
"""比较所有检索器的 Clean vs Noisy 性能"""

import json
import os
from pathlib import Path

results_dir = 'result/personal_query/13_retrieval'
retrievers = ['bm25', 'tfidf', 'dirichlet', 'bge', 'e5', 'ance', 'minilm', 'mpnet', 'star', 'dense', 'colbert']

print('=' * 100)
print('Stage 13 检索评估结果汇总 (GPU加速)')
print('=' * 100)
print()

# 表头
print(f"{'Retriever':<15} {'Clean P@1':<12} {'Noisy P@1':<12} {'Diff':<10} {'Clean MAP@3':<14} {'Noisy MAP@3':<14} {'Diff':<10} {'Status':<8}")
print('-' * 110)

summary = []
abnormal_count = 0

for ret in retrievers:
    clean_file = f'{results_dir}/retrieval_{ret}_clean_A13OFOB1394G31.json'
    noisy_file = f'{results_dir}/retrieval_{ret}_noisy_A13OFOB1394G31.json'

    if os.path.exists(clean_file) and os.path.exists(noisy_file):
        with open(clean_file) as f:
            clean_data = json.load(f)
        with open(noisy_file) as f:
            noisy_data = json.load(f)

        clean_p1 = clean_data['metrics']['P@1']
        noisy_p1 = noisy_data['metrics']['P@1']
        diff_p1 = noisy_p1 - clean_p1

        clean_map = clean_data['metrics']['MAP@3']
        noisy_map = noisy_data['metrics']['MAP@3']
        diff_map = noisy_map - clean_map

        # 标记异常（noisy应该低于clean）
        status = '✅ 正常' if diff_p1 <= 0 else '❌ 异常'
        if diff_p1 > 0:
            abnormal_count += 1

        summary.append((ret, clean_p1, noisy_p1, diff_p1, clean_map, noisy_map, diff_map))

        print(f"{ret:<15} {clean_p1:<12.4f} {noisy_p1:<12.4f} {diff_p1:+<10.4f} {clean_map:<14.4f} {noisy_map:<14.4f} {diff_map:+<10.4f} {status:<8}")

print()
print('=' * 100)
print(f'总结: {abnormal_count}/{len(retrievers)} 个检索器出现异常（Noisy > Clean）')
print('=' * 100)

# 详细分析异常的检索器
if abnormal_count > 0:
    print()
    print('⚠️  异常检索器详细分析:')
    print('-' * 100)
    for ret, clean_p1, noisy_p1, diff_p1, clean_map, noisy_map, diff_map in summary:
        if diff_p1 > 0:
            print(f'{ret}:')
            print(f'  Clean P@1: {clean_p1:.4f} -> Noisy P@1: {noisy_p1:.4f} (提升 {diff_p1:.4f})')
            print(f'  Clean MAP@3: {clean_map:.4f} -> Noisy MAP@3: {noisy_map:.4f} (提升 {diff_map:.4f})')
            print()
else:
    print()
    print('✅ 所有检索器表现正常！Noisy查询性能均低于或等于Clean查询。')
    print()

# 数据源检查
print()
print('=' * 100)
print('数据源一致性检查')
print('=' * 100)

sample_file = f'{results_dir}/retrieval_bm25_clean_A13OFOB1394G31.json'
if os.path.exists(sample_file):
    with open(sample_file) as f:
        data = json.load(f)
        if 'data_tracking' in data:
            tracking = data['data_tracking']
            print(f"数据源: {tracking.get('data_source', 'N/A')}")
            print(f"时间戳: {tracking.get('data_timestamp', 'N/A')}")
            if 'doc_fingerprints_sample' in tracking:
                print(f"文档指纹样本数: {len(tracking['doc_fingerprints_sample'])}")
        else:
            print("⚠️  未找到数据追踪信息")
else:
    print("⚠️  未找到结果文件")

print()
