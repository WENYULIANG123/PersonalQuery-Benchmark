#!/usr/bin/env python3
"""分析异常检索器（Noisy > Clean）"""

import json
import os

results_dir = 'result/personal_query/13_retrieval'
abnormal_retrievers = ['ance', 'star']  # tfidf 没有候选文件

print('=' * 100)
print('异常检索器详细分析')
print('=' * 100)
print()

for ret in abnormal_retrievers:
    clean_file = f'{results_dir}/{ret}_candidates_clean_A13OFOB1394G31_target.json'
    noisy_file = f'{results_dir}/{ret}_candidates_noisy_A13OFOB1394G31_target.json'

    if os.path.exists(clean_file) and os.path.exists(noisy_file):
        with open(clean_file) as f:
            clean_candidates = json.load(f)
        with open(noisy_file) as f:
            noisy_candidates = json.load(f)

        print('=' * 100)
        print(f'{ret.upper()} - 查询级别分析')
        print('=' * 100)

        # 分析每个查询的排名变化
        rank_changes = []
        queries_better_in_noisy = []
        queries_worse_in_noisy = []
        queries_same = []

        for clean_q, noisy_q in zip(clean_candidates['candidates'], noisy_candidates['candidates']):
            query = clean_q['query']
            target_asin = clean_q['asin']

            # 找到目标ASIN的排名
            clean_rank = None
            noisy_rank = None

            for i, candidate in enumerate(clean_q.get('candidates', [])):
                if candidate[0] == target_asin:  # candidate is [asin, score]
                    clean_rank = i + 1
                    break

            for i, candidate in enumerate(noisy_q.get('candidates', [])):
                if candidate[0] == target_asin:  # candidate is [asin, score]
                    noisy_rank = i + 1
                    break

            if clean_rank and noisy_rank:
                diff = noisy_rank - clean_rank
                rank_changes.append({
                    'query': query[:60] + '...' if len(query) > 60 else query,
                    'clean_rank': clean_rank,
                    'noisy_rank': noisy_rank,
                    'diff': diff,
                    'target_asin': target_asin
                })

                if noisy_rank < clean_rank:
                    queries_better_in_noisy.append((query, clean_rank, noisy_rank, diff, target_asin))
                elif noisy_rank > clean_rank:
                    queries_worse_in_noisy.append((query, clean_rank, noisy_rank, diff, target_asin))
                else:
                    queries_same.append((query, clean_rank, noisy_rank, target_asin))

        print(f'总查询数: {len(rank_changes)}')
        print(f'Noisy更好: {len(queries_better_in_noisy)} 个查询')
        print(f'Noisy更差: {len(queries_worse_in_noisy)} 个查询')
        print(f'相同: {len(queries_same)} 个查询')
        print()

        # 显示在Noisy模式下表现更好的查询
        if queries_better_in_noisy:
            print('⚠️  在Noisy模式下表现更好的查询 (排名提升):')
            print('-' * 100)
            count = 0
            for query, clean_r, noisy_r, diff, asin in sorted(queries_better_in_noisy, key=lambda x: x[3]):
                if count >= 5:  # 只显示前5个
                    break
                print(f'  ASIN: {asin}')
                print(f'  Query: {query[:70]}...')
                print(f'  Clean排名: {clean_r} -> Noisy排名: {noisy_r} (提升 {abs(diff)} 位)')
                print()
                count += 1

        # 统计有多少查询排名1发生变化
        clean_top1 = sum(1 for _, cr, _, _, _ in queries_better_in_noisy if cr > 1)
        noisy_top1 = sum(1 for _, _, nr, _, _ in queries_better_in_noisy if nr == 1)

        print(f'统计: {clean_top1} 个查询从Top1外进入Top1，{noisy_top1} 个查询原本就在Top1')
        print()
