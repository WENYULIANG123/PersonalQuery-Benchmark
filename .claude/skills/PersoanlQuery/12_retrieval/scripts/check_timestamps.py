#!/usr/bin/env python3
"""检查数据时间戳一致性"""

import json
import os
from datetime import datetime

results_dir = 'result/personal_query/13_retrieval'
retrievers = ['bm25', 'tfidf', 'bge', 'e5', 'ance', 'minilm', 'mpnet', 'star', 'dense', 'colbert']

print('=' * 110)
print('数据时间戳一致性检查')
print('=' * 110)
print()

print(f"{'Retriever':<15} {'Clean Data Time':<25} {'Noisy Data Time':<25} {'Diff (min)':<12} {'Status':<10}")
print('-' * 110)

all_consistent = True

for ret in retrievers:
    clean_file = f'{results_dir}/retrieval_{ret}_clean_A13OFOB1394G31.json'
    noisy_file = f'{results_dir}/retrieval_{ret}_noisy_A13OFOB1394G31.json'

    if os.path.exists(clean_file) and os.path.exists(noisy_file):
        with open(clean_file) as f:
            clean_data = json.load(f)
        with open(noisy_file) as f:
            noisy_data = json.load(f)

        clean_time = clean_data.get('data_timestamp', 'N/A')
        noisy_time = noisy_data.get('data_timestamp', 'N/A')

        if clean_time != 'N/A' and noisy_time != 'N/A':
            clean_dt = datetime.fromisoformat(clean_time)
            noisy_dt = datetime.fromisoformat(noisy_time)
            time_diff = (noisy_dt - clean_dt).total_seconds() / 60  # 分钟

            if time_diff < 1:
                status = 'OK'
            else:
                status = 'WARN'
                all_consistent = False

            print(f'{ret:<15} {clean_time:<25} {noisy_time:<25} {time_diff:<12.1f} {status:<10}')
        else:
            print(f'{ret:<15} {clean_time:<25} {noisy_time:<25} {"N/A":<12} {"MISS":<10}')
            all_consistent = False

print()
if all_consistent:
    print('✅ 所有检索器的Clean和Noisy模式使用相同的数据源（时间差 < 1分钟）')
else:
    print('⚠️  部分检索器的时间戳不一致，可能使用了不同的数据加载')
print()
