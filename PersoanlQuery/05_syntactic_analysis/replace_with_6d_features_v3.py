#!/usr/bin/env python3
"""将14D特征替换为6D深层特征(最终版)"""
import json
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

input_file = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/batch_generation_18templates_50products.json'
output_file = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/batch_generation_18templates_6d_final.json'

with open(input_file, 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} products")

import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')


def get_6d_features_final(text):
    """提取最终版6维深层特征

    设计原则：
    1. coordination_ratio: 并列结构（cc+conj），多数模板有
    2. subordinate_ratio: 所有从句弧比例（advcl+acl+ccomp+xcomp+relcl）
    3. advcl_ratio: 状语从句（because/when/if等从属连词引导）
    4. ccomp_ratio: 补语从句（that/whether等引导）
    5. avg_fanout: 平均扇出（树结构复杂度）
    6. prep_density: 介词密度（嵌套程度）
    """
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    n = len(tokens)
    if n == 0:
        return None

    # 1. coordination: cc+conj弧比例
    coord_count = sum(1 for t in tokens if t.dep_ in ['cc', 'conj'])
    coordination = coord_count / n

    # 2. subordinate: 从句总比例
    subord_count = sum(1 for t in tokens if t.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp', 'relcl'])
    subordinate = subord_count / n

    # 3. advcl: 状语从句
    advcl_count = sum(1 for t in tokens if t.dep_ == 'advcl')
    advcl = advcl_count / n

    # 4. ccomp: 补语从句（用that/whether等引导的宾语从句/表语从句）
    ccomp_count = sum(1 for t in tokens if t.dep_ in ['ccomp', 'xcomp'])
    ccomp = ccomp_count / n

    # 5. avg_fanout
    fanouts = [len(list(t.children)) for t in tokens]
    avg_fanout = np.mean(fanouts) if fanouts else 0

    # 6. prep_density
    prep_count = sum(1 for t in tokens if t.dep_ == 'prep')
    prep_density = prep_count / n

    return {
        'coordination': coordination,
        'subordinate': subordinate,
        'advcl': advcl,
        'ccomp': ccomp,
        'avg_fanout': avg_fanout,
        'prep_density': prep_density,
    }


# 替换
print("\n开始替换特征...")
total_queries = 0
success_count = 0

for r in results:
    for q in r['queries']:
        total_queries += 1
        if q['query'] and q['word_count'] > 5:
            feats = get_6d_features_final(q['query'])
            if feats:
                q['features_14d'] = feats
                success_count += 1
            else:
                q['features_14d'] = {k: 0 for k in ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']}
        else:
            q['features_14d'] = {k: 0 for k in ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']}

print(f"处理完成: {success_count}/{total_queries} 条查询成功提取6D特征")

# 保存
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n已保存到: {output_file}")

# 零值统计
feature_names = ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']
zero_counts = {name: 0 for name in feature_names}

for r in results:
    for q in r['queries']:
        if q['query'] and q['word_count'] > 5:
            feats = q.get('features_14d', {})
            for name in feature_names:
                if feats.get(name, 0) == 0:
                    zero_counts[name] += 1

print("\n" + "="*70)
print("最终6D特征零值分析:")
print("="*70)
for name in feature_names:
    zero_pct = zero_counts[name] / success_count * 100 if success_count > 0 else 0
    print(f"{name}: 零值占比 {zero_pct:.1f}%")