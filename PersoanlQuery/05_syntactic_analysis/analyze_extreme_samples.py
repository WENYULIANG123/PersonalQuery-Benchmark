#!/usr/bin/env python3
"""分析极端样本的聚类效果"""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

input_file = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/batch_generation_18templates_6d_final.json'
with open(input_file, 'r') as f:
    results = json.load(f)

feature_names = ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']

# 收集所有查询的6D特征
all_data = []
all_tids = []
for r in results:
    for q in r['queries']:
        if q['query'] and q['word_count'] > 5:
            feats = q.get('features_14d', {})
            vec = [feats.get(name, 0) for name in feature_names]
            all_data.append(vec)
            all_tids.append(q['template_id'])

X = np.array(all_data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算每个样本的极端程度（到原点的欧几里得距离）
extremeness = np.linalg.norm(X_scaled, axis=1)

print('极端程度分布:')
print(f'  最小值: {extremeness.min():.4f}')
print(f'  最大值: {extremeness.max():.4f}')
print(f'  均值: {extremeness.mean():.4f}')
print(f'  标准差: {extremeness.std():.4f}')

# 划分极端程度
extreme_mask = extremeness > 1.5
mild_mask = (extremeness >= 0.5) & (extremeness <= 1.0)
very_mild_mask = extremeness < 0.5

print()
print(f'极端 (>1.5): {extreme_mask.sum()} ({extreme_mask.sum()/len(X)*100:.1f}%)')
print(f'中等 (0.5-1.0): {mild_mask.sum()} ({mild_mask.sum()/len(X)*100:.1f}%)')
print(f'温和 (<0.5): {very_mild_mask.sum()} ({very_mild_mask.sum()/len(X)*100:.1f}%)')

# 极端样本聚类
print()
print('='*60)
print('极端样本聚类分析:')
print('='*60)

X_extreme = X_scaled[extreme_mask]
print(f'极端样本数: {len(X_extreme)}')

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_extreme)

    sil = silhouette_score(X_extreme, labels)
    ch = calinski_harabasz_score(X_extreme, labels)
    db = davies_bouldin_score(X_extreme, labels)

    print(f'K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')

# 中等样本聚类
print()
print('='*60)
print('中等样本聚类分析:')
print('='*60)

X_mild = X_scaled[mild_mask]
print(f'中等样本数: {len(X_mild)}')

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_mild)

    sil = silhouette_score(X_mild, labels)
    ch = calinski_harabasz_score(X_mild, labels)
    db = davies_bouldin_score(X_mild, labels)

    print(f'K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')