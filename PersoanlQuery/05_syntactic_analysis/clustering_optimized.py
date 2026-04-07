#!/usr/bin/env python3
"""使用优化后的标准化方法进行聚类"""
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

input_file = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/batch_generation_18templates_6d_final.json'
output_file = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/batch_generation_clustered.json'

with open(input_file, 'r') as f:
    results = json.load(f)

feature_names = ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']

# 收集所有查询的6D特征
all_data = []
all_tids = []
query_indices = []  # 记录每个query在结果中的位置
product_indices = []

idx = 0
for pi, r in enumerate(results):
    for qi, q in enumerate(r['queries']):
        if q['query'] and q['word_count'] > 5:
            feats = q.get('features_14d', {})
            vec = [feats.get(name, 0) for name in feature_names]
            all_data.append(vec)
            all_tids.append(q['template_id'])
            query_indices.append((pi, qi))
            product_indices.append(idx)
            idx += 1

X = np.array(all_data)

print('='*60)
print('聚类方案对比:')
print('='*60)

# 方案1: MinMaxScaler + KMeans
print()
print('方案1: MinMaxScaler + KMeans')
print('-'*40)
scaler1 = MinMaxScaler()
X_scaled1 = scaler1.fit_transform(X)

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled1)
    sil = silhouette_score(X_scaled1, labels)
    ch = calinski_harabasz_score(X_scaled1, labels)
    db = davies_bouldin_score(X_scaled1, labels)
    print(f'K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')

# 方案2: PCA(n=2) + KMeans
print()
print('方案2: PCA(n=2) + KMeans')
print('-'*40)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    print(f'K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')

# 选择最佳方案: PCA(n=2) + K=4
print()
print('='*60)
print('选择最佳方案: PCA(n=2) + K=4')
print('='*60)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f'PCA解释方差比例: {pca.explained_variance_ratio_}')

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

# 最终评估
sil = silhouette_score(X_pca, labels)
ch = calinski_harabasz_score(X_pca, labels)
db = davies_bouldin_score(X_pca, labels)
print(f'轮廓系数: {sil:.4f}')
print(f'CH指数: {ch:.2f}')
print(f'DB指数: {db:.4f}')

# 为每个query添加cluster标签
for i, (pi, qi) in enumerate(query_indices):
    results[pi]['queries'][qi]['cluster_id'] = int(labels[i])

# 保存结果
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n已保存聚类结果到: {output_file}')

# 统计各cluster分布
print()
print('='*60)
print('Cluster分布:')
print('='*60)
cluster_counts = {}
for label in labels:
    cluster_counts[label] = cluster_counts.get(label, 0) + 1
for c in sorted(cluster_counts.keys()):
    print(f'Cluster {c}: {cluster_counts[c]} ({cluster_counts[c]/len(labels)*100:.1f}%)')

# 各cluster的模板分布
print()
print('='*60)
print('各Cluster的模板分布:')
print('='*60)
from collections import defaultdict
cluster_template = defaultdict(lambda: defaultdict(int))

for i, tid in enumerate(all_tids):
    cluster_template[labels[i]][tid] += 1

for c in sorted(cluster_template.keys()):
    print(f'\nCluster {c}:')
    dist = cluster_template[c]
    total_c = sum(dist.values())
    for tid, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        if cnt / total_c * 100 > 3:
            print(f'  {tid}: {cnt} ({cnt/total_c*100:.1f}%)')