#!/usr/bin/env python3
"""分析特征分布并尝试减少分散"""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

print('='*60)
print('原始特征分布:')
print('='*60)
for i, name in enumerate(feature_names):
    print(f'{name}: min={X[:,i].min():.4f}, max={X[:,i].max():.4f}, mean={X[:,i].mean():.4f}, std={X[:,i].std():.4f}')

# 分析相关性
print()
print('='*60)
print('特征相关性矩阵:')
print('='*60)
corr = np.corrcoef(X.T)
for i, name in enumerate(feature_names):
    row = ' '.join([f'{corr[i,j]:.2f}' for j in range(len(feature_names))])
    print(f'{name[:8]}: {row}')

# 标准化方法比较
print()
print('='*60)
print('不同标准化方法的聚类效果比较 (K=4):')
print('='*60)

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
}

for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f'{name}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')

# PCA降维分析
print()
print('='*60)
print('PCA方差解释比例:')
print('='*60)
pca = PCA()
pca.fit(X)
for i, var in enumerate(pca.explained_variance_ratio_):
    cumsum = sum(pca.explained_variance_ratio_[:i+1])
    print(f'PC{i+1}: {var:.4f} (累计: {cumsum:.4f})')

# PCA后聚类
print()
print('='*60)
print('PCA降维后聚类效果 (K=4):')
print('='*60)
for n_comp in [2, 3, 4, 5]:
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)

    print(f'PCA(n={n_comp}): 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}')