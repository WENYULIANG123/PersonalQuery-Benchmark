#!/usr/bin/env python3
"""
PCA降维 + KMeans聚类分析
使用新6维特征，对500用户进行聚类
"""
import json
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings('ignore')

import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_features(text):
    """提取6维句法特征"""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    n = len(tokens)
    if n == 0:
        return None

    coord = sum(1 for t in tokens if t.dep_ in ['cc', 'conj']) / n
    subordinate = sum(1 for t in tokens if t.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp', 'relcl']) / n
    advcl = sum(1 for t in tokens if t.dep_ == 'advcl') / n
    ccomp = sum(1 for t in tokens if t.dep_ in ['ccomp', 'xcomp']) / n
    fanout = np.mean([len(list(t.children)) for t in tokens]) if tokens else 0
    prep = sum(1 for t in tokens if t.dep_ == 'prep') / n

    return {
        'coordination': coord,
        'subordinate': subordinate,
        'advcl': advcl,
        'ccomp': ccomp,
        'avg_fanout': fanout,
        'prep_density': prep
    }


def process_user(user_file):
    """处理单个用户文件"""
    with open(user_file) as f:
        data = json.load(f)

    reviews = []
    for product in data.get('results', []):
        reviews.extend(product.get('target_reviews', []))

    feats = [get_features(r) for r in reviews if r and len(r.strip()) > 10]
    feats = [f for f in feats if f]

    if not feats:
        return None

    return {
        'user_id': data['user_id'],
        'review_count': len(feats),
        **{k: np.mean([f[k] for f in feats]) for k in feats[0].keys()}
    }


def describe_style(stats):
    """根据特征值生成风格描述"""
    parts = []
    if stats['coordination'] > 0.09:
        parts.append('并列丰富')
    elif stats['coordination'] < 0.07:
        parts.append('并列简洁')
    if stats['subordinate'] > 0.08:
        parts.append('从句丰富')
    elif stats['subordinate'] < 0.06:
        parts.append('从句简洁')
    if stats['advcl'] > 0.025:
        parts.append('状语从句多')
    if stats['ccomp'] > 0.035:
        parts.append('补语从句多')
    if stats['prep_density'] > 0.09:
        parts.append('介词密集')
    elif stats['prep_density'] < 0.07:
        parts.append('介词简洁')
    return '/'.join(parts) if parts else '中等复杂度'


def main():
    # 配置
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 处理用户
    print("处理用户数据...")
    user_files = sorted(glob.glob(f'{input_dir}/reviews_*.json'))[:500]
    users = [process_user(f) for f in user_files]
    users = [u for u in users if u]
    print(f"成功处理 {len(users)} 个用户\n")

    # 2. 提取特征
    feats_6d = ['coordination', 'subordinate', 'advcl', 'ccomp', 'avg_fanout', 'prep_density']
    X = np.array([[u[f] for f in feats_6d] for u in users])

    # 3. PCA降维 + KMeans聚类
    print("="*70)
    print("PCA2 + KMeans聚类")
    print("="*70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA2方差解释率: {sum(pca.explained_variance_ratio_)*100:.1f}%\n")

    # K=3聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    # 4. 评估指标
    print("="*70)
    print("聚类评估指标")
    print("="*70)
    sil = silhouette_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    print(f"轮廓系数 (Silhouette): {sil:.4f}")
    print(f"Calinski-Harabasz指数: {ch:.4f}")
    print(f"Davies-Bouldin指数: {db:.4f}")
    print()

    # 5. 簇统计分析
    print("="*70)
    print("各簇详细特征")
    print("="*70)

    cluster_stats = {}
    for cid in range(3):
        mask = labels == cid
        cluster_users = [users[i] for i in range(len(users)) if labels[i] == cid]
        stats = {f: np.mean([u[f] for u in cluster_users]) for f in feats_6d}
        stats['size'] = len(cluster_users)
        stats['style'] = describe_style(stats)
        cluster_stats[cid] = stats

        print(f"\n【簇 {cid}】{len(cluster_users)} 用户 - {stats['style']}")
        for f in feats_6d:
            vals = [u[f] for u in cluster_users]
            print(f"  {f:15s}: {stats[f]:.4f} (std={np.std(vals):.4f})")

        # 代表用户
        sample = sorted(cluster_users, key=lambda u: u['review_count'], reverse=True)[:5]
        print(f"  代表用户: {[u['user_id'][:12] for u in sample]}")

    # 6. 簇间距离
    print("\n" + "="*70)
    print("簇间距离矩阵")
    print("="*70)

    centroids = []
    for cid in range(3):
        mask = labels == cid
        centroids.append(X_pca[mask].mean(axis=0))
    centroids = np.array(centroids)

    print("\n原始6维空间中的簇间距离:")
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  簇{i} - 簇{j}: {dist:.4f}")

    # 7. 保存结果
    output = {
        'method': 'PCA2+KMeans',
        'n_components': 2,
        'k': 3,
        'variance_explained': float(sum(pca.explained_variance_ratio_)),
        'metrics': {
            'silhouette': float(sil),
            'calinski_harabasz': float(ch),
            'davies_bouldin': float(db)
        },
        'feature_names': feats_6d,
        'clusters': {},
        'all_users': []
    }

    for cid in range(3):
        mask = labels == cid
        cluster_users = [users[i] for i in range(len(users)) if labels[i] == cid]
        output['clusters'][str(cid)] = {
            'size': len(cluster_users),
            'style': cluster_stats[cid]['style'],
            'features': {f: float(cluster_stats[cid][f]) for f in feats_6d},
            'users': [{'user_id': u['user_id'], 'review_count': u['review_count']} for u in cluster_users]
        }

    for i, u in enumerate(users):
        output['all_users'].append({
            'user_id': u['user_id'],
            'cluster': int(labels[i]),
            'review_count': u['review_count']
        })

    output_file = os.path.join(output_dir, 'pca_kmeans_500users.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
