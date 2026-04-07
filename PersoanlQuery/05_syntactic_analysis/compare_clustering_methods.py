#!/usr/bin/env python3
"""
对比不同聚类方法的效果
1. 特征降维 - 去掉冗余特征
2. 尝试不同K值 (K=2,3,4,5)
3. 层次聚类
"""
import json
import numpy as np
from collections import defaultdict
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 6维深层特征定义
DEEP_ARC_TYPES = {
    'subordinate_clause': ['advcl', 'acl', 'ccomp', 'xcomp'],
    'relative_clause': ['relcl'],
    'coordination': ['cc', 'conj'],
}


def get_6d_features(text):
    """提取6维深层句法特征"""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    n = len(tokens)
    if n == 0:
        return None

    coord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['coordination'])
    coordination = coord_count / n

    subclause_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'] + DEEP_ARC_TYPES['relative_clause'])
    subclause_ratio = subclause_count / n

    relcl_count = sum(1 for t in tokens if t.dep_ == 'relcl')
    relative_clause = relcl_count / n

    subord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'])
    subordinate_clause = subord_count / n

    fanouts = [len(list(t.children)) for t in tokens]
    avg_fanout = np.mean(fanouts) if fanouts else 0

    prep_count = sum(1 for t in tokens if t.dep_ == 'prep')
    prep_density = prep_count / n

    return {
        'coordination': coordination,
        'subclause_ratio': subclause_ratio,
        'relative_clause': relative_clause,
        'subordinate_clause': subordinate_clause,
        'avg_fanout': avg_fanout,
        'prep_density': prep_density,
        'token_count': n,
    }


def process_user_reviews(user_file):
    """处理单个用户的评论文件"""
    with open(user_file, 'r') as f:
        data = json.load(f)

    user_id = data['user_id']
    all_reviews = []

    for product in data.get('results', []):
        all_reviews.extend(product.get('target_reviews', []))

    if not all_reviews:
        return None

    review_features = []
    for review in all_reviews:
        if review and len(review.strip()) > 10:
            feats = get_6d_features(review)
            if feats:
                review_features.append(feats)

    if not review_features:
        return None

    user_features = {
        'user_id': user_id,
        'coordination': np.mean([f['coordination'] for f in review_features]),
        'subclause_ratio': np.mean([f['subclause_ratio'] for f in review_features]),
        'relative_clause': np.mean([f['relative_clause'] for f in review_features]),
        'subordinate_clause': np.mean([f['subordinate_clause'] for f in review_features]),
        'avg_fanout': np.mean([f['avg_fanout'] for f in review_features]),
        'prep_density': np.mean([f['prep_density'] for f in review_features]),
        'review_count': len(review_features),
    }

    return user_features


def evaluate_clustering(X_scaled, labels, n_clusters):
    """计算聚类评估指标"""
    if len(np.unique(labels)) < 2:
        return None

    silhouette = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    return {
        'silhouette': silhouette,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score,
        'n_clusters': n_clusters
    }


def main():
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    # 获取前100个用户
    user_files = glob.glob(os.path.join(input_dir, 'reviews_*.json'))[:100]
    print(f"处理 {len(user_files)} 个用户\n")

    # 处理用户
    all_user_features = []
    for user_file in user_files:
        user_feats = process_user_reviews(user_file)
        if user_feats:
            all_user_features.append(user_feats)

    print(f"成功处理 {len(all_user_features)} 个用户\n")

    # ===== 实验1: 原始6维特征 =====
    print("="*80)
    print("【实验1】原始6维特征")
    print("="*80)

    feature_names_6d = ['coordination', 'subclause_ratio', 'relative_clause',
                        'subordinate_clause', 'avg_fanout', 'prep_density']

    X_6d = np.array([[u[feat] for feat in feature_names_6d] for u in all_user_features])
    scaler_6d = StandardScaler()
    X_6d_scaled = scaler_6d.fit_transform(X_6d)

    results_6d = {}
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_6d_scaled)
        metrics = evaluate_clustering(X_6d_scaled, labels, k)
        results_6d[k] = metrics
        print(f"K={k}: 轮廓={metrics['silhouette']:.4f}, CH={metrics['calinski_harabasz']:.4f}, DB={metrics['davies_bouldin']:.4f}")

    # ===== 实验2: 降维后5维特征 (合并subclause_ratio和subordinate_clause) =====
    print("\n" + "="*80)
    print("【实验2】降维5维特征 (合并subclause_ratio和subordinate_clause -> clause_ratio)")
    print("="*80)

    # 创建合并后的特征
    for u in all_user_features:
        u['clause_ratio'] = (u['subclause_ratio'] + u['subordinate_clause']) / 2

    feature_names_5d = ['coordination', 'clause_ratio', 'relative_clause',
                        'avg_fanout', 'prep_density']

    X_5d = np.array([[u[feat] for feat in feature_names_5d] for u in all_user_features])
    scaler_5d = StandardScaler()
    X_5d_scaled = scaler_5d.fit_transform(X_5d)

    results_5d = {}
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_5d_scaled)
        metrics = evaluate_clustering(X_5d_scaled, labels, k)
        results_5d[k] = metrics
        print(f"K={k}: 轮廓={metrics['silhouette']:.4f}, CH={metrics['calinski_harabasz']:.4f}, DB={metrics['davies_bouldin']:.4f}")

    # ===== 实验3: 层次聚类 (原始6维) =====
    print("\n" + "="*80)
    print("【实验3】层次聚类 (Ward方法, 原始6维)")
    print("="*80)

    results_hier = {}
    for k in [2, 3, 4, 5]:
        hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hier.fit_predict(X_6d_scaled)
        metrics = evaluate_clustering(X_6d_scaled, labels, k)
        results_hier[k] = metrics
        print(f"K={k}: 轮廓={metrics['silhouette']:.4f}, CH={metrics['calinski_harabasz']:.4f}, DB={metrics['davies_bouldin']:.4f}")

    # ===== 实验4: 层次聚类 (5维降维) =====
    print("\n" + "="*80)
    print("【实验4】层次聚类 (Ward方法, 5维降维)")
    print("="*80)

    results_hier_5d = {}
    for k in [2, 3, 4, 5]:
        hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hier.fit_predict(X_5d_scaled)
        metrics = evaluate_clustering(X_5d_scaled, labels, k)
        results_hier_5d[k] = metrics
        print(f"K={k}: 轮廓={metrics['silhouette']:.4f}, CH={metrics['calinski_harabasz']:.4f}, DB={metrics['davies_bouldin']:.4f}")

    # ===== 找出最佳配置 =====
    print("\n" + "="*80)
    print("【最佳配置汇总】")
    print("="*80)

    all_results = []

    for k, metrics in results_6d.items():
        all_results.append(('KMeans-6d', k, metrics))

    for k, metrics in results_5d.items():
        all_results.append(('KMeans-5d', k, metrics))

    for k, metrics in results_hier.items():
        all_results.append(('Hier-6d', k, metrics))

    for k, metrics in results_hier_5d.items():
        all_results.append(('Hier-5d', k, metrics))

    # 按轮廓系数排序
    all_results.sort(key=lambda x: x[2]['silhouette'], reverse=True)

    print("\n按轮廓系数排名 (越高越好):")
    print("-" * 70)
    print(f"{'排名':<4} {'方法':<15} {'K':<4} {'轮廓':<10} {'CH指数':<12} {'DB指数':<10}")
    print("-" * 70)
    for i, (method, k, metrics) in enumerate(all_results[:10], 1):
        print(f"{i:<4} {method:<15} {k:<4} {metrics['silhouette']:<10.4f} {metrics['calinski_harabasz']:<12.4f} {metrics['davies_bouldin']:<10.4f}")

    # 按DB指数排序 (越低越好)
    all_results_db = sorted(all_results, key=lambda x: x[2]['davies_bouldin'])
    print("\n按Davies-Bouldin指数排名 (越低越好):")
    print("-" * 70)
    print(f"{'排名':<4} {'方法':<15} {'K':<4} {'轮廓':<10} {'CH指数':<12} {'DB指数':<10}")
    print("-" * 70)
    for i, (method, k, metrics) in enumerate(all_results_db[:10], 1):
        print(f"{i:<4} {method:<15} {k:<4} {metrics['silhouette']:<10.4f} {metrics['calinski_harabasz']:<12.4f} {metrics['davies_bouldin']:<10.4f}")

    # ===== 使用最佳配置生成详细分析 =====
    best_method, best_k, best_metrics = all_results[0]
    print("\n" + "="*80)
    print(f"【最佳配置】{best_method}, K={best_k}")
    print("="*80)

    if 'KMeans' in best_method:
        if '5d' in best_method:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_5d_scaled)
        else:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_6d_scaled)
    else:
        if '5d' in best_method:
            hier = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
            labels = hier.fit_predict(X_5d_scaled)
        else:
            hier = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
            labels = hier.fit_predict(X_6d_scaled)

    # 详细簇分析
    feature_names = feature_names_5d if '5d' in best_method else feature_names_6d

    print(f"\n簇分析 (使用{best_method}特征):")
    print("-" * 80)

    cluster_users = defaultdict(list)
    for i, u in enumerate(all_user_features):
        cluster_users[labels[i]].append(u)

    for cluster_id in range(best_k):
        users = cluster_users[cluster_id]
        print(f"\n簇 {cluster_id} ({len(users)} 用户):")

        # 特征均值
        cluster_feats = {feat: np.mean([u[feat] for u in users]) for feat in feature_names}
        for feat, val in cluster_feats.items():
            print(f"  {feat}: {val:.4f}")

        # 代表用户
        sample = sorted(users, key=lambda u: u['review_count'], reverse=True)[:3]
        print(f"  代表用户: {[u['user_id'][:12] for u in sample]}")

    # 保存最终结果
    output_data = {
        'best_method': best_method,
        'best_k': best_k,
        'metrics': best_metrics,
        'feature_names': feature_names,
        'labels': labels.tolist(),
        'users': [{'user_id': u['user_id'], 'cluster': int(labels[i]), 'features': {feat: u[feat] for feat in feature_names}} for i, u in enumerate(all_user_features)]
    }

    output_file = os.path.join(output_dir, 'best_clustering_result.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n最佳结果已保存: {output_file}")


if __name__ == '__main__':
    main()
