#!/usr/bin/env python3
"""
基于6维深层句法特征对stage0用户评论风格进行聚类分析

6维特征:
- coordination: cc+conj弧比例
- subclause_ratio: 从句弧比例
- relative_clause: relcl弧比例
- subordinate_clause: advcl/ccomp/xcomp弧比例
- avg_fanout: 平均子节点数
- prep_density: 介词短语密度
"""
import json
import numpy as np
from collections import defaultdict
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

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

    # 1. coordination: cc+conj弧比例
    coord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['coordination'])
    coordination = coord_count / n

    # 2. subclause_ratio: 从句弧比例
    subclause_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'] + DEEP_ARC_TYPES['relative_clause'])
    subclause_ratio = subclause_count / n

    # 3. relative_clause: relcl弧比例
    relcl_count = sum(1 for t in tokens if t.dep_ == 'relcl')
    relative_clause = relcl_count / n

    # 4. subordinate_clause: advcl/ccomp/xcomp弧比例
    subord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'])
    subordinate_clause = subord_count / n

    # 5. avg_fanout: 平均子节点数
    fanouts = [len(list(t.children)) for t in tokens]
    avg_fanout = np.mean(fanouts) if fanouts else 0

    # 6. prep_density: 介词短语密度
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
    """处理单个用户的评论文件，返回6维特征统计"""
    with open(user_file, 'r') as f:
        data = json.load(f)

    user_id = data['user_id']
    all_reviews = []

    # 收集所有target_reviews
    for product in data.get('results', []):
        all_reviews.extend(product.get('target_reviews', []))

    if not all_reviews:
        return None

    # 计算每条评论的6维特征
    review_features = []
    for review in all_reviews:
        if review and len(review.strip()) > 10:
            feats = get_6d_features(review)
            if feats:
                review_features.append(feats)

    if not review_features:
        return None

    # 计算用户的平均6维特征
    user_features = {
        'user_id': user_id,
        'coordination': np.mean([f['coordination'] for f in review_features]),
        'subclause_ratio': np.mean([f['subclause_ratio'] for f in review_features]),
        'relative_clause': np.mean([f['relative_clause'] for f in review_features]),
        'subordinate_clause': np.mean([f['subordinate_clause'] for f in review_features]),
        'avg_fanout': np.mean([f['avg_fanout'] for f in review_features]),
        'prep_density': np.mean([f['prep_density'] for f in review_features]),
        'review_count': len(review_features),
        'total_token_count': np.mean([f['token_count'] for f in review_features]),
    }

    return user_features


def main():
    # 配置
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有用户评论文件，只处理前100个
    user_files = glob.glob(os.path.join(input_dir, 'reviews_*.json'))[:100]
    print(f"只处理前100个用户，共找到 {len(user_files)} 个用户评论文件")

    # 处理每个用户
    all_user_features = []
    for i, user_file in enumerate(user_files):
        if (i + 1) % 50 == 0:
            print(f"处理进度: {i + 1}/{len(user_files)}")
        user_feats = process_user_reviews(user_file)
        if user_feats:
            all_user_features.append(user_feats)

    print(f"\n成功处理 {len(all_user_features)} 个用户")

    if len(all_user_features) == 0:
        print("没有找到有效用户数据")
        return

    # 特征名称
    feature_names = ['coordination', 'subclause_ratio', 'relative_clause',
                     'subordinate_clause', 'avg_fanout', 'prep_density']

    # ========== 1. 统计分析 ==========
    print("\n" + "="*90)
    print("6维句法特征全局统计:")
    print("="*90)

    for feat in feature_names:
        vals = [u[feat] for u in all_user_features]
        print(f"{feat:20s}: 均值={np.mean(vals):.4f}, 标准差={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # ========== 2. 聚类分析 ==========
    print("\n" + "="*90)
    print("K-means聚类分析 (K=4):")
    print("="*90)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = np.array([[u[feat] for feat in feature_names] for u in all_user_features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 尝试不同的K值
    for k in [3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        print(f"\n--- K={k} 聚类结果 ---")

        # 统计每个簇的用户数
        cluster_users = defaultdict(list)
        for i, u in enumerate(all_user_features):
            cluster_users[labels[i]].append(u)

        for cluster_id in range(k):
            users = cluster_users[cluster_id]
            print(f"\n簇 {cluster_id} ({len(users)} 用户):")

            # 打印该簇的特征均值
            cluster_feats = {feat: np.mean([u[feat] for u in users]) for feat in feature_names}
            print(f"  特征: " + ", ".join([f"{feat}={v:.4f}" for feat, v in cluster_feats.items()]))

            # 打印代表性用户ID
            sample_users = sorted(users, key=lambda u: u['review_count'], reverse=True)[:3]
            print(f"  代表用户: {', '.join([u['user_id'][:15] for u in sample_users])}")

        # 计算簇间距离
        centers = kmeans.cluster_centers_
        print(f"\n  簇间距离矩阵:")
        for i in range(k):
            for j in range(k):
                if i < j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    print(f"    簇{i}-簇{j}: {dist:.4f}")

    # ========== 3. 基于K=4的详细分析 ==========
    print("\n" + "="*90)
    print("详细用户风格分类 (K=4):")
    print("="*90)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 为每个簇命名
    cluster_stats = []
    for cluster_id in range(4):
        users = [u for i, u in enumerate(all_user_features) if labels[i] == cluster_id]
        cluster_feats = {feat: np.mean([u[feat] for u in users]) for feat in feature_names}
        cluster_stats.append({
            'cluster_id': cluster_id,
            'user_count': len(users),
            'features': cluster_feats,
            'users': users
        })

    # 根据特征为簇命名
    def classify_style(feats):
        style_parts = []
        if feats['coordination'] > 0.08:
            style_parts.append("并列复杂")
        if feats['subordinate_clause'] > 0.05:
            style_parts.append("从句丰富")
        if feats['relative_clause'] > 0.03:
            style_parts.append("定语从句多")
        if feats['prep_density'] > 0.15:
            style_parts.append("介词密集")
        if feats['avg_fanout'] > 2.5:
            style_parts.append("树形复杂")
        if feats['subclause_ratio'] < 0.05:
            style_parts.append("简单句")
        return "/".join(style_parts) if style_parts else "中等复杂度"

    for cs in sorted(cluster_stats, key=lambda x: x['features']['subclause_ratio'], reverse=True):
        print(f"\n【簇 {cs['cluster_id']}】({cs['user_count']} 用户)")
        style_type = classify_style(cs['features'])
        print(f"  风格类型: {style_type}")
        print(f"  特征均值:")
        for feat in feature_names:
            print(f"    {feat}: {cs['features'][feat]:.4f}")
        print(f"  用户示例:")
        sample = sorted(cs['users'], key=lambda u: u['review_count'], reverse=True)[:5]
        for u in sample:
            print(f"    {u['user_id']}: {u['review_count']}条评论")

    # ========== 4. 保存结果 ==========
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'user_count': len(all_user_features),
        'feature_names': feature_names,
        'clustering': {
            'method': 'KMeans',
            'n_clusters': 4,
            'labels': labels.tolist(),
        },
        'users': []
    }

    for i, u in enumerate(all_user_features):
        output_data['users'].append({
            'user_id': u['user_id'],
            'cluster': int(labels[i]),
            'review_count': u['review_count'],
            'features': {feat: u[feat] for feat in feature_names},
        })

    # 保存详细结果
    output_file = os.path.join(output_dir, 'user_style_clustering_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

    # 保存用户风格摘要CSV
    csv_file = os.path.join(output_dir, 'user_style_summary.csv')
    with open(csv_file, 'w') as f:
        header = 'user_id,cluster,review_count,' + ','.join(feature_names)
        f.write(header + '\n')
        for i, u in enumerate(all_user_features):
            row = [u['user_id'], str(labels[i]), str(u['review_count'])]
            row += [f"{u[feat]:.6f}" for feat in feature_names]
            f.write(','.join(row) + '\n')
    print(f"摘要CSV已保存到: {csv_file}")

    # ========== 5. 特征重要性分析 ==========
    print("\n" + "="*90)
    print("特征相关性分析:")
    print("="*90)

    from scipy.stats import pearsonr

    for i, f1 in enumerate(feature_names):
        for f2 in feature_names[i+1:]:
            corr, pval = pearsonr([u[f1] for u in all_user_features],
                                  [u[f2] for u in all_user_features])
            if abs(corr) > 0.3:
                print(f"  {f1} <-> {f2}: r={corr:.4f} (p={pval:.4f})")

    print("\n" + "="*90)
    print("聚类分析完成!")
    print("="*90)


if __name__ == '__main__':
    main()
