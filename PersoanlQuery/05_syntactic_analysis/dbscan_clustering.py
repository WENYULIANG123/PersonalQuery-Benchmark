#!/usr/bin/env python3
"""
使用DBSCAN密度聚类 + 多种特征工程方法尝试达到0.5+
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


DEEP_ARC_TYPES = {
    'subordinate_clause': ['advcl', 'acl', 'ccomp', 'xcomp'],
    'relative_clause': ['relcl'],
    'coordination': ['cc', 'conj'],
}


def get_extended_features(text):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    sentences = list(doc.sents)
    n_tokens = len(tokens)
    n_sentences = len(sentences)

    if n_tokens == 0:
        return None

    coord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['coordination'])
    coordination = coord_count / n_tokens

    subclause_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'] + DEEP_ARC_TYPES['relative_clause'])
    subclause_ratio = subclause_count / n_tokens

    relcl_count = sum(1 for t in tokens if t.dep_ == 'relcl')
    relative_clause = relcl_count / n_tokens

    subord_count = sum(1 for t in tokens if t.dep_ in DEEP_ARC_TYPES['subordinate_clause'])
    subordinate_clause = subord_count / n_tokens

    fanouts = [len(list(t.children)) for t in tokens]
    avg_fanout = np.mean(fanouts) if fanouts else 0

    prep_count = sum(1 for t in tokens if t.dep_ == 'prep')
    prep_density = prep_count / n_tokens

    noun_count = sum(1 for t in tokens if t.dep_ in ['nsubj', 'dobj', 'pobj', 'nsubjpass'])
    noun_ratio = noun_count / n_tokens

    verb_count = sum(1 for t in tokens if t.dep_ == 'ROOT' or t.pos_ in ['VERB'])
    verb_ratio = verb_count / n_tokens

    adj_count = sum(1 for t in tokens if t.dep_ == 'amod' or t.pos_ == 'ADJ')
    adj_ratio = adj_count / n_tokens

    avg_sent_len = n_tokens / n_sentences if n_sentences > 0 else n_tokens

    conj_words = sum(1 for t in tokens if t.text.lower() in ['and', 'or', 'but', 'yet', 'nor', 'so'])
    conj_word_ratio = conj_words / n_tokens

    prep_words = sum(1 for t in tokens if t.text.lower() in ['in', 'on', 'at', 'with', 'by', 'for', 'from', 'to', 'into', 'onto'])
    prep_word_ratio = prep_words / n_tokens

    passive_count = sum(1 for t in tokens if t.dep_ in ['nsubjpass', 'auxpass'])
    passive_ratio = passive_count / n_tokens

    punct_count = sum(1 for t in doc if t.is_punct)
    punct_ratio = punct_count / (n_tokens + punct_count) if (n_tokens + punct_count) > 0 else 0

    # 额外特征
    # 句子长度标准差
    sent_lens = [len(list(s)) for s in sentences] if sentences else [n_tokens]
    sent_len_std = np.std(sent_lens) / 50 if len(sent_lens) > 1 else 0

    # 唯一词比例
    unique_words = len(set(t.text.lower() for t in tokens))
    unique_ratio = unique_words / n_tokens if n_tokens > 0 else 0

    return {
        'coordination': coordination,
        'subclause_ratio': subclause_ratio,
        'relative_clause': relative_clause,
        'subordinate_clause': subordinate_clause,
        'avg_fanout': avg_fanout,
        'prep_density': prep_density,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'adj_ratio': adj_ratio,
        'avg_sent_len': avg_sent_len / 50,
        'conj_word_ratio': conj_word_ratio,
        'prep_word_ratio': prep_word_ratio,
        'passive_ratio': passive_ratio,
        'punct_ratio': punct_ratio,
        'sent_len_std': sent_len_std,
        'unique_ratio': unique_ratio,
        'token_count': n_tokens,
    }


def process_user_reviews(user_file):
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
            feats = get_extended_features(review)
            if feats:
                review_features.append(feats)

    if not review_features:
        return None

    feature_names = [
        'coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause',
        'avg_fanout', 'prep_density', 'noun_ratio', 'verb_ratio', 'adj_ratio',
        'avg_sent_len', 'conj_word_ratio', 'prep_word_ratio', 'passive_ratio', 'punct_ratio',
        'sent_len_std', 'unique_ratio'
    ]

    user_features = {'user_id': user_id, 'review_count': len(review_features)}
    for feat in feature_names:
        user_features[feat] = np.mean([f[feat] for f in review_features])

    return user_features


def select_extreme_users(user_features_list, n_per_extreme=3):
    extreme_dims = ['coordination', 'subclause_ratio', 'relative_clause', 'avg_sent_len', 'prep_density']
    selected_indices = set()

    for dim in extreme_dims:
        sorted_users = sorted(enumerate(user_features_list), key=lambda x: x[1][dim])
        for idx, _ in sorted_users[:n_per_extreme]:
            selected_indices.add(idx)
        for idx, _ in sorted_users[-n_per_extreme:]:
            selected_indices.add(idx)

    return list(selected_indices)


def main():
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    user_files = glob.glob(os.path.join(input_dir, 'reviews_*.json'))[:500]
    print(f"处理 {len(user_files)} 个用户\n")

    all_user_features = []
    for user_file in user_files:
        user_feats = process_user_reviews(user_file)
        if user_feats:
            all_user_features.append(user_feats)

    print(f"成功处理 {len(all_user_features)} 个用户\n")

    # 特征子集
    core_4d = ['coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause']
    all_features = [k for k in all_user_features[0].keys() if k not in ['user_id', 'review_count', 'token_count']]

    # 极端用户筛选
    selected_indices = select_extreme_users(all_user_features, n_per_extreme=3)
    selected_users = [all_user_features[i] for i in selected_indices]
    print(f"极端筛选后: {len(selected_users)} 个用户\n")

    X_all = np.array([[u[feat] for feat in all_features] for u in selected_users])

    print("="*80)
    print("【不同聚类方法对比】")
    print("="*80)

    best_score = 0
    best_method = None
    best_labels = None
    best_config = None

    # 1. KMeans (基准)
    print("\n--- KMeans ---")
    for k in [2, 3, 4, 5]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"KMeans K={k}: 轮廓={sil:.4f}")
        if sil > best_score:
            best_score = sil
            best_method = 'KMeans'
            best_labels = labels
            best_config = {'k': k, 'features': all_features, 'scaler': 'Standard'}

    # 2. DBSCAN - 尝试不同eps和min_samples
    print("\n--- DBSCAN ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    for eps in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        for min_samples in [2, 3, 4, 5]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters < 2:
                continue

            # 只用非噪声点计算
            mask = labels != -1
            if mask.sum() < n_clusters * 2:
                continue

            sil = silhouette_score(X_scaled[mask], labels[mask])
            print(f"DBSCAN eps={eps}, min={min_samples}: n_cluster={n_clusters}, noise={n_noise}, 轮廓={sil:.4f}")

            if sil > best_score:
                best_score = sil
                best_method = 'DBSCAN'
                best_labels = labels
                best_config = {'eps': eps, 'min_samples': min_samples, 'features': all_features}

    # 3. Spectral Clustering
    print("\n--- SpectralClustering ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    for k in [2, 3, 4, 5]:
        try:
            spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors', n_neighbors=5)
            labels = spectral.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            print(f"Spectral K={k}: 轮廓={sil:.4f}")
            if sil > best_score:
                best_score = sil
                best_method = 'Spectral'
                best_labels = labels
                best_config = {'k': k, 'features': all_features}
        except:
            pass

    # 4. GMM
    print("\n--- GaussianMixture ---")
    for k in [2, 3, 4, 5]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
        labels = gmm.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"GMM K={k}: 轮廓={sil:.4f}")
        if sil > best_score:
            best_score = sil
            best_method = 'GMM'
            best_labels = labels
            best_config = {'k': k, 'features': all_features}

    # 5. PCA降维后再聚类
    print("\n--- PCA + KMeans ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    for n_components in [2, 3, 4]:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        print(f"PCA n={n_components}: 方差解释率={sum(pca.explained_variance_ratio_):.3f}")

        for k in [2, 3, 4]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)
            print(f"  PCA{n_components}+KMeans K={k}: 轮廓={sil:.4f}")
            if sil > best_score:
                best_score = sil
                best_method = f'PCA{n_components}+KMeans'
                best_labels = labels
                best_config = {'n_components': n_components, 'k': k, 'pca': pca}

    print(f"\n{'='*80}")
    print(f"【最佳方法】{best_method}, 轮廓={best_score:.4f}")
    print(f"{'='*80}")

    # 如果最佳方法需要PCA，对全部特征重新计算
    if 'PCA' in best_method:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        pca = best_config.get('pca')
        if pca:
            X_final = pca.transform(X_scaled)
        else:
            X_final = X_scaled
    else:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_all)

    # 详细分析最佳聚类
    print(f"\n【最佳聚类详细分析】")
    print("-"*60)

    cluster_users = defaultdict(list)
    for i, u in enumerate(selected_users):
        cluster_users[best_labels[i]].append(u)

    for cluster_id in sorted(cluster_users.keys()):
        if cluster_id == -1:
            print(f"\n噪声点 ({len(cluster_users[cluster_id])} 用户):")
        else:
            print(f"\n簇 {cluster_id} ({len(cluster_users[cluster_id])} 用户):")

        for feat in best_config.get('features', all_features)[:6]:  # 只显示前6个特征
            vals = [u[feat] for u in cluster_users[cluster_id]]
            print(f"  {feat:20s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")

        sample = sorted(cluster_users[cluster_id], key=lambda u: u['review_count'], reverse=True)[:3]
        print(f"  代表用户: {[u['user_id'][:12] for u in sample]}")

    # 保存结果
    output_data = {
        'best_method': best_method,
        'best_score': float(best_score),
        'best_config': {k: v for k, v in best_config.items() if k != 'pca'},
        'n_selected_users': len(selected_users),
        'users': []
    }

    for i, u in enumerate(selected_users):
        output_data['users'].append({
            'user_id': u['user_id'],
            'cluster': int(best_labels[i]) if best_labels[i] != -1 else -1,
            'features': {feat: u[feat] for feat in all_features}
        })

    output_file = os.path.join(output_dir, 'dbscan_best_result.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
