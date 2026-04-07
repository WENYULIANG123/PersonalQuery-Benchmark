#!/usr/bin/env python3
"""
基于极端特征值筛选用户：对每个维度，只保留该维度上极端（最高/最低）的用户
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans


DEEP_ARC_TYPES = {
    'subordinate_clause': ['advcl', 'acl', 'ccomp', 'xcomp'],
    'relative_clause': ['relcl'],
    'coordination': ['cc', 'conj'],
}


def get_extended_features(text):
    """提取扩展句法特征"""
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
        'avg_sent_len', 'conj_word_ratio', 'prep_word_ratio', 'passive_ratio', 'punct_ratio'
    ]

    user_features = {'user_id': user_id, 'review_count': len(review_features)}
    for feat in feature_names:
        user_features[feat] = np.mean([f[feat] for f in review_features])

    return user_features


def select_extreme_users(user_features_list, n_per_extreme=5):
    """
    对每个关键维度，选择该维度上极端（最高/最低）的用户
    关键维度：coordination, subclause_ratio, relative_clause, avg_sent_len
    """
    extreme_dims = ['coordination', 'subclause_ratio', 'relative_clause', 'avg_sent_len', 'prep_density']

    selected_indices = set()

    for dim in extreme_dims:
        # 获取该维度上最高和最低的n_per_extreme个用户
        sorted_users = sorted(enumerate(user_features_list), key=lambda x: x[1][dim])

        # 最低的
        for idx, _ in sorted_users[:n_per_extreme]:
            selected_indices.add(idx)

        # 最高的
        for idx, _ in sorted_users[-n_per_extreme:]:
            selected_indices.add(idx)

    return list(selected_indices)


def main():
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    # 前100个用户
    user_files = glob.glob(os.path.join(input_dir, 'reviews_*.json'))[:500]
    print(f"处理 {len(user_files)} 个用户\n")

    # 处理用户
    all_user_features = []
    for user_file in user_files:
        user_feats = process_user_reviews(user_file)
        if user_feats:
            all_user_features.append(user_feats)

    print(f"成功处理 {len(all_user_features)} 个用户\n")

    feature_names = [
        'coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause',
        'avg_fanout', 'prep_density', 'noun_ratio', 'verb_ratio', 'adj_ratio',
        'avg_sent_len', 'conj_word_ratio', 'prep_word_ratio', 'passive_ratio', 'punct_ratio'
    ]

    X_all = np.array([[u[feat] for feat in feature_names] for u in all_user_features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ===== 实验：不同极端用户筛选数量 =====
    print("="*80)
    print("【极端用户筛选 + KMeans聚类】")
    print("="*80)

    best_result = None
    best_score = 0

    for n_per_extreme in [3, 5, 8, 10, 12, 15]:
        selected_indices = select_extreme_users(all_user_features, n_per_extreme)
        X_selected = X_scaled[selected_indices]

        for k in [2, 3, 4]:
            if k >= len(selected_indices):
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_selected)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            sil = silhouette_score(X_selected, labels)
            ch = calinski_harabasz_score(X_selected, labels)
            db = davies_bouldin_score(X_selected, labels)

            print(f"n_per_dim={n_per_extreme:2d}, n_users={len(selected_indices):2d}, K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")

            if sil > best_score:
                best_score = sil
                best_result = {
                    'n_per_extreme': n_per_extreme,
                    'n_select': len(selected_indices),
                    'k': k,
                    'indices': selected_indices,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                    'labels': labels,
                }

    print(f"\n最佳: n_per_extreme={best_result['n_per_extreme']}, n_users={best_result['n_select']}, K={best_result['k']}, 轮廓={best_result['silhouette']:.4f}")

    # ===== 尝试不同特征子集 =====
    print("\n" + "="*80)
    print("【不同特征子集效果】")
    print("="*80)

    feature_subsets = {
        'core_4d': ['coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause'],
        'core_6d': ['coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause', 'avg_fanout', 'prep_density'],
        'syntax_6d': ['coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause', 'avg_sent_len', 'conj_word_ratio'],
        'pos_4d': ['noun_ratio', 'verb_ratio', 'adj_ratio', 'passive_ratio'],
        'all_14d': feature_names,
    }

    for subset_name, subset_features in feature_subsets.items():
        # 提取子集
        X_subset = np.array([[u[feat] for feat in subset_features] for u in all_user_features])
        scaler_sub = StandardScaler()
        X_subset_scaled = scaler_sub.fit_transform(X_subset)

        # 用最佳筛选参数
        selected_indices = select_extreme_users(all_user_features, best_result['n_per_extreme'])
        X_sel = X_subset_scaled[selected_indices]

        for k in [2, 3]:
            if k >= len(selected_indices):
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_sel)

            sil = silhouette_score(X_sel, labels)

            print(f"{subset_name:15s}, K={k}: 轮廓={sil:.4f}")

            if sil > best_score:
                best_score = sil
                best_result['feature_subset'] = subset_name
                best_result['feature_names'] = subset_features
                best_result['scaler'] = scaler_sub

    print(f"\n最终最佳: 轮廓={best_score:.4f}")

    # ===== 使用最佳配置详细分析 =====
    print("\n" + "="*80)
    print(f"【最佳配置详细分析】")
    print("="*80)

    selected_users = [all_user_features[i] for i in best_result['indices']]
    labels = best_result['labels']

    # 标准化特征名
    if 'feature_names' not in best_result:
        best_result['feature_names'] = feature_names

    for cluster_id in range(best_result['k']):
        users = [selected_users[j] for j in range(len(selected_users)) if labels[j] == cluster_id]
        print(f"\n簇 {cluster_id} ({len(users)} 用户):")
        for feat in best_result['feature_names']:
            vals = [u[feat] for u in users]
            print(f"  {feat:20s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")
        sample = sorted(users, key=lambda u: u['review_count'], reverse=True)[:3]
        print(f"  代表用户: {[u['user_id'][:12] for u in sample]}")

    # 保存结果
    # 构建可序列化的结果
    serializable_result = {
        'n_per_extreme': best_result['n_per_extreme'],
        'n_select': best_result['n_select'],
        'k': best_result['k'],
        'silhouette': float(best_result['silhouette']),
        'calinski_harabasz': float(best_result['calinski_harabasz']),
        'davies_bouldin': float(best_result['davies_bouldin']),
        'feature_subset': best_result.get('feature_subset', 'unknown'),
    }

    output_data = {
        'best_config': serializable_result,
        'all_users_count': len(all_user_features),
        'feature_names': best_result['feature_names'],
        'selected_users': []
    }

    for i, u in enumerate(selected_users):
        output_data['selected_users'].append({
            'user_id': u['user_id'],
            'cluster': int(labels[i]),
            'review_count': u['review_count'],
            'features': {feat: u[feat] for feat in best_result['feature_names']}
        })

    output_file = os.path.join(output_dir, 'extreme_users_result.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
