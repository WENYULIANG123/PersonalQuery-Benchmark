#!/usr/bin/env python3
"""
筛选风格差异最大的代表性用户，然后进行聚类
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


# 扩展特征：增加更多有区分力的句法特征
DEEP_ARC_TYPES = {
    'subordinate_clause': ['advcl', 'acl', 'ccomp', 'xcomp'],
    'relative_clause': ['relcl'],
    'coordination': ['cc', 'conj'],
}


def get_extended_features(text):
    """提取扩展句法特征（更多维度增加区分力）"""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    sentences = list(doc.sents)
    n_tokens = len(tokens)
    n_sentences = len(sentences)

    if n_tokens == 0:
        return None

    # 基础6维
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

    # 扩展特征
    # 7. 名词比率 (nsubj, dobj, pobj)
    noun_count = sum(1 for t in tokens if t.dep_ in ['nsubj', 'dobj', 'pobj', 'nsubjpass'])
    noun_ratio = noun_count / n_tokens

    # 8. 动词比率 (ROOT, VB, VBD, VBG, VBN, VBP, VBZ)
    verb_count = sum(1 for t in tokens if t.dep_ == 'ROOT' or t.pos_ in ['VERB'])
    verb_ratio = verb_count / n_tokens

    # 9. 形容词比率 (amod, adj)
    adj_count = sum(1 for t in tokens if t.dep_ == 'amod' or t.pos_ == 'ADJ')
    adj_ratio = adj_count / n_tokens

    # 10. 平均句子长度
    avg_sent_len = n_tokens / n_sentences if n_sentences > 0 else n_tokens

    # 11. 并列连词密度 (and, or, but)
    conj_words = sum(1 for t in tokens if t.text.lower() in ['and', 'or', 'but', 'yet', 'nor', 'so'])
    conj_word_ratio = conj_words / n_tokens

    # 12. 介词短语密度 (in, on, at, with, by, for, from, to)
    prep_words = sum(1 for t in tokens if t.text.lower() in ['in', 'on', 'at', 'with', 'by', 'for', 'from', 'to', 'into', 'onto', 'upon'])
    prep_word_ratio = prep_words / n_tokens

    # 13. 被动语态比例
    passive_count = sum(1 for t in tokens if t.dep_ in ['nsubjpass', 'auxpass'])
    passive_ratio = passive_count / n_tokens

    # 14. 标点密度
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
        'avg_sent_len': avg_sent_len / 50,  # 归一化
        'conj_word_ratio': conj_word_ratio,
        'prep_word_ratio': prep_word_ratio,
        'passive_ratio': passive_ratio,
        'punct_ratio': punct_ratio,
        'token_count': n_tokens,
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
            feats = get_extended_features(review)
            if feats:
                review_features.append(feats)

    if not review_features:
        return None

    # 扩展特征 (14维)
    extended_features = [
        'coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause',
        'avg_fanout', 'prep_density', 'noun_ratio', 'verb_ratio', 'adj_ratio',
        'avg_sent_len', 'conj_word_ratio', 'prep_word_ratio', 'passive_ratio', 'punct_ratio'
    ]

    user_features = {'user_id': user_id, 'review_count': len(review_features)}
    for feat in extended_features:
        user_features[feat] = np.mean([f[feat] for f in review_features])

    return user_features


def maximin_sampling(X, user_ids, n_select, random_state=42):
    """Maximin距离采样：选择彼此距离最远的点"""
    np.random.seed(random_state)
    n_users = len(X)

    if n_select >= n_users:
        return list(range(n_users))

    # 选择第一个点：随机或选择最极端的点
    # 选择最极端的点（到质心最远的）
    centroid = np.mean(X, axis=0)
    distances_to_centroid = np.linalg.norm(X - centroid, axis=1)
    selected = [np.argmax(distances_to_centroid)]
    remaining = set(range(n_users)) - {selected[0]}

    for _ in range(n_select - 1):
        # 对于每个剩余点，计算到已选点的最小距离
        min_distances = []
        for idx in remaining:
            dists = [np.linalg.norm(X[idx] - X[s]) for s in selected]
            min_distances.append((idx, min(dists)))

        # 选择距离已选点最近距离最大的点（即最极端的点）
        min_distances.sort(key=lambda x: x[1], reverse=True)
        next_selected = min_distances[0][0]
        selected.append(next_selected)
        remaining.remove(next_selected)

    return selected


def compute_pairwise_distances(X):
    """计算所有用户两两之间的距离矩阵"""
    n = len(X)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(X[i] - X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def main():
    input_dir = '/fs04/ar57/wenyu/result/personal_query/00_data_preparation'
    output_dir = '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/user_style_clusters'
    os.makedirs(output_dir, exist_ok=True)

    # 获取前100个用户
    user_files = glob.glob(os.path.join(input_dir, 'reviews_*.json'))[:100]
    print(f"只处理前100个用户")

    # 处理用户
    all_user_features = []
    for user_file in user_files:
        user_feats = process_user_reviews(user_file)
        if user_feats:
            all_user_features.append(user_feats)

    print(f"成功处理 {len(all_user_features)} 个用户\n")

    # 14维扩展特征
    feature_names_14d = [
        'coordination', 'subclause_ratio', 'relative_clause', 'subordinate_clause',
        'avg_fanout', 'prep_density', 'noun_ratio', 'verb_ratio', 'adj_ratio',
        'avg_sent_len', 'conj_word_ratio', 'prep_word_ratio', 'passive_ratio', 'punct_ratio'
    ]

    X_all = np.array([[u[feat] for feat in feature_names_14d] for u in all_user_features])

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ===== 实验：不同筛选比例的效果 =====
    print("="*80)
    print("【Maximin采样 + KMeans聚类】不同用户数效果对比")
    print("="*80)

    n_users_list = [20, 30, 40, 50, 60, 80, 100]
    best_result = None
    best_score = 0

    for n_select in n_users_list:
        # Maximin采样
        selected_indices = maximin_sampling(X_scaled, [u['user_id'] for u in all_user_features], n_select)
        X_selected = X_scaled[selected_indices]

        # 尝试不同K值
        for k in [2, 3, 4]:
            if k >= n_select:
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_selected)

            # 计算指标
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            sil = silhouette_score(X_selected, labels)
            ch = calinski_harabasz_score(X_selected, labels)
            db = davies_bouldin_score(X_selected, labels)

            print(f"n={n_select:3d}, K={k}: 轮廓={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")

            if sil > best_score:
                best_score = sil
                best_result = {
                    'n_select': n_select,
                    'k': k,
                    'indices': selected_indices,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                    'labels': labels,
                    'feature_names': feature_names_14d,
                }

    print(f"\n最佳配置: n={best_result['n_select']}, K={best_result['k']}, 轮廓={best_result['silhouette']:.4f}")

    # ===== 使用最佳配置进行详细分析 =====
    print("\n" + "="*80)
    print(f"【最佳配置详细分析】n={best_result['n_select']}, K={best_result['k']}")
    print("="*80)

    selected_users = [all_user_features[i] for i in best_result['indices']]
    labels = best_result['labels']

    cluster_users = defaultdict(list)
    for i, u in enumerate(selected_users):
        cluster_users[labels[i]].append(u)

    for cluster_id in range(best_result['k']):
        users = cluster_users[cluster_id]
        print(f"\n簇 {cluster_id} ({len(users)} 用户):")

        # 特征均值
        for feat in feature_names_14d:
            vals = [u[feat] for u in users]
            print(f"  {feat:20s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")

        # 代表用户
        sample = sorted(users, key=lambda u: u['review_count'], reverse=True)[:3]
        print(f"  代表用户: {[u['user_id'][:12] for u in sample]}")

    # ===== 与原始全部用户对比 =====
    print("\n" + "="*80)
    print("【对比：全部用户 vs 筛选后用户】")
    print("="*80)

    # 全部用户 K=2
    kmeans_all = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_all = kmeans_all.fit_predict(X_scaled)
    sil_all = silhouette_score(X_scaled, labels_all)

    # 筛选后用户
    sil_sel = best_result['silhouette']

    print(f"全部用户 (n={len(all_user_features)}, K=2): 轮廓={sil_all:.4f}")
    print(f"筛选用户 (n={best_result['n_select']}, K={best_result['k']}): 轮廓={sil_sel:.4f}")
    print(f"提升: {(sil_sel - sil_all):.4f}")

    # ===== 保存结果 =====
    output_data = {
        'best_config': {
            'method': 'Maximin + KMeans',
            'n_selected': best_result['n_select'],
            'k': best_result['k'],
            'metrics': {
                'silhouette': best_result['silhouette'],
                'calinski_harabasz': best_result['calinski_harabasz'],
                'davies_bouldin': best_result['davies_bouldin'],
            }
        },
        'all_users_count': len(all_user_features),
        'feature_names': feature_names_14d,
        'selected_users': []
    }

    for i, u in enumerate(selected_users):
        output_data['selected_users'].append({
            'user_id': u['user_id'],
            'cluster': int(labels[i]),
            'review_count': u['review_count'],
            'features': {feat: u[feat] for feat in feature_names_14d}
        })

    output_file = os.path.join(output_dir, 'maximin_sampling_result.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
