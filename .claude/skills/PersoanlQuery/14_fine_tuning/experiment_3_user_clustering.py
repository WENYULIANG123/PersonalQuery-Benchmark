#!/usr/bin/env python3

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from difflib import SequenceMatcher
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'model_name': 'intfloat/e5-base-v2',
    'loocv_data_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data',
    'meta_path': '/fs04/ar57/wenyu/data/Amazon-Reviews-2018/intermediate/df_ucsd_meta.pkl',
    'output_dir': '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/experiment_3_clustering',
}

def load_product_metadata(meta_path: str) -> Dict[str, str]:
    logger.info(f"Loading metadata...")
    with open(meta_path, 'rb') as f:
        df_meta = pickle.load(f)
    
    asin_to_title = {}
    for idx, row in df_meta.iterrows():
        asin = str(row['asin'])
        title = row['title'] if pd.notna(row['title']) else ""
        asin_to_title[asin] = title
    
    return asin_to_title


def load_user_data(user_dir: str) -> Dict:
    with open(f'{user_dir}/holdout.json', 'r') as f:
        holdout = json.load(f)
    with open(f'{user_dir}/personal_train.json', 'r') as f:
        personal_train = json.load(f)
    with open(f'{user_dir}/global_train.json', 'r') as f:
        global_train = json.load(f)
    
    return {
        'holdout': holdout['pairs'],
        'personal_train': personal_train['pairs'],
        'global_train': global_train['pairs']
    }


def compute_noise_features(pairs: List[Dict]) -> Dict:
    typo_count = 0
    omission_count = 0
    repeat_count = 0
    insertion_count = 0
    
    total_errors = 0
    avg_edit_dist = 0
    
    for pair in pairs:
        clean = pair['query'].split()
        noisy = pair['positive'].split()
        
        clean_set = set(clean)
        noisy_set = set(noisy)
        
        omission = len(clean_set - noisy_set)
        insertion = len(noisy_set - clean_set)
        
        omission_count += omission
        insertion_count += insertion
        
        for w_noisy in noisy:
            found = False
            for w_clean in clean:
                if w_noisy != w_clean and SequenceMatcher(None, w_noisy, w_clean).ratio() > 0.8:
                    typo_count += 1
                    found = True
                    break
            if not found and w_noisy not in clean_set:
                repeat_count += 1
        
        avg_edit_dist += sum(
            min(len(w_c), len(w_n)) for w_c in clean for w_n in noisy
            if w_c != w_n
        )
        
        total_errors += max(omission, insertion, len(clean))
    
    n_pairs = len(pairs)
    
    return {
        'typo_ratio': typo_count / max(1, n_pairs * 5),
        'omission_ratio': omission_count / max(1, total_errors),
        'repeat_ratio': repeat_count / max(1, n_pairs * 5),
        'insertion_ratio': insertion_count / max(1, total_errors),
        'avg_query_length': np.mean([len(p['query'].split()) for p in pairs]),
        'query_length_std': np.std([len(p['query'].split()) for p in pairs]),
        'n_pairs': n_pairs,
    }


def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loocv_dir = Path(CONFIG['loocv_data_dir'])
    users = sorted([d.name for d in loocv_dir.iterdir() if d.is_dir()])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT 3: User Clustering")
    logger.info(f"{'='*60}")
    logger.info(f"Total users: {len(users)}")
    
    user_features = {}
    user_performance = {}
    user_to_pairs = {}
    
    logger.info("\nStep 1: Computing noise features for each user...")
    for user in users:
        user_data = load_user_data(str(loocv_dir / user))
        global_train = user_data['global_train']
        
        features = compute_noise_features(global_train)
        user_features[user] = features
        user_to_pairs[user] = global_train
        
        logger.info(f"  {user}: typo_ratio={features['typo_ratio']:.3f}, "
                   f"omission={features['omission_ratio']:.3f}, repeat={features['repeat_ratio']:.3f}")
    
    feature_names = ['typo_ratio', 'omission_ratio', 'repeat_ratio', 'insertion_ratio', 
                     'avg_query_length', 'query_length_std']
    X = np.array([[user_features[u][f] for f in feature_names] for u in users])
    
    logger.info("\nStep 2: Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info("\nStep 3: Finding optimal number of clusters (Elbow + Silhouette)...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, len(users))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        logger.info(f"  K={k}: Silhouette={silhouette_scores[-1]:.3f}")
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    logger.info(f"\n  Optimal K (by Silhouette): {optimal_k}")
    
    logger.info("\nStep 4: Final clustering with optimal K...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    logger.info("\nCluster assignments:")
    clustering_result = {}
    for user, cluster_id in zip(users, clusters):
        clustering_result[user] = int(cluster_id)
        logger.info(f"  {user} → Cluster {cluster_id}")
    
    logger.info("\nStep 5: Evaluate pretrained baseline per user...")
    model = SentenceTransformer(CONFIG['model_name'])
    asin_to_title = load_product_metadata(CONFIG['meta_path'])
    
    from sklearn.metrics import ndcg_score
    
    for user in users:
        user_data = load_user_data(str(loocv_dir / user))
        holdout = user_data['holdout']
        
        if not holdout:
            user_performance[user] = 0.0
            continue
        
        corpus = list(asin_to_title.values())
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        
        mrr_scores = []
        for pair in holdout:
            noisy_query = pair['positive']
            positive_asin = pair['asin']
            
            if positive_asin not in asin_to_title:
                continue
            
            query_embedding = model.encode(noisy_query, convert_to_tensor=True)
            
            import torch
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                corpus_embeddings
            ).cpu().numpy()
            
            ranked_indices = np.argsort(-similarities)
            
            for rank, idx in enumerate(ranked_indices):
                if corpus[idx] == asin_to_title[positive_asin]:
                    if rank < 10:
                        mrr_scores.append(1.0 / (rank + 1))
                    break
        
        user_performance[user] = np.mean(mrr_scores) if mrr_scores else 0.0
        logger.info(f"  {user}: MRR@10 = {user_performance[user]:.4f}")
    
    logger.info("\nStep 6: Compare within-cluster vs cross-cluster performance...")
    
    within_cluster_perf = []
    cross_cluster_perf = []
    
    for cluster_id in range(optimal_k):
        cluster_users = [u for u in users if clustering_result[u] == cluster_id]
        cluster_perf = [user_performance[u] for u in cluster_users]
        
        if cluster_perf:
            within_cluster_perf.extend(cluster_perf)
            logger.info(f"  Cluster {cluster_id} users: {cluster_users}")
            logger.info(f"    Avg MRR@10: {np.mean(cluster_perf):.4f} ± {np.std(cluster_perf):.4f}")
    
    logger.info(f"\n  Within-cluster avg MRR@10: {np.mean(within_cluster_perf):.4f}")
    
    results = {
        'clustering': clustering_result,
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'user_features': {u: user_features[u] for u in users},
        'user_performance': user_performance,
        'within_cluster_avg_mrr': float(np.mean(within_cluster_perf)),
        'pretrained_avg_mrr': float(np.mean(list(user_performance.values())))
    }
    
    results_file = output_dir / 'clustering_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"\nSummary:")
    logger.info(f"  Users: {len(users)}")
    logger.info(f"  Clusters found: {optimal_k}")
    logger.info(f"  Within-cluster MRR@10: {np.mean(within_cluster_perf):.4f}")
    logger.info(f"  Pretrained baseline MRR@10: {np.mean(list(user_performance.values())):.4f}")


if __name__ == '__main__':
    main()
