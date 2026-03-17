#!/usr/bin/env python3
"""
Clean vs Noisy Query Evaluation
Objective: Compare e5-base-v2 retrieval performance on clean vs noisy queries
without any fine-tuning.

Key question: Does using noisy queries (with typos, omissions) naturally degrade
ranking performance? How much?
"""

import json
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def compute_mrr_at_k(relevant_ranks, k=10):
    """Compute MRR@k where relevant_ranks is sorted list of relevant document positions (0-indexed)"""
    if not relevant_ranks or len(relevant_ranks) == 0:
        return 0.0
    first_relevant_rank = relevant_ranks[0] + 1
    if first_relevant_rank <= k:
        return 1.0 / first_relevant_rank
    return 0.0

def compute_ndcg_at_k(relevant_ranks, k=10):
    """Compute NDCG@k"""
    if not relevant_ranks or len(relevant_ranks) == 0:
        return 0.0
    
    dcg = 0.0
    for idx, rank in enumerate(relevant_ranks):
        if rank < k:
            dcg += 1.0 / np.log2(rank + 2)
    
    idcg = 0.0
    for i in range(min(len(relevant_ranks), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def load_product_corpus():
    """Load product corpus from pickle file"""
    corpus_path = "/fs04/ar57/wenyu/data/Amazon-Reviews-2018/intermediate/df_ucsd_meta.pkl"
    
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}", file=sys.stderr)
        return None, None
    
    print(f"Loading corpus from {corpus_path}...")
    df = pickle.load(open(corpus_path, 'rb'))
    print(f"  Loaded {len(df)} products")
    
    asin_to_title = {}
    for idx, row in df.iterrows():
        asin = row.get('asin') or row.get('ASIN')
        title = row.get('title') or row.get('Title')
        if asin and title:
            asin_to_title[asin] = title
    
    print(f"  Extracted {len(asin_to_title)} asin->title mappings")
    return df, asin_to_title

def evaluate_user(user_id, asin_to_title, model, k=10):
    """Evaluate one user's clean vs noisy queries"""
    
    user_dir = f"/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data/user_{user_id}"
    
    if not os.path.exists(user_dir):
        print(f"User dir not found: {user_dir}")
        return None
    
    holdout_path = f"{user_dir}/holdout.json"
    if not os.path.exists(holdout_path):
        return None
    
    with open(holdout_path) as f:
        holdout_data = json.load(f)
    
    pairs = holdout_data.get('pairs', [])
    if not pairs:
        return None
    
    print(f"  Evaluating {user_id}: {len(pairs)} holdout pairs")
    
    results = {
        'user_id': user_id,
        'n_pairs': len(pairs),
        'clean_mrr': [],
        'noisy_mrr': [],
        'clean_ndcg': [],
        'noisy_ndcg': [],
    }
    
    all_corpus_products = list(asin_to_title.keys())
    print(f"    Corpus size: {len(all_corpus_products)} products")
    
    for pair_idx, pair in enumerate(pairs):
        clean_query = pair.get('query', '')
        noisy_query = pair.get('positive', '')
        relevant_asin = pair.get('asin', '')
        
        if not clean_query or not noisy_query or not relevant_asin:
            continue
        
        if relevant_asin not in asin_to_title:
            continue
        
        relevant_title = asin_to_title[relevant_asin]
        
        try:
            clean_emb = model.encode(clean_query, convert_to_numpy=True)
            noisy_emb = model.encode(noisy_query, convert_to_numpy=True)
        except Exception as e:
            print(f"      Encoding error on pair {pair_idx}: {e}", file=sys.stderr)
            continue
        
        if pair_idx % max(1, len(pairs)//3) == 0:
            print(f"      Processing pair {pair_idx}/{len(pairs)}...", flush=True)
        
        try:
            corpus_embs = model.encode(
                [asin_to_title[asin] for asin in all_corpus_products],
                convert_to_numpy=True,
                batch_size=256
            )
        except Exception as e:
            print(f"      Corpus encoding error: {e}", file=sys.stderr)
            continue
        
        clean_scores = cosine_similarity([clean_emb], corpus_embs)[0]
        noisy_scores = cosine_similarity([noisy_emb], corpus_embs)[0]
        
        clean_ranks = np.argsort(-clean_scores)
        noisy_ranks = np.argsort(-noisy_scores)
        
        relevant_idx = all_corpus_products.index(relevant_asin)
        
        clean_relevant_rank = np.where(clean_ranks == relevant_idx)[0]
        noisy_relevant_rank = np.where(noisy_ranks == relevant_idx)[0]
        
        clean_mrr = compute_mrr_at_k(clean_relevant_rank if len(clean_relevant_rank) > 0 else [], k=k)
        noisy_mrr = compute_mrr_at_k(noisy_relevant_rank if len(noisy_relevant_rank) > 0 else [], k=k)
        
        clean_ndcg = compute_ndcg_at_k(clean_relevant_rank if len(clean_relevant_rank) > 0 else [], k=k)
        noisy_ndcg = compute_ndcg_at_k(noisy_relevant_rank if len(noisy_relevant_rank) > 0 else [], k=k)
        
        results['clean_mrr'].append(clean_mrr)
        results['noisy_mrr'].append(noisy_mrr)
        results['clean_ndcg'].append(clean_ndcg)
        results['noisy_ndcg'].append(noisy_ndcg)
    
    if not results['clean_mrr']:
        return None
    
    results['clean_mrr_mean'] = np.mean(results['clean_mrr'])
    results['clean_mrr_std'] = np.std(results['clean_mrr'])
    results['noisy_mrr_mean'] = np.mean(results['noisy_mrr'])
    results['noisy_mrr_std'] = np.std(results['noisy_mrr'])
    results['mrr_degradation'] = results['clean_mrr_mean'] - results['noisy_mrr_mean']
    
    results['clean_ndcg_mean'] = np.mean(results['clean_ndcg'])
    results['noisy_ndcg_mean'] = np.mean(results['noisy_ndcg'])
    
    return results

def main():
    print("=" * 100)
    print("CLEAN vs NOISY QUERY EVALUATION (e5-base-v2, no fine-tuning)")
    print("=" * 100)
    
    print("\n[1] Loading e5-base-v2 model...")
    model = SentenceTransformer('sentence-transformers/e5-base-v2')
    print(f"  Loaded: {model}")
    
    print("\n[2] Loading product corpus...")
    df, asin_to_title = load_product_corpus()
    if asin_to_title is None:
        sys.exit(1)
    
    print("\n[3] Evaluating all users...")
    all_results = {}
    
    users = [
        "A13OFOB1394G31", "A1GYEGLX3P2Y7P", "A1PAGHECG401K1", 
        "A211W8JLJFDIC0", "A24FX30B20WLMV", "A2GJX2KCUSR0EI",
        "A2MNB77YGJ3CN0", "A2U6VP21H9UVV3", "A3E5V5TSTAY3R9",
        "A3RZ23PMNZGQC1", "ALYZJ7W14YS26"
    ]
    
    for user_id in users:
        result = evaluate_user(user_id, asin_to_title, model, k=10)
        if result:
            all_results[user_id] = result
            print(f"✅ {user_id}")
        else:
            print(f"⚠️  {user_id} (skipped)")
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"\n{'User ID':25s} {'Clean MRR@10':15s} {'Noisy MRR@10':15s} {'Degradation':15s}")
    print("-" * 100)
    
    clean_mrrs = []
    noisy_mrrs = []
    degradations = []
    
    for user_id, result in sorted(all_results.items()):
        clean = result['clean_mrr_mean']
        noisy = result['noisy_mrr_mean']
        deg = result['mrr_degradation']
        
        clean_mrrs.append(clean)
        noisy_mrrs.append(noisy)
        degradations.append(deg)
        
        print(f"{user_id:25s} {clean:15.4f} {noisy:15.4f} {deg:15.4f}")
    
    print("-" * 100)
    print(f"{'AVERAGE':25s} {np.mean(clean_mrrs):15.4f} {np.mean(noisy_mrrs):15.4f} {np.mean(degradations):15.4f}")
    print(f"{'STD':25s} {np.std(clean_mrrs):15.4f} {np.std(noisy_mrrs):15.4f} {np.std(degradations):15.4f}")
    
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    
    avg_clean = np.mean(clean_mrrs)
    avg_noisy = np.mean(noisy_mrrs)
    avg_deg = np.mean(degradations)
    deg_pct = (avg_deg / avg_clean * 100) if avg_clean > 0 else 0
    
    print(f"\nAverage Clean Query MRR@10:  {avg_clean:.4f}")
    print(f"Average Noisy Query MRR@10:  {avg_noisy:.4f}")
    print(f"Average Degradation:         {avg_deg:.4f} ({deg_pct:.1f}%)")
    
    if avg_clean < 0.05:
        print("\n🔴 WARNING: Even clean queries show very low MRR (<0.05)")
        print("   This suggests a data quality issue, not just noise impact")
    
    if deg_pct < 10:
        print(f"\n✅ Finding: Noisy queries degrade performance by {deg_pct:.1f}% only")
        print("   Conclusion: Query noise is NOT the primary factor")
    elif deg_pct < 50:
        print(f"\n⚠️  Finding: Noisy queries degrade performance by {deg_pct:.1f}%")
        print("   Conclusion: Query noise has moderate impact")
    else:
        print(f"\n❌ Finding: Noisy queries degrade performance by {deg_pct:.1f}%")
        print("   Conclusion: Query noise has severe impact")
    
    output_file = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/clean_vs_noisy_results.json"
    summary = {
        'average_clean_mrr': float(np.mean(clean_mrrs)),
        'average_noisy_mrr': float(np.mean(noisy_mrrs)),
        'average_degradation': float(np.mean(degradations)),
        'degradation_percent': float(deg_pct),
        'all_users': all_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
