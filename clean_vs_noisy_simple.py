#!/usr/bin/env python3
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

loocv_base = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data')
metadata_file = Path("/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl")

print("=" * 100)
print("CLEAN vs NOISY QUERY BASELINE PERFORMANCE (e5-base-v2, NO fine-tuning)")
print("=" * 100)

print("\nLoading model...")
try:
    model = SentenceTransformer("intfloat/e5-base-v2", local_files_only=True)
except:
    model = SentenceTransformer("intfloat/e5-base-v2")

print("Loading product metadata...")
with open(metadata_file, 'rb') as f:
    product_metadata = pickle.load(f)

all_product_asins = list(product_metadata.keys())
print(f"Corpus size: {len(all_product_asins)}")

print("Computing product embeddings (using titles)...")
product_embeddings = {}
batch_size = 32

for i in range(0, len(all_product_asins), batch_size):
    batch_asins = all_product_asins[i:i+batch_size]
    batch_titles = []
    
    for asin in batch_asins:
        title = product_metadata.get(asin, {}).get('title', '')
        batch_titles.append(title)
    
    if batch_titles:
        embeddings = model.encode(batch_titles, batch_size=batch_size, 
                                convert_to_numpy=True, show_progress_bar=False)
        for asin, emb in zip(batch_asins, embeddings):
            product_embeddings[asin] = emb

print(f"✓ {len(product_embeddings)} product embeddings ready\n")

def evaluate_queries(holdout_pairs, query_type='clean'):
    metrics_10 = []
    
    for pair in holdout_pairs:
        clean_q = pair.get('query', '')
        noisy_q = pair.get('positive', '')
        target_asin = pair.get('asin', '')
        
        query = clean_q if query_type == 'clean' else noisy_q
        
        if not (query and target_asin):
            continue
        if target_asin not in product_embeddings:
            continue
        
        query_emb = model.encode(query, convert_to_numpy=True)
        
        scores = []
        for asin, prod_emb in product_embeddings.items():
            if prod_emb is not None:
                sim = np.dot(query_emb, prod_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(prod_emb) + 1e-8
                )
                scores.append((asin, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_asins = [asin for asin, _ in scores]
        
        if target_asin in ranked_asins[:10]:
            position = ranked_asins[:10].index(target_asin)
            mrr = 1.0 / (position + 1)
        else:
            mrr = 0.0
        
        metrics_10.append(mrr)
    
    return np.mean(metrics_10) if metrics_10 else 0.0, len(metrics_10)

users = [
    "A13OFOB1394G31", "A1GYEGLX3P2Y7P", "A1PAGHECG401K1",
    "A211W8JLJFDIC0", "A24FX30B20WLMV", "A2GJX2KCUSR0EI",
    "A2MNB77YGJ3CN0", "A2U6VP21H9UVV3", "A3E5V5TSTAY3R9",
    "A3RZ23PMNZGQC1", "ALYZJ7W14YS26"
]

print("Evaluating each user...")
print("=" * 100)
print(f"{'User ID':25s} {'Clean MRR@10':15s} {'Noisy MRR@10':15s} {'Degradation':15s} {'N':8s}")
print("-" * 100)

results = {}
clean_mrrs = []
noisy_mrrs = []

for user_id in users:
    user_dir = loocv_base / f"user_{user_id}"
    holdout_file = user_dir / "holdout.json"
    
    if not holdout_file.exists():
        print(f"{user_id:25s} ⚠️  holdout.json not found")
        continue
    
    with open(holdout_file) as f:
        holdout_data = json.load(f).get('pairs', [])
    
    clean_mrr, clean_n = evaluate_queries(holdout_data, 'clean')
    noisy_mrr, noisy_n = evaluate_queries(holdout_data, 'noisy')
    
    degradation = clean_mrr - noisy_mrr
    
    results[user_id] = {
        'clean_mrr': clean_mrr,
        'noisy_mrr': noisy_mrr,
        'degradation': degradation,
        'n_pairs': clean_n
    }
    
    clean_mrrs.append(clean_mrr)
    noisy_mrrs.append(noisy_mrr)
    
    print(f"{user_id:25s} {clean_mrr:15.4f} {noisy_mrr:15.4f} {degradation:15.4f} {clean_n:8d}")

print("-" * 100)
avg_clean = np.mean(clean_mrrs)
avg_noisy = np.mean(noisy_mrrs)
avg_deg = np.mean([r['degradation'] for r in results.values()])

print(f"{'AVERAGE':25s} {avg_clean:15.4f} {avg_noisy:15.4f} {avg_deg:15.4f}")
print(f"{'STD':25s} {np.std(clean_mrrs):15.4f} {np.std(noisy_mrrs):15.4f} {np.std([r['degradation'] for r in results.values()]):15.4f}")

print("\n" + "=" * 100)
print("CONCLUSIONS")
print("=" * 100)

deg_pct = (avg_deg / avg_clean * 100) if avg_clean > 0 else 0

print(f"\nAverage Clean Query MRR@10:  {avg_clean:.4f}")
print(f"Average Noisy Query MRR@10:  {avg_noisy:.4f}")
print(f"Average Degradation:         {avg_deg:.4f} ({deg_pct:.1f}%)")

if deg_pct < 5:
    print("\n✅ FINDING: Noisy queries degrade performance by <5%")
    print("   → Query noise is NOT the primary limiting factor")
elif deg_pct < 20:
    print(f"\n⚠️  FINDING: Noisy queries degrade performance by {deg_pct:.1f}%")
    print("   → Query noise has minor-to-moderate impact")
else:
    print(f"\n❌ FINDING: Noisy queries degrade performance by {deg_pct:.1f}%")
    print("   → Query noise significantly impacts ranking")

output_file = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/clean_vs_noisy_baseline.json')
with open(output_file, 'w') as f:
    json.dump({
        'average_clean_mrr': float(avg_clean),
        'average_noisy_mrr': float(avg_noisy),
        'average_degradation': float(avg_deg),
        'degradation_percent': float(deg_pct),
        'all_users': results
    }, f, indent=2)

print(f"\n✅ Results saved: {output_file}")
