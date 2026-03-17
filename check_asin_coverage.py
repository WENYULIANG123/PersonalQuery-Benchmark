#!/usr/bin/env python3
import json
import pickle
from pathlib import Path
from collections import Counter

# Load product metadata
metadata_file = Path('/fs04/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl')
print("[*] Loading product metadata...")
with open(metadata_file, 'rb') as f:
    product_metadata = pickle.load(f)
catalog_asins = set(product_metadata.keys())
print(f"✓ Catalog has {len(catalog_asins)} products")

# Check holdout ASINs coverage
loocv_data_dir = Path('/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/loocv_data')
users = sorted([d.name.replace('user_', '') for d in loocv_data_dir.glob('user_*')])

coverage_stats = []
for user_id in users:
    holdout_file = loocv_data_dir / f'user_{user_id}' / 'holdout.json'
    with open(holdout_file) as f:
        holdout_data = json.load(f)
    
    pairs = holdout_data.get('pairs', [])
    holdout_asins = [p['asin'] for p in pairs]
    
    found = sum(1 for asin in holdout_asins if asin in catalog_asins)
    missing = len(holdout_asins) - found
    coverage = found / len(holdout_asins) if holdout_asins else 0
    
    coverage_stats.append({
        'user': user_id,
        'holdout_count': len(holdout_asins),
        'found_in_catalog': found,
        'missing': missing,
        'coverage_%': round(coverage * 100, 1)
    })

print("\n" + "="*80)
print("ASIN COVERAGE ANALYSIS BY USER")
print("="*80)
for stat in coverage_stats:
    status = "✓" if stat['coverage_%'] == 100 else "✗"
    print(f"{status} {stat['user']:20s} | Holdout: {stat['holdout_count']:3d} | Found: {stat['found_in_catalog']:3d} | Missing: {stat['missing']:3d} | Coverage: {stat['coverage_%']:5.1f}%")

# Summary
total_holdout = sum(s['holdout_count'] for s in coverage_stats)
total_found = sum(s['found_in_catalog'] for s in coverage_stats)
total_missing = sum(s['missing'] for s in coverage_stats)
overall_coverage = total_found / total_holdout if total_holdout > 0 else 0

print("\n" + "="*80)
print(f"OVERALL: {total_found}/{total_holdout} ASINs found ({overall_coverage*100:.1f}% coverage)")
print(f"Missing ASINs: {total_missing}")
print("="*80)

# Find first few missing ASINs
all_missing = []
for stat in coverage_stats:
    holdout_file = loocv_data_dir / f'user_{stat["user"]}' / 'holdout.json'
    with open(holdout_file) as f:
        holdout_data = json.load(f)
    for pair in holdout_data.get('pairs', []):
        asin = pair['asin']
        if asin not in catalog_asins:
            all_missing.append((stat['user'], asin))

if all_missing:
    print(f"\nFirst 10 missing ASINs:")
    for user, asin in all_missing[:10]:
        print(f"  {user:20s} -> {asin}")
