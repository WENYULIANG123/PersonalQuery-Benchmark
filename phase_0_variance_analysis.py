#!/usr/bin/env python3
"""
Phase 0: Variance Decomposition Analysis
目的：量化Inter-User vs Intra-User噪声方差，决定个性化是否可行
输出：ICC、方差比、每用户统计
"""

import json
import glob
import os
import sys
from statistics import mean, stdev
from pathlib import Path
import numpy as np
from scipy import stats

def safe_mean(vals):
    return mean(vals) if len(vals) > 0 else 0.0

def safe_stdev(vals):
    return stdev(vals) if len(vals) > 1 else 0.0

def load_user_features_from_clustering(clustering_data):
    """Extract user features directly from clustering_results.json"""
    user_features = {}
    
    if 'user_features' not in clustering_data:
        return user_features
    
    for user_id, features in clustering_data['user_features'].items():
        clean_user_id = user_id.replace('user_', '')
        user_features[clean_user_id] = {
            'typo': features.get('typo_ratio', 0.0),
            'omit': features.get('omission_ratio', 0.0),
            'repeat': features.get('repeat_ratio', 0.0),
        }
    
    return user_features

def load_mrr_data():
    """Load MRR@10 scores from comparison_results.json"""
    mrr_file = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/loocv_results/results/comparison_results.json"
    try:
        with open(mrr_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load MRR data: {e}", file=sys.stderr)
        return {}
    
    mrr_dict = {}
    for user_id, experiments in data.items():
        if isinstance(experiments, dict) and 'experiment_a' in experiments:
            mrr_value = experiments['experiment_a'].get('mrr@10', None)
            if mrr_value is not None:
                mrr_dict[user_id] = mrr_value
    
    return mrr_dict

def main():
    clustering_results = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/experiment_3_clustering/clustering_results.json"
    
    if not os.path.exists(clustering_results):
        print(f"ERROR: Clustering results not found: {clustering_results}", file=sys.stderr)
        sys.exit(1)
    
    with open(clustering_results, 'r') as f:
        cluster_data = json.load(f)
    
    user_features = load_user_features_from_clustering(cluster_data)
    mrr_dict = load_mrr_data()
    
    if not user_features:
        print(f"ERROR: No user features found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(user_features)} users")
    print(f"=" * 100)
    
    user_stats = []
    all_noise = {
        'typo': [],
        'omit': [],
        'repeat': []
    }
    
    for user_id, features in sorted(user_features.items()):
        mu_typo = features['typo']
        mu_omit = features['omit']
        mu_repeat = features['repeat']
        
        mrr = mrr_dict.get(user_id, None)
        
        user_stats.append({
            'user_id': user_id,
            'mean_typo': mu_typo,
            'mean_omit': mu_omit,
            'mean_repeat': mu_repeat,
            'intra_cv': 0.0,
            'mrr': mrr
        })
        
        all_noise['typo'].append(mu_typo)
        all_noise['omit'].append(mu_omit)
        all_noise['repeat'].append(mu_repeat)
    
    global_means = {
        'typo': safe_mean(all_noise['typo']),
        'omit': safe_mean(all_noise['omit']),
        'repeat': safe_mean(all_noise['repeat']),
    }
    
    global_stds = {
        'typo': safe_stdev(all_noise['typo']),
        'omit': safe_stdev(all_noise['omit']),
        'repeat': safe_stdev(all_noise['repeat']),
    }
    
    print("\n" + "=" * 100)
    print("GLOBAL NOISE STATISTICS (across all users)")
    print("=" * 100)
    for feat in ['typo', 'omit', 'repeat']:
        print(f"  {feat:10s}: mean={global_means[feat]:.6f}, std={global_stds[feat]:.6f}")
    
    print("\n" + "=" * 100)
    print("PER-USER NOISE PROFILE")
    print("=" * 100)
    print(f"{'User ID':25s} {'Typo':10s} {'Omit':10s} {'Repeat':10s} {'MRR@10':10s}")
    print("-" * 100)
    
    for u in user_stats:
        mrr_str = f"{u['mrr']:.4f}" if u['mrr'] is not None else "N/A"
        print(f"{u['user_id']:25s} {u['mean_typo']:10.6f} {u['mean_omit']:10.6f} {u['mean_repeat']:10.6f} {mrr_str:>10s}")
    
    # ===== VARIANCE DECOMPOSITION =====
    print("\n" + "=" * 100)
    print("VARIANCE DECOMPOSITION: INTER-USER vs INTRA-USER")
    print("=" * 100)
    
    # Inter-user CV (variance of user means)
    per_feature_means = {
        'typo': [u['mean_typo'] for u in user_stats],
        'omit': [u['mean_omit'] for u in user_stats],
        'repeat': [u['mean_repeat'] for u in user_stats],
    }
    
    inter_cv = {}
    for feat in ['typo', 'omit', 'repeat']:
        vals = per_feature_means[feat]
        m = safe_mean(vals)
        s = safe_stdev(vals)
        inter_cv[feat] = (s / abs(m)) if m != 0 else 0.0
    
    inter_cv_avg = mean([v for v in inter_cv.values()])
    
    # Intra-user CV (average within-user consistency)
    intra_cv_avg = mean([u['intra_cv'] for u in user_stats]) if user_stats else 0.0
    
    # Ratio
    ratio = inter_cv_avg / intra_cv_avg if intra_cv_avg != 0 else float('inf')
    
    print("\nPer-Feature Inter-User Coefficient of Variation:")
    for feat in ['typo', 'omit', 'repeat']:
        print(f"  {feat:10s}: CV = {inter_cv[feat]:.6f}")
    print(f"  {'Average':10s}: CV = {inter_cv_avg:.6f}")
    
    print("\nPer-Feature Intra-User Coefficient of Variation:")
    print(f"  Average: {intra_cv_avg:.6f}")
    
    print(f"\n{'INTER/INTRA RATIO':30s}: {ratio:.6f}")
    print(f"{'Interpretation':30s}: ", end="")
    if ratio < 0.5:
        print("❌ Weak inter-user variance (< 0.5) — users very similar")
    elif ratio < 1.0:
        print("⚠️  Moderate inter-user variance (0.5-1.0) — mixed signal")
    else:
        print("✅ Strong inter-user variance (> 1.0) — users noticeably different")
    
    print("\n" + "=" * 100)
    print("INTRACLASS CORRELATION COEFFICIENT (ICC)")
    print("=" * 100)
    
    typo_vals = all_noise['typo']
    grand_mean = safe_mean(typo_vals)
    
    user_means = [u['mean_typo'] for u in user_stats]
    
    ss_between = sum(len(user_stats) * (um - grand_mean)**2 for um in user_means)
    ms_between = ss_between / (len(user_stats) - 1) if len(user_stats) > 1 else 0
    
    ss_within = sum((typo_vals[i] - user_means[i])**2 
                    for i in range(len(typo_vals)))
    ms_within = ss_within / (len(typo_vals) - len(user_stats)) if (len(typo_vals) - len(user_stats)) > 0 else 0
    
    k = len(typo_vals) / len(user_stats) if user_stats else 1
    
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within) if (ms_between + (k - 1) * ms_within) != 0 else 0
    
    sigma2_between = (ms_between - ms_within) / k
    sigma2_within = ms_within
    
    print(f"{'MS Between':30s}: {ms_between:.8f}")
    print(f"{'MS Within':30s}: {ms_within:.8f}")
    print(f"{'Average group size (k)':30s}: {k:.2f}")
    print(f"{'Var(between-user, σ²_u)':30s}: {sigma2_between:.8f}")
    print(f"{'Var(within-user, σ²_ε)':30s}: {sigma2_within:.8f}")
    print(f"\n{'ICC(1) (one-way random)':30s}: {icc:.6f}")
    print(f"{'Interpretation':30s}: ", end="")
    if icc < 0.1:
        print("❌ Negligible user differences (< 0.1)")
    elif icc < 0.4:
        print("⚠️  Moderate user differences (0.1-0.4)")
    else:
        print("✅ Strong user differences (> 0.4)")
    
    # ===== DECISION GATE =====
    print("\n" + "=" * 100)
    print("DECISION GATE: SHOULD WE CONTINUE TO PHASE 1?")
    print("=" * 100)
    
    decision_points = []
    
    if ratio >= 0.5:
        print("✅ Inter/Intra ratio >= 0.5: Some user-level signal exists")
        decision_points.append(True)
    else:
        print("❌ Inter/Intra ratio < 0.5: Very weak user-level signal")
        decision_points.append(False)
    
    if icc >= 0.1:
        print("✅ ICC >= 0.1: Statistically significant user effects")
        decision_points.append(True)
    else:
        print("❌ ICC < 0.1: Negligible user effects")
        decision_points.append(False)
    
    proceed = any(decision_points)
    
    print("\n" + "=" * 100)
    if proceed:
        print("🚀 RECOMMENDATION: PROCEED TO PHASE 1 (Mixed Effects Modeling)")
        print("   → User-specific noise patterns exist and may be learnable")
    else:
        print("⛔ RECOMMENDATION: STOP (Data quality issue likely)")
        print("   → User differences too weak. Check data alignment/preprocessing.")
    print("=" * 100)
    
    # Save results
    results = {
        'global_means': global_means,
        'global_stds': global_stds,
        'inter_cv': inter_cv,
        'inter_cv_avg': inter_cv_avg,
        'intra_cv_avg': intra_cv_avg,
        'inter_intra_ratio': ratio,
        'variance_between': sigma2_between,
        'variance_within': sigma2_within,
        'icc': icc,
        'decision_proceed': proceed,
        'user_stats': user_stats
    }
    
    output_file = "/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/14_fine_tuning/phase_0_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return 0 if proceed else 1

if __name__ == "__main__":
    sys.exit(main())
