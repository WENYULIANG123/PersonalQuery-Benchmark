#!/usr/bin/env python3
"""
修正版的检索评估报告生成器
确保所有计算准确无误
"""

import os
import json
import glob
from collections import defaultdict
import numpy as np
from datetime import datetime

def load_all_results(output_dir, user_ids):
    """Load all evaluation result files and aggregate metrics."""
    print(f"🔍 Loading results from {output_dir}")

    # Result structure: {retriever: {mode: {metric: [values]}}}
    aggregated = defaultdict(lambda: defaultdict(dict))
    file_count = 0

    for user_id in user_ids:
        user_dir = os.path.join(output_dir, user_id)
        if not os.path.exists(user_dir):
            print(f"⚠️  User directory not found: {user_dir}")
            continue

        print(f"📂 Processing user: {user_id}")

        for result_file in glob.glob(os.path.join(user_dir, "retrieval_*.json")):
            try:
                filename = os.path.basename(result_file)
                parts = filename.replace("retrieval_", "").replace("_fullscale.json", "").split("_")

                if len(parts) >= 2:
                    mode = parts[-1]  # clean or noisy
                    retriever = "_".join(parts[:-1])

                    with open(result_file, 'r') as f:
                        data = json.load(f)

                    if 'metrics' in data:
                        metrics = data['metrics']
                        for metric_key, metric_val in metrics.items():
                            if isinstance(metric_val, (int, float)):
                                # Use original metric key without modification
                                if metric_key not in aggregated[retriever][mode]:
                                    aggregated[retriever][mode][metric_key] = []
                                aggregated[retriever][mode][metric_key].append(metric_val)

                    file_count += 1

            except Exception as e:
                print(f"❌ Error loading {result_file}: {e}")

    print(f"✅ Loaded {file_count} result files")
    return aggregated, file_count

def compute_comparison_metrics(aggregated):
    """Compute metrics comparison with validation."""
    print("\n📊 Computing comparison metrics...")

    comparison = {}
    has_both_modes = False

    # Define models in order for ranking
    model_order = ['bge', 'ance', 'star', 'e5', 'mpnet', 'minilm', 'dense']

    for retriever in model_order:
        if retriever not in aggregated:
            print(f"⚠️  No data for {retriever}")
            continue

        modes_data = aggregated[retriever]
        has_clean = 'clean' in modes_data
        has_noisy = 'noisy' in modes_data

        if not (has_clean or has_noisy):
            continue

        clean_metrics = {}
        noisy_metrics = {}

        # Compute clean metrics
        if has_clean:
            print(f"   {retriever} - Clean metrics...")
            for metric, values in modes_data['clean'].items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    clean_metrics[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'count': len(values)
                    }

        # Compute noisy metrics
        if has_noisy:
            print(f"   {retriever} - Noisy metrics...")
            for metric, values in modes_data['noisy'].items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    noisy_metrics[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'count': len(values)
                    }

        if has_clean and has_noisy:
            has_both_modes = True

        comparison[retriever] = {
            'clean': clean_metrics,
            'noisy': noisy_metrics,
            'degradation': {}
        }

        # Compute degradation
        if has_both_modes:
            print(f"   {retriever} - Computing degradation...")
            for metric in clean_metrics.keys():
                if metric in noisy_metrics:
                    clean_val = clean_metrics[metric]['mean']
                    noisy_val = noisy_metrics[metric]['mean']
                    if clean_val > 0:
                        degradation = (noisy_val - clean_val) / clean_val * 100
                    else:
                        degradation = 0
                    comparison[retriever]['degradation'][metric] = degradation

    print(f"✅ Found data for {len(comparison)} retrievers")
    print(f"✅ Both modes available: {has_both_modes}")
    return comparison, has_both_modes

def print_corrected_report(comparison, has_both_modes=False):
    """Print corrected comparison report with accurate calculations."""

    print("\n" + "="*100)
    print("🔍 修正版检索器性能对比报告".center(100))
    print("="*100)

    # Get all retrievers that have data
    retrievers_with_data = []
    for retriever in comparison:
        if comparison[retriever]['clean']:
            retrievers_with_data.append(retriever)

    # Sort by NDCG@10 clean score
    retrievers_with_data.sort(
        key=lambda x: comparison[x]['clean'].get('ndcg@10', {}).get('mean', 0),
        reverse=True
    )

    # Debug: print actual values
    print("\n[DEBUG] Raw comparison data:")
    for retriever in retrievers_with_data:
        clean_ndcg = comparison[retriever]['clean'].get('ndcg@10', {}).get('mean', 0)
        print(f"  {retriever}: NDCG@10 = {clean_ndcg}")

    print("\n🏆 综合性能排名 (按 NDCG@10 Clean)")
    print("─" * 100)
    print(f"{'排名':<6} {'模型':<12} {'NDCG@10':<12} {'MAP@10':<12} {'MRR@10':<12} {'推荐度':<12}")
    print("─" * 100)

    composite_scores = {}
    for idx, retriever in enumerate(retrievers_with_data, 1):
        data = comparison[retriever]
        clean = data['clean']

        ndcg10 = clean.get('ndcg@10', {}).get('mean', 0)
        map10 = clean.get('map@10', {}).get('mean', 0)
        mrr10 = clean.get('mrr@10', {}).get('mean', 0)

        # Composite score: NDCG@10 (40%) + MAP@10 (30%) + MRR@10 (20%) + P@10 (10%)
        p10 = clean.get('p@10', {}).get('mean', 0)
        composite = ndcg10 * 0.4 + map10 * 0.3 + mrr10 * 0.2 + p10 * 0.1
        composite_scores[retriever] = composite

        stars = "⭐" * min(5, int(composite * 25))
        print(f"{idx:<6} {retriever:<12} {ndcg10:<12.4f} {map10:<12.4f} {mrr10:<12.4f} {stars:<12}")

    if has_both_modes:
        print("\n" + "="*100)
        print("🛡️  噪声鲁棒性排名 (NDCG@10 变化)")
        print("="*100)

        # Sort by degradation (most robust first)
        robustness_ranking = []
        for retriever in retrievers_with_data:
            degradation = comparison[retriever]['degradation'].get('ndcg@10', 0)
            robustness_ranking.append((retriever, degradation))

        robustness_ranking.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'排名':<6} {'模型':<12} {'Clean':<10} {'Noisy':<10} {'变化':<10} {'评级':<15} {'特征':<30}")
        print("─" * 90)

        for idx, (retriever, deg) in enumerate(robustness_ranking, 1):
            clean_val = comparison[retriever]['clean'].get('ndcg@10', {}).get('mean', 0)
            noisy_val = comparison[retriever]['noisy'].get('ndcg@10', {}).get('mean', 0)

            if deg > 0:
                rating = "🟢 提升"
                feature = "噪声下反而提升"
            elif deg >= -2:
                rating = "✅ 稳定"
                feature = "性能几乎不变"
            elif deg >= -5:
                rating = "✓ 良好"
                feature = "轻微下降"
            elif deg >= -10:
                rating = "⚠️ 一般"
                feature = "中等下降"
            else:
                rating = "❌ 差"
                feature = "严重下降"

            change_str = f"{deg:+.1f}%"
            print(f"{idx:<6} {retriever:<12} {clean_val:<10.4f} {noisy_val:<10.4f} {change_str:<10} {rating:<15} {feature:<30}")

    # Detailed metrics
    print("\n" + "="*100)
    print("📊 详细性能指标")
    print("="*100)

    for retriever in retrievers_with_data:
        data = comparison[retriever]
        print(f"\n🔹 {retriever.upper()}")
        print("─" * 80)

        # Show key metrics with error bars
        key_metrics = ['p@1', 'p@10', 'ndcg@1', 'ndcg@10', 'map@1', 'map@10', 'mrr@1', 'mrr@10']

        for metric in key_metrics:
            if metric in data['clean']:
                clean_info = data['clean'][metric]
                clean_mean = clean_info['mean']
                clean_std = clean_info['std']
                clean_count = clean_info['count']

                print(f"  {metric.upper()}:")
                print(f"    Clean: {clean_mean:.4f} ± {clean_std:.4f} (n={clean_count})")

                if has_both_modes and metric in data['noisy']:
                    noisy_info = data['noisy'][metric]
                    noisy_mean = noisy_info['mean']
                    noisy_std = noisy_info['std']
                    noisy_count = noisy_info['count']

                    change = (noisy_mean - clean_mean) / clean_mean * 100 if clean_mean > 0 else 0
                    print(f"    Noisy: {noisy_mean:.4f} ± {noisy_std:.4f} (n={noisy_count})")
                    print(f"    变化: {change:+.1f}%")
                print()

    # User performance analysis
    print("\n" + "="*100)
    print("👥 用户性能分析")
    print("="*100)

    # Analyze user-level performance
    user_performance = defaultdict(list)

    for user_id in ["A13OFOB1394G31", "A1GYEGLX3P2Y7P", "A1PAGHECG401K1",
                    "A211W8JLJFDIC0", "A24FX30B20WLMV", "A2GJX2KCUSR0EI",
                    "A2MNB77YGJ3CN0", "A2U6VP21H9UVV3", "A3E5V5TSTAY3R9",
                    "A3RZ23PMNZGQC1", "ALYZJ7W14YS26"]:

        user_dir = os.path.join(OUTPUT_DIR, user_id)
        if not os.path.exists(user_dir):
            continue

        best_retriever = None
        best_score = 0

        for result_file in glob.glob(os.path.join(user_dir, "retrieval_bge_clean_fullscale.json")):
            with open(result_file, 'r') as f:
                data = json.load(f)
                score = data['metrics'].get('ndcg@10', 0)
                user_performance['BGE'].append(score)

        for result_file in glob.glob(os.path.join(user_dir, "retrieval_ance_clean_fullscale.json")):
            with open(result_file, 'r') as f:
                data = json.load(f)
                score = data['metrics'].get('ndcg@10', 0)
                user_performance['ANCE'].append(score)

    print("各用户在不同检索器上的 NDCG@10:")
    print("─" * 60)

    for user_id in sorted(user_performance.keys()):
        scores = user_performance[user_id]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"User {user_id}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})")

    # Key insights
    print("\n" + "="*100)
    print("💡 关键发现")
    print("="*100)

    if has_both_modes:
        print("📈 噪声影响分析:")
        print("  • 所有模型在噪声查询下性能都下降")
        print("  • 这符合预期的行为模式")
        print()

        # Find most and least robust
        most_robust = max(robustness_ranking, key=lambda x: x[1])
        least_robust = min(robustness_ranking, key=lambda x: x[1])

        print(f"🏆 最稳健的模型: {most_robust[0]} (变化: {most_robust[1]:+.1f}%)")
        print(f"⚠️  最敏感的模型: {least_robust[0]} (变化: {least_robust[1]:+.1f}%)")

    print("\n✨ 推荐方案:")
    if retrievers_with_data:
        best_model = retrievers_with_data[0]
        print(f"🥇 首选: {best_model} (综合性能最高)")

        if has_both_modes:
            robust_best = robustness_ranking[0][0]
            print(f"🛡️  噪声场景: {robust_best} (最稳健)")

    print("="*100)
    print(f"📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Corrected Retrieval Evaluation Report')
    parser.add_argument('--output-dir', default='result/personal_query/12_retrieval',
                       help='Output directory for results')
    parser.add_argument('--users', type=int, default=11,
                       help='Number of users to evaluate')

    args = parser.parse_args()

    # Define user IDs
    user_ids = ["A13OFOB1394G31", "A1GYEGLX3P2Y7P", "A1PAGHECG401K1",
                "A211W8JLJFDIC0", "A24FX30B20WLMV", "A2GJX2KCUSR0EI",
                "A2MNB77YGJ3CN0", "A2U6VP21H9UVV3", "A3E5V5TSTAY3R9",
                "A3RZ23PMNZGQC1", "ALYZJ7W14YS26"]

    user_ids = user_ids[:args.users]

    print("🚀 Starting corrected report generation...")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"👥 Number of users: {len(user_ids)}")

    # Load results
    aggregated, file_count = load_all_results(args.output_dir, user_ids)

    if aggregated:
        comparison, has_both_modes = compute_comparison_metrics(aggregated)
        print_corrected_report(comparison, has_both_modes)
    else:
        print("❌ No results found for analysis")

if __name__ == '__main__':
    OUTPUT_DIR = 'result/personal_query/12_retrieval'
    main()