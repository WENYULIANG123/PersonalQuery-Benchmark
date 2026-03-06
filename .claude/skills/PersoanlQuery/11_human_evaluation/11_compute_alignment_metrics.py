#!/usr/bin/env python3
"""
Stage 11: Human Evaluation - Compute Alignment Metrics

This script calculates alignment metrics between LLM and human evaluations:
1. Spearman's Rank Correlation (ρ)
2. Cohen's Kappa (κ)
3. Recall @ Human Preference
4. Mean Absolute Error (MAE)
5. Systematic Bias Analysis
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")


def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_human_evaluation(human_results_path):
    """Load human evaluation results"""
    log_with_timestamp(f"Loading human evaluation results from: {human_results_path}")
    with open(human_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log_with_timestamp(f"Loaded {len(data)} evaluation tasks")
    return data


def load_llm_evaluation(llm_results_path, llm_dir):
    """Load LLM evaluation results"""
    log_with_timestamp(f"Loading LLM evaluation results from: {llm_results_path}")

    # Load summary
    with open(llm_results_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # Load detailed results per user
    detailed_results = {}
    for filename in os.listdir(llm_dir):
        if filename.startswith('evaluation_') and filename.endswith('.json') and filename != 'evaluation_summary.json':
            filepath = os.path.join(llm_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                user_id = data.get('user_id')
                if user_id:
                    # Create a mapping from ASIN to result
                    for result in data.get('results', []):
                        key = f"{user_id}_{result.get('asin')}"
                        detailed_results[key] = result

    log_with_timestamp(f"Loaded LLM summary and detailed results for {len(detailed_results)} query pairs")
    return summary, detailed_results


def compute_spearman_correlation(human_results, llm_summary):
    """Compute Spearman's rank correlation between LLM and human average scores"""
    log_with_timestamp("Computing Spearman's Rank Correlation...")

    # Build per-user average scores from human results
    human_user_scores = defaultdict(list)
    for task in human_results:
        user_id = task.get('user_id')
        human_eval = task.get('human_evaluation', {})
        if human_eval.get('personalized_query_score'):
            human_user_scores[user_id].append(human_eval['personalized_query_score'])

    # Calculate average per user
    human_avg_scores = {user_id: np.mean(scores) for user_id, scores in human_user_scores.items()}

    # Get LLM average scores from summary
    llm_user_scores = {}
    for user_data in llm_summary.get('by_user', []):
        user_id = user_data.get('user_id')
        llm_user_scores[user_id] = user_data.get('avg_personalized', 0)

    # Align users
    common_users = list(set(human_avg_scores.keys()) & set(llm_user_scores.keys()))
    if len(common_users) < 2:
        log_with_timestamp("Warning: Not enough common users for Spearman correlation")
        return None

    human_scores = [human_avg_scores[user] for user in common_users]
    llm_scores = [llm_user_scores[user] for user in common_users]

    # Compute Spearman correlation
    rho, p_value = spearmanr(llm_scores, human_scores)

    # Interpretation
    if abs(rho) > 0.8:
        interpretation = "强相关"
    elif abs(rho) > 0.6:
        interpretation = "中等相关"
    elif abs(rho) > 0.4:
        interpretation = "弱相关"
    else:
        interpretation = "极弱相关"

    result = {
        "correlation_coefficient": round(float(rho), 4),
        "p_value": round(float(p_value), 4),
        "interpretation": f"{interpretation} ({'正' if rho > 0 else '负'}相关 ρ = {rho:.3f})",
        "n_users": len(common_users),
        "human_scores": {user: round(human_avg_scores[user], 2) for user in common_users},
        "llm_scores": {user: round(llm_user_scores[user], 2) for user in common_users}
    }

    log_with_timestamp(f"  Spearman ρ = {rho:.4f} (p = {p_value:.4f}) - {interpretation}")
    return result


def compute_cohens_kappa(human_results):
    """Compute Cohen's Kappa for LLM-Human agreement on preferences"""
    log_with_timestamp("Computing Cohen's Kappa...")

    llm_preferences = []
    human_preferences = []

    # Build confusion matrix
    confusion = {
        'llm_personalized_human_personalized': 0,
        'llm_personalized_human_public': 0,
        'llm_personalized_human_tie': 0,
        'llm_public_human_personalized': 0,
        'llm_public_human_public': 0,
        'llm_public_human_tie': 0,
        'llm_tie_human_personalized': 0,
        'llm_tie_human_public': 0,
        'llm_tie_human_tie': 0
    }

    for task in human_results:
        human_eval = task.get('human_evaluation', {})
        llm_scores = task.get('llm_scores', {})

        human_pref = human_eval.get('preferred_query')
        llm_pref = llm_scores.get('llm_prefers')

        if not human_pref or not llm_pref:
            continue

        human_preferences.append(human_pref)
        llm_preferences.append(llm_pref)

        # Update confusion matrix
        key = f'llm_{llm_pref}_human_{human_pref}'
        if key in confusion:
            confusion[key] += 1

    if len(llm_preferences) == 0:
        log_with_timestamp("Warning: No valid preference data for Cohen's Kappa")
        return None

    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(llm_preferences, human_preferences)

    # Interpretation
    if kappa > 0.8:
        interpretation = "极高的一致性"
    elif kappa > 0.6:
        interpretation = "显著一致"
    elif kappa > 0.4:
        interpretation = "中等一致性"
    elif kappa > 0.2:
        interpretation = "一般一致性"
    else:
        interpretation = "一致性不足"

    # Calculate agreement rate
    total = sum(confusion.values())
    agreement = (confusion['llm_personalized_human_personalized'] +
                 confusion['llm_public_human_public'] +
                 confusion['llm_tie_human_tie'])
    agreement_rate = agreement / total if total > 0 else 0

    result = {
        "kappa_score": round(float(kappa), 4),
        "interpretation": f"{interpretation} (κ = {kappa:.3f})",
        "agreement_rate": round(agreement_rate, 4),
        "confusion_matrix": confusion,
        "total_comparisons": total
    }

    log_with_timestamp(f"  Cohen's κ = {kappa:.4f} - {interpretation}")
    log_with_timestamp(f"  Agreement rate: {agreement_rate:.1%}")
    return result


def compute_recall_at_human_preference(human_results):
    """Compute Recall @ Human Preference"""
    log_with_timestamp("Computing Recall @ Human Preference...")

    human_personalized_count = 0
    llm_agree_count = 0

    for task in human_results:
        human_eval = task.get('human_evaluation', {})
        llm_scores = task.get('llm_scores', {})

        human_pref = human_eval.get('preferred_query')
        llm_pref = llm_scores.get('llm_prefers')

        if not human_pref:
            continue

        if human_pref == 'personalized':
            human_personalized_count += 1
            if llm_pref == 'personalized':
                llm_agree_count += 1

    if human_personalized_count == 0:
        log_with_timestamp("Warning: No human preferences for personalized queries")
        return None

    recall = llm_agree_count / human_personalized_count

    result = {
        "recall_score": round(recall, 4),
        "human_personalized_count": human_personalized_count,
        "llm_agreement_count": llm_agree_count,
        "interpretation": f"当真人认为个性化查询更好时，LLM 有 {recall:.1%} 的概率也选"
    }

    log_with_timestamp(f"  Recall @ Human Preference = {recall:.4f}")
    return result


def compute_mae(human_results):
    """Compute Mean Absolute Error between LLM and human scores"""
    log_with_timestamp("Computing Mean Absolute Error (MAE)...")

    public_errors = []
    personalized_errors = []
    all_errors = []

    for task in human_results:
        human_eval = task.get('human_evaluation', {})
        llm_scores = task.get('llm_scores', {})

        # Public query MAE
        human_public = human_eval.get('public_query_score')
        llm_public = llm_scores.get('public_score')
        if human_public and llm_public:
            error = abs(human_public - llm_public)
            public_errors.append(error)
            all_errors.append(error)

        # Personalized query MAE
        human_personalized = human_eval.get('personalized_query_score')
        llm_personalized = llm_scores.get('personalized_score')
        if human_personalized and llm_personalized:
            error = abs(human_personalized - llm_personalized)
            personalized_errors.append(error)
            all_errors.append(error)

    if not all_errors:
        log_with_timestamp("Warning: No valid score data for MAE calculation")
        return None

    mae_public = np.mean(public_errors) if public_errors else 0
    mae_personalized = np.mean(personalized_errors) if personalized_errors else 0
    mae_overall = np.mean(all_errors)

    result = {
        "mae": round(float(mae_overall), 4),
        "mae_public": round(float(mae_public), 4),
        "mae_personalized": round(float(mae_personalized), 4),
        "n_public_pairs": len(public_errors),
        "n_personalized_pairs": len(personalized_errors),
        "interpretation": f"LLM 与真人评分平均相差 {mae_overall:.2f} 分（满分 10 分）"
    }

    log_with_timestamp(f"  Overall MAE = {mae_overall:.4f}")
    log_with_timestamp(f"  Public query MAE = {mae_public:.4f}")
    log_with_timestamp(f"  Personalized query MAE = {mae_personalized:.4f}")
    return result


def compute_systematic_bias(human_results):
    """Compute systematic bias analysis"""
    log_with_timestamp("Computing Systematic Bias Analysis...")

    llm_personalized_wins = 0
    human_personalized_wins = 0
    total = 0

    for task in human_results:
        human_eval = task.get('human_evaluation', {})
        llm_scores = task.get('llm_scores', {})

        human_pref = human_eval.get('preferred_query')
        llm_pref = llm_scores.get('llm_prefers')

        if not human_pref or not llm_pref:
            continue

        total += 1

        if llm_pref == 'personalized':
            llm_personalized_wins += 1
        if human_pref == 'personalized':
            human_personalized_wins += 1

    if total == 0:
        log_with_timestamp("Warning: No valid preference data for bias analysis")
        return None

    llm_win_rate = llm_personalized_wins / total
    human_win_rate = human_personalized_wins / total
    bias = llm_win_rate - human_win_rate

    # Interpretation
    if bias > 0.2:
        interpretation = f"⚠️ LLM 存在显著正向偏见：LLM 觉得 {llm_win_rate:.1%} 的个性化查询好，但真人只觉得 {human_win_rate:.1%} 好"
    elif bias > 0.1:
        interpretation = f"ℹ️ LLM 轻微倾向于过度评价个性化查询"
    elif bias < -0.1:
        interpretation = f"ℹ️ LLM 轻微倾向于低估个性化查询"
    else:
        interpretation = f"✅ 未检测到显著偏见"

    result = {
        "llm_personalized_win_rate": round(llm_win_rate, 4),
        "human_personalized_win_rate": round(human_win_rate, 4),
        "bias_difference": round(float(bias), 4),
        "llm_personalized_wins": llm_personalized_wins,
        "human_personalized_wins": human_personalized_wins,
        "total_comparisons": total,
        "interpretation": interpretation
    }

    log_with_timestamp(f"  LLM personalized win rate: {llm_win_rate:.1%}")
    log_with_timestamp(f"  Human personalized win rate: {human_win_rate:.1%}")
    log_with_timestamp(f"  Bias: {bias:+.1%}")
    return result


def compute_per_user_analysis(human_results):
    """Compute per-user alignment statistics"""
    log_with_timestamp("Computing per-user analysis...")

    user_stats = defaultdict(lambda: {
        'llm_scores': [],
        'human_scores': [],
        'agreements': 0,
        'total': 0
    })

    for task in human_results:
        user_id = task.get('user_id')
        human_eval = task.get('human_evaluation', {})
        llm_scores = task.get('llm_scores', {})

        human_pref = human_eval.get('preferred_query')
        llm_pref = llm_scores.get('llm_prefers')

        if human_pref and llm_pref:
            user_stats[user_id]['total'] += 1
            if human_pref == llm_pref:
                user_stats[user_id]['agreements'] += 1

        # Collect scores
        human_score = human_eval.get('personalized_query_score')
        llm_score = llm_scores.get('personalized_score')
        if human_score and llm_score:
            user_stats[user_id]['human_scores'].append(human_score)
            user_stats[user_id]['llm_scores'].append(llm_score)

    # Compute per-user metrics
    per_user_results = []
    for user_id, stats in user_stats.items():
        avg_human = np.mean(stats['human_scores']) if stats['human_scores'] else 0
        avg_llm = np.mean(stats['llm_scores']) if stats['llm_scores'] else 0
        mae = np.mean([abs(h - l) for h, l in zip(stats['human_scores'], stats['llm_scores'])]) if stats['human_scores'] else 0
        agreement = stats['agreements'] / stats['total'] if stats['total'] > 0 else 0

        per_user_results.append({
            'user_id': user_id,
            'avg_human_score': round(float(avg_human), 2),
            'avg_llm_score': round(float(avg_llm), 2),
            'mae': round(float(mae), 2),
            'agreement_rate': round(float(agreement), 4),
            'n_evaluations': stats['total'],
            'n_agreements': stats['agreements']
        })

    # Sort by agreement rate
    per_user_results.sort(key=lambda x: x['agreement_rate'], reverse=True)

    log_with_timestamp(f"  Computed statistics for {len(per_user_results)} users")
    return per_user_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute alignment metrics between LLM and human evaluations"
    )
    parser.add_argument(
        "--human-results",
        required=True,
        help="Path to human evaluation results JSON file (downloaded from HTML interface)"
    )
    parser.add_argument(
        "--llm-results",
        default="/home/wlia0047/wenyu/result/user_profile/10_evaluation/evaluation_summary.json",
        help="Path to LLM evaluation summary JSON file"
    )
    parser.add_argument(
        "--llm-dir",
        default="/home/wlia0047/wenyu/result/user_profile/10_evaluation",
        help="Directory containing detailed LLM evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/wenyu/result/user_profile/11_human_evaluation/reports",
        help="Output directory for alignment metrics report"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    human_results = load_human_evaluation(args.human_results)
    llm_summary, llm_detailed = load_llm_evaluation(args.llm_results, args.llm_dir)

    # Compute all metrics
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("COMPUTING ALIGNMENT METRICS")
    log_with_timestamp("="*70)

    metrics = {
        'generated_at': datetime.now().isoformat(),
        'human_results_file': args.human_results,
        'llm_results_file': args.llm_results,
        'total_evaluations': len(human_results)
    }

    # 1. Spearman Correlation
    spearman_result = compute_spearman_correlation(human_results, llm_summary)
    if spearman_result:
        metrics['spearman_correlation'] = spearman_result

    # 2. Cohen's Kappa
    kappa_result = compute_cohens_kappa(human_results)
    if kappa_result:
        metrics['cohens_kappa'] = kappa_result

    # 3. Recall @ Human Preference
    recall_result = compute_recall_at_human_preference(human_results)
    if recall_result:
        metrics['recall_at_human_preference'] = recall_result

    # 4. MAE
    mae_result = compute_mae(human_results)
    if mae_result:
        metrics['mean_absolute_error'] = mae_result

    # 5. Systematic Bias
    bias_result = compute_systematic_bias(human_results)
    if bias_result:
        metrics['systematic_bias'] = bias_result

    # 6. Per-User Analysis
    per_user_result = compute_per_user_analysis(human_results)
    metrics['per_user_analysis'] = per_user_result

    # Save results
    output_path = os.path.join(args.output_dir, 'alignment_metrics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("METRICS COMPUTATION COMPLETE")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Results saved to: {output_path}")
    log_with_timestamp(f"\nNext step: Run 11_generate_report.py to generate the visualization report")
    log_with_timestamp("="*70)


if __name__ == "__main__":
    main()
