#!/usr/bin/env python3
"""
Stage 11: Human Evaluation - Generate Visualization Report

This script generates a comprehensive evaluation report with:
1. Markdown summary report
2. Visualization figures (Spearman correlation, confusion matrix, score distributions, etc.)
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")


# Set matplotlib to use a nice style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_metrics(metrics_dir):
    """Load alignment metrics"""
    log_with_timestamp("Loading alignment metrics...")
    metrics_path = os.path.join(metrics_dir, 'alignment_metrics.json')
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    log_with_timestamp("Metrics loaded successfully")
    return metrics


def plot_spearman_correlation(metrics, output_dir):
    """Generate Spearman correlation scatter plot"""
    log_with_timestamp("Generating Spearman correlation plot...")

    spearman_data = metrics.get('spearman_correlation')
    if not spearman_data:
        log_with_timestamp("  Skipping: No Spearman correlation data")
        return None

    human_scores = spearman_data.get('human_scores', {})
    llm_scores = spearman_data.get('llm_scores', {})

    # Get common users
    users = list(set(human_scores.keys()) & set(llm_scores.keys()))
    human_vals = [human_scores[u] for u in users]
    llm_vals = [llm_scores[u] for u in users]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(llm_vals, human_vals, s=100, alpha=0.6, edgecolors='black', linewidths=1.5, c='#667eea')

    # Add user labels
    for i, user in enumerate(users):
        ax.annotate(user, (llm_vals[i], human_vals[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    # Add diagonal line
    min_val = min(min(human_vals), min(llm_vals))
    max_val = max(max(human_vals), max(llm_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect Alignment')

    # Labels and title
    ax.set_xlabel('LLM Average Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Human Average Score', fontsize=12, fontweight='bold')
    ax.set_title(f"Spearman's Rank Correlation: ρ = {spearman_data['correlation_coefficient']:.3f} (p = {spearman_data['p_value']:.4f})\n{spearman_data['interpretation']}",
                 fontsize=13, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spearman_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_with_timestamp(f"  Saved: {output_path}")
    return 'spearman_correlation.png'


def plot_confusion_matrix(metrics, output_dir):
    """Generate confusion matrix heatmap"""
    log_with_timestamp("Generating confusion matrix plot...")

    kappa_data = metrics.get('cohens_kappa')
    if not kappa_data:
        log_with_timestamp("  Skipping: No Cohen's Kappa data")
        return None

    confusion = kappa_data.get('confusion_matrix', {})

    # Build matrix
    labels = ['Personalized', 'Public', 'Tie']
    matrix = np.zeros((3, 3))

    matrix[0, 0] = confusion.get('llm_personalized_human_personalized', 0)
    matrix[0, 1] = confusion.get('llm_personalized_human_public', 0)
    matrix[0, 2] = confusion.get('llm_personalized_human_tie', 0)
    matrix[1, 0] = confusion.get('llm_public_human_personalized', 0)
    matrix[1, 1] = confusion.get('llm_public_human_public', 0)
    matrix[1, 2] = confusion.get('llm_public_human_tie', 0)
    matrix[2, 0] = confusion.get('llm_tie_human_personalized', 0)
    matrix[2, 1] = confusion.get('llm_tie_human_public', 0)
    matrix[2, 2] = confusion.get('llm_tie_human_tie', 0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Human Preference', fontsize=12, fontweight='bold')
    ax.set_ylabel('LLM Preference', fontsize=12, fontweight='bold')
    ax.set_title(f"Cohen's Kappa: κ = {kappa_data['kappa_score']:.3f}\n{kappa_data['interpretation']}",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_with_timestamp(f"  Saved: {output_path}")
    return 'confusion_matrix.png'


def plot_score_distribution(metrics, output_dir):
    """Generate score distribution comparison"""
    log_with_timestamp("Generating score distribution plot...")

    mae_data = metrics.get('mean_absolute_error')
    if not mae_data:
        log_with_timestamp("  Skipping: No MAE data")
        return None

    # For this plot, we'd need individual scores, not just MAE
    # Since we don't have that in metrics, we'll create a summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Public Query', 'Personalized Query']
    mae_values = [mae_data.get('mae_public', 0), mae_data.get('mae_personalized', 0)]
    colors = ['#ff6b6b', '#4ecdc4']

    bars = ax.bar(categories, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Labels and title
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title(f"Score Error Analysis\n{mae_data.get('interpretation', '')}",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(mae_values) * 1.2)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_with_timestamp(f"  Saved: {output_path}")
    return 'score_distribution.png'


def plot_per_user_agreement(metrics, output_dir):
    """Generate per-user agreement bar chart"""
    log_with_timestamp("Generating per-user agreement plot...")

    per_user = metrics.get('per_user_analysis', [])
    if not per_user:
        log_with_timestamp("  Skipping: No per-user analysis data")
        return None

    # Sort by user_id for better readability
    per_user_sorted = sorted(per_user, key=lambda x: x['user_id'])

    user_ids = [u['user_id'][:12] for u in per_user_sorted]  # Truncate for display
    agreement_rates = [u['agreement_rate'] * 100 for u in per_user_sorted]
    maes = [u['mae'] for u in per_user_sorted]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Agreement rate plot
    colors1 = ['#28a745' if rate >= 80 else '#ffc107' if rate >= 60 else '#dc3545' for rate in agreement_rates]
    bars1 = ax1.bar(user_ids, agreement_rates, color=colors1, alpha=0.7, edgecolor='black', linewidth=1)

    ax1.set_ylabel('Agreement Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('LLM-Human Agreement Rate per User', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

    # MAE plot
    bars2 = ax2.bar(user_ids, maes, color='#667eea', alpha=0.7, edgecolor='black', linewidth=1)

    ax2.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('Score Error per User', fontsize=12, fontweight='bold')
    ax2.set_xlabel('User ID', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Rotate x labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_user_agreement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_with_timestamp(f"  Saved: {output_path}")
    return 'per_user_agreement.png'


def plot_systematic_bias(metrics, output_dir):
    """Generate systematic bias visualization"""
    log_with_timestamp("Generating systematic bias plot...")

    bias_data = metrics.get('systematic_bias')
    if not bias_data:
        log_with_timestamp("  Skipping: No systematic bias data")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['LLM', 'Human']
    win_rates = [bias_data.get('llm_personalized_win_rate', 0) * 100,
                 bias_data.get('human_personalized_win_rate', 0) * 100]
    colors = ['#667eea', '#28a745']

    bars = ax.bar(categories, win_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2, width=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add bias annotation
    bias_diff = bias_data.get('bias_difference', 0)
    ax.annotate(f'Bias: {bias_diff:+.1%}', xy=(0.5, max(win_rates) * 0.9),
               xycoords='data', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               horizontalalignment='center')

    # Labels and title
    ax.set_ylabel('Personalized Query Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f"Systematic Bias Analysis\n{bias_data.get('interpretation', '')}",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'systematic_bias.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_with_timestamp(f"  Saved: {output_path}")
    return 'systematic_bias.png'


def generate_markdown_report(metrics, figures, output_dir):
    """Generate comprehensive Markdown report"""
    log_with_timestamp("Generating Markdown report...")

    md_content = f"""# LLM-Human Alignment Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Evaluated:** {metrics.get('total_evaluations', 0)} query pairs

---

## Executive Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
"""

    # Add Spearman correlation
    if 'spearman_correlation' in metrics:
        sc = metrics['spearman_correlation']
        md_content += f"| **Spearman Correlation** | ρ = {sc['correlation_coefficient']:.3f} (p = {sc['p_value']:.4f}) | {sc['interpretation']} |\n"

    # Add Cohen's Kappa
    if 'cohens_kappa' in metrics:
        kappa = metrics['cohens_kappa']
        md_content += f"| **Cohen's Kappa** | κ = {kappa['kappa_score']:.3f} | {kappa['interpretation']} |\n"

    # Add Recall
    if 'recall_at_human_preference' in metrics:
        recall = metrics['recall_at_human_preference']
        md_content += f"| **Recall @ Human Preference** | {recall['recall_score']:.1%} | {recall['interpretation']} |\n"

    # Add MAE
    if 'mean_absolute_error' in metrics:
        mae = metrics['mean_absolute_error']
        md_content += f"| **MAE** | {mae['mae']:.2f}/10 | {mae['interpretation']} |\n"

    # Add bias
    if 'systematic_bias' in metrics:
        bias = metrics['systematic_bias']
        md_content += f"| **Systematic Bias** | {bias['bias_difference']:+.1%} | {bias['interpretation']} |\n"

    # Overall conclusion
    bias_val = metrics.get('systematic_bias', {}).get('bias_difference', 0)
    kappa_val = metrics.get('cohens_kappa', {}).get('kappa_score', 0)

    if kappa_val > 0.6 and abs(bias_val) < 0.2:
        conclusion = "✅ **Conclusion:** LLM 评估与真人评估具有高度一致性，可用于自动化评估。"
    elif kappa_val > 0.4:
        conclusion = "⚠️ **Conclusion:** LLM 评估与真人评估具有中等一致性，建议谨慎使用。"
    else:
        conclusion = "❌ **Conclusion:** LLM 评估与真人评估一致性不足，需要改进评估方法。"

    md_content += f"""

{conclusion}

---

"""

    # Add Spearman correlation section
    if 'spearman_correlation' in metrics and 'spearman_correlation.png' in figures:
        sc = metrics['spearman_correlation']
        md_content += f"""## 1. Spearman's Rank Correlation

![Spearman Correlation](figures/spearman_correlation.png)

**Interpretation:** {sc['interpretation']}

- **Correlation Coefficient (ρ):** {sc['correlation_coefficient']:.4f}
- **P-value:** {sc['p_value']:.4f}
- **Number of Users:** {sc['n_users']}

When ρ > 0.8, it indicates strong positive correlation, meaning LLM and human rank users similarly in terms of personalization quality.

---

"""

    # Add Cohen's Kappa section
    if 'cohens_kappa' in metrics and 'confusion_matrix.png' in figures:
        kappa = metrics['cohens_kappa']
        md_content += f"""## 2. Cohen's Kappa Analysis

![Confusion Matrix](figures/confusion_matrix.png)

**Interpretation:** {kappa['interpretation']}

- **Kappa Score (κ):** {kappa['kappa_score']:.4f}
- **Agreement Rate:** {kappa['agreement_rate']:.1%}
- **Total Comparisons:** {kappa['total_comparisons']}

**Confusion Matrix:**
| LLM \\ Human | Personalized | Public | Tie |
|--------------|--------------|--------|-----|
| **Personalized** | {kappa['confusion_matrix']['llm_personalized_human_personalized']} | {kappa['confusion_matrix']['llm_personalized_human_public']} | {kappa['confusion_matrix']['llm_personalized_human_tie']} |
| **Public** | {kappa['confusion_matrix']['llm_public_human_personalized']} | {kappa['confusion_matrix']['llm_public_human_public']} | {kappa['confusion_matrix']['llm_public_human_tie']} |
| **Tie** | {kappa['confusion_matrix']['llm_tie_human_personalized']} | {kappa['confusion_matrix']['llm_tie_human_public']} | {kappa['confusion_matrix']['llm_tie_human_tie']} |

Cohen's Kappa measures agreement while accounting for chance. κ > 0.6 indicates significant agreement.

---

"""

    # Add Recall section
    if 'recall_at_human_preference' in metrics:
        recall = metrics['recall_at_human_preference']
        md_content += f"""## 3. Recall @ Human Preference

当真人认为个性化查询更好时，LLM 有 **{recall['recall_score']:.1%}** 的概率也选择个性化查询。

- **Human Personalized Count:** {recall['human_personalized_count']}
- **LLM Agreement Count:** {recall['llm_agreement_count']}

This metric measures LLM's ability to identify personalized queries that humans prefer.

---

"""

    # Add MAE section
    if 'mean_absolute_error' in metrics and 'score_distribution.png' in figures:
        mae = metrics['mean_absolute_error']
        md_content += f"""## 4. Mean Absolute Error

![Score Distribution](figures/score_distribution.png)

**Overall MAE:** {mae['mae']:.2f}/10

- **Public Query MAE:** {mae['mae_public']:.2f}/10 ({mae['n_public_pairs']} pairs)
- **Personalized Query MAE:** {mae['mae_personalized']:.2f}/10 ({mae['n_personalized_pairs']} pairs)

**Interpretation:** {mae['interpretation']}

Lower MAE indicates better absolute score alignment. An MAE < 2.0 is generally acceptable for a 10-point scale.

---

"""

    # Add Systematic Bias section
    if 'systematic_bias' in metrics and 'systematic_bias.png' in figures:
        bias = metrics['systematic_bias']
        md_content += f"""## 5. Systematic Bias Analysis

![Systematic Bias](figures/systematic_bias.png)

**Personalized Win Rate:**
- **LLM:** {bias['llm_personalized_win_rate']:.1%}
- **Human:** {bias['human_personalized_win_rate']:.1%}
- **Bias Difference:** {bias['bias_difference']:+.1%}

**Finding:** {bias['interpretation']}

Systematic bias detection helps identify if LLM has inherent preferences (e.g., always preferring personalized queries).

---

"""

    # Add Per-User Analysis section
    if 'per_user_analysis' in metrics and 'per_user_agreement.png' in figures:
        per_user = metrics['per_user_analysis']
        md_content += f"""## 6. Per-User Analysis

![Per-User Agreement](figures/per_user_agreement.png)

| User ID | LLM Avg | Human Avg | MAE | Agreement Rate |
|---------|---------|-----------|-----|----------------|
"""
        for user in per_user[:10]:  # Show top 10
            md_content += f"| {user['user_id']} | {user['avg_llm_score']:.2f} | {user['avg_human_score']:.2f} | {user['mae']:.2f} | {user['agreement_rate']:.1%} |\n"

        md_content += "\n"

    # Add Recommendations section
    md_content += """---

## Recommendations

Based on the analysis above, here are our recommendations:

"""

    if bias_val > 0.15:
        md_content += "1. ⚠️ **Adjust LLM scoring prompt:** The LLM shows significant positive bias toward personalized queries. Consider adding instructions to be more objective.\n"
    elif kappa_val < 0.6:
        md_content += "1. ❌ **Improve evaluation criteria:** Low agreement suggests the evaluation criteria may need refinement.\n"
    else:
        md_content += "1. ✅ **LLM evaluation is reliable:** Continue using LLM for automated evaluation with current setup.\n"

    md_content += """2. 📊 **Regular human evaluation:** Conduct periodic human evaluations to monitor alignment over time.
3. 🔍 **Analyze edge cases:** Investigate cases where LLM and human disagree to understand the reasons.
4. 🎯 **Fine-tune evaluation:** Consider fine-tuning the LLM evaluation prompt based on insights from this analysis.

---

## Report Metadata

- **Generated by:** Stage 11: Human Evaluation Pipeline
- **Timestamp:** """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
- **Metrics file:** `alignment_metrics.json`
- **Figures directory:** `figures/`

"""

    # Save report
    output_path = os.path.join(output_dir, 'alignment_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    log_with_timestamp(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization report for LLM-Human alignment evaluation"
    )
    parser.add_argument(
        "--metrics-dir",
        default="/home/wlia0047/wenyu/result/user_profile/11_human_evaluation/reports",
        help="Directory containing alignment_metrics.json"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/wenyu/result/user_profile/11_human_evaluation/reports",
        help="Output directory for report and figures"
    )

    args = parser.parse_args()

    # Create figures directory
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Load metrics
    metrics = load_metrics(args.metrics_dir)

    # Generate all figures
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("GENERATING VISUALIZATION REPORT")
    log_with_timestamp("="*70)

    figures = {}
    figures['spearman_correlation.png'] = plot_spearman_correlation(metrics, figures_dir)
    figures['confusion_matrix.png'] = plot_confusion_matrix(metrics, figures_dir)
    figures['score_distribution.png'] = plot_score_distribution(metrics, figures_dir)
    figures['per_user_agreement.png'] = plot_per_user_agreement(metrics, figures_dir)
    figures['systematic_bias.png'] = plot_systematic_bias(metrics, figures_dir)

    # Filter out None values
    figures = {k: v for k, v in figures.items() if v is not None}

    # Generate markdown report
    report_path = generate_markdown_report(metrics, figures, args.output_dir)

    # Final summary
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("REPORT GENERATION COMPLETE")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Generated {len(figures)} figures:")
    for fig_name in figures.values():
        log_with_timestamp(f"  - figures/{fig_name}")
    log_with_timestamp(f"\nMarkdown report: {report_path}")
    log_with_timestamp(f"\nOpen {report_path} to view the complete evaluation report")
    log_with_timestamp("="*70)


if __name__ == "__main__":
    main()
