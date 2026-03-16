#!/usr/bin/env python3
"""
Comprehensive verification script for all retrieval evaluation results
Checks data consistency and performance metrics across all retrievers
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_tracking import verify_data_consistency


def load_all_results(output_dir, user_id):
    """Load all clean and noisy evaluation results"""
    retrievers = {
        'bm25': 'BM25',
        'tfidf': 'TF-IDF',
        'dirichlet': 'Dirichlet',
        'dense': 'Dense (MiniLM)',
        'e5': 'E5',
        'bge': 'BGE',
        'colbert': 'ColBERT',
        'ance': 'ANCE',
        'minilm': 'MiniLM',
        'mpnet': 'MPNet',
        'star': 'STAR',
        'hybrid_bm25_e5': 'Hybrid BM25+E5'
    }

    results = {}
    for key, name in retrievers.items():
        clean_file = os.path.join(output_dir, f'retrieval_{key}_clean_{user_id}.json')
        noisy_file = os.path.join(output_dir, f'retrieval_{key}_noisy_{user_id}.json')

        if os.path.exists(clean_file) and os.path.exists(noisy_file):
            try:
                with open(clean_file, 'r') as f:
                    clean_data = json.load(f)
                with open(noisy_file, 'r') as f:
                    noisy_data = json.load(f)

                results[key] = {
                    'name': name,
                    'clean': clean_data,
                    'noisy': noisy_data,
                    'verification': verify_data_consistency(clean_file, noisy_file)
                }
            except Exception as e:
                results[key] = {
                    'name': name,
                    'error': str(e)
                }

    return results


def analyze_metrics(results):
    """Analyze metrics across all retrievers"""
    print("=" * 90)
    print("性能指标分析 (Clean vs Noisy)")
    print("=" * 90)
    print()

    expected_behavior = []
    abnormal_behavior = []
    no_diff = []

    for key, data in results.items():
        if 'error' in data:
            continue

        clean_metrics = data['clean'].get('metrics', {})
        noisy_metrics = data['noisy'].get('metrics', {})

        # Calculate key metric differences
        p1_diff = noisy_metrics.get('P@1', 0) - clean_metrics.get('P@1', 0)
        map3_diff = noisy_metrics.get('MAP@3', 0) - clean_metrics.get('MAP@3', 0)
        ndcg3_diff = noisy_metrics.get('NDCG@3', 0) - clean_metrics.get('NDCG@3', 0)

        # Determine behavior
        is_expected = p1_diff <= 0.001 and map3_diff <= 0.001 and ndcg3_diff <= 0.001
        has_significant_diff = abs(p1_diff) > 0.001 or abs(map3_diff) > 0.001 or abs(ndcg3_diff) > 0.001

        retriever_info = {
            'key': key,
            'name': data['name'],
            'p1_clean': clean_metrics.get('P@1', 0),
            'p1_noisy': noisy_metrics.get('P@1', 0),
            'p1_diff': p1_diff,
            'map3_diff': map3_diff,
            'ndcg3_diff': ndcg3_diff,
            'verification': data.get('verification', {})
        }

        if not has_significant_diff:
            no_diff.append(retriever_info)
        elif is_expected:
            expected_behavior.append(retriever_info)
        else:
            abnormal_behavior.append(retriever_info)

    # Print results
    if expected_behavior:
        print("✅ 预期行为 (Noisy ≤ Clean):")
        for info in expected_behavior:
            print(f"   {info['name']:20s} P@1: {info['p1_clean']:.4f} → {info['p1_noisy']:.4f} ({info['p1_diff']:+.4f})")
        print()

    if no_diff:
        print("✅ 无显著差异:")
        for info in no_diff:
            print(f"   {info['name']:20s} P@1: {info['p1_clean']:.4f} → {info['p1_noisy']:.4f} ({info['p1_diff']:+.4f})")
        print()

    if abnormal_behavior:
        print("❌ 异常行为 (Noisy > Clean):")
        for info in abnormal_behavior:
            print(f"   {info['name']:20s} P@1: {info['p1_clean']:.4f} → {info['p1_noisy']:.4f} ({info['p1_diff']:+.4f})")
        print()

    return {
        'expected': len(expected_behavior),
        'no_diff': len(no_diff),
        'abnormal': len(abnormal_behavior),
        'abnormal_list': abnormal_behavior
    }


def analyze_data_consistency(results):
    """Analyze data source consistency across all retrievers"""
    print("=" * 90)
    print("数据一致性分析")
    print("=" * 90)
    print()

    consistent = []
    inconsistent = []
    unknown = []

    for key, data in results.items():
        if 'error' in data:
            continue

        verification = data.get('verification', {})

        if 'error' in verification:
            unknown.append((data['name'], verification['error']))
        elif verification.get('overall_consistent', False):
            consistent.append(data['name'])
        else:
            inconsistent.append((data['name'], verification))

    print(f"✅ 数据一致: {len(consistent)} 个")
    for name in consistent:
        print(f"   - {name}")

    if inconsistent:
        print(f"\n⚠️  数据不一致: {len(inconsistent)} 个")
        for name, info in inconsistent:
            print(f"   - {name}: Clean={info['clean_source']}, Noisy={info['noisy_source']}")

    if unknown:
        print(f"\n❓ 无法验证: {len(unknown)} 个")
        for name, error in unknown:
            print(f"   - {name}: {error}")

    print()

    return {
        'consistent': len(consistent),
        'inconsistent': len(inconsistent),
        'unknown': len(unknown)
    }


def generate_summary_report(results, metrics_analysis, consistency_analysis):
    """Generate comprehensive summary report"""
    print("=" * 90)
    print("总结报告")
    print("=" * 90)
    print()

    total = len(results)
    abnormal_count = metrics_analysis['abnormal']
    consistent_count = consistency_analysis['consistent']

    print(f"📊 评估统计:")
    print(f"   总检索器数: {total}")
    print(f"   数据一致: {consistent_count} ✅")
    print(f"   数据不一致: {total - consistent_count} ⚠️")
    print(f"   性能异常: {abnormal_count} ❌")
    print()

    # Overall status
    if consistent_count == total and abnormal_count == 0:
        print("🎉 所有检索器验证通过！")
        print("   ✅ 数据源完全一致")
        print("   ✅ 性能表现符合预期")
    elif consistent_count == total:
        print("✅ 数据一致性验证通过，但存在性能异常")
        print(f"   ℹ️  {abnormal_count} 个检索器的 Noisy 模式表现优于 Clean")
    else:
        print("⚠️  发现数据一致性问题")
        print(f"   ℹ️  {total - consistent_count} 个检索器数据源不一致")
        print(f"   ℹ️  建议重新运行受影响的检索器")

    print()
    print(f"报告生成时间: {datetime.now().isoformat()}")


def main():
    output_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/13_retrieval"
    user_id = "A13OFOB1394G31"

    print("=" * 90)
    print("Stage 13: 全面检索评估验证")
    print("=" * 90)
    print(f"输出目录: {output_dir}")
    print(f"用户 ID: {user_id}")
    print(f"验证时间: {datetime.now().isoformat()}")
    print()

    # Load all results
    results = load_all_results(output_dir, user_id)

    # Analyze metrics
    metrics_analysis = analyze_metrics(results)

    # Analyze data consistency
    consistency_analysis = analyze_data_consistency(results)

    # Generate summary
    generate_summary_report(results, metrics_analysis, consistency_analysis)

    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_retrievers': len(results),
        'metrics_analysis': metrics_analysis,
        'consistency_analysis': consistency_analysis,
        'detailed_results': {}
    }

    for key, data in results.items():
        if 'error' not in data:
            report['detailed_results'][key] = {
                'name': data['name'],
                'clean_p1': data['clean'].get('metrics', {}).get('P@1', 0),
                'noisy_p1': data['noisy'].get('metrics', {}).get('P@1', 0),
                'data_consistent': data.get('verification', {}).get('overall_consistent', False)
            }

    report_file = os.path.join(output_dir, 'verification_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 详细报告已保存: {report_file}")


if __name__ == "__main__":
    main()
