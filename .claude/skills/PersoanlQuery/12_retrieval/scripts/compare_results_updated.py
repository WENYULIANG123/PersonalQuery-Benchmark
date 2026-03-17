#!/usr/bin/env python3
"""比较所有检索器的 Clean vs Noisy 性能 - 支持新的用户子目录结构"""

import json
import os
from pathlib import Path
import argparse

def compare_user_results(results_dir, user_id):
    """比较特定用户的所有检索器结果"""
    retrievers = ['bm25', 'tfidf', 'dirichlet', 'bge', 'e5', 'ance', 'minilm', 'mpnet', 'star', 'dense', 'colbert']
    
    print(f"\n用户 {user_id} 的检索结果对比:")
    print('=' * 110)
    
    # 表头
    print(f"{'Retriever':<15} {'Clean P@1':<12} {'Noisy P@1':<12} {'Diff':<10} {'Clean MAP@3':<14} {'Noisy MAP@3':<14} {'Diff':<10} {'Status':<8}")
    print('-' * 110)
    
    summary = []
    abnormal_count = 0
    missing_files = []
    
    for ret in retrievers:
        # 新的文件路径结构
        clean_file = os.path.join(results_dir, user_id, f'retrieval_{ret}_clean.json')
        noisy_file = os.path.join(results_dir, user_id, f'retrieval_{ret}_noisy.json')
        
        if os.path.exists(clean_file) and os.path.exists(noisy_file):
            with open(clean_file) as f:
                clean_data = json.load(f)
            with open(noisy_file) as f:
                noisy_data = json.load(f)
            
            clean_p1 = clean_data['metrics']['P@1']
            noisy_p1 = noisy_data['metrics']['P@1']
            diff_p1 = noisy_p1 - clean_p1
            
            clean_map = clean_data['metrics']['MAP@3']
            noisy_map = noisy_data['metrics']['MAP@3']
            diff_map = noisy_map - clean_map
            
            # 标记异常（noisy应该低于clean）
            status = '✅ 正常' if diff_p1 <= 0 else '❌ 异常'
            if diff_p1 > 0:
                abnormal_count += 1
            
            summary.append((ret, clean_p1, noisy_p1, diff_p1, clean_map, noisy_map, diff_map))
            
            print(f"{ret:<15} {clean_p1:<12.4f} {noisy_p1:<12.4f} {diff_p1:+<10.4f} {clean_map:<14.4f} {noisy_map:<14.4f} {diff_map:+<10.4f} {status:<8}")
        else:
            if not os.path.exists(clean_file):
                missing_files.append(f"{ret}_clean")
            if not os.path.exists(noisy_file):
                missing_files.append(f"{ret}_noisy")
    
    if missing_files:
        print(f"\n缺失文件: {', '.join(missing_files)}")
    
    return summary, abnormal_count

def compare_all_users(results_dir):
    """比较所有用户的结果"""
    # 获取所有用户目录
    user_dirs = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('A')]
    
    print('=' * 100)
    print('Stage 12 检索评估结果汇总 (新目录结构)')
    print('=' * 100)
    print(f"找到 {len(user_dirs)} 个用户")
    
    all_summaries = {}
    total_abnormal = 0
    
    for user_id in sorted(user_dirs):
        summary, abnormal_count = compare_user_results(results_dir, user_id)
        all_summaries[user_id] = summary
        total_abnormal += abnormal_count
    
    # 汇总统计
    print('\n' + '=' * 100)
    print('整体统计')
    print('=' * 100)
    
    # 计算平均性能
    retriever_stats = {}
    for user_id, user_summary in all_summaries.items():
        for ret_data in user_summary:
            ret_name = ret_data[0]
            if ret_name not in retriever_stats:
                retriever_stats[ret_name] = {
                    'clean_p1': [], 'noisy_p1': [],
                    'clean_map': [], 'noisy_map': []
                }
            retriever_stats[ret_name]['clean_p1'].append(ret_data[1])
            retriever_stats[ret_name]['noisy_p1'].append(ret_data[2])
            retriever_stats[ret_name]['clean_map'].append(ret_data[4])
            retriever_stats[ret_name]['noisy_map'].append(ret_data[5])
    
    print(f"\n{'Retriever':<15} {'Avg Clean P@1':<15} {'Avg Noisy P@1':<15} {'Avg Diff':<12} {'Avg Clean MAP@3':<17} {'Avg Noisy MAP@3':<17}")
    print('-' * 100)
    
    for ret_name, stats in sorted(retriever_stats.items()):
        if stats['clean_p1']:  # 如果有数据
            avg_clean_p1 = sum(stats['clean_p1']) / len(stats['clean_p1'])
            avg_noisy_p1 = sum(stats['noisy_p1']) / len(stats['noisy_p1'])
            avg_clean_map = sum(stats['clean_map']) / len(stats['clean_map'])
            avg_noisy_map = sum(stats['noisy_map']) / len(stats['noisy_map'])
            
            print(f"{ret_name:<15} {avg_clean_p1:<15.4f} {avg_noisy_p1:<15.4f} "
                  f"{(avg_noisy_p1 - avg_clean_p1):+<12.4f} "
                  f"{avg_clean_map:<17.4f} {avg_noisy_map:<17.4f}")
    
    print(f"\n总异常情况数: {total_abnormal}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='比较检索结果')
    parser.add_argument('--results-dir', default='/fs04/ar57/wenyu/result/personal_query/12_retrieval',
                        help='结果目录路径')
    parser.add_argument('--user', default=None, help='特定用户ID (默认: 所有用户)')
    
    args = parser.parse_args()
    
    if args.user:
        compare_user_results(args.results_dir, args.user)
    else:
        compare_all_users(args.results_dir)