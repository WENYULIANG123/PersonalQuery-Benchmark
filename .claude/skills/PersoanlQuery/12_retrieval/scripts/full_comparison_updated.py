#!/usr/bin/env python3
"""完整对比所有检索器的所有指标 - 支持新的用户子目录结构"""

import json
import os
import argparse
from collections import defaultdict

def compare_user_full_metrics(results_dir, user_id):
    """对比单个用户的完整指标"""
    retrievers = ['bm25', 'tfidf', 'dirichlet', 'bge', 'e5', 'ance', 'minilm', 'mpnet', 'star', 'dense', 'colbert']
    
    # 所有指标
    metrics = ['P@1', 'R@1', 'MAP@1', 'NDCG@1', 'MRR@1',
               'P@3', 'R@3', 'MAP@3', 'NDCG@3', 'MRR@3',
               'P@5', 'R@5', 'MAP@5', 'NDCG@5', 'MRR@5',
               'P@10', 'R@10', 'MAP@10', 'NDCG@10', 'MRR@10']
    
    print(f'\n用户 {user_id} 的完整指标对比')
    print('=' * 160)
    
    # 为每个检索器收集数据
    summary = {}
    
    for ret in retrievers:
        clean_file = os.path.join(results_dir, user_id, f'retrieval_{ret}_clean.json')
        noisy_file = os.path.join(results_dir, user_id, f'retrieval_{ret}_noisy.json')
        
        if os.path.exists(clean_file) and os.path.exists(noisy_file):
            with open(clean_file) as f:
                clean_data = json.load(f)
            with open(noisy_file) as f:
                noisy_data = json.load(f)
            
            clean_metrics = clean_data['metrics']
            noisy_metrics = noisy_data['metrics']
            
            summary[ret] = {
                'clean': clean_metrics,
                'noisy': noisy_metrics
            }
    
    # 打印每个k值的对比
    for k in [1, 3, 5, 10]:
        print('=' * 160)
        print(f'@{k} 指标对比')
        print('=' * 160)
        
        # 表头
        header = f"{'Retriever':<15} {'Clean P@'+str(k):<12} {'Noisy P@'+str(k):<12} {'Diff':<10} {'Clean R@'+str(k):<12} {'Noisy R@'+str(k):<12} {'Diff':<10} {'Clean MAP@'+str(k):<14} {'Noisy MAP@'+str(k):<14} {'Diff':<10}"
        print(header)
        print('-' * 110)
        
        # 按clean MAP@k排序
        sorted_retrievers = sorted(summary.keys(), 
                                 key=lambda x: summary[x]['clean'][f'MAP@{k}'] if x in summary else 0, 
                                 reverse=True)
        
        for ret in sorted_retrievers:
            if ret in summary:
                clean = summary[ret]['clean']
                noisy = summary[ret]['noisy']
                
                p_clean = clean[f'P@{k}']
                p_noisy = noisy[f'P@{k}']
                p_diff = p_noisy - p_clean
                
                r_clean = clean[f'R@{k}']
                r_noisy = noisy[f'R@{k}']
                r_diff = r_noisy - r_clean
                
                map_clean = clean[f'MAP@{k}']
                map_noisy = noisy[f'MAP@{k}']
                map_diff = map_noisy - map_clean
                
                print(f"{ret:<15} {p_clean:<12.4f} {p_noisy:<12.4f} {p_diff:+<10.4f} {r_clean:<12.4f} {r_noisy:<12.4f} {r_diff:+<10.4f} {map_clean:<14.4f} {map_noisy:<14.4f} {map_diff:+<10.4f}")
        
        print()
    
    return summary

def compare_all_users_full(results_dir):
    """对比所有用户的完整指标"""
    # 获取所有用户目录
    user_dirs = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('A')]
    
    print('=' * 160)
    print('Stage 12 检索评估完整指标对比 (新目录结构)')
    print('=' * 160)
    print(f'找到 {len(user_dirs)} 个用户\n')
    
    # 收集所有用户数据
    all_user_data = {}
    for user_id in sorted(user_dirs):
        all_user_data[user_id] = compare_user_full_metrics(results_dir, user_id)
    
    # 计算跨用户平均值
    print('\n' + '=' * 160)
    print('跨用户平均性能')
    print('=' * 160)
    
    retriever_avg_metrics = defaultdict(lambda: defaultdict(list))
    
    for user_id, user_summary in all_user_data.items():
        for ret, data in user_summary.items():
            for metric_name in data['clean'].keys():
                retriever_avg_metrics[ret][f'clean_{metric_name}'].append(data['clean'][metric_name])
                retriever_avg_metrics[ret][f'noisy_{metric_name}'].append(data['noisy'][metric_name])
    
    # 打印平均值表格
    for k in [1, 3, 5, 10]:
        print(f'\n平均 @{k} 指标')
        print('-' * 110)
        header = f"{'Retriever':<15} {'Avg Clean P@'+str(k):<15} {'Avg Noisy P@'+str(k):<15} {'Avg Clean MAP@'+str(k):<17} {'Avg Noisy MAP@'+str(k):<17} {'MAP Drop':<10}"
        print(header)
        print('-' * 110)
        
        # 按平均clean MAP@k排序
        sorted_retrievers = sorted(retriever_avg_metrics.keys(),
                                 key=lambda x: sum(retriever_avg_metrics[x][f'clean_MAP@{k}']) / len(retriever_avg_metrics[x][f'clean_MAP@{k}']),
                                 reverse=True)
        
        for ret in sorted_retrievers:
            avg_clean_p = sum(retriever_avg_metrics[ret][f'clean_P@{k}']) / len(retriever_avg_metrics[ret][f'clean_P@{k}'])
            avg_noisy_p = sum(retriever_avg_metrics[ret][f'noisy_P@{k}']) / len(retriever_avg_metrics[ret][f'noisy_P@{k}'])
            avg_clean_map = sum(retriever_avg_metrics[ret][f'clean_MAP@{k}']) / len(retriever_avg_metrics[ret][f'clean_MAP@{k}'])
            avg_noisy_map = sum(retriever_avg_metrics[ret][f'noisy_MAP@{k}']) / len(retriever_avg_metrics[ret][f'noisy_MAP@{k}'])
            map_drop = (avg_noisy_map - avg_clean_map) / avg_clean_map * 100  # 百分比下降
            
            print(f"{ret:<15} {avg_clean_p:<15.4f} {avg_noisy_p:<15.4f} {avg_clean_map:<17.4f} {avg_noisy_map:<17.4f} {map_drop:+<9.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='完整对比检索结果')
    parser.add_argument('--results-dir', default='/fs04/ar57/wenyu/result/personal_query/12_retrieval',
                        help='结果目录路径')
    parser.add_argument('--user', default=None, help='特定用户ID (默认: 所有用户)')
    
    args = parser.parse_args()
    
    if args.user:
        compare_user_full_metrics(args.results_dir, args.user)
    else:
        compare_all_users_full(args.results_dir)