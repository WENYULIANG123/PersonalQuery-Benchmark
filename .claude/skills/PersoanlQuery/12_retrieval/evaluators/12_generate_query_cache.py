#!/usr/bin/env python3
"""
生成并预存储所有检索器的查询缓存

这个脚本将：
1. 加载所有用户的查询 (clean + noisy)
2. 为每个检索器编码每个查询
3. 保存缓存到磁盘以加速后续评估

使用方法：
    python3 generate_query_cache.py
    python3 generate_query_cache.py --retrievers ANCE Dense E5
    python3 generate_query_cache.py --users USER1 USER2 --modes clean
    
特点：
    - 默认为所有检索器生成clean和noisy两种模式的缓存
    - 预期收益: 查询评估时间 14.6s → 11-12s (20-30% improvement)
"""

import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

current_dir = Path(__file__).parent.resolve()
retrieval_root = current_dir.parent
personquery_root = retrieval_root.parent

sys.path.insert(0, str(retrieval_root))
sys.path.insert(0, str(personquery_root))

from utils.retrievers import (
    ANCERetriever, DenseRetriever, E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, MPNetRetriever
)

STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
CACHE_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/query_cache"
METADATA_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/metadata.pkl"

AVAILABLE_RETRIEVERS = {
    'ANCE': ANCERetriever,
    'Dense': DenseRetriever,
    'E5': E5Retriever,
    'BGE': BGERetriever,
    'STAR': STARRetriever,
    'MiniLM': MiniLMRetriever,
    'MPNet': MPNetRetriever,
}

def log_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def find_all_users() -> Set[str]:
    """查找所有用户"""
    users = set()
    for filename in os.listdir(STAGE9_DIR):
        if filename.startswith('noisy_queries_') and filename.endswith('.json'):
            user_id = filename.replace('noisy_queries_', '').replace('.json', '')
            users.add(user_id)
    return users

def load_user_queries(user_id: str) -> Dict[str, List[Dict]]:
    """加载用户的clean和noisy查询"""
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    
    if not os.path.exists(query_file):
        log_with_timestamp(f"⚠️  查询文件不存在: {query_file}")
        return {'clean': [], 'noisy': []}
    
    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    result = {'clean': [], 'noisy': []}
    
    for q in queries:
        asin = q.get('asin', '')
        if not asin:
            continue
        
        pq = q.get('personalized_query', {})
        
        clean_query = pq.get('original', '')
        if clean_query:
            result['clean'].append({
                'asin': asin,
                'query': clean_query,
                'is_noisy': False
            })
        
        noisy_query = pq.get('noisy', '')
        if noisy_query:
            result['noisy'].append({
                'asin': asin,
                'query': noisy_query,
                'is_noisy': True
            })
    
    return result

def encode_queries(retriever_instance, queries: List[Dict], retriever_name: str = "", user_id: str = "", mode: str = "") -> Dict[str, np.ndarray]:
    """Encode queries and return cache dict with query embeddings
    
    Args:
        retriever_instance: Initialized retriever with encode_query method
        queries: List of query dicts with 'query' key
        retriever_name: Name of retriever (for logging)
        user_id: User ID (for logging)
        mode: Query mode - clean or noisy (for logging)
        
    Returns:
        Dict mapping query_text -> embedding (numpy array)
    """
    cache = {}
    failed_count = 0
    
    for i, q in enumerate(queries):
        query_text = q.get('query', '')
        if not query_text:
            continue
        
        if query_text in cache:
            continue
        
        try:
            embedding = retriever_instance.encode_query(query_text)
            
            if not isinstance(embedding, np.ndarray):
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                else:
                    embedding = np.array(embedding)
            
            cache[query_text] = embedding
            
            progress = i + 1
            progress_pct = (progress / len(queries)) * 100
            
            if progress % 5 == 0 or progress == len(queries):
                log_with_timestamp(f"      编码进度 [{retriever_name}|{user_id}|{mode}]: {progress}/{len(queries)} ({progress_pct:.1f}%)")
        
        except Exception as e:
            failed_count += 1
            log_with_timestamp(f"      ❌ 编码失败 [{retriever_name}|{user_id}|{mode}] 查询: {query_text[:40]}... 错误: {str(e)[:50]}")
            continue
    
    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")
    
    return cache

def initialize_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")

def get_cache_file_path(retriever_name: str, user_id: str, mode: str) -> str:
    """获取缓存文件路径"""
    return os.path.join(
        CACHE_DIR,
        f"{retriever_name.lower()}_{user_id}_{mode}_cache.pkl"
    )

def cache_exists_for_query(retriever_name: str, user_id: str, mode: str) -> bool:
    """检查查询缓存文件是否已存在"""
    cache_file = get_cache_file_path(retriever_name, user_id, mode)
    return os.path.exists(cache_file)

def save_cache_for_retriever(retriever_name: str, user_id: str, mode: str, cache: Dict):
    cache_file = get_cache_file_path(retriever_name, user_id, mode)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    log_with_timestamp(f"      ✓ 缓存已保存到文件: {cache_file}")
    log_with_timestamp(f"        - 查询数: {len(cache)}")
    log_with_timestamp(f"        - 文件大小: {file_size_mb:.2f} MB")

def load_metadata() -> Dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def generate_cache_for_all_retrievers(
    retriever_names: List[str] = None,
    user_ids: List[str] = None,
    modes: List[str] = None
):
    """为所有检索器生成缓存 (默认: 所有检索器 + 所有用户 + clean和noisy两种模式)"""
    
    if retriever_names is None:
        retriever_names = list(AVAILABLE_RETRIEVERS.keys())
    if user_ids is None:
        user_ids = list(find_all_users())
    if modes is None:
        modes = ['clean', 'noisy']
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 开始生成查询缓存系统")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • 检索器: {len(retriever_names)} 个 - {', '.join(retriever_names)}")
    log_with_timestamp(f"  • 用户: {len(user_ids)} 个")
    log_with_timestamp(f"  • 模式: {', '.join(modes)}")
    log_with_timestamp(f"  • 预期缓存数: {len(retriever_names) * len(user_ids) * len(modes)} 个")
    log_with_timestamp(f"  • 缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"")
    
    initialize_cache_dir()
    
    start_time = time.time()
    
    stats = {
        'total_queries': 0,
        'total_cached': 0,
        'retrievers_processed': 0,
        'users_processed': 0,
    }
    
    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue
        
        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"【{stats['retrievers_processed'] + 1}/{len(retriever_names)}】正在处理检索器: {retriever_name}")
        log_with_timestamp(f"{'='*80}")
        
        retriever_class = AVAILABLE_RETRIEVERS[retriever_name]
        
        for user_idx, user_id in enumerate(user_ids):
            log_with_timestamp(f"  【用户 {user_idx + 1}/{len(user_ids)}】{user_id}")
            
            user_queries = load_user_queries(user_id)
            if not user_queries.get('clean') and not user_queries.get('noisy'):
                log_with_timestamp(f"    ⚠️  用户 {user_id} 没有查询数据，跳过")
                continue
            
            retriever = None
            
            for mode_idx, mode in enumerate(modes):
                queries = user_queries.get(mode, [])
                if not queries:
                    log_with_timestamp(f"    ⚠️  {mode} 模式无查询，跳过")
                    continue
                
                log_with_timestamp(f"    【模式 {mode_idx + 1}/{len(modes)}】{mode.upper()}: {len(queries)} 个查询")
                
                if cache_exists_for_query(retriever_name, user_id, mode):
                    cache_file = get_cache_file_path(retriever_name, user_id, mode)
                    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                    log_with_timestamp(f"      ✓ 缓存已存在，跳过编码")
                    log_with_timestamp(f"        - 文件: {cache_file}")
                    log_with_timestamp(f"        - 文件大小: {file_size_mb:.2f} MB")
                    log_with_timestamp(f"      ✓ {retriever_name}|{user_id}|{mode} 处理完成")
                    stats['total_cached'] += len(queries)
                    stats['total_queries'] += len(queries)
                    continue
                
                if retriever is None:
                    log_with_timestamp(f"      初始化检索器 {retriever_name}...")
                    retriever = retriever_class()
                    log_with_timestamp(f"      ✓ 检索器初始化完成")
                
                log_with_timestamp(f"      开始编码查询...")
                cache = encode_queries(retriever, queries, retriever_name, user_id, mode)
                
                if cache:
                    log_with_timestamp(f"      成功编码 {len(cache)} 个查询，开始保存...")
                    save_cache_for_retriever(retriever_name, user_id, mode, cache)
                    stats['total_cached'] += len(cache)
                    log_with_timestamp(f"      ✓ {retriever_name}|{user_id}|{mode} 处理完成")
                else:
                    log_with_timestamp(f"      ⚠️  未生成任何缓存")
                
                stats['total_queries'] += len(queries)
        
        log_with_timestamp(f"✓ 检索器 {retriever_name} 全部用户处理完成\n")
        stats['retrievers_processed'] += 1
    
    stats['users_processed'] = len(user_ids)
    
    elapsed = time.time() - start_time
    cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]) if os.path.exists(CACHE_DIR) else 0
    cache_dir_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')) / (1024*1024) if os.path.exists(CACHE_DIR) else 0
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("✅ 缓存生成完成!")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"")
    log_with_timestamp(f"⏱️  执行统计:")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp(f"  • 检索器处理数: {stats['retrievers_processed']}/{len(retriever_names)}")
    log_with_timestamp(f"  • 用户处理数: {stats['users_processed']}/{len(user_ids)}")
    log_with_timestamp(f"")
    log_with_timestamp(f"📊 数据统计:")
    log_with_timestamp(f"  • 总查询数: {stats['total_queries']}")
    log_with_timestamp(f"  • 已缓存查询: {stats['total_cached']}")
    log_with_timestamp(f"  • 缓存命中率: {(stats['total_cached']/stats['total_queries']*100 if stats['total_queries'] > 0 else 0):.1f}%")
    log_with_timestamp(f"")
    log_with_timestamp(f"💾 缓存存储:")
    log_with_timestamp(f"  • 缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"  • 缓存文件数: {cache_files}")
    log_with_timestamp(f"  • 总大小: {cache_dir_size:.2f} MB")
    log_with_timestamp(f"")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description='为所有检索器生成查询缓存 (默认为所有检索器、所有用户、clean和noisy两种模式生成缓存)'
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=None,
        help=f'要处理的检索器 (默认: 全部). 可选: {", ".join(AVAILABLE_RETRIEVERS.keys())}'
    )
    parser.add_argument(
        '--users',
        nargs='+',
        default=None,
        help='要处理的用户 (默认: 全部)'
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['clean', 'noisy'],
        help='查询模式 (默认: clean noisy)'
    )
    parser.add_argument(
        '--list-users',
        action='store_true',
        help='列出所有可用用户并退出'
    )
    parser.add_argument(
        '--list-retrievers',
        action='store_true',
        help='列出所有可用检索器并退出'
    )
    
    args = parser.parse_args()
    
    if args.list_users:
        users = find_all_users()
        log_with_timestamp(f"可用用户 ({len(users)}):")
        for user in sorted(users):
            log_with_timestamp(f"  - {user}")
        return
    
    if args.list_retrievers:
        log_with_timestamp("可用检索器:")
        for name in sorted(AVAILABLE_RETRIEVERS.keys()):
            log_with_timestamp(f"  - {name}")
        return
    
    stats = generate_cache_for_all_retrievers(
        retriever_names=args.retrievers,
        user_ids=args.users,
        modes=args.modes
    )

if __name__ == '__main__':
    main()
