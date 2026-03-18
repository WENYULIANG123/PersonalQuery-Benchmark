#!/usr/bin/env python3
"""
生成并预存储所有检索器的查询缓存

这个脚本将：
1. 加载所有用户的查询 (clean + noisy)
2. 为每个检索器编码每个查询
3. 保存缓存到磁盘以加速后续评估

使用方法：
    python3 generate_query_cache.py [--retrievers ANCE DENSE ...] [--users USER1 USER2 ...]
    
预期收益：
    - 查询评估时间: 14.6s → 10.1s (假设缓存命中率30%)
    - 后续运行更快（复用缓存）
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

# 添加retrievers路径
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery/12_retrieval')
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills/PersoanlQuery')

from utils.retrievers import (
    ANCERetriever, DenseRetriever, E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, MPNetRetriever, GritLMRetriever,
    TFIDFRetriever, DirichletPriorRetriever, ColBERTRetriever
)

# 常量定义
STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
CACHE_DIR = "/fs04/ar57/wenyu/.query_cache"  # 新的查询缓存目录
METADATA_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/metadata.pkl"

# 所有可用的检索器
AVAILABLE_RETRIEVERS = {
    'ANCE': ANCERetriever,
    'Dense': DenseRetriever,
    'E5': E5Retriever,
    'BGE': BGERetriever,
    'STAR': STARRetriever,
    'MiniLM': MiniLMRetriever,
    'MPNet': MPNetRetriever,
    'GritLM': GritLMRetriever,
    'TFIDF': TFIDFRetriever,
    'Dirichlet': DirichletPriorRetriever,
    'ColBERT': ColBERTRetriever,
}

def log_with_timestamp(msg: str):
    """带时间戳的日志"""
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
        
        # Clean查询
        clean_query = pq.get('original', '')
        if clean_query:
            result['clean'].append({
                'asin': asin,
                'query': clean_query,
                'is_noisy': False
            })
        
        # Noisy查询
        noisy_query = pq.get('noisy', '')
        if noisy_query:
            result['noisy'].append({
                'asin': asin,
                'query': noisy_query,
                'is_noisy': True
            })
    
    return result

def encode_queries(retriever_instance, queries: List[Dict]) -> Dict[str, np.ndarray]:
    """Encode queries and return cache dict with query embeddings
    
    Args:
        retriever_instance: Initialized retriever with encode_query method
        queries: List of query dicts with 'query' key
        
    Returns:
        Dict mapping query_text -> embedding (numpy array)
    """
    import numpy as np
    cache = {}
    
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
            
            if (i + 1) % 10 == 0:
                log_with_timestamp(f"    Encoded {i + 1}/{len(queries)} queries")
        
        except Exception as e:
            log_with_timestamp(f"❌ Encoding failed [{retriever_instance.__class__.__name__}] {query_text[:50]}: {e}")
            continue
    
    return cache

def initialize_cache_dir():
    """初始化缓存目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")

def save_cache_for_retriever(retriever_name: str, user_id: str, mode: str, cache: Dict):
    """保存单个检索器的缓存"""
    cache_file = os.path.join(
        CACHE_DIR, 
        f"{retriever_name.lower()}_{user_id}_{mode}_cache.pkl"
    )
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    log_with_timestamp(f"✓ 缓存已保存: {cache_file} ({len(cache)} queries)")

def load_metadata() -> Dict:
    """加载元数据（获取ASIN列表）"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def generate_cache_for_all_retrievers(
    retriever_names: List[str] = None,
    user_ids: List[str] = None,
    modes: List[str] = None
):
    """为所有检索器生成缓存"""
    
    # 默认参数
    if retriever_names is None:
        retriever_names = list(AVAILABLE_RETRIEVERS.keys())
    if user_ids is None:
        user_ids = list(find_all_users())
    if modes is None:
        modes = ['clean', 'noisy']
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("开始生成查询缓存")
    log_with_timestamp("=" * 80)
    
    log_with_timestamp(f"检索器: {', '.join(retriever_names)}")
    log_with_timestamp(f"用户: {len(user_ids)} 个")
    log_with_timestamp(f"模式: {', '.join(modes)}")
    
    initialize_cache_dir()
    
    start_time = time.time()
    
    # 统计信息
    stats = {
        'total_queries': 0,
        'total_cached': 0,
        'retrievers_processed': 0,
        'users_processed': 0,
    }
    
    # 遍历每个检索器
    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue
        
        log_with_timestamp(f"\n【处理检索器】{retriever_name}")
        
        retriever_class = AVAILABLE_RETRIEVERS[retriever_name]
        
        # 遍历每个用户
        for user_id in user_ids:
            log_with_timestamp(f"  加载用户查询: {user_id}")
            
            user_queries = load_user_queries(user_id)
            
            # 遍历每种模式
            for mode in modes:
                queries = user_queries.get(mode, [])
                if not queries:
                    continue
                
                log_with_timestamp(f"    {mode}: {len(queries)} queries")
                
                retriever = retriever_class()
                cache = encode_queries(retriever, queries)
                
                if cache:
                    save_cache_for_retriever(retriever_name, user_id, mode, cache)
                    stats['total_cached'] += len(cache)
                
                stats['total_queries'] += len(queries)
        
        stats['retrievers_processed'] += 1
    
    stats['users_processed'] = len(user_ids)
    
    elapsed = time.time() - start_time
    
    # 打印统计信息
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("缓存生成完成!")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"用时: {elapsed:.1f} 秒")
    log_with_timestamp(f"处理检索器: {stats['retrievers_processed']}")
    log_with_timestamp(f"处理用户: {stats['users_processed']}")
    log_with_timestamp(f"总查询数: {stats['total_queries']}")
    log_with_timestamp(f"已缓存: {stats['total_cached']}")
    log_with_timestamp(f"缓存目录: {CACHE_DIR}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description='为检索器生成查询缓存'
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
    
    # 列出用户
    if args.list_users:
        users = find_all_users()
        log_with_timestamp(f"可用用户 ({len(users)}):")
        for user in sorted(users):
            log_with_timestamp(f"  - {user}")
        return
    
    # 列出检索器
    if args.list_retrievers:
        log_with_timestamp("可用检索器:")
        for name in sorted(AVAILABLE_RETRIEVERS.keys()):
            log_with_timestamp(f"  - {name}")
        return
    
    # 生成缓存
    stats = generate_cache_for_all_retrievers(
        retriever_names=args.retrievers,
        user_ids=args.users,
        modes=args.modes
    )

if __name__ == '__main__':
    main()
