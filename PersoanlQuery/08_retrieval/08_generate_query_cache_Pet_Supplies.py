#!/usr/bin/env python3
"""
生成并预存储所有检索器的查询缓存

这个脚本将：
1. 加载所有用户的查询 (stage6 clean/noisy + stage7 clean/noisy)
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
os.environ["HF_HOME"] = "/root/hf_models"

import sys
import json
import pickle
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

current_dir = Path(__file__).parent.resolve()
retrieval_root = current_dir.parent
personquery_root = retrieval_root.parent

sys.path.insert(0, str(retrieval_root))
sys.path.insert(0, str(personquery_root))

from utils.retrievers import (
    E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, GritLMRetriever, BM25
)

# ============ 配置加载 ============
from config import get_category_config, get_global_paths

CATEGORY_NAME = "Pet_Supplies"
CAT_CONFIG = get_category_config(CATEGORY_NAME)
GLOBAL_PATHS = get_global_paths()

STAGE9_DIR = GLOBAL_PATHS['stage9_targeted_noisy_query']
STAGE7_DIR = GLOBAL_PATHS['stage7_iterative_refinement']
STAGE6_DIR = GLOBAL_PATHS['stage6_query']
QUERY_FILE = CAT_CONFIG['query_file']
ACL_QUERY_FILE = QUERY_FILE
CCOMP_QUERY_FILE = QUERY_FILE
CACHE_DIR = CAT_CONFIG['query_cache_dir']
BM25_RETRIEVER_CACHE_DIR = CAT_CONFIG['retriever_cache_dir']

AVAILABLE_RETRIEVERS = {
    'GRITLM': GritLMRetriever,  # 优先处理 GRITLM
    'BGE': BGERetriever,
    'E5': E5Retriever,
    'MiniLM': MiniLMRetriever,
    'STAR': STARRetriever,
    'BM25': None,  # BM25 单独处理
}

def log_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def find_all_users() -> Set[str]:
    """查找所有用户"""
    users = set()

    # 从Stage6查找
    if os.path.isdir(STAGE6_DIR):
        for filename in os.listdir(STAGE6_DIR):
            if filename.startswith('queries_') and filename.endswith('.json'):
                user_id = filename.replace('queries_', '').replace('.json', '')
                users.add(user_id)

    # 从Stage9查找
    if os.path.isdir(STAGE9_DIR):
        for filename in os.listdir(STAGE9_DIR):
            if filename.startswith('noisy_queries_') and filename.endswith('.json'):
                user_id = filename.replace('noisy_queries_', '').replace('.json', '')
                users.add(user_id)
            if filename.startswith('iterative_noisy_query_') and filename.endswith('.json'):
                user_id = filename.replace('iterative_noisy_query_', '').replace('.json', '')
                if user_id and user_id not in {'summary', 'all_users_summary'}:
                    users.add(user_id)

    return users


def load_bm25_retriever():
    """加载 BM25 检索器（从预构建的索引缓存）"""
    bm25_path = None
    for f in os.listdir(BM25_RETRIEVER_CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(BM25_RETRIEVER_CACHE_DIR, f)
            break
    if bm25_path is None:
        raise FileNotFoundError(f"BM25 retriever cache not found in {BM25_RETRIEVER_CACHE_DIR}")

    log_with_timestamp(f"  加载 BM25 索引: {os.path.getsize(bm25_path)/1024/1024:.1f} MB")
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25


def generate_bm25_query_cache_for_mode(queries: List[Dict], mode: str, top_k: int = 100) -> Dict[str, List]:
    """为单个模式生成 BM25 查询缓存

    Args:
        queries: 查询列表
        mode: 查询模式 (acl_correct, acl_noisy, ccomp_correct, ccomp_noisy)
        top_k: 预计算 top-k 结果

    Returns:
        Dict: 查询缓存，key=query文本, value=[(asin, score), ...]
    """
    log_with_timestamp(f"  开始生成 BM25 查询缓存 ({mode})...")

    # 加载 BM25 检索器
    bm25 = load_bm25_retriever()

    cache = {}
    for i, q in enumerate(queries):
        query_text = q.get('query', '')
        if not query_text or query_text in cache:
            continue

        try:
            results = bm25.search(query_text, top_k=top_k)
            cache[query_text] = results
        except Exception as e:
            log_with_timestamp(f"    ⚠️  BM25 搜索失败: {query_text[:40]}... - {e}")

        if (i + 1) % 100 == 0:
            log_with_timestamp(f"    进度: {i+1}/{len(queries)}")

    log_with_timestamp(f"  ✓ BM25 查询缓存生成完成: {len(cache)} 条")
    return cache


def save_bm25_query_cache(cache: Dict, mode: str):
    """保存 BM25 查询缓存到文件

    Args:
        cache: 查询缓存
        mode: 查询模式 (acl_correct, acl_noisy, ccomp_correct, ccomp_noisy)
    """
    subdir = get_cache_subdir(mode)
    cache_file = os.path.join(subdir, f"bm25__{mode}_cache.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    log_with_timestamp(f"  ✓ BM25 查询缓存已保存: {cache_file}")
    log_with_timestamp(f"    - 查询数: {len(cache)}")
    log_with_timestamp(f"    - 文件大小: {file_size_mb:.2f} MB")


def generate_bm25_cache_from_persona_source():
    """为 BM25 生成 correct 查询缓存 (ACL + CCOMP - correct)

    noisy 查询缓存由 09_noisy_retrieval 脚本单独生成
    """
    # 加载 ACL 查询
    acl_correct, _ = load_acl_queries()
    # 加载 CCOMP 查询
    ccomp_correct, _ = load_ccomp_queries()

    if not acl_correct and not ccomp_correct:
        log_with_timestamp("⚠️  没有从 acl_query.json 或 ccomp_query.json 加载到任何查询")
        return {'total_cached': 0}

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 开始生成 BM25 查询缓存 (ACL + CCOMP - correct)")
    log_with_timestamp("=" * 80)

    # 统计
    acl_correct_count = len(acl_correct)
    ccomp_correct_count = len(ccomp_correct)

    log_with_timestamp(f"")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • ACL correct 查询: {acl_correct_count} 条")
    log_with_timestamp(f"  • CCOMP correct 查询: {ccomp_correct_count} 条")
    log_with_timestamp(f"  • 缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"")

    start_time = time.time()
    total_cached = 0

    # 定义查询类型和模式（仅 correct）
    query_modes = [
        (acl_correct, 'acl_correct'),
        (ccomp_correct, 'ccomp_correct'),
    ]

    for queries, mode in query_modes:
        if queries:
            cache = generate_bm25_query_cache_for_mode(queries, mode, top_k=100)
            save_bm25_query_cache(cache, mode)
            total_cached += len(cache)
        else:
            log_with_timestamp(f"  (无 {mode} 查询，跳过)")

    elapsed = time.time() - start_time

    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✅ BM25 查询缓存生成完成!")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"")
    log_with_timestamp(f"⏱️  执行统计:")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp(f"  • 缓存查询总数: {total_cached}")
    log_with_timestamp(f"")

    return {'total_cached': total_cached}


def load_acl_queries() -> Tuple[List[Dict], List[Dict]]:
    """从 query.json 加载所有 ACL 查询，返回 (correct_queries, [])

    新格式：每个 item 直接包含 acl_query 字段
    noisy_queries 由 09_noisy_retrieval 脚本单独生成
    """
    if not os.path.exists(ACL_QUERY_FILE):
        log_with_timestamp(f"⚠️  文件不存在: {ACL_QUERY_FILE}")
        return [], []

    with open(ACL_QUERY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        log_with_timestamp(f"⚠️  query.json 格式错误：期望 list，实际 {type(data)}")
        return [], []

    correct_queries = []

    for item in data:
        user_id = item.get('user_id', '')
        asin = item.get('asin', '')

        # 新格式：直接有 acl_query 字段
        acl_query = item.get('acl_query')
        if isinstance(acl_query, dict):
            query_text = acl_query.get('query', '')
            if query_text:
                correct_queries.append({
                    'user_id': user_id,
                    'asin': asin,
                    'acl': acl_query.get('level', 0),
                    'is_ground_truth': True,
                    'query': query_text,
                })
        elif isinstance(acl_query, str) and acl_query:
            # 兼容字符串格式
            correct_queries.append({
                'user_id': user_id,
                'asin': asin,
                'acl': 0,
                'is_ground_truth': True,
                'query': acl_query,
            })

    log_with_timestamp(f"✓ 从 {ACL_QUERY_FILE} 加载了 {len(correct_queries)} 条 correct 查询 (ACL)")
    return correct_queries, []


def load_ccomp_queries() -> Tuple[List[Dict], List[Dict]]:
    """从 query.json 加载所有 CCOMP 查询，返回 (correct_queries, [])

    新格式：每个 item 直接包含 ccomp_query 字段
    noisy_queries 由 09_noisy_retrieval 脚本单独生成
    """
    if not os.path.exists(CCOMP_QUERY_FILE):
        log_with_timestamp(f"⚠️  文件不存在: {CCOMP_QUERY_FILE}")
        return [], []

    with open(CCOMP_QUERY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        log_with_timestamp(f"⚠️  query.json 格式错误：期望 list，实际 {type(data)}")
        return [], []

    correct_queries = []

    for item in data:
        user_id = item.get('user_id', '')
        asin = item.get('asin', '')

        # 新格式：直接有 ccomp_query 字段
        ccomp_query = item.get('ccomp_query')
        if isinstance(ccomp_query, dict):
            query_text = ccomp_query.get('query', '')
            if query_text:
                correct_queries.append({
                    'user_id': user_id,
                    'asin': asin,
                    'ccomp': ccomp_query.get('level', 0),
                    'is_ground_truth': True,
                    'query': query_text,
                })
        elif isinstance(ccomp_query, str) and ccomp_query:
            # 兼容字符串格式
            correct_queries.append({
                'user_id': user_id,
                'asin': asin,
                'ccomp': 0,
                'is_ground_truth': True,
                'query': ccomp_query,
            })

    log_with_timestamp(f"✓ 从 {CCOMP_QUERY_FILE} 加载了 {len(correct_queries)} 条 correct 查询 (CCOMP)")
    return correct_queries, []


def get_users_from_persona_generated() -> Set[str]:
    """从 acl_query.json 和 ccomp_query.json 获取所有用户 ID"""
    users = set()
    # ACL 用户
    acl_correct, _ = load_acl_queries()
    users.update(q.get('user_id', '') for q in acl_correct if q.get('user_id'))
    # CCOMP 用户
    ccomp_correct, _ = load_ccomp_queries()
    users.update(q.get('user_id', '') for q in ccomp_correct if q.get('user_id'))
    return users

def load_stage9_queries(user_id: str) -> Dict[str, List[Dict]]:
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")

    if not os.path.exists(query_file):
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
                'is_noisy': False,
                'source': 'stage9_clean'
            })

        noisy_query = pq.get('noisy', '')
        if noisy_query:
            result['noisy'].append({
                'asin': asin,
                'query': noisy_query,
                'is_noisy': True,
                'source': 'stage9_noisy'
            })

    return result


def load_stage6_queries(user_id: str) -> List[Dict]:
    """从Stage6加载查询（兼容新旧两种格式）"""
    query_file = os.path.join(STAGE6_DIR, f"queries_{user_id}.json")

    if not os.path.exists(query_file):
        return []

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    # 新格式：直接是顶层对象，target_user_query 是字符串
    if 'target_user_query' in data and isinstance(data.get('target_user_query'), str):
        query_text = data.get('target_user_query', '')
        asin = data.get('asin', '')
        if query_text:
            result.append({
                'asin': asin,
                'query': query_text,
                'is_noisy': False,
                'source': 'stage6'
            })
        return result

    # 旧格式：results 数组
    for q in data.get('results', []):
        target = q.get('target_user_query', {})
        if isinstance(target, dict):
            query_text = target.get('query', '')
        else:
            query_text = str(target)
        asin = q.get('asin', '')
        if query_text:
            result.append({
                'asin': asin,
                'query': query_text,
                'is_noisy': False,
                'source': 'stage6'
            })

    return result


def load_stage7_clean_queries(user_id: str) -> List[Dict]:
    query_file = os.path.join(STAGE9_DIR, f"iterative_noisy_query_{user_id}.json")
    if not os.path.exists(query_file):
        return []

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    result = []
    for q in data.get('queries', []):
        if not isinstance(q, dict):
            continue
        asin = q.get('asin', '')
        pq = q.get('personalized_query', {}) if isinstance(q.get('personalized_query', {}), dict) else {}
        query_text = pq.get('original', '')
        if asin and query_text:
            result.append({
                'asin': asin,
                'query': query_text,
                'is_noisy': False,
                'source': 'stage7_clean'
            })

    return result


def load_stage7_noisy_queries(user_id: str) -> List[Dict]:
    query_file = os.path.join(STAGE9_DIR, f"iterative_noisy_query_{user_id}.json")
    if not os.path.exists(query_file):
        return []

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    result = []
    for q in data.get('queries', []):
        if not isinstance(q, dict):
            continue
        asin = q.get('asin', '')
        pq = q.get('personalized_query', {}) if isinstance(q.get('personalized_query', {}), dict) else {}
        query_text = pq.get('noisy', '') or pq.get('original', '')
        if asin and query_text:
            result.append({
                'asin': asin,
                'query': query_text,
                'is_noisy': True,
                'source': 'stage7_noisy'
            })

    return result


def load_user_queries(user_id: str, modes: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """加载用户查询，支持 stage6/7/9 的 clean/noisy 模式。"""
    requested_modes = set(modes or ['clean', 'noisy'])
    result = {mode: [] for mode in requested_modes}

    # Stage6 查询
    if 'stage6' in requested_modes:
        result['stage6'] = load_stage6_queries(user_id)

    if 'clean' in requested_modes or 'noisy' in requested_modes:
        stage9_result = load_stage9_queries(user_id)
        if 'clean' in requested_modes:
            result['clean'] = stage9_result.get('clean', [])
        if 'noisy' in requested_modes:
            result['noisy'] = stage9_result.get('noisy', [])

    if 'stage7_clean' in requested_modes:
        result['stage7_clean'] = load_stage7_clean_queries(user_id)

    if 'stage7_noisy' in requested_modes:
        result['stage7_noisy'] = load_stage7_noisy_queries(user_id)

    if 'stage7' in requested_modes:
        result['stage7'] = load_stage7_noisy_queries(user_id)

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
            error_msg = str(e)
            log_with_timestamp(f"      ❌ 编码失败 [{retriever_name}|{user_id}|{mode}] 查询: {query_text[:40]}... 错误: {error_msg[:100]}")
            # 任何编码错误都立即结束脚本
            log_with_timestamp(f"      ⚠️  检测到编码错误，立即结束任务")
            log_with_timestamp(f"      ⚠️  当前检索器: {retriever_name}, 用户: {user_id}, 模式: {mode}")
            import sys
            sys.exit(1)
    
    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")
    
    return cache

def clear_cache() -> int:
    """删除旧的查询缓存文件（保留 noisy 查询缓存）"""
    if not os.path.exists(CACHE_DIR):
        return 0

    # noisy 查询缓存目录，不删除
    NOISY_DIRS = {'acl_noisy_query', 'ccomp_noisy_query', 'stage6_noisy_query', 'stage7_noisy_query', 'persona_noisy_query'}

    deleted_count = 0
    for root, _, files in os.walk(CACHE_DIR):
        # 获取相对于 CACHE_DIR 的路径
        rel_path = os.path.relpath(root, CACHE_DIR)
        # 如果在 noisy 目录下，跳过
        if rel_path in NOISY_DIRS or any(rel_path.startswith(f"{n}/") for n in NOISY_DIRS):
            continue

        for name in files:
            if name.endswith('.pkl') or name.endswith('.json'):
                filepath = os.path.join(root, name)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    log_with_timestamp(f"  ⚠️  删除失败: {filepath} - {e}")

    if deleted_count > 0:
        log_with_timestamp(f"✓ 已清理旧缓存: {deleted_count} 个文件（保留 noisy 查询缓存）")
    return deleted_count

def initialize_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "stage6_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "stage6_clean_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "stage6_noisy_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "stage7_clean_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "stage7_noisy_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "persona_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "persona_correct_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "persona_noisy_query"), exist_ok=True)
    # ACL 查询缓存
    os.makedirs(os.path.join(CACHE_DIR, "acl_correct_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "acl_noisy_query"), exist_ok=True)
    # CCOMP 查询缓存
    os.makedirs(os.path.join(CACHE_DIR, "ccomp_correct_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "ccomp_noisy_query"), exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")


def get_cache_subdir(mode: str) -> str:
    if mode == 'stage6':
        return os.path.join(CACHE_DIR, "stage6_query")
    if mode == 'clean':
        return os.path.join(CACHE_DIR, "stage6_clean_query")
    if mode == 'noisy':
        return os.path.join(CACHE_DIR, "stage6_noisy_query")
    if mode == 'stage7_clean':
        return os.path.join(CACHE_DIR, "stage7_clean_query")
    if mode in {'stage7_noisy', 'stage7'}:
        return os.path.join(CACHE_DIR, "stage7_noisy_query")
    if mode == 'persona':
        return os.path.join(CACHE_DIR, "persona_query")
    if mode == 'persona_correct':
        return os.path.join(CACHE_DIR, "persona_correct_query")
    if mode == 'persona_noisy':
        return os.path.join(CACHE_DIR, "persona_noisy_query")
    if mode == 'acl_correct':
        return os.path.join(CACHE_DIR, "acl_correct_query")
    if mode == 'acl_noisy':
        return os.path.join(CACHE_DIR, "acl_noisy_query")
    if mode == 'ccomp_correct':
        return os.path.join(CACHE_DIR, "ccomp_correct_query")
    if mode == 'ccomp_noisy':
        return os.path.join(CACHE_DIR, "ccomp_noisy_query")
    raise ValueError(f"Unsupported mode for cache subdir: {mode}")


def get_mode_suffix(mode: str) -> str:
    if mode == 'stage6':
        return 'stage6'
    if mode in {'clean', 'stage7_clean'}:
        return 'clean'
    if mode in {'noisy', 'stage7_noisy', 'stage7'}:
        return 'noisy'
    if mode == 'persona':
        return 'persona'
    if mode == 'persona_correct':
        return 'persona_correct'
    if mode == 'persona_noisy':
        return 'persona_noisy'
    if mode == 'acl_correct':
        return 'acl_correct'
    if mode == 'acl_noisy':
        return 'acl_noisy'
    if mode == 'ccomp_correct':
        return 'ccomp_correct'
    if mode == 'ccomp_noisy':
        return 'ccomp_noisy'
    raise ValueError(f"Unsupported mode for filename suffix: {mode}")

def get_cache_file_path(retriever_name: str, user_id: str, mode: str) -> str:
    """获取缓存文件路径

    persona 模式下，所有用户共享一个文件：{retriever_name}_{suffix}_cache.pkl
    其他模式下，每个用户一个文件：{retriever_name}_{user_id}_{suffix}_cache.pkl
    """
    suffix = get_mode_suffix(mode)
    subdir = get_cache_subdir(mode)
    if mode == 'persona':
        return os.path.join(subdir, f"{retriever_name.lower()}_{suffix}_cache.pkl")
    return os.path.join(subdir, f"{retriever_name.lower()}_{user_id}_{suffix}_cache.pkl")

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


def _build_queries_by_user(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """将查询列表按 user_id 分组"""
    queries_by_user: Dict[str, List[Dict]] = {}
    for item in queries:
        user_id = item.get('user_id', '')
        query_text = item.get('query', '')
        asin = item.get('asin', '')
        if not user_id or not query_text:
            continue
        if user_id not in queries_by_user:
            queries_by_user[user_id] = []
        queries_by_user[user_id].append({
            'query': query_text,
            'asin': asin,
            'user_id': user_id,
            'source': 'persona',
        })
    return queries_by_user


def _encode_and_save_cache(
    retriever_name: str,
    all_queries: List[Dict],
    queries_by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为单个检索器编码并保存缓存，返回缓存的查询数"""
    retriever_class = AVAILABLE_RETRIEVERS[retriever_name]
    log_with_timestamp(f"  初始化检索器 {retriever_name}...")
    retriever = retriever_class()
    log_with_timestamp(f"  ✓ 检索器初始化完成，模型已加载")

    log_with_timestamp(f"  开始编码 {len(all_queries)} 条查询 ({mode})...")
    full_cache = encode_queries(retriever, all_queries, retriever_name, 'all_users', mode)

    if not full_cache:
        log_with_timestamp(f"  ⚠️  未生成任何缓存 ({mode})")
        return 0

    # 按 user_id 重新组织缓存
    result_cache: Dict[str, Dict[str, np.ndarray]] = {uid: {} for uid in queries_by_user.keys()}
    for query_text, embedding in full_cache.items():
        for uid, queries in queries_by_user.items():
            if any(q['query'] == query_text for q in queries):
                result_cache[uid][query_text] = embedding
                break

    cache_file = get_cache_file_path(retriever_name, '', mode)
    log_with_timestamp(f"  保存缓存到: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result_cache, f)

    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    total_embedded = sum(len(v) for v in result_cache.values())
    log_with_timestamp(f"  ✓ 缓存已保存 ({mode})")
    log_with_timestamp(f"    - 用户数: {len(result_cache)}")
    log_with_timestamp(f"    - 查询数: {total_embedded}")
    log_with_timestamp(f"    - 文件大小: {file_size_mb:.2f} MB")

    return total_embedded


def generate_cache_from_persona_source(retriever_names: Optional[List[str]] = None, clear_cache_before: bool = False):
    """从 acl_query.json 和 ccomp_query.json 生成 correct 查询缓存

    noisy 查询缓存由 09_noisy_retrieval 脚本单独生成
    """
    if retriever_names is None:
        retriever_names = list(AVAILABLE_RETRIEVERS.keys())

    # 加载 ACL 查询
    acl_correct, _ = load_acl_queries()
    # 加载 CCOMP 查询
    ccomp_correct, _ = load_ccomp_queries()

    if not acl_correct and not ccomp_correct:
        log_with_timestamp("⚠️  没有从 acl_query.json 或 ccomp_query.json 加载到任何查询")
        return {'total_queries': 0, 'total_cached': 0, 'retrievers_processed': 0}

    # 构建按用户分组的查询
    acl_correct_by_user = _build_queries_by_user(acl_correct)
    ccomp_correct_by_user = _build_queries_by_user(ccomp_correct)

    # 统计 ACL
    acl_correct_count = sum(len(v) for v in acl_correct_by_user.values())
    # 统计 CCOMP
    ccomp_correct_count = sum(len(v) for v in ccomp_correct_by_user.values())

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 开始生成查询缓存 (ACL + CCOMP - correct)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • 检索器: {len(retriever_names)} 个 - {', '.join(retriever_names)}")
    log_with_timestamp(f"  • ACL correct 用户: {len(acl_correct_by_user)} 个, 查询: {acl_correct_count} 条")
    log_with_timestamp(f"  • CCOMP correct 用户: {len(ccomp_correct_by_user)} 个, 查询: {ccomp_correct_count} 条")
    log_with_timestamp(f"  • 缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"")

    if clear_cache_before:
        clear_cache()
    initialize_cache_dir()

    start_time = time.time()
    stats = {
        'total_correct_queries': acl_correct_count + ccomp_correct_count,
        'total_cached': 0,
        'retrievers_processed': 0,
    }

    # 定义查询类型和模式（仅 correct）
    query_types = [
        ('ACL', acl_correct, acl_correct_by_user, 'acl_correct'),
        ('CCOMP', ccomp_correct, ccomp_correct_by_user, 'ccomp_correct'),
    ]

    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue

        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"【{stats['retrievers_processed'] + 1}/{len(retriever_names)}】正在处理检索器: {retriever_name}")
        log_with_timestamp(f"{'='*80}")

        for query_type, queries, by_user, mode in query_types:
            if queries:
                total = _encode_and_save_cache(
                    retriever_name,
                    queries,
                    by_user,
                    mode,
                )
                stats['total_cached'] += total
                log_with_timestamp(f"  ✓ {query_type} {mode} 缓存: {total} 条")
            else:
                log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")

        log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成\n")
        stats['retrievers_processed'] += 1

    elapsed = time.time() - start_time

    # 统计缓存目录
    cache_files = 0
    cache_dir_size = 0.0
    for subdir_name in ["acl_correct_query", "ccomp_correct_query"]:
        subdir = os.path.join(CACHE_DIR, subdir_name)
        if os.path.exists(subdir):
            for name in os.listdir(subdir):
                if name.endswith('.pkl'):
                    cache_files += 1
                    cache_dir_size += os.path.getsize(os.path.join(subdir, name))
    cache_dir_size /= (1024 * 1024)

    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("✅ 缓存生成完成!")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"")
    log_with_timestamp(f"⏱️  执行统计:")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp(f"  • 检索器处理数: {stats['retrievers_processed']}/{len(retriever_names)}")
    log_with_timestamp(f"")
    log_with_timestamp(f"📊 数据统计:")
    log_with_timestamp(f"  • ACL correct 查询数: {acl_correct_count}")
    log_with_timestamp(f"  • CCOMP correct 查询数: {ccomp_correct_count}")
    log_with_timestamp(f"  • 检索器数量: {stats['retrievers_processed']}")
    log_with_timestamp(f"  • 编码查询总数: {stats['total_cached']} (= 查询数 × 检索器数)")
    log_with_timestamp(f"")
    log_with_timestamp(f"💾 缓存存储:")
    log_with_timestamp(f"  • ACL correct 缓存目录: {os.path.join(CACHE_DIR, 'acl_correct_query')}")
    log_with_timestamp(f"  • CCOMP correct 缓存目录: {os.path.join(CACHE_DIR, 'ccomp_correct_query')}")
    log_with_timestamp(f"  • 缓存文件数: {cache_files}")
    log_with_timestamp(f"  • 总大小: {cache_dir_size:.2f} MB")
    log_with_timestamp(f"")

    return stats


def generate_cache_for_all_retrievers(
    retriever_names: Optional[List[str]] = None,
    user_ids: Optional[List[str]] = None,
    modes: Optional[List[str]] = None
):
    """为所有检索器生成缓存 (默认: 所有检索器 + 所有用户 + clean和noisy两种模式)"""
    
    if retriever_names is None:
        retriever_names = list(AVAILABLE_RETRIEVERS.keys())
    if user_ids is None:
        user_ids = list(find_all_users())
    if modes is None:
        modes = ['stage6']
    
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
    
    clear_cache()
    initialize_cache_dir()
    
    start_time = time.time()
    
    stats = {
        'total_queries': 0,
        'total_cached': 0,
        'cache_hits': 0,  # 缓存命中次数
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

        # ⚡ 性能优化：在用户循环外初始化检索器，每个检索器只加载一次模型
        log_with_timestamp(f"  初始化检索器 {retriever_name}...")
        retriever = retriever_class()
        log_with_timestamp(f"  ✓ 检索器初始化完成，模型已加载")

        for user_idx, user_id in enumerate(user_ids):
            log_with_timestamp(f"  【用户 {user_idx + 1}/{len(user_ids)}】{user_id}")

            user_queries = load_user_queries(user_id, modes)
            if not any(user_queries.get(m, []) for m in modes):
                log_with_timestamp(f"    ⚠️  用户 {user_id} 没有查询数据，跳过")
                continue

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
                    stats['cache_hits'] += 1  # 缓存命中
                    stats['total_queries'] += 1  # 每个retriever-user-mode组合算一次处理
                    continue

                log_with_timestamp(f"      开始编码查询...")
                cache = encode_queries(retriever, queries, retriever_name, user_id, mode)

                if cache:
                    log_with_timestamp(f"      成功编码 {len(cache)} 个查询，开始保存...")
                    save_cache_for_retriever(retriever_name, user_id, mode, cache)
                    stats['total_cached'] += len(cache)
                    log_with_timestamp(f"      ✓ {retriever_name}|{user_id}|{mode} 处理完成")
                else:
                    log_with_timestamp(f"      ⚠️  未生成任何缓存")

                stats['total_queries'] += 1  # 每个retriever-user-mode组合算一次处理

        log_with_timestamp(f"✓ 检索器 {retriever_name} 全部用户处理完成\n")
        stats['retrievers_processed'] += 1
    
    stats['users_processed'] = len(user_ids)
    
    elapsed = time.time() - start_time
    cache_files = 0
    cache_dir_size = 0.0
    if os.path.exists(CACHE_DIR):
        for root, _, files in os.walk(CACHE_DIR):
            for name in files:
                if name.endswith('.pkl'):
                    cache_files += 1
                    cache_dir_size += os.path.getsize(os.path.join(root, name))
        cache_dir_size /= (1024 * 1024)
    
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
    log_with_timestamp(f"  • 总处理组合数: {stats['total_queries']} (检索器×用户×模式)")
    log_with_timestamp(f"  • 缓存命中次数: {stats['cache_hits']}")
    log_with_timestamp(f"  • 新编码次数: {stats['total_queries'] - stats['cache_hits']}")
    log_with_timestamp(f"  • 缓存命中率: {(stats['cache_hits']/stats['total_queries']*100 if stats['total_queries'] > 0 else 0):.1f}%")
    log_with_timestamp(f"")
    log_with_timestamp(f"💾 缓存存储:")
    log_with_timestamp(f"  • 缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"  • 缓存文件数: {cache_files}")
    log_with_timestamp(f"  • 总大小: {cache_dir_size:.2f} MB")
    log_with_timestamp(f"")
    
    return stats

def main():
    # ==================== 硬编码配置 ====================
    # 检索器列表：核心5个 + GritLM
    RETRIEVER_NAMES = ['GRITLM', 'BGE', 'E5', 'MiniLM', 'STAR']  # GRITLM 优先
    # 是否清理旧缓存
    CLEAR_CACHE_BEFORE = True
    # 数据源：persona_generated_queries.json
    PERSONA_SOURCE = True
    # 是否生成 BM25 查询缓存
    GENERATE_BM25_CACHE = True
    # =================================================

    # 打印环境变量
    log_with_timestamp(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    log_with_timestamp(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")

    if PERSONA_SOURCE:
        log_with_timestamp("📋 使用 persona_generated_queries.json 作为数据源")

        # 先初始化缓存目录
        if CLEAR_CACHE_BEFORE:
            clear_cache()
        initialize_cache_dir()

        # 生成密集检索器查询缓存
        stats = generate_cache_from_persona_source(
            retriever_names=RETRIEVER_NAMES,
            clear_cache_before=False,  # 目录已初始化
        )

        # 生成 BM25 查询缓存
        if GENERATE_BM25_CACHE:
            log_with_timestamp("")
            bm25_stats = generate_bm25_cache_from_persona_source()
            log_with_timestamp(f"✓ BM25 查询缓存生成完成: {bm25_stats['total_cached']} 条")
        return

    # 默认旧逻辑（保留但不使用）
    stats = generate_cache_for_all_retrievers(
        retriever_names=RETRIEVER_NAMES,
        user_ids=None,
        modes=['stage6']
    )

if __name__ == '__main__':
    main()
