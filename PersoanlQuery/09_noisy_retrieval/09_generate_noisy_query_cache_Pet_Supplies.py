#!/usr/bin/env python3
"""
生成并预存储检索器的 noisy 查询缓存

这个脚本将：
1. 从 07_inject_noisy 的 combined noisy_query.json 加载 ACL 和 CCOMP noisy queries
2. 为每个检索器编码每个 noisy 查询
3. 保存缓存到磁盘以加速后续评估

使用方法：
    python3 09_generate_noisy_query_cache_Pet_Supplies.py
    python3 09_generate_noisy_query_cache_Pet_Supplies.py --retrievers GRITLM BGE
"""

import os
os.environ["HF_HOME"] = "/root/hf_models"

import sys
import json
import pickle
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

current_dir = Path(__file__).parent.resolve()
retrieval_root = current_dir.parent / "08_retrieval"
personquery_root = retrieval_root.parent

sys.path.insert(0, str(retrieval_root))
sys.path.insert(0, str(personquery_root))

from utils.retrievers import (
    E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, GritLMRetriever, BM25
)
from config import get_category_config, get_global_paths

# ============ 配置加载 ============
CATEGORY_NAME = "Pet_Supplies"
CAT_CONFIG = get_category_config(CATEGORY_NAME)
GLOBAL_PATHS = get_global_paths()

# Noisy query 文件（07 生成的 combined 文件）
NOISY_QUERY_FILE = f"{GLOBAL_PATHS['inject_noisy']}/{CATEGORY_NAME}/noisy_query.json"

# 缓存目录 - 使用与 08 相同的目录结构
CACHE_DIR = CAT_CONFIG['query_cache_dir']
BM25_RETRIEVER_CACHE_DIR = CAT_CONFIG['retriever_cache_dir']

AVAILABLE_RETRIEVERS = {
    'GRITLM': GritLMRetriever,
    'BGE': BGERetriever,
    'E5': E5Retriever,
    'MiniLM': MiniLMRetriever,
    'STAR': STARRetriever,
    'BM25': None,
}


def log_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_noisy_queries() -> Tuple[List[Dict], List[Dict]]:
    """从 combined noisy_query.json 加载 ACL 和 CCOMP noisy queries

    Returns:
        (acl_noisy_queries, ccomp_noisy_queries)
    """
    acl_noisy = []
    ccomp_noisy = []

    if not os.path.exists(NOISY_QUERY_FILE):
        log_with_timestamp(f"⚠️  noisy query 文件不存在: {NOISY_QUERY_FILE}")
        return [], []

    try:
        with open(NOISY_QUERY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            log_with_timestamp(f"⚠️  noisy query 文件为空: {NOISY_QUERY_FILE}")
            return [], []

        # 解析 JSON（支持 JSON Lines、pretty-printed 或 JSON array）
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = []
            depth = 0
            start = -1
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            data.append(json.loads(content[start:i+1]))
                        except:
                            pass
                        start = -1

        for item in data:
            query_cat = item.get('query_category', '')
            noisy_text = item.get('noisy_query', '')
            if not noisy_text:
                continue

            entry = {
                'user_id': item.get('user_id', ''),
                'asin': item.get('asin', ''),
                'is_ground_truth': True,
                'query': noisy_text,
            }

            if query_cat == 'acl':
                entry['acl'] = item.get('level', 0)
                acl_noisy.append(entry)
            elif query_cat == 'ccomp':
                entry['ccomp'] = item.get('level', 0)
                ccomp_noisy.append(entry)

        log_with_timestamp(f"✓ 从 {NOISY_QUERY_FILE} 加载了 {len(acl_noisy)} 条 ACL noisy, {len(ccomp_noisy)} 条 CCOMP noisy")
    except Exception as e:
        log_with_timestamp(f"⚠️  读取 noisy query 文件失败: {e}")
        return [], []

    return acl_noisy, ccomp_noisy


def _build_queries_by_user(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """将查询列表按用户ID分组"""
    by_user = defaultdict(list)
    for q in queries:
        uid = q.get('user_id', '')
        if uid:
            by_user[uid].append(q)
    return dict(by_user)


def get_retriever(retriever_name: str):
    """获取检索器实例"""
    if retriever_name == 'BM25':
        return BM25()
    retriever_class = AVAILABLE_RETRIEVERS.get(retriever_name)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever: {retriever_name}")
    return retriever_class()


def initialize_cache_dir():
    """初始化缓存目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "acl_noisy_query"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "ccomp_noisy_query"), exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")


def get_cache_file_path(retriever_name: str, user_id: str, mode: str) -> str:
    """获取缓存文件路径"""
    subdir = os.path.join(CACHE_DIR, f"{mode}_query")
    filename = f"{retriever_name}_{user_id}.pkl"
    return os.path.join(subdir, filename)


def _encode_and_save_cache(
    retriever_name: str,
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为检索器编码并保存查询缓存"""
    if retriever_name == 'BM25':
        # BM25 使用不同的处理方式
        return _encode_and_save_bm25_cache(queries, by_user, mode)

    retriever_class = AVAILABLE_RETRIEVERS[retriever_name]
    log_with_timestamp(f"  初始化检索器 {retriever_name}...")
    retriever = retriever_class()
    log_with_timestamp(f"  ✓ 检索器初始化完成，模型已加载")

    cache = {}
    failed_count = 0

    for user_id, user_queries in by_user.items():
        user_cache = []
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    continue
                embedding = retriever.encode_query(text)

                if not isinstance(embedding, np.ndarray):
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    else:
                        embedding = np.array(embedding)

                user_cache.append({
                    'query': text,
                    'vector': embedding,
                    'user_id': user_id,
                    'asin': q.get('asin', ''),
                    'level': q.get('acl') or q.get('ccomp', 0),
                    'is_ground_truth': q.get('is_ground_truth', True),
                })
            except Exception as e:
                log_with_timestamp(f"      ❌ 编码失败 [{retriever_name}] 查询: {text[:40]}... 错误: {str(e)[:100]}")
                failed_count += 1
                import sys
                sys.exit(1)

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        cache_file = get_cache_file_path(retriever_name, "", mode)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")

    return len(cache)


def _encode_and_save_bm25_cache(
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为 BM25 编码并保存查询缓存"""
    log_with_timestamp(f"  初始化 BM25...")

    # 加载 BM25 检索器
    bm25_path = None
    for f in os.listdir(BM25_RETRIEVER_CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(BM25_RETRIEVER_CACHE_DIR, f)
            break
    if bm25_path is None:
        log_with_timestamp(f"  ⚠️  BM25 retriever cache not found in {BM25_RETRIEVER_CACHE_DIR}")
        return 0

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    log_with_timestamp(f"  ✓ BM25 加载完成")

    cache = {}
    failed_count = 0

    for user_id, user_queries in by_user.items():
        user_cache = []
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    continue
                results = bm25.search(text, top_k=100)
                user_cache.append({
                    'query': text,
                    'results': results,
                    'user_id': user_id,
                    'asin': q.get('asin', ''),
                    'level': q.get('acl') or q.get('ccomp', 0),
                    'is_ground_truth': q.get('is_ground_truth', True),
                })
            except Exception as e:
                log_with_timestamp(f"      ❌ BM25 搜索失败: {text[:40]}... 错误: {str(e)[:100]}")
                failed_count += 1
                import sys
                sys.exit(1)

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        cache_file = get_cache_file_path("bm25", "", mode)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ BM25 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")

    return len(cache)


def clear_noisy_cache() -> int:
    """删除旧的 noisy 查询缓存文件"""
    if not os.path.exists(CACHE_DIR):
        return 0

    deleted_count = 0
    for subdir in ["acl_noisy_query", "ccomp_noisy_query"]:
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if os.path.exists(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for name in files:
                    if name.endswith('.pkl') or name.endswith('.json'):
                        filepath = os.path.join(root, name)
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception as e:
                            log_with_timestamp(f"  ⚠️  删除失败: {filepath} - {e}")

    if deleted_count > 0:
        log_with_timestamp(f"✓ 已清理旧缓存: {deleted_count} 个文件")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='生成 noisy 查询缓存')
    parser.add_argument('--retrievers', type=str, nargs='+',
                        choices=list(AVAILABLE_RETRIEVERS.keys()),
                        default=list(AVAILABLE_RETRIEVERS.keys()),
                        help='指定要处理的检索器')
    parser.add_argument('--clear', action='store_true',
                        help='清理旧缓存后再生成')
    args = parser.parse_args()

    retriever_names = args.retriever_names if hasattr(args, 'retriever_names') else args.retrievers
    clear_cache_before = args.clear

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 生成 Noisy 查询缓存")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"类别: {CATEGORY_NAME}")
    log_with_timestamp(f"Noisy 文件: {NOISY_QUERY_FILE}")
    log_with_timestamp(f"缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"检索器: {', '.join(retriever_names)}")
    log_with_timestamp("")

    # 加载 noisy queries
    acl_noisy, ccomp_noisy = load_noisy_queries()

    if not acl_noisy and not ccomp_noisy:
        log_with_timestamp("⚠️  没有加载到任何 noisy 查询")
        return

    # 按用户分组
    acl_noisy_by_user = _build_queries_by_user(acl_noisy)
    ccomp_noisy_by_user = _build_queries_by_user(ccomp_noisy)

    log_with_timestamp("")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • ACL noisy 用户: {len(acl_noisy_by_user)} 个, 查询: {sum(len(v) for v in acl_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • CCOMP noisy 用户: {len(ccomp_noisy_by_user)} 个, 查询: {sum(len(v) for v in ccomp_noisy_by_user.values())} 条")
    log_with_timestamp("")

    if clear_cache_before:
        clear_noisy_cache()
    initialize_cache_dir()

    start_time = time.time()
    total_cached = 0

    # 定义查询类型
    query_types = [
        ('ACL', acl_noisy, acl_noisy_by_user, 'acl_noisy'),
        ('CCOMP', ccomp_noisy, ccomp_noisy_by_user, 'ccomp_noisy'),
    ]

    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue

        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"正在处理检索器: {retriever_name}")
        log_with_timestamp(f"{'='*80}")

        for query_type, queries, by_user, mode in query_types:
            if queries:
                total = _encode_and_save_cache(
                    retriever_name,
                    queries,
                    by_user,
                    mode,
                )
                total_cached += total
                log_with_timestamp(f"  ✓ {query_type} {mode} 缓存: {total} 用户")
            else:
                log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")

        log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成")

    elapsed = time.time() - start_time

    # 统计缓存
    cache_files = 0
    cache_dir_size = 0.0
    for subdir in ["acl_noisy_query", "ccomp_noisy_query"]:
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if os.path.exists(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for name in files:
                    if name.endswith('.pkl'):
                        cache_files += 1
                        cache_dir_size += os.path.getsize(os.path.join(root, name)) / (1024 * 1024)

    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✨ 完成!")
    log_with_timestamp(f"  • 处理检索器: {len(retriever_names)} 个")
    log_with_timestamp(f"  • ACL noisy 缓存: {sum(len(v) for v in acl_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • CCOMP noisy 缓存: {sum(len(v) for v in ccomp_noisy_by_user.values())} 条")
    log_with_timestamp(f"  • 缓存文件: {cache_files} 个")
    log_with_timestamp(f"  • 缓存大小: {cache_dir_size:.1f} MB")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
