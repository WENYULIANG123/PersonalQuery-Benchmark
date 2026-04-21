#!/usr/bin/env python3
"""
快速全量评估脚本 - 支持多检索器 + ACL/CCOMP 分组交叉对比
包括: bge, e5, minilm, star, gritlm (密集) + bm25 (稀疏)
包含 ACL/CCOMP 混淆因素分析 (Check 1-4 + Bootstrap CI)
"""

# 完全离线模式 - 避免 HuggingFace 网络验证
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
import pickle
import json
import gzip
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from scipy import stats
import statsmodels.formula.api as smf

# 设置路径
sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery/12_retrieval')

# ============ 配置加载 ============
from config import get_category_config, get_retriever_config

CATEGORY_NAME = "Pet_Supplies"
CAT_CONFIG = get_category_config(CATEGORY_NAME)

CACHE_DIR = CAT_CONFIG['retriever_cache_dir']
QUERY_CACHE_BASE_DIR = CAT_CONFIG['query_cache_dir']
QUERY_TYPES = ['correct', 'noisy']  # 两种查询类型
QUERY_CATEGORIES = ['acl', 'ccomp']  # 两种查询类别
ACL_QUERIES_FILE = CAT_CONFIG['query_file']
CCOMP_QUERIES_FILE = CAT_CONFIG['query_file']
OUTPUT_DIR = CAT_CONFIG['output_dir']
META_FILE = CAT_CONFIG['corpus_file']

RETRIEVER_CONFIG = get_retriever_config()
RETRIEVERS = RETRIEVER_CONFIG['retrievers']
DENSE_RETRIEVERS = RETRIEVER_CONFIG['dense_retrievers']

# IDF 分层配置
IDF_BINS = [(2.5, 3.5), (3.5, 4.5), (4.5, 5.0), (5.0, float('inf'))]
IDF_BIN_LABELS = RETRIEVER_CONFIG['idf_bin_labels']

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============ 缓存完整性检查 ============
def validate_cache() -> bool:
    """检查缓存目录中的文件是否完整且有效"""
    log("\n检查缓存完整性...")

    if not os.path.exists(CACHE_DIR):
        log(f"  错误: 缓存目录不存在: {CACHE_DIR}")
        return False

    issues = []

    # 检查密集检索器文件及数据完整性
    for retriever in DENSE_RETRIEVERS:
        # 查找该检索器的所有版本
        # 文件格式: {retriever}_{hash}_{suffix}.npy/.pkl
        # hash 本身可能包含下划线，所以需要从后往前推断
        retriever_files = {}
        for f in os.listdir(CACHE_DIR):
            if not (f.startswith(f'{retriever}_') and f.endswith(('.npy', '.pkl'))):
                continue

            # 提取 suffix（最后一部分）
            suffix = f.rsplit('_', 1)[-1]

            # 提取 hash（去掉 retriever_ 前缀和 suffix 后缀）
            # 例如: bge_457d1871f380782c05a5d94e656fef2c_embeddings.npy
            # -> 去掉前缀 retriever_ 和后缀 _embeddings.npy
            middle = f[len(f'{retriever}_'):]
            hash_id = middle[:-len(suffix) - 1]  # -1 for the underscore before suffix

            if hash_id not in retriever_files:
                retriever_files[hash_id] = set()
            retriever_files[hash_id].add(suffix)

        if not retriever_files:
            issues.append(f"  缺失: {retriever} 检索器缓存文件")
            continue

        # 检查每个版本是否完整
        for hash_id, suffixes in retriever_files.items():
            required_files = ['embeddings.npy', 'doc_ids.pkl', 'config.pkl', 'metadata.pkl']
            for suffix in required_files:
                full_file = f"{retriever}_{hash_id}_{suffix}"
                file_path = os.path.join(CACHE_DIR, full_file)
                if suffix not in suffixes:
                    issues.append(f"  缺失: {retriever} ({hash_id[:8]}...) - {suffix}")
                elif os.path.getsize(file_path) == 0:
                    issues.append(f"  空文件: {full_file}")

            # 验证 embeddings 和 doc_ids 数量是否匹配
            embeddings_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_embeddings.npy")
            doc_ids_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_doc_ids.pkl")
            if os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
                try:
                    embeddings = np.load(embeddings_path, mmap_mode='r')
                    n_embeddings = embeddings.shape[0]
                    with open(doc_ids_path, 'rb') as f:
                        doc_ids = pickle.load(f)
                    n_doc_ids = len(doc_ids)

                    if n_embeddings != n_doc_ids:
                        issues.append(f"  数据不一致: {retriever} ({hash_id[:8]}...) - embeddings数量({n_embeddings}) != doc_ids数量({n_doc_ids})")

                    # 检查 doc_ids 是否有重复
                    if len(doc_ids) != len(set(doc_ids)):
                        duplicates = len(doc_ids) - len(set(doc_ids))
                        issues.append(f"  数据错误: {retriever} ({hash_id[:8]}...) - doc_ids中有 {duplicates} 个重复项")

                    log(f"  {retriever} ({hash_id[:8]}...): embeddings={n_embeddings}, doc_ids={n_doc_ids}")
                except Exception as e:
                    issues.append(f"  验证失败: {retriever} ({hash_id[:8]}...) - {str(e)}")

    # 检查 BM25 文件
    bm25_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('bm25_') and f.endswith('.pkl')]
    if not bm25_files:
        issues.append("  缺失: bm25 检索器缓存文件")
    else:
        for f in bm25_files:
            file_path = os.path.join(CACHE_DIR, f)
            if os.path.getsize(file_path) == 0:
                issues.append(f"  空文件: {f}")
            else:
                # 验证 BM25 数据可加载
                try:
                    with open(file_path, 'rb') as fp:
                        bm25 = pickle.load(fp)
                    # 检查 BM25 是否有 search 方法
                    if not hasattr(bm25, 'search'):
                        issues.append(f"  数据错误: {f} - BM25对象缺少search方法")
                    else:
                        log(f"  bm25 ({f.split('_')[1][:8]}...): 可正常加载")
                except Exception as e:
                    issues.append(f"  验证失败: {f} - {str(e)}")

    # 检查查询缓存
    log("  检查查询缓存...")
    for query_type in QUERY_TYPES:
        query_cache_dir = os.path.join(QUERY_CACHE_BASE_DIR, f'persona_{query_type}_query')
        if not os.path.exists(query_cache_dir):
            issues.append(f"  缺失: 查询缓存目录 persona_{query_type}_query")
            continue

        for retriever in RETRIEVERS:
            cache_file = os.path.join(query_cache_dir, f'{retriever}__persona_{query_type}_cache.pkl')
            if not os.path.exists(cache_file):
                issues.append(f"  缺失: {retriever} ({query_type}) 查询缓存")
            else:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    n_users = len(cache_data)
                    log(f"  {retriever} ({query_type}): {n_users} 用户")
                except Exception as e:
                    issues.append(f"  验证失败: {retriever} ({query_type}) - {str(e)}")

    if issues:
        log("  缓存完整性检查未通过:")
        for issue in issues:
            log(issue)
        return False

    log("  缓存完整性检查通过 ✓")
    return True

# ============ ACL/CCOMP Paired Analysis ============
# 模块级变量
UNIQUE_LEVELS = [0, 1, 2, 3]  # 默认值，会在 load_paired_queries 时更新
GROUP_FIELD = 'ccomp'  # 默认值，会在 load_paired_queries 时更新


def load_paired_queries(query_category: str = 'acl', filter_same_level: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载 ACL 和 CCOMP 查询数据用于配对分析，返回 (acl_df, ccomp_df)

    每条记录包含: user_id, asin, level, retriever, P@10, query, word_count, mean_idf
    """
    global UNIQUE_LEVELS

    # ACL 查询文件
    with open(ACL_QUERIES_FILE, 'r') as f:
        acl_data = json.load(f)

    # CCOMP 查询文件
    with open(CCOMP_QUERIES_FILE, 'r') as f:
        ccomp_data = json.load(f)

    # 如果启用筛选，先找出 ACL 和 CCOMP level 一致的 (user_id, asin) 对
    valid_pairs = set()
    if filter_same_level:
        valid_pairs = _find_same_level_pairs(acl_data)

    # 动态收集所有唯一的 level 值
    acl_levels = set()
    ccomp_levels = set()

    acl_records = []
    for item in acl_data:
        if 'queries' in item:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            for q in item['queries']:
                level = q.get('acl', 0)
                acl_levels.add(level)
                query_text = q.get('filled_query', '') or q.get('generated_query', '') or q.get('query', '')
                word_count = q.get('word_count', 0)
                if query_text and asin:
                    acl_records.append({
                        'user_id': user_id,
                        'asin': asin,
                        'level': level,
                        'query': query_text,
                        'word_count': word_count,
                    })
        elif 'acl_query' in item:
            # 新格式: acl_query 是嵌套对象
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            acl_query = item.get('acl_query', {})
            level = acl_query.get('level', 0)
            acl_levels.add(level)
            query_text = acl_query.get('query', '')
            word_count = acl_query.get('word_count', 0)
            if query_text and asin:
                acl_records.append({
                    'user_id': user_id,
                    'asin': asin,
                    'level': level,
                    'query': query_text,
                    'word_count': word_count,
                })
        else:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            level = item.get('target_acl', item.get('acl', 0))
            acl_levels.add(level)
            query_text = item.get('filled_query', '') or item.get('generated_query', '') or item.get('query', '')
            word_count = item.get('word_count') or 0
            if query_text and asin:
                acl_records.append({
                    'user_id': user_id,
                    'asin': asin,
                    'level': level,
                    'query': query_text,
                    'word_count': word_count,
                })

    ccomp_records = []
    for item in ccomp_data:
        if 'queries' in item:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            for q in item['queries']:
                level = q.get('ccomp', 0)
                ccomp_levels.add(level)
                query_text = q.get('filled_query', '') or q.get('generated_query', '') or q.get('query', '')
                word_count = q.get('word_count', 0)
                if query_text and asin:
                    ccomp_records.append({
                        'user_id': user_id,
                        'asin': asin,
                        'level': level,
                        'query': query_text,
                        'word_count': word_count,
                    })
        elif 'ccomp_query' in item:
            # 新格式: ccomp_query 是嵌套对象
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            ccomp_query = item.get('ccomp_query', {})
            level = ccomp_query.get('level', 0)
            ccomp_levels.add(level)
            query_text = ccomp_query.get('query', '')
            word_count = ccomp_query.get('word_count', 0)
            if query_text and asin:
                ccomp_records.append({
                    'user_id': user_id,
                    'asin': asin,
                    'level': level,
                    'query': query_text,
                    'word_count': word_count,
                })
        else:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            level = item.get('target_ccomp', item.get('ccomp', 0))
            ccomp_levels.add(level)
            query_text = item.get('filled_query', '') or item.get('generated_query', '') or item.get('query', '')
            word_count = item.get('word_count') or 0
            if query_text and asin:
                ccomp_records.append({
                    'user_id': user_id,
                    'asin': asin,
                    'level': level,
                    'query': query_text,
                    'word_count': word_count,
                })

    UNIQUE_LEVELS = sorted(acl_levels | ccomp_levels)
    acl_df = pd.DataFrame(acl_records)
    ccomp_df = pd.DataFrame(ccomp_records)

    return acl_df, ccomp_df


def compute_query_idf_simple(query: str, word_idf: Dict[str, float]) -> float:
    """计算查询的平均IDF"""
    words = query.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]
    return np.mean(idf_values)

# ============ 评估指标计算 ============
def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f'P@{k}'] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f'N@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'MR@{k}'] = 1.0 / rank
        else:
            metrics[f'N@{k}'] = 0.0
            metrics[f'MR@{k}'] = 0.0
        metrics[f'H@{k}'] = 1.0 if relevant_asin in top_k else 0.0
    return metrics

def compute_average_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f'P@{k}'] = np.mean([m.get(f'P@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'N@{k}'] = np.mean([m.get(f'N@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'MR@{k}'] = np.mean([m.get(f'MR@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'H@{k}'] = np.mean([m.get(f'H@{k}', 0.0) for m in all_metrics])
    return avg_metrics

# ============ Bootstrap CI ============
def compute_bootstrap_ci(all_metrics: List[Dict], k_values: List[int], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """计算Bootstrap置信区间"""
    np.random.seed(42)
    n_samples = len(all_metrics)
    if n_samples < 2:
        return {}

    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    metric_keys = [f'P@{k}' for k in k_values] + [f'N@{k}' for k in k_values] + [f'MR@{k}' for k in k_values] + [f'H@{k}' for k in k_values]

    bootstrap_results = {}
    for key in metric_keys:
        values = np.array([m.get(key, 0.0) for m in all_metrics])
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_values = values[sample_indices]
            bootstrapped_means.append(np.mean(sample_values))
        bootstrapped_means = np.array(bootstrapped_means)
        bootstrap_results[key] = {
            'mean': np.mean(bootstrapped_means),
            'std': np.std(bootstrapped_means),
            'ci_lower': np.percentile(bootstrapped_means, lower_percentile),
            'ci_upper': np.percentile(bootstrapped_means, upper_percentile),
        }
    return bootstrap_results

def print_bootstrap_ci_table(all_results: List[Dict], k_values: List[int]):
    log("\n" + "=" * 100)
    log("Bootstrap CI (95%) - P@10")
    log("=" * 100)

    header = f"{'检索器':<12} {GROUP_FIELD.upper():<10} {'Mean':<10} {'Std':<10} {'CI Lower':<12} {'CI Upper':<12}"
    log(header)
    log("-" * 100)

    for r in all_results:
        retriever = r['retriever']
        for ccomp in UNIQUE_LEVELS:
            ci = r['bootstrap_ci'].get(ccomp, {}).get('P@10', {})
            if ci:
                row = f"{retriever:<12} {GROUP_FIELD.upper()}{ccomp}   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}"
                log(row)

    log("-" * 100)
    # 总体 CI
    for r in all_results:
        ci = r['bootstrap_ci'].get('overall', {}).get('P@10', {})
        if ci:
            log(f"{r['retriever']:<12} overall   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}")

# ============ 数据加载 ============
def load_dense_retriever(retriever_name: str) -> Tuple[np.ndarray, List[str], int]:
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith(f'{retriever_name}_') and f.endswith('_embeddings.npy'):
            embeddings_path = os.path.join(CACHE_DIR, f)
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} embeddings not found")

    log(f"  [{retriever_name}] 加载: {os.path.getsize(embeddings_path)/1024/1024:.1f} MB")
    mmap_array = np.load(embeddings_path, mmap_mode='r')
    embeddings = mmap_array[:].copy()

    doc_ids_path = embeddings_path.replace('_embeddings.npy', '_doc_ids.pkl')
    with open(doc_ids_path, 'rb') as f:
        doc_ids = pickle.load(f)

    return embeddings, doc_ids, embeddings.shape[1]

def load_bm25_retriever():
    """加载 BM25 检索器"""
    bm25_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(CACHE_DIR, f)
            break
    if bm25_path is None:
        raise FileNotFoundError("BM25 cache not found")

    log(f"  [bm25] 加载: {os.path.getsize(bm25_path)/1024/1024:.1f} MB")
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25

def load_query_cache(retriever_name: str, query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    """加载查询缓存

    Args:
        retriever_name: 检索器名称
        query_type: 查询类型 ('correct' 或 'noisy')
        query_category: 查询类别 ('acl' 或 'ccomp')
    """
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'{retriever_name}__{query_category}_{query_type}_cache.pkl'
    )
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def load_bm25_query_cache(query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    """加载 BM25 查询缓存

    Args:
        query_type: 查询类型 ('correct' 或 'noisy')
        query_category: 查询类别 ('acl' 或 'ccomp')

    Returns:
        Dict: 查询缓存，key 为 query 字符串，value 为检索结果 [(asin, score), ...]
    """
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'bm25__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def _find_same_level_pairs(data: list) -> set:
    """找出 ACL 和 CCOMP level 一致的 (user_id, asin) 对"""
    pairs = {}  # (user_id, asin) -> {acl_level, ccomp_level}
    for item in data:
        user_id = item.get('user_id', '')
        asin = item.get('asin', '')
        key = (user_id, asin)

        acl_query = item.get('acl_query', {})
        ccomp_query = item.get('ccomp_query', {})

        if isinstance(acl_query, dict) and isinstance(ccomp_query, dict):
            acl_level = acl_query.get('level', -1)
            ccomp_level = ccomp_query.get('level', -1)

            if key not in pairs:
                pairs[key] = {'acl_level': acl_level, 'ccomp_level': ccomp_level}

            # 只有 level 完全一致才保留
            if acl_level != ccomp_level:
                pairs[key]['invalid'] = True

    # 只返回 level 一致的对
    valid_pairs = set()
    for key, info in pairs.items():
        if not info.get('invalid', False):
            valid_pairs.add(key)
    return valid_pairs


def load_user_queries(query_type: str = 'correct', query_category: str = 'acl', filter_same_level: bool = True) -> Tuple[Dict[str, List[Dict]], Dict[str, int], List[Tuple[int, float, float]]]:
    """加载用户查询，每个查询项包含word_count和group_ratio（POS ratio代理）

    Args:
        query_type: 查询类型 ('correct' 使用 filled_query, 'noisy' 使用 noisy_query)
        query_category: 查询类别 ('acl' 或 'ccomp')
        filter_same_level: 是否只保留 ACL 和 CCOMP level 一致的用户（默认 True）
    """
    global GROUP_FIELD, UNIQUE_LEVELS

    # 根据 query_category 选择查询文件
    queries_file = ACL_QUERIES_FILE if query_category == 'acl' else CCOMP_QUERIES_FILE
    with open(queries_file, 'r') as f:
        data = json.load(f)
    user_queries = {}
    user_to_group = {}
    all_query_metadata = []  # (user_idx, word_count, group_ratio)
    idx = 0

    # query_category 直接决定 GROUP_FIELD
    GROUP_FIELD = query_category

    # 动态收集所有唯一的 group 值
    group_values = set()
    for item in data:
        if 'queries' in item:
            for q in item['queries']:
                gv = q.get(GROUP_FIELD, 0)
                group_values.add(gv)
        elif f'{query_category}_query' in item:
            # 新嵌套格式: acl_query 或 ccomp_query 对象
            query_obj = item.get(f'{query_category}_query', {})
            gv = query_obj.get('level', 0)
            group_values.add(gv)
        else:
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            group_values.add(gv)
    UNIQUE_LEVELS = sorted(group_values)

    # 如果启用筛选，先找出 ACL 和 CCOMP level 一致的 (user_id, asin) 对
    valid_pairs = set()
    if filter_same_level:
        # 只加载一次数据来找出有效对
        valid_pairs = _find_same_level_pairs(data)

    # 根据 query_type 确定查询文本字段
    # correct: correct_query (ground truth) / filled_query / generated_query / query
    # noisy: noisy_query
    def get_query_text(q):
        if query_type == 'noisy':
            # noisy 模式只返回有明确 noisy_query 字段的查询
            return q.get('noisy_query', '')
        else:
            # 对于 ground truth (acl=0), 使用 correct_query
            # 对于其他版本, 使用 filled_query
            if q.get('is_ground_truth', False):
                return q.get('correct_query', '') or q.get('filled_query', '') or q.get('query', '')
            return q.get('filled_query', '') or q.get('generated_query', '') or q.get('query', '')

    # 支持两种格式：
    # 1. 新嵌套格式：[{"user_id": ..., "asin": ..., "queries": [{filled_query, acl/ccomp, word_count}, ...]}]
    # 2. 旧平铺格式：[{"user_id": ..., "filled_query": ..., "target_acl/ccomp": ...}]
    items = data if isinstance(data, list) else []

    for item in items:
        # 新嵌套格式 (queries 数组)
        if 'queries' in item:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            for q in item['queries']:
                query_text = get_query_text(q)
                gv = q.get(GROUP_FIELD, 0)
                word_count = q.get('word_count', 0)
                if user_id not in user_queries:
                    user_queries[user_id] = []
                    user_to_group[user_id] = gv
                if query_text and asin:
                    user_queries[user_id].append({
                        'query': query_text,
                        'asin': asin,
                        'word_count': word_count,
                        f'{GROUP_FIELD}_ratio': 0.0,
                        f'{GROUP_FIELD}': gv
                    })
                    all_query_metadata.append((idx, word_count, 0.0))
                    idx += 1
        # 新嵌套格式 (acl_query/ccomp_query 对象)
        elif f'{query_category}_query' in item:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            query_obj = item.get(f'{query_category}_query', {})
            query_text = query_obj.get('query', '')
            gv = query_obj.get('level', 0)
            word_count = query_obj.get('word_count', 0)
            if user_id not in user_queries:
                user_queries[user_id] = []
                user_to_group[user_id] = gv
            if query_text and asin:
                user_queries[user_id].append({
                    'query': query_text,
                    'asin': asin,
                    'word_count': word_count,
                    f'{GROUP_FIELD}_ratio': 0.0,
                    f'{GROUP_FIELD}': gv
                })
                all_query_metadata.append((idx, word_count, 0.0))
                idx += 1
        else:
            # 旧平铺格式
            user_id = item.get('user_id')
            if not user_id:
                continue
            asin = item.get('asin', '')
            # 过滤：只保留 ACL 和 CCOMP level 一致的对
            if filter_same_level and (user_id, asin) not in valid_pairs:
                continue
            query_text = get_query_text(item)
            asin = item.get('asin', '')
            gv = item.get(f'target_{GROUP_FIELD}', item.get(GROUP_FIELD, 0))
            word_count = item.get('word_count') or 0
            group_ratio = item.get('persona', {}).get(f'{GROUP_FIELD}_sentence_ratio', 0.0)
            if user_id not in user_queries:
                user_queries[user_id] = []
                user_to_group[user_id] = gv
            if query_text and asin:
                user_queries[user_id].append({
                    'query': query_text,
                    'asin': asin,
                    'word_count': word_count,
                    f'{GROUP_FIELD}_ratio': group_ratio,
                    f'{GROUP_FIELD}': gv
                })
                all_query_metadata.append((idx, word_count, group_ratio))
                idx += 1
    return user_queries, user_to_group, all_query_metadata

def build_word_idf_dict(meta_file: str, sample_size: int = 50000) -> Dict[str, float]:
    """从商品元数据语料库构建词的IDF字典（采样版本加速）"""
    import gzip
    word_doc_freq = defaultdict(int)
    total_sampled = 0

    log(f"Building word IDF from corpus (sampling {sample_size} docs)...")
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                item = json.loads(line)
                # 从 title + brand + description 提取词
                text = ' '.join(filter(None, [
                    item.get('title', ''),
                    item.get('brand', ''),
                    ' '.join(item.get('description', []))
                ])).lower()
                words = set(text.split())
                for w in words:
                    if len(w) > 1:  # 过滤单字符
                        word_doc_freq[w] += 1
                total_sampled += 1
            except Exception:
                continue

    N = total_sampled
    word_idf = {}
    for w, df in word_doc_freq.items():
        word_idf[w] = np.log(N / (df + 1))  # +1 平滑

    # 也计算字符ngram（用于商品品牌等专有名词）
    for w, df in word_doc_freq.items():
        if len(w) >= 4 and df < 10:
            # 罕见词给高IDF
            word_idf[w] = max(word_idf.get(w, 0), np.log(N / 10))

    log(f"  IDF vocabulary: {len(word_idf)} words, {total_sampled} docs sampled")
    return word_idf


def compute_query_idf(query_text: str, word_idf: Dict[str, float]) -> float:
    """计算查询的平均IDF（使用预计算的词IDF）"""
    words = query_text.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]  # 未知词给中等IDF=5
    return np.mean(idf_values)


def compute_idf(queries: List[str], doc_count: int) -> float:
    """计算查询的平均IDF（简化版：使用词频）"""
    from collections import Counter
    words = ' '.join(queries).lower().split()
    word_freq = Counter(words)
    if not word_freq:
        return 0.0
    # 平均 IDF = log(N / df)，这里简化为平均词频的倒数
    avg_df = sum(word_freq.values()) / len(word_freq) if word_freq else 1
    return np.log(doc_count / avg_df + 1)

def compute_oracle_random_baseline(relevant_asin: str, doc_ids: List[str], n_trials: int = 100, seed: int = 42) -> Dict:
    """计算oracle-aware随机基线：给定相关文档在随机位置时的期望性能"""
    np.random.seed(seed)
    n_docs = len(doc_ids)
    if relevant_asin not in doc_ids:
        return {'P@10': 0.0, 'N@10': 0.0}
    rel_idx = doc_ids.index(relevant_asin)
    p10_list = []
    n10_list = []
    for _ in range(n_trials):
        # 随机打乱位置（保留relevant_asin在某个随机位置）
        random_pos = np.random.randint(0, n_docs)
        top10_positions = list(range(random_pos, min(random_pos + 10, n_docs)))
        if rel_idx in top10_positions:
            rank = top10_positions.index(rel_idx) + 1
            p10_list.append(1.0)
            n10_list.append(1.0 / np.log2(rank + 1))
        else:
            p10_list.append(0.0)
            n10_list.append(0.0)
    return {'P@10': np.mean(p10_list), 'N@10': np.mean(n10_list)}

# ============ 搜索器 ============
class DenseSearcher:
    """密集检索器搜索器 (GPU 矩阵乘法 + 余弦相似度)"""
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str], retriever_name: str):
        self.doc_ids = doc_ids
        self.retriever_name = retriever_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 归一化 doc embeddings 以支持余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            return []
        query_tensor = torch.from_numpy(np.array(query_embeddings)).float().to(self.device)
        # 归一化 query embeddings
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
        # 余弦相似度 = 归一化点积
        scores = torch.mm(query_tensor, self.embeddings_tensor.T)
        results = []
        for i in range(len(query_embeddings)):
            top_scores, top_indices = torch.topk(scores[i], min(top_k, len(self.doc_ids)))
            results.append([(self.doc_ids[idx.item()], top_scores[j].item()) for j, idx in enumerate(top_indices)])
        return results

class BM25Searcher:
    """BM25 搜索器 (文本搜索)"""
    def __init__(self, bm25_retriever):
        self.bm25 = bm25_retriever

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        results = []
        for query in queries:
            # BM25 search returns [(asin, score), ...]
            search_results = self.bm25.search(query, top_k=top_k)
            results.append(search_results)
        return results

# ============ 评估 ============
def evaluate_dense_retriever(retriever_name: str, user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    global GROUP_FIELD, UNIQUE_LEVELS
    # 确保 GROUP_FIELD 与 query_category 一致
    GROUP_FIELD = query_category
    # 清理 CUDA 缓存，避免 cuBLAS 上下文冲突
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    log(f"\n{'='*60}")
    log(f"检索器: {retriever_name.upper()} (密集) - {query_category.upper()}/{query_type.upper()}")
    log(f"{'='*60}")

    embeddings, doc_ids, dim = load_dense_retriever(retriever_name)
    query_cache = load_query_cache(retriever_name, query_type, query_category)
    searcher = DenseSearcher(embeddings, doc_ids, retriever_name)

    matched_users = [uid for uid in user_queries.keys() if uid in query_cache]
    log(f"  匹配用户: {len(matched_users)}")

    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []
    eval_start = time.time()

    # 分组统计: word_count bins 和 group_ratio bins
    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']

    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    # IDF × group 交叉分组: {(idf_label, group): [metrics]}
    idf_group_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]
        cached_queries = query_cache[user_id]
        group = user_to_group.get(user_id, 0)

        query_embeddings = []
        query_asins = []
        query_texts = []
        query_word_counts = []
        query_group_ratios = []
        query_idf_values = []
        query_groups = []  # 每条查询自己的group值

        for q in queries:
            query_text = q['query']
            relevant_asin = q['asin']
            word_count = q.get('word_count', 0)
            group_ratio = q.get(f'{GROUP_FIELD}_ratio', 0.0)
            q_group = q.get(GROUP_FIELD, 0)  # 每条查询自己的group值
            if query_text in cached_queries:
                query_embeddings.append(cached_queries[query_text])
                query_asins.append(relevant_asin)
                query_texts.append(query_text)
                query_word_counts.append(word_count)
                query_group_ratios.append(group_ratio)
                query_groups.append(q_group)
                # 计算 IDF
                q_idf = compute_query_idf(query_text, word_idf) if word_idf else 0.0
                query_idf_values.append(q_idf)

        if not query_embeddings:
            continue

        results = searcher.search_batch(query_embeddings, top_k=max(k_values))

        for i, (retrieved, relevant_asin) in enumerate(zip(results, query_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            group = query_groups[i]  # 使用该查询自己的group值
            all_metrics.append(metrics)
            group_groups[group].append(metrics)

            # 记录每条 query 的原始数据（用于 OLS 回归和 Paired Difference 分析）
            all_query_records.append({
                'user_id': user_id,
                'asin': relevant_asin,
                f'{GROUP_FIELD}': group,
                'mean_idf': query_idf_values[i],
                'query_length': query_word_counts[i],
                f'{GROUP_FIELD}_ratio': query_group_ratios[i],
                'p_at1': float(metrics.get('P@1', 0.0)),
                'p_at3': float(metrics.get('P@3', 0.0)),
                'p_at5': float(metrics.get('P@5', 0.0)),
                'p_at10': float(metrics.get('P@10', 0.0)),
                'n_at10': float(metrics.get('N@10', 0.0)),
                'mrr_at10': float(metrics.get('MR@10', 0.0)),
                'hit_at10': float(metrics.get('H@10', 0.0)),
            })

            # word_count 分组
            wc = query_word_counts[i]
            for (low, high), label in zip(word_bins, word_bin_labels):
                if low <= wc < high:
                    word_count_groups[label].append(metrics)
                    break

            # group_ratio (POS proxy) 分组
            cr = query_group_ratios[i]
            for (low, high), label in zip(ratio_bins, ratio_bin_labels):
                if low <= cr < high:
                    group_ratio_groups[label].append(metrics)
                    break

            # IDF 分组
            q_idf = query_idf_values[i]
            for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
                if low <= q_idf < high:
                    idf_bin_groups[label].append(metrics)
                    idf_group_cross[(label, group)].append(metrics)
                    break

        if (user_idx + 1) % 100 == 0:
            elapsed = time.time() - eval_start
            log(f"    进度: {user_idx+1}/{len(matched_users)} ({100*(user_idx+1)/len(matched_users):.1f}%)")

    eval_time = time.time() - eval_start
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 group_ratio 分组统计
    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × group 交叉分组统计
    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    # 显式释放 GPU 内存
    del embeddings
    del searcher
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        'retriever': retriever_name, 'dim': dim, 'type': 'dense', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }

def evaluate_bm25_retriever(user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    global GROUP_FIELD, UNIQUE_LEVELS
    # 确保 GROUP_FIELD 与 query_category 一致
    GROUP_FIELD = query_category
    log(f"\n{'='*60}")
    log(f"检索器: BM25 (稀疏) - {query_category.upper()}/{query_type.upper()}")
    log(f"{'='*60}")

    bm25 = load_bm25_retriever()
    searcher = BM25Searcher(bm25)

    matched_users = list(user_queries.keys())
    log(f"  用户数: {len(matched_users)}")

    eval_start = time.time()

    # ========== 优化：批量搜索所有查询 ==========
    # 先收集所有查询和元数据
    all_query_texts = []
    all_query_asins = []
    all_query_users = []
    all_query_word_counts = []
    all_query_group_ratios = []
    all_query_groups = []
    all_query_idf_values = []
    user_query_ranges = []  # (start_idx, end_idx) for each user

    for user_id in matched_users:
        queries = user_queries[user_id]
        start_idx = len(all_query_texts)
        for q in queries:
            all_query_texts.append(q['query'])
            all_query_asins.append(q['asin'])
            all_query_users.append(user_id)
            all_query_word_counts.append(q.get('word_count', 0))
            all_query_group_ratios.append(q.get(f'{GROUP_FIELD}_ratio', 0.0))
            all_query_groups.append(q.get(GROUP_FIELD, 0))
            all_query_idf_values.append(compute_query_idf(q['query'], word_idf) if word_idf else 0.0)
        end_idx = len(all_query_texts)
        user_query_ranges.append((start_idx, end_idx))

    log(f"  总查询数: {len(all_query_texts)}")

    # 尝试加载查询缓存
    log(f"  尝试加载查询缓存...")
    query_cache = load_bm25_query_cache(query_type, query_category)

    if query_cache is not None:
        # 使用查询缓存，直接查表获取结果
        log(f"  使用查询缓存 (共 {len(query_cache)} 条记录)...")
        cache_hit_start = time.time()
        all_results = []
        cache_hits = 0
        for query_text in all_query_texts:
            if query_text in query_cache:
                all_results.append(query_cache[query_text])
                cache_hits += 1
            else:
                # 缓存未命中，在线计算
                all_results.append(searcher.bm25.search(query_text, top_k=max(k_values)))
        cache_time = time.time() - cache_hit_start
        log(f"  缓存查找完成: {cache_hits}/{len(all_query_texts)} 命中, 耗时: {cache_time:.1f}秒")
    else:
        # 无缓存，在线搜索
        log(f"  开始批量搜索...")
        batch_start = time.time()
        all_results = searcher.search_batch(all_query_texts, top_k=max(k_values))
        batch_time = time.time() - batch_start
        log(f"  批量搜索完成，耗时: {batch_time:.1f}秒")

    # 分组统计
    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []

    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_group_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    # 处理每个查询的结果
    for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_query_asins)):
        retrieved_asins = [r[0] for r in retrieved]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        group = all_query_groups[i]
        all_metrics.append(metrics)
        group_groups[group].append(metrics)

        # 记录每条 query 的原始数据
        all_query_records.append({
            'user_id': all_query_users[i],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': group,
            'mean_idf': all_query_idf_values[i],
            'query_length': all_query_word_counts[i],
            f'{GROUP_FIELD}_ratio': all_query_group_ratios[i],
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # word_count 分组
        wc = all_query_word_counts[i]
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= wc < high:
                word_count_groups[label].append(metrics)
                break

        # group_ratio 分组
        cr = all_query_group_ratios[i]
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= cr < high:
                group_ratio_groups[label].append(metrics)
                break

        # IDF 分组
        q_idf = all_query_idf_values[i]
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, group)].append(metrics)
                break

        if (i + 1) % 100 == 0:
            elapsed = time.time() - eval_start
            log(f"    进度: {i+1}/{len(all_results)} ({100*(i+1)/len(all_results):.1f}%)")

    eval_time = time.time() - eval_start
    log(f"  评估完成，总耗时: {eval_time:.1f}秒")
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 group_ratio 分组统计
    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × group 交叉分组统计
    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': 'bm25', 'dim': 0, 'type': 'sparse', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }

# ============ 表格打印 ============


def print_summary_table_wide(all_results: List[Dict], query_type: str = 'CORRECT'):
    """宽格式汇总表：每行一个检索器，列是 (指标 × 分组) 的所有组合

    格式：
    检索器 | P@1_ACL0 | P@1_ACL1 | ... | P@10_ACL3 | 平均
    """
    global GROUP_FIELD
    log(f"\n{'='*100}")
    log(f"汇总表（宽格式）| {query_type} | 每行一个检索器，列为 (指标×分组)")
    log(f"{'='*100}")

    metrics = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']
    groups = [f"{GROUP_FIELD.upper()}{g}" for g in UNIQUE_LEVELS]

    # 构建表头 - 使用固定宽度列（每列12字符，右对齐）
    COL_W = 12
    header = f"{'检索器':<10}"
    for m in metrics:
        for g in groups:
            label = f"{m}_{g}"
            header += f" {label:>{COL_W}}"
        header += f" {m:>{COL_W}}"
    header += f" {'总平均':>{COL_W}}"
    log(header)
    log("-" * 100)

    # 构建数据
    retrievers = sorted(set(r['retriever'] for r in all_results))

    for retriever in retrievers:
        r = next((x for x in all_results if x['retriever'] == retriever), None)
        if not r:
            continue

        row = f"{retriever:<10}"
        all_vals = []

        for m in metrics:
            metric_vals = []
            for group in UNIQUE_LEVELS:
                val = r['group_metrics'].get(group, {}).get(m, 0.0)
                row += f" {val:>{COL_W}.4f}"
                metric_vals.append(val)
                all_vals.append(val)
            metric_avg = sum(metric_vals) / len(metric_vals) if metric_vals else 0.0
            row += f" {metric_avg:>{COL_W}.4f}"
            all_vals.append(metric_avg)

        total_avg = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row += f" {total_avg:>{COL_W}.4f}"
        log(row)

    log("-" * 100)


# ============ 实验 1: Paired t-test on ACL_k − CCOMP_k ============
def run_paired_ttest_analysis(all_results_by_category_and_type: Dict, word_idf: Dict[str, float]):
    """实验 1: Paired t-test on ACL_k − CCOMP_k

    对每个 (retriever, level, domain) 跑配对 t-test
    输出格式: Retriever | Level | Domain | Mean Diff | 95% CI | p-value
    """
    log("\n" + "=" * 100)
    log("实验 1: Paired t-test on ACL_k − CCOMP_k (P@10)")
    log("=" * 100)
    log("按 (user_id, asin, level) 配对，计算 diff = P@10(ACL_k) - P@10(CCOMP_k)")
    log("")

    results_table = []

    for (category, qt), results in all_results_by_category_and_type.items():
        if qt != 'correct':  # 只用 correct 查询
            continue

        domain = category  # 'acl' 或 'ccomp'

        for r in results:
            retriever = r['retriever']
            query_records = r.get('all_query_records', [])

            if not query_records:
                continue

            # 按 level 分组
            level_groups = defaultdict(list)
            for rec in query_records:
                level = rec.get('acl', rec.get('ccomp', 0))
                level_groups[level].append(rec)

            for level, records in level_groups.items():
                # 同一 retriever 下，按 (user_id, asin) 配对
                # 由于我们是在同一个结果文件里，需要找对应的 ACL 和 CCOMP 记录
                # 这里我们比较同一 level 下 ACL vs CCOMP 的 P@10
                pass  # 需要 ACL 和 CCOMP 分别的结果来配对

    # 重新组织数据：分别收集 ACL 和 CCOMP 的结果
    acl_results_by_retriever = {}  # {retriever: {level: {key: p10}}}
    ccomp_results_by_retriever = {}

    for (category, qt), results in all_results_by_category_and_type.items():
        if qt != 'correct':
            continue

        is_acl = (category == 'acl')

        for r in results:
            retriever = r['retriever']

            if is_acl:
                if retriever not in acl_results_by_retriever:
                    acl_results_by_retriever[retriever] = defaultdict(dict)
                target_dict = acl_results_by_retriever[retriever]
            else:
                if retriever not in ccomp_results_by_retriever:
                    ccomp_results_by_retriever[retriever] = defaultdict(dict)
                target_dict = ccomp_results_by_retriever[retriever]

            for rec in r.get('all_query_records', []):
                # ACL 查询的 level 字段是 'acl'，CCOMP 是 'ccomp'
                if is_acl:
                    level = rec.get('acl', 0)
                else:
                    level = rec.get('ccomp', 0)
                key = (rec.get('user_id', ''), rec.get('asin', ''))
                target_dict[level][key] = rec.get('p_at10', 0.0)

    # 对每个 (retriever, level) 跑配对 t-test
    log(f"\n{'Retriever':<12} {'Level':<8} {'N_pairs':<10} {'Mean_Diff':<12} {'95% CI':<18} {'p-value':<12} {'Sig':<6}")
    log("-" * 100)

    all_results = []

    for retriever in sorted(set(list(acl_results_by_retriever.keys()) + list(ccomp_results_by_retriever.keys()))):
        acl_levels = acl_results_by_retriever.get(retriever, {})
        ccomp_levels = ccomp_results_by_retriever.get(retriever, {})

        all_levels = sorted(set(list(acl_levels.keys()) + list(ccomp_levels.keys())))

        for level in all_levels:
            acl_dict = acl_levels.get(level, {})
            ccomp_dict = ccomp_levels.get(level, {})

            # 找共同 key
            common_keys = set(acl_dict.keys()) & set(ccomp_dict.keys())

            if len(common_keys) < 3:
                continue

            diffs = [acl_dict[k] - ccomp_dict[k] for k in common_keys]
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            # Bootstrap 95% CI
            np.random.seed(42)
            n_bootstrap = 1000
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(diffs, size=len(diffs), replace=True)
                boot_means.append(np.mean(sample))
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)

            # Paired t-test against 0
            t_stat, p_value = stats.ttest_1samp(diffs, 0)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            mean_str = f"{mean_diff:+.4f}"
            ci_str = f"[{ci_lower:+.2f}, {ci_upper:+.2f}]"
            p_str = f"{p_value:.3e}" if p_value < 0.001 else f"{p_value:.4f}"

            log(f"{retriever:<12} {level:<8} {len(common_keys):<10} {mean_str:<12} {ci_str:<18} {p_str:<12} {sig:<6}")

            all_results.append({
                'retriever': retriever,
                'level': level,
                'n_pairs': len(common_keys),
                'mean_diff': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                't_stat': t_stat,
                'sig': sig
            })

    # 汇总行
    if all_results:
        log("-" * 100)
        pooled_diffs = [r['mean_diff'] * r['n_pairs'] for r in all_results]
        total_n = sum(r['n_pairs'] for r in all_results)
        pooled_mean = sum(pooled_diffs) / total_n if total_n > 0 else 0

        # 简单的汇总结论
        sig_count = sum(1 for r in all_results if r['sig'])
        log(f"\n  汇总: {len(all_results)} 个 (retriever, level) 组合中，{sig_count} 个显著 (p<0.05)")

    return all_results


# ============ 实验 2: Query 长度和 IDF 的对称性验证 ============
def run_symmetry_check(acl_df: pd.DataFrame, ccomp_df: pd.DataFrame, word_idf: Dict[str, float]):
    """实验 2: 验证 ACL_k 和 CCOMP_k 的长度/IDF 对称性

    对每对 (ACL_k, CCOMP_k) query，计算：
    - len_diff = len_acl - len_ccomp
    - idf_diff = idf_acl - idf_ccomp

    期望: mean len_diff ≈ 0, mean idf_diff ≈ 0
    """
    log("\n" + "=" * 100)
    log("实验 2: Query 长度和 IDF 的对称性验证")
    log("=" * 100)
    log("目的: 验证 ACL_k 和 CCOMP_k 长度/IDF 是否可比（无系统性偏斜）")
    log("")

    # 计算 IDF
    log("  计算查询 IDF...")
    acl_df['mean_idf'] = acl_df['query'].apply(lambda q: compute_query_idf_simple(q, word_idf))
    ccomp_df['mean_idf'] = ccomp_df['query'].apply(lambda q: compute_query_idf_simple(q, word_idf))

    # 配对: 按 (user_id, asin, level)
    merged = pd.merge(
        acl_df, ccomp_df,
        on=['user_id', 'asin', 'level'],
        suffixes=('_acl', '_ccomp')
    )

    if merged.empty:
        log("  警告: 无法配对 ACL 和 CCOMP 查询，跳过对称性检验")
        return None

    log(f"  配对数: {len(merged)}")
    log("")

    # 计算差值
    merged['len_diff'] = merged['word_count_acl'] - merged['word_count_ccomp']
    merged['idf_diff'] = merged['mean_idf_acl'] - merged['mean_idf_ccomp']

    # 按 level 报告
    log(f"{'Level':<8} {'N':<10} {'Mean_len_diff':<14} {'Std_len':<10} {'Mean_idf_diff':<14} {'Std_idf':<10}")
    log("-" * 80)

    symmetry_results = []
    for level in sorted(merged['level'].unique()):
        subset = merged[merged['level'] == level]
        n = len(subset)

        mean_len_diff = subset['len_diff'].mean()
        std_len_diff = subset['len_diff'].std()
        mean_idf_diff = subset['idf_diff'].mean()
        std_idf_diff = subset['idf_diff'].std()

        log(f"{level:<8} {n:<10} {mean_len_diff:+14.2f} {std_len_diff:<10.2f} {mean_idf_diff:+14.4f} {std_idf_diff:<10.4f}")

        symmetry_results.append({
            'level': level,
            'n': n,
            'mean_len_diff': mean_len_diff,
            'std_len_diff': std_len_diff,
            'mean_idf_diff': mean_idf_diff,
            'std_idf_diff': std_idf_diff
        })

    # 总体对称性
    log("-" * 80)
    overall_len_diff = merged['len_diff'].mean()
    overall_std_len = merged['len_diff'].std()
    overall_idf_diff = merged['idf_diff'].mean()
    overall_std_idf = merged['idf_diff'].std()
    log(f"{'Overall':<8} {len(merged):<10} {overall_len_diff:+14.2f} {overall_std_len:<10.2f} {overall_idf_diff:+14.4f} {overall_std_idf:<10.4f}")

    log("")
    log("  解读:")
    log("  • |Mean len_diff| < 0.5 词 → 长度基本对称 ✓")
    log("  • |Mean idf_diff| < 0.1 → IDF 基本对称 ✓")
    if abs(overall_len_diff) > 1.0:
        log(f"  ⚠️ 警告: ACL 平均比 CCOMP 长 {overall_len_diff:.1f} 词，存在长度 confound!")
    if abs(overall_idf_diff) > 0.2:
        log("  ⚠️ 警告: ACL 和 CCOMP 的 IDF 分布存在系统性差异!")

    return symmetry_results


# ============ 实验 3: OLS 控制 len_diff + idf_diff ============
def run_controlled_ols_analysis(all_results_by_category_and_type: Dict, word_idf: Dict[str, float]):
    """实验 3: OLS 控制 len_diff + idf_diff 后的纯方向偏好效应

    如果实验 2 发现 ACL 和 CCOMP 长度/IDF 不完全对称，
    运行 OLS: diff_p10 ~ len_diff + idf_diff
    截距 = 控制长度和 IDF 后的"纯方向偏好"
    """
    log("\n" + "=" * 100)
    log("实验 3: OLS 控制 len_diff + idf_diff 后的纯方向偏好 (P@10)")
    log("=" * 100)
    log("模型: diff_p10 ~ len_diff + idf_diff")
    log("  • 截距 = 控制 len_diff 和 idf_diff 后的纯 ACL-CCOMP 方向偏好")
    log("  • len_diff 系数 = 长度每多 1 词，diff_p10 变化多少")
    log("  • idf_diff 系数 = IDF 每多 1，diff_p10 变化多少")
    log("")

    # 重新组织数据：分别收集 ACL 和 CCOMP 的结果
    acl_results_by_retriever = {}
    ccomp_results_by_retriever = {}

    for (category, qt), results in all_results_by_category_and_type.items():
        if qt != 'correct':
            continue

        is_acl = (category == 'acl')

        for r in results:
            retriever = r['retriever']

            if is_acl:
                if retriever not in acl_results_by_retriever:
                    acl_results_by_retriever[retriever] = defaultdict(dict)
                target_dict = acl_results_by_retriever[retriever]
            else:
                if retriever not in ccomp_results_by_retriever:
                    ccomp_results_by_retriever[retriever] = defaultdict(dict)
                target_dict = ccomp_results_by_retriever[retriever]

            for rec in r.get('all_query_records', []):
                if is_acl:
                    level = rec.get('acl', 0)
                else:
                    level = rec.get('ccomp', 0)
                key = (rec.get('user_id', ''), rec.get('asin', ''))
                target_dict[level][key] = {
                    'p_at10': rec.get('p_at10', 0.0),
                    'query_length': rec.get('query_length', 0.0),
                    'mean_idf': rec.get('mean_idf', 0.0),
                    'query': rec.get('query', '') if 'query' in rec else ''
                }

    # 配对并构建 OLS 数据
    ols_records = []

    for retriever in sorted(set(list(acl_results_by_retriever.keys()) + list(ccomp_results_by_retriever.keys()))):
        acl_levels = acl_results_by_retriever.get(retriever, {})
        ccomp_levels = ccomp_results_by_retriever.get(retriever, {})

        all_levels = sorted(set(list(acl_levels.keys()) + list(ccomp_levels.keys())))

        for level in all_levels:
            acl_dict = acl_levels.get(level, {})
            ccomp_dict = ccomp_levels.get(level, {})

            common_keys = set(acl_dict.keys()) & set(ccomp_dict.keys())

            if len(common_keys) < 3:
                continue

            for key in common_keys:
                acl_rec = acl_dict[key]
                ccomp_rec = ccomp_dict[key]

                ols_records.append({
                    'retriever': retriever,
                    'level': level,
                    'diff_p10': acl_rec['p_at10'] - ccomp_rec['p_at10'],
                    'len_acl': acl_rec['query_length'],
                    'len_ccomp': ccomp_rec['query_length'],
                    'idf_acl': acl_rec['mean_idf'],
                    'idf_ccomp': ccomp_rec['mean_idf'],
                    'len_diff': acl_rec['query_length'] - ccomp_rec['query_length'],
                    'idf_diff': acl_rec['mean_idf'] - ccomp_rec['mean_idf'],
                })

    if not ols_records:
        log("  警告: 没有足够的配对数据，跳过 OLS 分析")
        return None

    df = pd.DataFrame(ols_records)
    log(f"  总配对数: {len(df)}")

    # 分 (retriever, level) 跑 OLS
    log(f"\n{'Retriever':<12} {'Level':<8} {'N':<8} {'R2':<8} {'Intercept':<12} {'P>|t|':<10} {'Coef_len':<10} {'P>|t|':<10} {'Coef_idf':<10} {'P>|t|':<10}")
    log("-" * 120)

    ols_results = []

    for retriever in sorted(df['retriever'].unique()):
        for level in sorted(df['level'].unique()):
            subset = df[(df['retriever'] == retriever) & (df['level'] == level)].copy()

            if len(subset) < 10:
                continue

            # OLS: diff_p10 ~ len_diff + idf_diff
            formula = 'diff_p10 ~ len_diff + idf_diff'
            try:
                model = smf.ols(formula, data=subset).fit()

                intercept = model.params.get('Intercept', float('nan'))
                intercept_p = model.pvalues.get('Intercept', float('nan'))
                coef_len = model.params.get('len_diff', float('nan'))
                coef_len_p = model.pvalues.get('len_diff', float('nan'))
                coef_idf = model.params.get('idf_diff', float('nan'))
                coef_idf_p = model.pvalues.get('idf_diff', float('nan'))

                sig_int = "***" if intercept_p < 0.001 else "**" if intercept_p < 0.01 else "*" if intercept_p < 0.05 else ""

                int_str = f"{intercept:+.5f}" if not np.isnan(intercept) else "NaN"
                int_p_str = f"{intercept_p:.3e}" if intercept_p < 0.001 else f"{intercept_p:.4f}" if not np.isnan(intercept_p) else "NaN"
                len_str = f"{coef_len:+.5f}" if not np.isnan(coef_len) else "NaN"
                len_p_str = f"{coef_len_p:.4f}" if not np.isnan(coef_len_p) else "NaN"
                idf_str = f"{coef_idf:+.5f}" if not np.isnan(coef_idf) else "NaN"
                idf_p_str = f"{coef_idf_p:.4f}" if not np.isnan(coef_idf_p) else "NaN"

                log(f"{retriever:<12} {level:<8} {len(subset):<8} {model.rsquared:<8.4f} {int_str:<12} {int_p_str:<10} {len_str:<10} {len_p_str:<10} {idf_str:<10} {idf_p_str:<10}")

                ols_results.append({
                    'retriever': retriever,
                    'level': level,
                    'n': len(subset),
                    'r2': model.rsquared,
                    'intercept': intercept,
                    'intercept_p': intercept_p,
                    'coef_len': coef_len,
                    'coef_len_p': coef_len_p,
                    'coef_idf': coef_idf,
                    'coef_idf_p': coef_idf_p,
                    'sig_int': sig_int
                })
            except Exception as e:
                log(f"{retriever:<12} {level:<8} {len(subset):<8} ERR: {e}")

    # 汇总: 所有 (retriever, level) 合并
    if ols_results:
        log("-" * 120)
        log("\n  解读:")
        log("  • Intercept = 控制 len_diff 和 idf_diff 后，ACL - CCOMP 的 P@10 纯效应")
        log("  • 如果 Intercept 显著为正 → 用户对 ACL 风格有明显偏好")
        log("  • 如果 Intercept 接近 0 → ACL/CCOMP 偏好差异来自长度/IDF confounds")

    return ols_results


# ============ 主流程 ============
def print_query_type_comparison(all_results_by_type: Dict[str, List[Dict]], k_values: List[int], category_name: str = ''):
    """打印 correct vs noisy 配对比较表（宽格式）

    对于每个 noisy 查询，找到其同 CCOMP 级别的 correct 查询进行配对
    配对 key: (user_id, asin, ccomp)
    输出宽格式表格，包含所有指标
    """
    log("\n" + "=" * 100)
    log(f"{category_name} CORRECT vs NOISY 配对比较（宽格式 | 基于 (user_id, asin, {GROUP_FIELD.upper()}) 匹配）")
    log("=" * 100)

    # 获取所有检索器
    retrievers = set()
    for qt_results in all_results_by_type.values():
        for r in qt_results:
            retrievers.add(r['retriever'])
    retrievers = sorted(retrievers)

    # 定义所有指标及其对应的字段
    METRICS = [
        ('P@1', 'p_at1'),
        ('P@3', 'p_at3'),
        ('P@5', 'p_at5'),
        ('P@10', 'p_at10'),
        ('N@10', 'n_at10'),
        ('MR@10', 'mrr_at10'),
        ('H@10', 'hit_at10'),
    ]

    group_field = GROUP_FIELD  # 'ccomp' or 'acl'

    # 构建宽格式表头 - 每个指标3个子列，固定宽度
    # 格式: 检索器 | CORR | NOISY | DIFF | CORR | NOISY | DIFF | ...
    header = f"{'检索器':<10}"
    sep = " " * 1
    for metric_name, _ in METRICS:
        header += sep + f"{metric_name:>7} {metric_name:>7} {metric_name:>7}"
    log(header)
    log("-" * 100)

    # 对每个检索器计算所有指标的配对比较
    for retriever in retrievers:
        correct_results = next((x for x in all_results_by_type.get('correct', []) if x['retriever'] == retriever), None)
        noisy_results = next((x for x in all_results_by_type.get('noisy', []) if x['retriever'] == retriever), None)

        if not correct_results or not noisy_results:
            continue

        row = f"{retriever:<10}"

        for metric_name, metric_field in METRICS:
            correct_dict = {}
            noisy_dict = {}

            for rec in correct_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                correct_dict[key] = rec.get(metric_field, 0)

            for rec in noisy_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                noisy_dict[key] = rec.get(metric_field, 0)

            # 找到共同的查询对
            common_keys = set(correct_dict.keys()) & set(noisy_dict.keys())

            if not common_keys:
                row += sep + f"{'N/A':>7} {'N/A':>7} {'N/A':>7}"
                continue

            # 计算配对差异
            correct_vals = [correct_dict[k] for k in common_keys]
            noisy_vals = [noisy_dict[k] for k in common_keys]
            diffs = [noisy_vals[i] - correct_vals[i] for i in range(len(common_keys))]

            mean_correct = sum(correct_vals) / len(correct_vals)
            mean_noisy = sum(noisy_vals) / len(noisy_vals)
            mean_diff = sum(diffs) / len(diffs)

            row += sep + f"{mean_correct:7.4f} {mean_noisy:7.4f} {mean_diff:+7.4f}"
        log(row)

    log("-" * 100)


def main():
    log("=" * 60)
    log(f"快速全量评估 - 多检索器 + ACL/CCOMP 双类别 + 交叉对比")
    log(f"类别: {CATEGORY_NAME}")
    log("=" * 60)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]

    # 构建词IDF字典（用于分层分析）- 只需构建一次
    log("\n构建词IDF字典（用于分层分析）...")
    word_idf = build_word_idf_dict(META_FILE, sample_size=50000)

    # 加载 doc_ids 用于随机基线
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bge_') and f.endswith('_doc_ids.pkl'):
            embeddings_path = os.path.join(CACHE_DIR, f)
            break
    with open(embeddings_path, 'rb') as f:
        doc_ids = pickle.load(f)
    n_docs = len(doc_ids)
    log(f"  总文档数: {n_docs}")

    # ========== 对每个查询类别分别进行评估 ==========
    # all_results_by_category_and_type: {(category, query_type): [results]}
    all_results_by_category_and_type = {}
    # all_results_by_category: {category: [all results for that category]}
    all_results_by_category = {}

    for query_category in QUERY_CATEGORIES:
        global GROUP_FIELD
        GROUP_FIELD = query_category
        log("\n" + "=" * 80)
        log(f"========== 开始评估 {query_category.upper()} ==========")
        log("=" * 80)

        # 先加载 correct 版本获取统计信息
        user_queries_correct, user_to_group, _ = load_user_queries('correct', query_category)
        log(f"  [{query_category.upper()}] 用户数: {len(user_queries_correct)}")

        # 按每条查询的group值计数
        group_dist = defaultdict(int)
        for user_qs in user_queries_correct.values():
            for q in user_qs:
                group_dist[q.get(GROUP_FIELD, 0)] += 1
        log(f"  [{query_category.upper()}] {GROUP_FIELD.upper()}分布: {dict(sorted(group_dist.items()))}")

        # 计算查询长度分组统计
        all_word_counts = [q.get('word_count') or 0 for user_qs in user_queries_correct.values() for q in user_qs]
        if not all_word_counts:
            log(f"  [{query_category.upper()}] 警告: 没有查询数据，跳过此类别")
            continue
        q25, q50, q75 = np.percentile(all_word_counts, [25, 50, 75])
        log(f"  [{query_category.upper()}] Query长度分布: min={min(all_word_counts)}, Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}, max={max(all_word_counts)}")

        # 检查IDF分布
        sample_idfs = [compute_query_idf(q['query'], word_idf) for user_qs in user_queries_correct.values() for q in user_qs]
        if sample_idfs:
            log(f"  [{query_category.upper()}] Query IDF分布: min={min(sample_idfs):.2f}, mean={np.mean(sample_idfs):.2f}, max={max(sample_idfs):.2f}")

        # 分别对 correct 和 noisy 进行评估
        all_results_by_type = {}

        for query_type in QUERY_TYPES:
            log("\n" + "#" * 80)
            log(f"# [{query_category.upper()}] 查询类型: {query_type.upper()}")
            log("#" * 80)

            # 根据 query_type 和 query_category 加载对应的查询
            user_queries, user_to_group, _ = load_user_queries(query_type, query_category)
            log(f"  用户数: {len(user_queries)}")

            query_type_results = []

            # 评估密集检索器
            for retriever_name in DENSE_RETRIEVERS:
                try:
                    result = evaluate_dense_retriever(retriever_name, user_queries, user_to_group, k_values, word_idf, query_type, query_category)
                    result['query_type'] = query_type
                    result['query_category'] = query_category
                    query_type_results.append(result)
                except FileNotFoundError as e:
                    log(f"  跳过 {retriever_name}: {e}")
                except Exception as e:
                    log(f"  错误 {retriever_name}: {e}")

            # 评估 BM25
            try:
                result = evaluate_bm25_retriever(user_queries, user_to_group, k_values, word_idf, query_type, query_category)
                result['query_type'] = query_type
                result['query_category'] = query_category
                query_type_results.append(result)
            except Exception as e:
                log(f"  BM25 错误: {e}")

            all_results_by_type[query_type] = query_type_results

        # 保存该类别的所有结果
        all_results_by_category[query_category] = all_results_by_type.get('correct', []) + all_results_by_type.get('noisy', [])

        # 存储到全局结构
        for qt, results in all_results_by_type.items():
            all_results_by_category_and_type[(query_category, qt)] = results

    # =========================================================
    # ========== 所有评估完成，开始输出结果 ==========
    # =========================================================
    log("\n")
    log("=" * 100)
    log(f"========== 评估结果汇总 [{CATEGORY_NAME}] ==========")
    log("=" * 100)

    # ========== 输出每个类别的结果 ==========
    for query_category in QUERY_CATEGORIES:
        # 确保 GROUP_FIELD 与当前 query_category 一致
        GROUP_FIELD = query_category
        all_results_by_type = {
            'correct': all_results_by_category_and_type.get((query_category, 'correct'), []),
            'noisy': all_results_by_category_and_type.get((query_category, 'noisy'), [])
        }

        log("\n" + "=" * 80)
        log(f"========== {query_category.upper()} 评估结果 ==========")
        log("=" * 80)

        # 该类别的 CORRECT 组间性能比较
        if all_results_by_type.get('correct'):
            print_summary_table_wide(all_results_by_type['correct'], f'{CATEGORY_NAME} {query_category.upper()}-CORRECT')

        # 该类别的 NOISY 组间性能比较
        if all_results_by_type.get('noisy'):
            print_summary_table_wide(all_results_by_type['noisy'], f'{CATEGORY_NAME} {query_category.upper()}-NOISY')

        # 该类别的 CORRECT vs NOISY 配对比较
        if all_results_by_type.get('noisy') and all_results_by_type.get('correct'):
            print_query_type_comparison(all_results_by_type, k_values, CATEGORY_NAME)

    # ========== 全局 CORRECT vs NOISY 配对比较 (所有类别) ==========
    # 打印一个综合的 CORRECT vs NOISY 对比表
    log("\n" + "=" * 100)
    log("GLOBAL CORRECT vs NOISY 配对比较（所有类别综合）")
    log("=" * 100)

    # 合并所有类别的结果
    all_results_combined = []
    for (category, qt), results in all_results_by_category_and_type.items():
        for r in results:
            r_copy = r.copy()
            r_copy['query_category'] = category
            all_results_combined.append(r_copy)

    # =========================================================
    # ========== 新增三个核心实验 ==========
    # =========================================================

    # 加载配对数据用于实验 2
    log("\n加载 ACL/CCOMP 配对数据用于对称性检验...")
    acl_df, ccomp_df = load_paired_queries('acl')

    # 实验 1: Paired t-test on ACL_k − CCOMP_k (最核心)
    pttest_results = run_paired_ttest_analysis(all_results_by_category_and_type, word_idf)

    # 实验 2: Query 长度和 IDF 的对称性验证
    symmetry_results = run_symmetry_check(acl_df, ccomp_df, word_idf)

    # 实验 3: OLS 控制 len_diff + idf_diff 后的纯方向偏好
    ols_results = run_controlled_ols_analysis(all_results_by_category_and_type, word_idf)

    if all_results_combined:
        # 计算并打印 Bootstrap CI
        log("\n计算 Bootstrap CI (n=1000, CI=95%)...")
        for r in all_results_combined:
            r['bootstrap_ci'] = {}
            if r.get('all_raw_metrics'):
                r['bootstrap_ci']['overall'] = compute_bootstrap_ci(
                    r['all_raw_metrics'], k_values, n_bootstrap=1000, ci=0.95
                )

    # 打印 Oracle-Aware Random Baseline
    log("\n" + "=" * 80)
    log("Oracle-Aware Random Baseline")
    log("=" * 80)
    if all_results_combined and all_results_combined[0].get('all_raw_metrics'):
        log(f"  理论随机P@10 = 10/{n_docs} = {10/n_docs:.6f}")
        log(f"  理论随机N@10 ≈ {np.mean([1/np.log2(r+2) if r < 10 else 0 for r in range(n_docs)])*10:.6f}")

    # 保存结果（处理 tuple key 等不可 JSON 序列化的问题）
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    output_file = os.path.join(OUTPUT_DIR, "retrieval_all_summary.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'category_name': CATEGORY_NAME,
            'query_types': QUERY_TYPES,
            'query_categories': QUERY_CATEGORIES,
            'results_by_category_and_type': sanitize_for_json(all_results_by_category_and_type),
            'all_results_combined': sanitize_for_json(all_results_combined),
            'experiment1_paired_ttest': sanitize_for_json(pttest_results) if pttest_results else None,
            'experiment2_symmetry_check': sanitize_for_json(symmetry_results) if symmetry_results else None,
            'experiment3_controlled_ols': sanitize_for_json(ols_results) if ols_results else None,
        }, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n" + "=" * 60)
    log("评估完成!")
    log("=" * 60)

if __name__ == "__main__":
    main()
