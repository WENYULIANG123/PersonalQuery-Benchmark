#!/usr/bin/env python3
"""
快速全量评估脚本 - 支持多检索器 + ACL/CCOMP 分组交叉对比
包括: bge, e5, minilm, star, gritlm (密集) + bm25 (稀疏)
包含 ACL/CCOMP 混淆因素分析 (Check 1-4 + Bootstrap CI)
"""

import os
os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
COLBERTV2_CUDA_HOME = "/usr/local/cuda-12.5"
COLBERTV2_TORCH_EXTENSIONS_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/torch_extensions"
COLBERTV2_HOST_CC = "/usr/bin/gcc"
COLBERTV2_HOST_CXX = "/usr/bin/g++"
if not os.path.exists(os.path.join(COLBERTV2_CUDA_HOME, "include", "cuda_runtime.h")):
    raise FileNotFoundError(f"Required CUDA header not found under {COLBERTV2_CUDA_HOME}")
if not os.path.exists(os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")):
    raise FileNotFoundError(f"Required nvcc not found under {COLBERTV2_CUDA_HOME}")
if not os.path.exists(COLBERTV2_HOST_CC):
    raise FileNotFoundError(f"Required host C compiler not found: {COLBERTV2_HOST_CC}")
if not os.path.exists(COLBERTV2_HOST_CXX):
    raise FileNotFoundError(f"Required host C++ compiler not found: {COLBERTV2_HOST_CXX}")
os.environ["CUDA_HOME"] = COLBERTV2_CUDA_HOME
os.environ["CUDA_PATH"] = COLBERTV2_CUDA_HOME
os.environ["CUDACXX"] = os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")
os.environ["CC"] = COLBERTV2_HOST_CC
os.environ["CXX"] = COLBERTV2_HOST_CXX
os.environ["CUDAHOSTCXX"] = COLBERTV2_HOST_CXX
for _env_name, _env_value in [
    ("PATH", os.path.join(COLBERTV2_CUDA_HOME, "bin")),
    ("CPATH", os.path.join(COLBERTV2_CUDA_HOME, "include")),
    ("LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
    ("LD_LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
]:
    _existing_env_value = os.environ.get(_env_name)
    os.environ[_env_name] = _env_value if not _existing_env_value else f"{_env_value}:{_existing_env_value}"
import sys
import importlib.util
import time
import pickle
import json
import gzip
import shutil
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
from revised_query_utils import load_revised_query_map

CATEGORY_NAME = "Baby_Products"
CAT_CONFIG = get_category_config(CATEGORY_NAME)

CACHE_DIR = CAT_CONFIG['retriever_cache_dir']
QUERY_CACHE_BASE_DIR = CAT_CONFIG['query_cache_dir']
QUERY_TYPES = ['correct']  # 只评估 correct
QUERY_CATEGORIES = ['acl', 'ccomp']  # 两种查询类别
ACL_QUERIES_FILE = CAT_CONFIG['query_file']
CCOMP_QUERIES_FILE = CAT_CONFIG['query_file']
OUTPUT_DIR = CAT_CONFIG['output_dir']
META_FILE = CAT_CONFIG['corpus_file']

RETRIEVER_CONFIG = get_retriever_config()
RETRIEVERS = RETRIEVER_CONFIG['retrievers']
DENSE_RETRIEVERS = RETRIEVER_CONFIG['dense_retrievers']
SPARSE_RETRIEVERS = RETRIEVER_CONFIG.get('sparse_retrievers', [])
COLBERTV2_RETRIEVERS = ['colbertv2']
DENSE_SEARCH_BATCH_SIZE = 64
SPLADE_QUERY_BATCH_SIZE = 128

# IDF 分层配置
IDF_BINS = [(2.5, 3.5), (3.5, 4.5), (4.5, 5.0), (5.0, float('inf'))]
IDF_BIN_LABELS = RETRIEVER_CONFIG['idf_bin_labels']

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def prepare_colbert_torch_extensions_dir() -> str:
    ext_dir = os.path.join(
        COLBERTV2_TORCH_EXTENSIONS_BASE_DIR,
        f"colbertv2_cuda125_{CATEGORY_NAME}_{os.getpid()}"
    )
    if os.path.exists(ext_dir):
        shutil.rmtree(ext_dir)
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log(f"[BOOT] ColBERT TORCH_EXTENSIONS_DIR = {ext_dir}")
    return ext_dir


def log_colbert_extension_dir_state(ext_dir: str, prefix: str):
    if not os.path.exists(ext_dir):
        log(f"{prefix} extension dir missing: {ext_dir}")
        return
    entries = sorted(os.listdir(ext_dir))
    preview = entries[:20]
    log(f"{prefix} extension dir entries ({len(entries)}): {preview}")


COLBERTV2_TORCH_EXTENSIONS_DIR = prepare_colbert_torch_extensions_dir()

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
    for query_category in QUERY_CATEGORIES:
        for query_type in QUERY_TYPES:
            query_cache_dir = os.path.join(QUERY_CACHE_BASE_DIR, f'{query_category}_{query_type}_query')
            if not os.path.exists(query_cache_dir):
                issues.append(f"  缺失: 查询缓存目录 {query_category}_{query_type}_query")
                continue

            for retriever in RETRIEVERS:
                cache_file = os.path.join(query_cache_dir, f'{retriever}__{query_category}_{query_type}_cache.pkl')
                if not os.path.exists(cache_file):
                    issues.append(f"  缺失: {retriever} ({query_category}/{query_type}) 查询缓存")
                else:
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        n_users = len(cache_data)
                        log(f"  {retriever} ({query_category}/{query_type}): {n_users} 用户")
                    except Exception as e:
                        issues.append(f"  验证失败: {retriever} ({query_category}/{query_type}) - {str(e)}")

    if issues:
        log("  缓存完整性检查未通过:")
        for issue in issues:
            log(issue)
        return False

    log("  缓存完整性检查通过 ✓")
    return True

# ============ ACL/CCOMP Paired Analysis ============
# 模块级变量，动态从查询文件获取
UNIQUE_LEVELS = [0, 1, 2, 3]  # 默认值，会在 load_paired_queries 时更新
GROUP_FIELD = 'ccomp'  # 默认值，会在 load_paired_queries 时更新


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


def load_paired_queries(query_category: str = 'acl', filter_same_level: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def load_result_query_cache(retriever_name: str, query_type: str, query_category: str) -> Dict:
    """加载已经生成好的 query -> [(asin, score), ...] 结果缓存"""
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'{retriever_name}__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{retriever_name} query result cache not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise TypeError(f"{retriever_name} query result cache must be dict, got {type(cache).__name__}: {cache_path}")

    return cache


def load_colbertv2_build_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, f"08_build_retriever_indices_{CATEGORY_NAME}.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Required ColBERTv2 build module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"build_retriever_indices_{CATEGORY_NAME.lower()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ColBERTv2 build module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_colbertv2_template_module"):
        return module.get_colbertv2_template_module()
    return module


def resolve_colbertv2_output_root() -> str:
    helper = load_colbertv2_build_module()

    metadata_file = CAT_CONFIG['metadata_cache_file']
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = helper.load_fullscale_metadata(CAT_CONFIG['raw_corpus_file'])

    documents, _ = helper.build_fullscale_documents(CATEGORY_NAME, metadata)
    doc_hash = helper.compute_document_hash(documents)
    is_valid, error_msg = helper.validate_retriever_cache(
        'colbertv2',
        doc_hash,
        CAT_CONFIG['retriever_cache_dir'],
        len(documents),
    )
    if not is_valid:
        raise RuntimeError(f"ColBERTv2 index cache is invalid: {error_msg}")

    paths = helper.get_cache_paths('colbertv2', doc_hash, CAT_CONFIG['retriever_cache_dir'])
    return paths['index_root']


def load_colbertv2_doc_ids(output_root: str) -> List[str]:
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Required ColBERTv2 doc id mapping not found: {doc_ids_path}")

    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)

    if not isinstance(doc_ids, list):
        raise TypeError(f"doc_ids.pkl must contain a list, got {type(doc_ids).__name__}")
    if not doc_ids:
        raise ValueError(f"doc_ids.pkl is empty: {doc_ids_path}")
    return doc_ids


def resolve_local_hf_snapshot(repo_id: str) -> str:
    repo_cache_dir = os.path.join(
        os.environ["HF_HOME"],
        "models--" + repo_id.replace("/", "--")
    )
    ref_file = os.path.join(repo_cache_dir, "refs", "main")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Hugging Face ref file not found for {repo_id}: {ref_file}")
    with open(ref_file, "r") as f:
        snapshot_hash = f.read().strip()
    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", snapshot_hash)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Hugging Face snapshot not found for {repo_id}: {snapshot_dir}")
    return snapshot_dir


def build_colbertv2_searcher(output_root: str, doc_ids: List[str]):
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher

    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    checkpoint_path = resolve_local_hf_snapshot("colbert-ir/colbertv2.0")
    log(f"[ColBERT] checkpoint_path = {checkpoint_path}")
    log(f"[ColBERT] output_root = {output_root}")
    log_colbert_extension_dir_state(COLBERTV2_TORCH_EXTENSIONS_DIR, "[ColBERT][before Searcher]")
    searcher_start = time.time()
    with Run().context(RunConfig(experiment="colbertv2_index", root=output_root)):
        config = ColBERTConfig(root=output_root)
        searcher = Searcher(
            index="colbertv2_index",
            checkpoint=checkpoint_path,
            collection=collection,
            config=config,
        )
    log(f"[ColBERT] Searcher loaded in {time.time() - searcher_start:.1f}s")
    log_colbert_extension_dir_state(COLBERTV2_TORCH_EXTENSIONS_DIR, "[ColBERT][after Searcher]")
    return searcher


def load_colbertv2_query_cache(query_type: str, query_category: str) -> Dict[str, Dict[str, np.ndarray]]:
    cache = load_query_cache("colbertv2", query_type, query_category)
    if not isinstance(cache, dict):
        raise TypeError(f"ColBERTv2 query embedding cache must be dict, got {type(cache).__name__}")
    return cache


def colbertv2_search_from_cached_embedding(searcher, doc_ids: List[str], query_embedding, top_k: int) -> List[Tuple[str, float]]:
    if not isinstance(query_embedding, np.ndarray):
        raise TypeError(f"ColBERTv2 cached query embedding must be numpy.ndarray, got {type(query_embedding).__name__}")
    if query_embedding.ndim != 2:
        raise ValueError(f"ColBERTv2 cached query embedding must be 2D, got shape {query_embedding.shape}")

    query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
    pids, _, scores = searcher.dense_search(query_tensor, k=top_k)

    results = []
    for pid, score in zip(pids, scores):
        pid_int = int(pid)
        if pid_int < 0 or pid_int >= len(doc_ids):
            raise IndexError(f"ColBERTv2 pid {pid_int} is outside doc_ids range 0..{len(doc_ids)-1}")
        results.append((doc_ids[pid_int], float(score)))
    return results

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


def load_user_queries(query_type: str = 'correct', query_category: str = 'acl', filter_same_level: bool = False) -> Tuple[Dict[str, List[Dict]], Dict[str, int], List[Tuple[int, float, float]]]:
    """加载用户查询，每个查询项包含word_count和group_ratio（POS ratio代理）

    Args:
        query_type: 查询类型 ('correct' 使用 filled_query, 'noisy' 使用 noisy_query)
        query_category: 查询类别 ('acl' 或 'ccomp')
        filter_same_level: 是否只保留 ACL 和 CCOMP level 一致的用户（默认 False）
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

    revised_query_map = {}
    if query_type != 'noisy':
        revised_query_map = load_revised_query_map(CATEGORY_NAME)

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
                if query_type != 'noisy':
                    revised_query = revised_query_map.get((user_id, asin, query_category))
                    if revised_query:
                        query_text = revised_query
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
            if query_type != 'noisy':
                revised_query = revised_query_map.get((user_id, asin, query_category))
                if revised_query:
                    query_text = revised_query
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
            if query_type != 'noisy':
                revised_query = revised_query_map.get((user_id, asin, query_category))
                if revised_query:
                    query_text = revised_query
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

def build_word_idf_dict(meta_file: str, sample_size: int | None = None) -> Dict[str, float]:
    """从商品元数据语料库构建词的IDF字典。

    当 `sample_size` 为 `None` 时，遍历完整商品语料。
    """
    import gzip
    word_doc_freq = defaultdict(int)
    total_sampled = 0

    if sample_size is None:
        log("Building word IDF from full corpus...")
    else:
        log(f"Building word IDF from corpus (sampling {sample_size} docs)...")
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size is not None and i >= sample_size:
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

    log(f"  IDF vocabulary: {len(word_idf)} words, {total_sampled} docs processed")
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


def save_word_idf_dict(word_idf: Dict[str, float], sample_size: int, output_dir: str) -> Tuple[str, str]:
    """保存构建好的词级 IDF 字典。

    保存两份文件：
    1. `word_idf.pkl`：完整词典，供后续直接加载复用
    2. `word_idf_summary.json`：概要信息，便于快速检查
    """
    os.makedirs(output_dir, exist_ok=True)

    idf_pickle_path = os.path.join(output_dir, "word_idf.pkl")
    idf_summary_path = os.path.join(output_dir, "word_idf_summary.json")

    with open(idf_pickle_path, "wb") as f:
        pickle.dump(word_idf, f, protocol=pickle.HIGHEST_PROTOCOL)

    sorted_items = sorted(word_idf.items(), key=lambda x: x[1], reverse=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "category_name": CATEGORY_NAME,
        "sample_size": sample_size,
        "vocab_size": len(word_idf),
        "top_100_highest_idf": [
            {"token": token, "idf": float(idf_value)}
            for token, idf_value in sorted_items[:100]
        ]
    }
    with open(idf_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return idf_pickle_path, idf_summary_path

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
        self.device = torch.device('cuda')
        # 归一化 doc embeddings 以支持余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10, batch_size: int = DENSE_SEARCH_BATCH_SIZE) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            return []
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0, 实际为 {batch_size}")

        results = []
        n = len(query_embeddings)
        top_k = min(top_k, len(self.doc_ids))

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_embeddings = query_embeddings[start:end]
                query_tensor = torch.from_numpy(np.array(batch_embeddings)).float().to(self.device)
                # 归一化 query embeddings
                q_norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                q_norms = np.where(q_norms == 0, 1, q_norms)
                query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
                # 余弦相似度 = 归一化点积
                scores = torch.mm(query_tensor, self.embeddings_tensor.T)
                for i in range(scores.shape[0]):
                    top_scores, top_indices = torch.topk(scores[i], top_k)
                    results.append([(self.doc_ids[idx.item()], top_scores[j].item()) for j, idx in enumerate(top_indices)])
                del scores
                del query_tensor

        if len(results) != n:
            raise RuntimeError(f"批量搜索结果条数不匹配: results={len(results)} queries={n}")
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

    # 第一步：收集所有用户的查询信息
    log(f"  收集所有查询信息...")
    all_query_data = []  # [(user_id, query_embedding, relevant_asin, word_count, group_ratio, q_group, q_idf)]

    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]
        cached_queries = query_cache[user_id]

        for q in queries:
            query_text = q['query']
            relevant_asin = q['asin']
            word_count = q.get('word_count', 0)
            group_ratio = q.get(f'{GROUP_FIELD}_ratio', 0.0)
            q_group = q.get(GROUP_FIELD, 0)
            if query_text in cached_queries:
                q_idf = compute_query_idf(query_text, word_idf) if word_idf else 0.0
                all_query_data.append((user_id, cached_queries[query_text], relevant_asin, word_count, group_ratio, q_group, q_idf))

    log(f"  总查询数: {len(all_query_data)}")

    # 第二步：一次性批量搜索所有查询
    log(f"  批量搜索所有查询...")
    if all_query_data:
        all_embeddings = [q[1] for q in all_query_data]
        results = searcher.search_batch(all_embeddings, top_k=max(k_values), batch_size=DENSE_SEARCH_BATCH_SIZE)

    # 第三步：遍历搜索结果，进行分组统计
    log(f"  处理结果并分组统计...")
    for i, (result, (user_id, _, relevant_asin, word_count, group_ratio, q_group, q_idf)) in enumerate(zip(results, all_query_data)):
        retrieved_asins = [r[0] for r in result]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        all_metrics.append(metrics)
        group_groups[q_group].append(metrics)

        # 记录每条 query 的原始数据
        all_query_records.append({
            'user_id': user_id,
            'asin': relevant_asin,
            f'{GROUP_FIELD}': q_group,
            'mean_idf': q_idf,
            'query_length': word_count,
            f'{GROUP_FIELD}_ratio': group_ratio,
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # word_count 分组
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= word_count < high:
                word_count_groups[label].append(metrics)
                break

        # group_ratio 分组
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        # IDF 分组
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, q_group)].append(metrics)
                break

        if (i + 1) % 500 == 0:
            log(f"    进度: {i+1}/{len(all_query_data)} ({100*(i+1)/len(all_query_data):.1f}%)")

    log(f"    进度: {len(all_query_data)}/{len(all_query_data)} (100.0%)")

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


def evaluate_cached_result_retriever(retriever_name: str, user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float], query_type: str, query_category: str) -> Dict:
    """评估缓存为 user -> query -> token embedding 的 ColBERTv2 查询缓存"""
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = query_category

    required_k_values = {1, 3, 5, 10}
    if not required_k_values.issubset(set(k_values)):
        raise ValueError(f"{retriever_name} evaluation requires k_values to contain {sorted(required_k_values)}")
    if word_idf is None:
        raise ValueError(f"{retriever_name} evaluation requires word_idf")

    log(f"\n{'='*60}")
    if retriever_name != "colbertv2":
        raise ValueError(f"Token-embedding evaluation only supports colbertv2, got {retriever_name}")

    log(f"检索器: {retriever_name.upper()} (query token embedding 缓存) - {query_category.upper()}/{query_type.upper()}")
    log(f"{'='*60}")

    query_cache = load_colbertv2_query_cache(query_type, query_category)
    missing_users = sorted(set(user_queries.keys()) - set(query_cache.keys()))
    if missing_users:
        raise KeyError(
            f"{retriever_name} query embedding cache missing {len(missing_users)} users "
            f"for {query_category}/{query_type}; examples: {missing_users[:5]}"
        )

    matched_users = list(user_queries.keys())
    output_root = resolve_colbertv2_output_root()
    index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")

    log(f"  用户数: {len(matched_users)}")
    log(f"  ColBERTv2 索引目录: {index_dir}")
    doc_ids = load_colbertv2_doc_ids(output_root)
    searcher = build_colbertv2_searcher(output_root, doc_ids)

    eval_start = time.time()

    all_query_texts = []
    all_query_embeddings = []
    all_query_asins = []
    all_query_users = []
    all_query_word_counts = []
    all_query_group_ratios = []
    all_query_groups = []
    all_query_idf_values = []

    for user_id in matched_users:
        cached_queries = query_cache[user_id]
        if not isinstance(cached_queries, dict):
            raise TypeError(f"{retriever_name} query embedding cache for user {user_id} must be dict")
        for q in user_queries[user_id]:
            query_text = q['query']
            if query_text not in cached_queries:
                raise KeyError(
                    f"{retriever_name} query embedding cache missing query for user={user_id}, "
                    f"{query_category}/{query_type}: {query_text}"
                )
            all_query_texts.append(query_text)
            all_query_embeddings.append(cached_queries[query_text])
            all_query_asins.append(q['asin'])
            all_query_users.append(user_id)
            all_query_word_counts.append(q['word_count'])
            all_query_group_ratios.append(q[f'{GROUP_FIELD}_ratio'])
            all_query_groups.append(q[GROUP_FIELD])
            all_query_idf_values.append(compute_query_idf(query_text, word_idf))

    if not all_query_texts:
        raise ValueError(f"{retriever_name} has no queries for {query_category}/{query_type}")

    log(f"  总查询数: {len(all_query_texts)}")
    log(f"  使用 query token embedding 缓存 (用户数 {len(query_cache)})...")

    all_results = []
    for index, query_embedding in enumerate(all_query_embeddings):
        all_results.append(
            colbertv2_search_from_cached_embedding(
                searcher,
                doc_ids,
                query_embedding,
                max(k_values),
            )
        )
        if (index + 1) % 500 == 0:
            log(f"    ColBERTv2 搜索进度: {index+1}/{len(all_query_embeddings)} ({100*(index+1)/len(all_query_embeddings):.1f}%)")

    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []

    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_group_cross = defaultdict(list)
    all_query_records = []

    for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_query_asins)):
        retrieved_asins = [r[0] for r in retrieved]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        group = all_query_groups[i]
        all_metrics.append(metrics)
        group_groups[group].append(metrics)

        all_query_records.append({
            'user_id': all_query_users[i],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': group,
            'mean_idf': all_query_idf_values[i],
            'query_length': all_query_word_counts[i],
            f'{GROUP_FIELD}_ratio': all_query_group_ratios[i],
            'p_at1': float(metrics['P@1']),
            'p_at3': float(metrics['P@3']),
            'p_at5': float(metrics['P@5']),
            'p_at10': float(metrics['P@10']),
            'n_at10': float(metrics['N@10']),
            'mrr_at10': float(metrics['MR@10']),
            'hit_at10': float(metrics['H@10']),
        })

        wc = all_query_word_counts[i]
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= wc < high:
                word_count_groups[label].append(metrics)
                break

        group_ratio = all_query_group_ratios[i]
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        q_idf = all_query_idf_values[i]
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, group)].append(metrics)
                break

        if (i + 1) % 500 == 0:
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

    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': retriever_name, 'dim': 128, 'type': 'late_interaction_query_embedding_cache',
        'num_users': len(matched_users), 'num_queries': len(all_metrics),
        'eval_time_seconds': eval_time, 'metrics': overall_metrics,
        'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }


def load_splade_retriever():
    """加载 SPLADE 检索器（稀疏向量格式）"""
    from utils.retrievers import SPLADERetriever

    splade_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('splade_') and f.endswith('.pkl'):
            splade_path = os.path.join(CACHE_DIR, f)
            break
    if splade_path is None:
        raise FileNotFoundError("SPLADE cache not found")

    log(f"  [splade] 加载: {os.path.getsize(splade_path)/1024/1024:.1f} MB")
    with open(splade_path, 'rb') as f:
        retriever = pickle.load(f)
    return retriever


def load_splade_query_cache(query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    """加载 SPLADE 查询缓存

    Returns:
        Dict: {user_id: {query_text: {doc_idx: score}}} - 稀疏分数格式
    """
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'splade__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def evaluate_splade_retriever(user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    """评估 SPLADE 检索器

    由于 SPLADE 的 doc_vectors 没有被 pickle 保存（会导致文件过大），
    我们直接调用 search() 方法进行检索，而不是依赖缓存。
    """
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = query_category

    log(f"\n{'='*60}")
    log(f"检索器: SPLADE (稀疏) - {query_category.upper()}/{query_type.upper()}")
    log(f"{'='*60}")

    # 加载 SPLADE 检索器
    retriever = load_splade_retriever()
    doc_ids = retriever.doc_ids

    # SPLADE 的 doc_vectors 没有被 pickle 保存，需要先调用 fit() 重建索引
    if retriever.doc_vectors is None:
        log(f"  [SPLADE] doc_vectors 未保存，正在重建索引...")
        corpus_file = CAT_CONFIG.get('corpus_file', '')
        if os.path.exists(corpus_file):
            documents = []
            with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    if 'asin' in doc and 'title' in doc:
                        documents.append(doc)
                    if len(documents) >= 500000:
                        break
            log(f"  加载了 {len(documents)} 个文档，正在构建 SPLADE 索引...")
            retriever.fit(documents, None)
            log(f"  [SPLADE] 索引构建完成")
        else:
            raise FileNotFoundError(f"SPLADE 索引重建失败: 找不到语料文件 {corpus_file}")

    matched_users = list(user_queries.keys())
    log(f"  用户数: {len(matched_users)}")

    # 加载 SPLADE 查询缓存（预编码的查询向量）
    splade_cache = load_splade_query_cache(query_type, query_category)
    if splade_cache is None:
        raise FileNotFoundError(f"SPLADE query cache not found for {query_category}/{query_type}")
    log(f"  [SPLADE] 使用查询缓存 (共 {len(splade_cache)} 条记录)...")

    # 确保倒排索引已构建（触发 lazy initialization）
    retriever.search(["dummy"], top_k=1)

    eval_start = time.time()

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
    all_query_records = []

    # 获取最大 k 值
    max_k = max(k_values)

    # 收集所有查询用于批量评分
    all_q_data = []
    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]
        for q in queries:
            query_text = q['query']
            if splade_cache and user_id in splade_cache and query_text in splade_cache[user_id]:
                all_q_data.append({
                    'user_id': user_id,
                    'query': query_text,
                    'relevant_asin': q['asin'],
                    'word_count': q.get('word_count', 0),
                    'group_ratio': q.get(f'{GROUP_FIELD}_ratio', 0.0),
                    'q_group': q.get(GROUP_FIELD, 0),
                    'q_vec': splade_cache[user_id][query_text]
                })

    log(f"  批量评分: {len(all_q_data)} 个查询")

    # 预热：确保倒排索引已构建
    inverted_index = retriever._inverted_index
    n_docs = len(retriever.doc_ids)
    log(f"  构建稀疏矩阵...")

    # 构建 term×doc 的稀疏矩阵 (CSR格式)
    from scipy import sparse
    import numpy as np

    # 从倒排索引构建: term_id -> [(doc_idx, weight)]
    # 构建行索引、列索引、数据值
    row_indices = []
    col_indices = []
    data_values = []
    max_doc_term_id = -1
    for term_id, doc_list in inverted_index.items():
        term_id = int(term_id)
        for doc_idx, d_weight in doc_list:
            row_indices.append(term_id)
            col_indices.append(doc_idx)
            data_values.append(d_weight)
            if term_id > max_doc_term_id:
                max_doc_term_id = term_id

    if max_doc_term_id < 0:
        raise ValueError("SPLADE 文档向量为空，无法构建稀疏矩阵")

    max_query_term_id = -1
    for q_data_item in all_q_data:
        for term_id in q_data_item['q_vec'].keys():
            term_id = int(term_id)
            if term_id > max_query_term_id:
                max_query_term_id = term_id

    n_terms = max(max_doc_term_id, max_query_term_id) + 1

    # 构建稀疏矩阵 (term × doc)
    doc_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_terms, n_docs),
        dtype=np.float32
    )
    log(f"  稀疏矩阵构建完成: {n_terms} terms × {n_docs} docs, {len(data_values)} non-zeros")

    n_queries = len(all_q_data)
    log(f"  开始分块矩阵乘法与 top-k 提取...")
    batch_scores = []
    for batch_start in range(0, n_queries, SPLADE_QUERY_BATCH_SIZE):
        batch_end = min(batch_start + SPLADE_QUERY_BATCH_SIZE, n_queries)
        q_rows = []
        q_cols = []
        q_data = []

        for local_q_idx, q_data_item in enumerate(all_q_data[batch_start:batch_end]):
            for term_id, q_weight in q_data_item['q_vec'].items():
                term_id = int(term_id)
                q_rows.append(local_q_idx)
                q_cols.append(term_id)
                q_data.append(q_weight)

        query_matrix = sparse.csr_matrix(
            (q_data, (q_rows, q_cols)),
            shape=(batch_end - batch_start, n_terms),
            dtype=np.float32
        )
        score_matrix = query_matrix @ doc_matrix

        for row_idx in range(score_matrix.shape[0]):
            global_idx = batch_start + row_idx
            if (global_idx + 1) % 500 == 0 or global_idx + 1 == n_queries:
                log(f"    Top-k提取进度: {global_idx + 1}/{n_queries}")

            row = score_matrix.getrow(row_idx)
            if row.nnz == 0:
                batch_scores.append([])
                continue

            if row.nnz <= max_k:
                order = np.argsort(row.data)[::-1]
            else:
                partition = np.argpartition(row.data, -max_k)[-max_k:]
                order = partition[np.argsort(row.data[partition])[::-1]]

            top_doc_indices = row.indices[order]
            top_scores = row.data[order]
            batch_scores.append([
                (retriever.doc_ids[int(doc_idx)], float(score))
                for doc_idx, score in zip(top_doc_indices, top_scores)
            ])

        del query_matrix
        del score_matrix

    if len(batch_scores) != len(all_q_data):
        raise RuntimeError(f"SPLADE result count mismatch: scores={len(batch_scores)}, queries={len(all_q_data)}")

    # 重新组织结果用于后续统计。只处理实际进入批量评分的缓存查询。
    for q_idx, q in enumerate(all_q_data):
        query_text = q['query']
        relevant_asin = q['relevant_asin']
        word_count = q['word_count']
        group_ratio = q['group_ratio']
        q_group = q['q_group']

        retrieved_with_scores = batch_scores[q_idx]

        retrieved_asins = [asin for asin, score in retrieved_with_scores]

        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        all_metrics.append(metrics)
        group_groups[q_group].append(metrics)

        # 计算 IDF
        q_idf = compute_query_idf(query_text, word_idf) if word_idf else 0.0

        # 记录原始数据
        all_query_records.append({
            'user_id': q['user_id'],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': q_group,
            'mean_idf': q_idf,
            'query_length': word_count,
            f'{GROUP_FIELD}_ratio': group_ratio,
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # 分组统计
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= word_count < high:
                word_count_groups[label].append(metrics)
                break

        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, q_group)].append(metrics)
                break

        if (q_idx + 1) % 100 == 0 or q_idx + 1 == len(all_q_data):
            elapsed = time.time() - eval_start
            log(f"    进度: {q_idx+1}/{len(all_q_data)} ({100*(q_idx+1)/len(all_q_data):.1f}%)")

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
        'retriever': 'splade', 'dim': 0, 'type': 'sparse', 'num_users': len(matched_users),
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


# ============ 实验 2: Query 长度和 IDF 的统计分布 ============
def run_symmetry_check(acl_df: pd.DataFrame, ccomp_df: pd.DataFrame, word_idf: Dict[str, float]):
    """实验 2: 统计 ACL 和 CCOMP 各 level 的查询长度和 IDF 分布

    直接计算 ACL0/1/2/3 和 CCOMP0/1/2/3 各组的平均长度和平均 IDF
    """
    log("\n" + "=" * 100)
    log("实验 2: Query 长度和 IDF 统计分布")
    log("=" * 100)
    log("目的: 统计 ACL 和 CCOMP 各 level 的查询长度和 IDF 分布")
    log("")

    # 计算 IDF
    log("  计算查询 IDF...")
    acl_df['mean_idf'] = acl_df['query'].apply(lambda q: compute_query_idf_simple(q, word_idf))
    ccomp_df['mean_idf'] = ccomp_df['query'].apply(lambda q: compute_query_idf_simple(q, word_idf))

    symmetry_results = []

    # ========== ACL Family ==========
    log("\n--- ACL Family ---")
    log(f"\n{'Level':<10} {'N':<10} {'Mean_len':<12} {'Std_len':<10} {'Mean_IDF':<12} {'Std_IDF':<10}")
    log("-" * 70)

    for level in sorted(acl_df['level'].unique()):
        level_data = acl_df[acl_df['level'] == level]
        n = len(level_data)
        mean_len = level_data['word_count'].mean()
        std_len = level_data['word_count'].std()
        mean_idf = level_data['mean_idf'].mean()
        std_idf = level_data['mean_idf'].std()

        log(f"ACL{level:<7} {n:<10} {mean_len:<12.2f} {std_len:<10.2f} {mean_idf:<12.4f} {std_idf:<10.4f}")

        symmetry_results.append({
            'family': 'ACL',
            'level': level,
            'n': n,
            'mean_len': mean_len,
            'std_len': std_len,
            'mean_idf': mean_idf,
            'std_idf': std_idf
        })

    # ========== CCOMP Family ==========
    log("\n--- CCOMP Family ---")
    log(f"\n{'Level':<10} {'N':<10} {'Mean_len':<12} {'Std_len':<10} {'Mean_IDF':<12} {'Std_IDF':<10}")
    log("-" * 70)

    for level in sorted(ccomp_df['level'].unique()):
        level_data = ccomp_df[ccomp_df['level'] == level]
        n = len(level_data)
        mean_len = level_data['word_count'].mean()
        std_len = level_data['word_count'].std()
        mean_idf = level_data['mean_idf'].mean()
        std_idf = level_data['mean_idf'].std()

        log(f"CCOMP{level:<5} {n:<10} {mean_len:<12.2f} {std_len:<10.2f} {mean_idf:<12.4f} {std_idf:<10.4f}")

        symmetry_results.append({
            'family': 'CCOMP',
            'level': level,
            'n': n,
            'mean_len': mean_len,
            'std_len': std_len,
            'mean_idf': mean_idf,
            'std_idf': std_idf
        })

    log("")
    log("  解读:")
    log("  • 查看各 level 的 Mean_len 是否有显著差异")
    log("  • 查看各 level 的 Mean_IDF 是否有显著差异")
    log("  • 如果某 level 与其他 level 差异过大，可能存在 confound")

    return symmetry_results


# ============ 实验 3: OLS 控制 len_diff + idf_diff ============
# 所有需要进行 OLS 分析的指标
OLS_METRICS = ['p_at1', 'p_at3', 'p_at5', 'p_at10', 'n_at10', 'mrr_at10', 'hit_at10']
METRIC_DISPLAY_NAMES = {
    'p_at1': 'P@1', 'p_at3': 'P@3', 'p_at5': 'P@5', 'p_at10': 'P@10',
    'n_at10': 'N@10', 'mrr_at10': 'MR@10', 'hit_at10': 'H@10'
}

def run_controlled_ols_analysis(all_results_by_category_and_type: Dict, word_idf: Dict[str, float]):
    """实验 3: OLS 控制 len_diff + idf_diff 后的纯方向偏好效应

    对多个指标运行 OLS: diff_metric ~ len_diff + idf_diff
    截距 = 控制长度和 IDF 后的"纯方向偏好"
    """
    log("\n" + "=" * 100)
    log("实验 3: OLS 控制 len_diff + idf_diff 后的纯方向偏好")
    log("=" * 100)
    log(f"分析指标: {', '.join(METRIC_DISPLAY_NAMES.values())}")
    log("模型: diff_metric ~ len_diff + idf_diff")
    log("  • 截距 = 控制 len_diff 和 idf_diff 后的纯 ACL-CCOMP 方向偏好")
    log("  • len_diff 系数 = 长度每多 1 词，diff_metric 变化多少")
    log("  • idf_diff 系数 = IDF 每多 1，diff_metric 变化多少")
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
                record = {
                    'query_length': rec.get('query_length', 0.0),
                    'mean_idf': rec.get('mean_idf', 0.0),
                }
                # 收集所有 OLS 指标
                for metric in OLS_METRICS:
                    record[metric] = rec.get(metric, 0.0)
                target_dict[level][key] = record

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

                record = {
                    'retriever': retriever,
                    'level': level,
                    'len_acl': acl_rec['query_length'],
                    'len_ccomp': ccomp_rec['query_length'],
                    'idf_acl': acl_rec['mean_idf'],
                    'idf_ccomp': ccomp_rec['mean_idf'],
                    'len_diff': acl_rec['query_length'] - ccomp_rec['query_length'],
                    'idf_diff': acl_rec['mean_idf'] - ccomp_rec['mean_idf'],
                }
                # 添加所有指标的 diff
                for metric in OLS_METRICS:
                    record[f'diff_{metric}'] = acl_rec[metric] - ccomp_rec[metric]
                ols_records.append(record)

    if not ols_records:
        log("  警告: 没有足够的配对数据，跳过 OLS 分析")
        return None

    df = pd.DataFrame(ols_records)
    log(f"  总配对数: {len(df)}")

    # 对每个指标分别跑 OLS
    ols_results = []

    for metric in OLS_METRICS:
        metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        log(f"\n--- {metric_name} ---")
        log(f"{'Retriever':<12} {'Level':<8} {'N':<8} {'R2':<8} {'Intercept':<12} {'P>|t|':<10} {'Coef_len':<10} {'P>|t|':<10} {'Coef_idf':<10} {'P>|t|':<10}")
        log("-" * 100)

        for retriever in sorted(df['retriever'].unique()):
            for level in sorted(df['level'].unique()):
                subset = df[(df['retriever'] == retriever) & (df['level'] == level)].copy()

                if len(subset) < 10:
                    continue

                diff_col = f'diff_{metric}'
                formula = f'{diff_col} ~ len_diff + idf_diff'
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
                        'metric': metric_name,
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

    # 汇总
    if ols_results:
        log("\n" + "=" * 100)
        log("  解读:")
        log("  • Intercept = 控制 len_diff 和 idf_diff 后，ACL - CCOMP 的纯方向偏好")
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
    idf_sample_size = None
    word_idf = build_word_idf_dict(META_FILE, sample_size=idf_sample_size)
    idf_pickle_path, idf_summary_path = save_word_idf_dict(word_idf, idf_sample_size, OUTPUT_DIR)
    log(f"  词级IDF已保存到: {idf_pickle_path}")
    log(f"  IDF概要已保存到: {idf_summary_path}")

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

            # 评估 ColBERTv2（使用预生成的 query token embedding 缓存）
            for retriever_name in COLBERTV2_RETRIEVERS:
                result = evaluate_cached_result_retriever(
                    retriever_name, user_queries, user_to_group, k_values, word_idf, query_type, query_category
                )
                result['query_type'] = query_type
                result['query_category'] = query_category
                query_type_results.append(result)

            # 评估稀疏检索器（SPLADE）
            for retriever_name in SPARSE_RETRIEVERS:
                try:
                    result = evaluate_splade_retriever(user_queries, user_to_group, k_values, word_idf, query_type, query_category)
                    result['query_type'] = query_type
                    result['query_category'] = query_category
                    query_type_results.append(result)
                except FileNotFoundError as e:
                    log(f"  跳过 {retriever_name}: {e}")
                except Exception as e:
                    log(f"  {retriever_name} 错误: {e}")

            all_results_by_type[query_type] = query_type_results

        # 保存该类别的所有结果（只有 correct）
        all_results_by_category[query_category] = all_results_by_type.get('correct', [])

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
            'correct': all_results_by_category_and_type.get((query_category, 'correct'), [])
        }

        log("\n" + "=" * 80)
        log(f"========== {query_category.upper()} 评估结果 ==========")
        log("=" * 80)

        # 该类别的 CORRECT 组间性能比较
        if all_results_by_type.get('correct'):
            print_summary_table_wide(all_results_by_type['correct'], f'{CATEGORY_NAME} {query_category.upper()}-CORRECT')

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

    # 实验 4: Within-Family OLS (ACL 和 CCOMP 分别分析)
    within_family_ols_results, within_family_models = run_within_family_ols_analysis(all_results_by_category_and_type)

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
            'word_idf_pickle_file': idf_pickle_path,
            'word_idf_summary_file': idf_summary_path,
            'results_by_category_and_type': sanitize_for_json(all_results_by_category_and_type),
            'all_results_combined': sanitize_for_json(all_results_combined),
            'experiment1_paired_ttest': sanitize_for_json(pttest_results) if pttest_results else None,
            'experiment2_symmetry_check': sanitize_for_json(symmetry_results) if symmetry_results else None,
            'experiment3_controlled_ols': sanitize_for_json(ols_results) if ols_results else None,
            'experiment4_within_family_ols': sanitize_for_json(within_family_ols_results.to_dict() if within_family_ols_results is not None else None),
        }, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n" + "=" * 60)
    log("评估完成!")
    log("=" * 60)

# ============================================================
# Within-Family OLS: ACL 和 CCOMP 分别的 OLS 分析（中心化版本）
# ============================================================
def fit_within_family_ols_centered(
    subdf: pd.DataFrame,
    metric_col: str = "p10",
    len_col: str = "qlen",
    idf_col: str = "qidf",
):
    """
    对单个 domain × retriever × family 子集拟合：
        p10 ~ C(level) + len_c + idf_c

    len 和 idf 会先做中心化：
        len_c = len - mean(len)
        idf_c = idf - mean(idf)

    这样：
    - Intercept(L0) = 平均长度、平均IDF下，L0 的预测 P@10
    - 其余 level 系数表示相对 L0 的净变化
    """
    subdf = subdf.copy()
    subdf = subdf[subdf["level"].isin([0, 1, 2, 3])].dropna(
        subset=[metric_col, "level", len_col, idf_col]
    )

    if len(subdf) < 20:
        return None, None

    # level 设为分类变量，L0 做 reference
    subdf["level"] = pd.Categorical(subdf["level"], categories=[0, 1, 2, 3], ordered=True)

    # 中心化
    subdf["len_c"] = subdf[len_col] - subdf[len_col].mean()
    subdf["idf_c"] = subdf[idf_col] - subdf[idf_col].mean()

    model = smf.ols(
        f"{metric_col} ~ C(level, Treatment(reference=0)) + len_c + idf_c",
        data=subdf
    ).fit()

    params = model.params
    pvals = model.pvalues
    conf = model.conf_int()

    key_l1 = "C(level, Treatment(reference=0))[T.1]"
    key_l2 = "C(level, Treatment(reference=0))[T.2]"
    key_l3 = "C(level, Treatment(reference=0))[T.3]"

    beta_l1 = params.get(key_l1, float("nan"))
    beta_l2 = params.get(key_l2, float("nan"))
    beta_l3 = params.get(key_l3, float("nan"))
    p_l1 = pvals.get(key_l1, float("nan"))
    p_l2 = pvals.get(key_l2, float("nan"))
    p_l3 = pvals.get(key_l3, float("nan"))

    ci_l1 = conf.loc[key_l1].tolist() if key_l1 in conf.index else [float("nan"), float("nan")]
    ci_l2 = conf.loc[key_l2].tolist() if key_l2 in conf.index else [float("nan"), float("nan")]
    ci_l3 = conf.loc[key_l3].tolist() if key_l3 in conf.index else [float("nan"), float("nan")]

    # L3 vs L1
    l3_minus_l1 = beta_l3 - beta_l1
    try:
        test = model.t_test(f"{key_l3} - {key_l1} = 0")
        p_l3_vs_l1 = float(test.pvalue)
    except Exception:
        p_l3_vs_l1 = float("nan")

    # L3 vs L2
    l3_minus_l2 = beta_l3 - beta_l2
    try:
        test = model.t_test(f"{key_l3} - {key_l2} = 0")
        p_l3_vs_l2 = float(test.pvalue)
    except Exception:
        p_l3_vs_l2 = float("nan")

    # L2 vs L1
    l2_minus_l1 = beta_l2 - beta_l1
    try:
        test = model.t_test(f"{key_l2} - {key_l1} = 0")
        p_l2_vs_l1 = float(test.pvalue)
    except Exception:
        p_l2_vs_l1 = float("nan")

    result = {
        "n": len(subdf),
        "r2": model.rsquared,
        "intercept_l0_at_mean_covariates": params.get("Intercept", float("nan")),
        "p_intercept_l0": pvals.get("Intercept", float("nan")),
        "delta_l1_vs_l0": beta_l1,
        "p_l1_vs_l0": p_l1,
        "ci_l1_vs_l0_low": ci_l1[0],
        "ci_l1_vs_l0_high": ci_l1[1],
        "delta_l2_vs_l0": beta_l2,
        "p_l2_vs_l0": p_l2,
        "ci_l2_vs_l0_low": ci_l2[0],
        "ci_l2_vs_l0_high": ci_l2[1],
        "delta_l3_vs_l0": beta_l3,
        "p_l3_vs_l0": p_l3,
        "ci_l3_vs_l0_low": ci_l3[0],
        "ci_l3_vs_l0_high": ci_l3[1],
        "delta_l2_vs_l1": l2_minus_l1,
        "p_l2_vs_l1": p_l2_vs_l1,
        "delta_l3_vs_l1": l3_minus_l1,
        "p_l3_vs_l1": p_l3_vs_l1,
        "delta_l3_vs_l2": l3_minus_l2,
        "p_l3_vs_l2": p_l3_vs_l2,
        "coef_len_c": params.get("len_c", float("nan")),
        "p_len_c": pvals.get("len_c", float("nan")),
        "coef_idf_c": params.get("idf_c", float("nan")),
        "p_idf_c": pvals.get("idf_c", float("nan")),
        "mean_len": subdf[len_col].mean(),
        "mean_idf": subdf[idf_col].mean(),
    }
    return result, model


def run_all_within_family_ols_centered(
    df: pd.DataFrame,
    metric_col: str = "p10",
    len_col: str = "qlen",
    idf_col: str = "qidf",
):
    results = []
    models = {}

    for (domain, retriever, family), subdf in df.groupby(["domain", "retriever", "family"]):
        result, model = fit_within_family_ols_centered(
            subdf,
            metric_col=metric_col,
            len_col=len_col,
            idf_col=idf_col,
        )
        if result is None:
            continue
        row = {"domain": domain, "retriever": retriever, "family": family, **result}
        results.append(row)
        models[(domain, retriever, family)] = model

    results_df = pd.DataFrame(results)
    return results_df, models


def sign_with_threshold(x: float, threshold: float = 0.01) -> str:
    if pd.isna(x):
        return "NA"
    if x >= threshold:
        return "+"
    if x <= -threshold:
        return "-"
    return "0"


def add_direction_columns(results_df: pd.DataFrame, threshold: float = 0.01):
    out = results_df.copy()
    out["dir_l1_vs_l0"] = out["delta_l1_vs_l0"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    out["dir_l2_vs_l0"] = out["delta_l2_vs_l0"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    out["dir_l3_vs_l0"] = out["delta_l3_vs_l0"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    out["dir_l2_vs_l1"] = out["delta_l2_vs_l1"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    out["dir_l3_vs_l1"] = out["delta_l3_vs_l1"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    out["dir_l3_vs_l2"] = out["delta_l3_vs_l2"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0" if x == 0 else "NA")
    return out


def run_within_family_ols_analysis(all_results_by_category_and_type: Dict):
    """构建 within-family OLS 分析数据并执行分析（中心化版本）"""
    log("\n" + "=" * 100)
    log("实验 4: Within-Family OLS (ACL/CCOMP 分别分析)")
    log("=" * 100)
    log("模型: p10 ~ C(level) + len_c + idf_c (level=0 作为 reference, covariates 中心化)")
    log("")

    # 重新组织 all_results_by_category_and_type 数据
    all_query_records = []

    for (category, qt), results in all_results_by_category_and_type.items():
        if qt != 'correct':
            continue

        family = category.upper()  # 'ACL' or 'CCOMP'

        for r in results:
            retriever = r['retriever']
            for rec in r.get('all_query_records', []):
                level = rec.get('acl', rec.get('ccomp', 0))
                if level not in [0, 1, 2, 3]:
                    continue
                all_query_records.append({
                    'domain': CATEGORY_NAME,
                    'retriever': retriever,
                    'family': family,
                    'level': level,
                    'p10': rec.get('p_at10', 0.0),
                    'qlen': rec.get('query_length', 0),
                    'qidf': rec.get('mean_idf', 0.0),
                })

    if not all_query_records:
        log("  警告: 没有足够的 query 记录进行 OLS 分析")
        return None, None

    df = pd.DataFrame(all_query_records)
    log(f"  总记录数: {len(df)}")

    results_df, models = run_all_within_family_ols_centered(df)
    results_df = add_direction_columns(results_df, threshold=0.01)

    # 分别打印 ACL 和 CCOMP 结果
    for family in ['ACL', 'CCOMP']:
        fam_df = results_df[results_df['family'] == family].sort_values(['retriever'])
        if fam_df.empty:
            continue

        log(f"\n--- {family} Within-Family OLS (Centered) ---")
        log(f"{'Retriever':<12} {'N':<6} {'R2':<6} {'L1vsL0':<10} {'p':<8} {'L2vsL0':<10} {'p':<8} {'L3vsL0':<10} {'p':<8} {'L2vsL1':<10} {'p':<8} {'L3vsL1':<10} {'p':<8} {'L3vsL2':<10} {'p':<8}")
        log("-" * 150)

        for _, row in fam_df.iterrows():
            retriever = row['retriever']
            n = row['n']
            r2 = row['r2']

            # L1vsL0
            delta_l1 = row.get('delta_l1_vs_l0', float('nan'))
            p_l1 = row.get('p_l1_vs_l0', float('nan'))
            # L2vsL0
            delta_l2 = row.get('delta_l2_vs_l0', float('nan'))
            p_l2 = row.get('p_l2_vs_l0', float('nan'))
            # L3vsL0
            delta_l3 = row.get('delta_l3_vs_l0', float('nan'))
            p_l3 = row.get('p_l3_vs_l0', float('nan'))
            # L2vsL1
            delta_l2_l1 = row.get('delta_l2_vs_l1', float('nan'))
            p_l2_l1 = row.get('p_l2_vs_l1', float('nan'))
            # L3vsL1
            delta_l3_l1 = row.get('delta_l3_vs_l1', float('nan'))
            p_l3_l1 = row.get('p_l3_vs_l1', float('nan'))
            # L3vsL2
            delta_l3_l2 = row.get('delta_l3_vs_l2', float('nan'))
            p_l3_l2 = row.get('p_l3_vs_l2', float('nan'))

            d1_str = f"{delta_l1:+.4f}" if not pd.isna(delta_l1) else "NaN"
            p1_str = f"{p_l1:.3e}" if not pd.isna(p_l1) and p_l1 < 0.001 else f"{p_l1:.4f}" if not pd.isna(p_l1) else "NaN"
            d2_str = f"{delta_l2:+.4f}" if not pd.isna(delta_l2) else "NaN"
            p2_str = f"{p_l2:.3e}" if not pd.isna(p_l2) and p_l2 < 0.001 else f"{p_l2:.4f}" if not pd.isna(p_l2) else "NaN"
            d3_str = f"{delta_l3:+.4f}" if not pd.isna(delta_l3) else "NaN"
            p3_str = f"{p_l3:.3e}" if not pd.isna(p_l3) and p_l3 < 0.001 else f"{p_l3:.4f}" if not pd.isna(p_l3) else "NaN"
            d21_str = f"{delta_l2_l1:+.4f}" if not pd.isna(delta_l2_l1) else "NaN"
            p21_str = f"{p_l2_l1:.3e}" if not pd.isna(p_l2_l1) and p_l2_l1 < 0.001 else f"{p_l2_l1:.4f}" if not pd.isna(p_l2_l1) else "NaN"
            d31_str = f"{delta_l3_l1:+.4f}" if not pd.isna(delta_l3_l1) else "NaN"
            p31_str = f"{p_l3_l1:.3e}" if not pd.isna(p_l3_l1) and p_l3_l1 < 0.001 else f"{p_l3_l1:.4f}" if not pd.isna(p_l3_l1) else "NaN"
            d32_str = f"{delta_l3_l2:+.4f}" if not pd.isna(delta_l3_l2) else "NaN"
            p32_str = f"{p_l3_l2:.3e}" if not pd.isna(p_l3_l2) and p_l3_l2 < 0.001 else f"{p_l3_l2:.4f}" if not pd.isna(p_l3_l2) else "NaN"

            log(f"{retriever:<12} {n:<6} {r2:<6.3f} {d1_str:<10} {p1_str:<8} {d2_str:<10} {p2_str:<8} {d3_str:<10} {p3_str:<8} {d21_str:<10} {p21_str:<8} {d31_str:<10} {p31_str:<8} {d32_str:<10} {p32_str:<8}")

    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, "within_family_ols_results.csv")
    results_df.to_csv(output_file, index=False)
    log(f"\n  结果已保存到: {output_file}")

    return results_df, models


if __name__ == "__main__":
    main()
