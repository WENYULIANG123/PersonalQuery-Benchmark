#!/usr/bin/env python3
"""
评估脚本 - 评估 Pet_Supplies 类别的 correct vs noisy 查询性能
只评估有 noisy 配对的 correct 查询
"""

import os
import sys
import importlib.util
os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
import pickle
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Set

# 设置路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery/08_retrieval')
from config import get_category_config

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def enforce_hf_offline_mode() -> None:
    os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
    os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# ============ 配置加载 ============
CATEGORY_NAME = "Pet_Supplies"
BASE_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query"

try:
    CAT_CONFIG = get_category_config(CATEGORY_NAME)
    CACHE_DIR = CAT_CONFIG['retriever_cache_dir']
    QUERY_CACHE_BASE_DIR = CAT_CONFIG['query_cache_dir']
    QUERY_FILE = CAT_CONFIG['query_file']
    META_FILE = CAT_CONFIG['corpus_file']
except:
    CACHE_DIR = f"{BASE_DIR}/08_retrieval/retriever_{CATEGORY_NAME}_cache"
    QUERY_CACHE_BASE_DIR = f"{BASE_DIR}/08_retrieval/query_cache_{CATEGORY_NAME}"
    QUERY_FILE = f"{BASE_DIR}/06_query/{CATEGORY_NAME}/query.json"
    META_FILE = f"/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_{CATEGORY_NAME}.jsonl.gz"

OUTPUT_DIR = f"{BASE_DIR}/09_noisy_retrieval/{CATEGORY_NAME}"

RETRIEVERS = ['bge', 'e5', 'minilm', 'star', 'ance', 'colbertv2', 'splade', 'bm25']
QUERY_CATEGORIES = ('acl', 'ccomp')

# Noisy 查询文件路径
NOISY_QUERY_BASE = "/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy"

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

def compute_metrics_by_query_category(
    all_metrics: List[Dict],
    all_categories: List[str],
    k_values: List[int]
) -> Dict[str, Dict]:
    if len(all_metrics) != len(all_categories):
        raise ValueError(
            f"Metric/category length mismatch: {len(all_metrics)} metrics vs "
            f"{len(all_categories)} categories"
        )

    metrics_by_category = {}
    for query_category in QUERY_CATEGORIES:
        category_metrics = [
            metrics
            for metrics, category in zip(all_metrics, all_categories)
            if category == query_category
        ]
        if category_metrics:
            metrics_by_category[query_category] = {
                'num_queries': len(category_metrics),
                'metrics': compute_average_metrics(category_metrics, k_values),
            }
        else:
            metrics_by_category[query_category] = {
                'num_queries': 0,
                'metrics': {},
            }
    return metrics_by_category

def build_retriever_result(
    retriever_name: str,
    all_metrics: List[Dict],
    all_categories: List[str],
    k_values: List[int]
) -> Dict:
    return {
        'retriever': retriever_name,
        'num_queries': len(all_metrics),
        'metrics': compute_average_metrics(all_metrics, k_values),
        'metrics_by_category': compute_metrics_by_query_category(all_metrics, all_categories, k_values),
    }

# ============ 数据加载 ============
def load_dense_retriever(retriever_name: str) -> Tuple[np.ndarray, List[str], int]:
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith(f'{retriever_name.lower()}_') and f.endswith('_embeddings.npy'):
            embeddings_path = os.path.join(CACHE_DIR, f)
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} embeddings not found")

    mmap_array = np.load(embeddings_path, mmap_mode='r')
    embeddings = mmap_array[:].copy()

    doc_ids_path = embeddings_path.replace('_embeddings.npy', '_doc_ids.pkl')
    with open(doc_ids_path, 'rb') as f:
        doc_ids = pickle.load(f)

    return embeddings, doc_ids, embeddings.shape[1]

def load_bm25_retriever():
    bm25_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(CACHE_DIR, f)
            break
    if bm25_path is None:
        raise FileNotFoundError("BM25 cache not found")

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25

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

def load_noisy_queries() -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """加载 noisy 查询，返回 (queries, user_asin_pairs)"""
    queries = []
    user_asin_pairs = set()

    noisy_file = f"{NOISY_QUERY_BASE}/{CATEGORY_NAME}/noisy_query.json"
    if not os.path.exists(noisy_file):
        log(f"  警告: Noisy 查询文件不存在: {noisy_file}")
        return [], set()

    with open(noisy_file, 'r') as f:
        content = f.read().strip()

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
        user_id = item.get('user_id', '')
        asin = item.get('asin', '')
        query_text = item.get('noisy_query', '')
        query_category = item.get('query_category', 'acl')
        level = item.get('level', 0)

        user_asin_pairs.add((user_id, asin))

        if query_text:
            queries.append({
                'user_id': user_id,
                'asin': asin,
                'query': query_text,
                'query_category': query_category,
                'level': level,
                'query_type': 'noisy'
            })

    return queries, user_asin_pairs

def load_correct_queries_for_pairs(user_asin_pairs: Set[Tuple[str, str]], noisy_queries: List[Dict]) -> List[Dict]:
    """只加载有 noisy 配对的 correct 查询（类别需匹配）"""
    # 构建 (user_id, asin, query_category) -> noisy query 存在的集合
    noisy_keys = set()
    for q in noisy_queries:
        noisy_keys.add((q['user_id'], q['asin'], q['query_category']))

    queries = []

    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)

    for item in data:
        user_id = item.get('user_id', '')
        asin = item.get('asin', '')

        if (user_id, asin) not in user_asin_pairs:
            continue

        acl_query = item.get('acl_query', {})
        if acl_query:
            query_text = acl_query.get('query', '')
            level = acl_query.get('level', 0)
            if query_text and (user_id, asin, 'acl') in noisy_keys:
                queries.append({
                    'user_id': user_id,
                    'asin': asin,
                    'query': query_text,
                    'query_category': 'acl',
                    'level': level,
                    'query_type': 'correct'
                })

        ccomp_query = item.get('ccomp_query', {})
        if ccomp_query:
            query_text = ccomp_query.get('query', '')
            level = ccomp_query.get('level', 0)
            if query_text and (user_id, asin, 'ccomp') in noisy_keys:
                queries.append({
                    'user_id': user_id,
                    'asin': asin,
                    'query': query_text,
                    'query_category': 'ccomp',
                    'level': level,
                    'query_type': 'correct'
                })

    return queries

def load_correct_query_cache(retriever_name: str, query_category: str = 'acl') -> Dict:
    """加载 correct 查询缓存（按 user_id 索引，值是 {query_text: embedding}）"""
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_correct_query',
        f'{retriever_name.lower()}__{query_category}_correct_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_bm25_correct_cache(query_category: str = 'acl') -> Dict:
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_correct_query',
        f'bm25__{query_category}_correct_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

# Noisy 缓存文件名映射（配置名称 -> 实际文件名）
NOISY_CACHE_NAME_MAP = {
    'bge': 'BGE_.pkl',
    'e5': 'E5_.pkl',
    'minilm': 'MiniLM_.pkl',
    'star': 'STAR_.pkl',
    'ance': 'ANCE_.pkl',
}

def load_noisy_cache(retriever_name: str, query_category: str = 'acl') -> Dict:
    """加载 noisy 查询缓存（按 user_id 索引）"""
    cache_file = NOISY_CACHE_NAME_MAP.get(retriever_name.lower(), f'{retriever_name.upper()}_.pkl')
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_noisy_query',
        cache_file
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_bm25_noisy_cache(query_category: str = 'acl') -> Dict:
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_noisy_query',
        f'bm25__{query_category}_noisy_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_colbertv2_query_cache(query_type: str, query_category: str) -> Dict[str, Dict[str, np.ndarray]]:
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'colbertv2__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"ColBERTv2 query embedding cache not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise TypeError(f"ColBERTv2 query embedding cache must be dict, got {type(cache).__name__}: {cache_path}")
    return cache

def load_colbertv2_build_module():
    retrieval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '08_retrieval')
    module_path = os.path.abspath(os.path.join(retrieval_dir, f"08_build_retriever_indices_{CATEGORY_NAME}.py"))
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Required ColBERTv2 build module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"build_retriever_indices_{CATEGORY_NAME.lower()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ColBERTv2 build module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enforce_hf_offline_mode()
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

def configure_colbertv2_runtime() -> None:
    from utils.retrievers import (
        select_cuda_toolkit_for_colbert_extension_build,
        configure_host_compiler_for_colbert_extension_build,
        validate_cuda_toolkit_for_colbert,
        configure_cuda_env_for_colbert_extension_build,
        preflight_colbert_cuda_extension_build,
    )

    select_cuda_toolkit_for_colbert_extension_build()
    configure_host_compiler_for_colbert_extension_build()
    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()
    preflight_colbert_cuda_extension_build()

def build_colbertv2_searcher(output_root: str, doc_ids: List[str]):
    configure_colbertv2_runtime()

    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher

    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    with Run().context(RunConfig(experiment="colbertv2_index", root=output_root)):
        config = ColBERTConfig(root=output_root)
        return Searcher(
            index="colbertv2_index",
            checkpoint="colbert-ir/colbertv2.0",
            collection=collection,
            config=config,
        )

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

def build_word_idf_dict(meta_file: str, sample_size: int = 50000) -> Dict[str, float]:
    word_doc_freq = defaultdict(int)
    total_sampled = 0

    log(f"Building word IDF from corpus (sampling {sample_size} docs)...")
    import gzip
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                item = json.loads(line)
                text = ' '.join(filter(None, [
                    item.get('title', ''),
                    item.get('brand', ''),
                    ' '.join(item.get('description', []))
                ])).lower()
                words = set(text.split())
                for w in words:
                    if len(w) > 1:
                        word_doc_freq[w] += 1
                total_sampled += 1
            except Exception:
                continue

    N = total_sampled
    word_idf = {}
    for w, df in word_doc_freq.items():
        word_idf[w] = np.log(N / (df + 1))

    log(f"  IDF vocabulary: {len(word_idf)} words, {total_sampled} docs sampled")
    return word_idf

# ============ 搜索器 ============
class DenseSearcher:
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str]):
        self.doc_ids = doc_ids
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            return []
        query_tensor = torch.from_numpy(np.array(query_embeddings)).float().to(self.device)
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
        scores = torch.mm(query_tensor, self.embeddings_tensor.T)
        results = []
        for i in range(len(query_embeddings)):
            top_scores, top_indices = torch.topk(scores[i], min(top_k, len(self.doc_ids)))
            results.append([(self.doc_ids[idx.item()], top_scores[j].item()) for j, idx in enumerate(top_indices)])
        return results

class BM25Searcher:
    def __init__(self, bm25_retriever):
        self.bm25 = bm25_retriever

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        results = []
        for query in queries:
            search_results = self.bm25.search(query, top_k=top_k)
            results.append(search_results)
        return results

def evaluate_colbertv2_queries(queries: List[Dict], k_values: List[int], query_type: str) -> Dict:
    log(f"\n{'='*60}")
    log(f"评估 {query_type.upper()} - COLBERTV2")
    log(f"{'='*60}")

    acl_cache = load_colbertv2_query_cache(query_type, 'acl')
    ccomp_cache = load_colbertv2_query_cache(query_type, 'ccomp')

    output_root = resolve_colbertv2_output_root()
    index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")

    log(f"  查询数: {len(queries)}")
    log(f"  ColBERTv2 索引目录: {index_dir}")
    doc_ids = load_colbertv2_doc_ids(output_root)
    searcher = build_colbertv2_searcher(output_root, doc_ids)

    all_query_embeddings = []
    all_asins = []
    all_categories = []
    for q in queries:
        user_id = q['user_id']
        query_text = q['query']
        category = q['query_category']
        cache = acl_cache if category == 'acl' else ccomp_cache

        if user_id not in cache:
            raise KeyError(f"ColBERTv2 {query_type} cache missing user={user_id}, category={category}")
        if query_text not in cache[user_id]:
            raise KeyError(
                f"ColBERTv2 {query_type} cache missing query for user={user_id}, "
                f"category={category}: {query_text}"
            )

        all_query_embeddings.append(cache[user_id][query_text])
        all_asins.append(q['asin'])
        all_categories.append(category)

    all_metrics = []
    for index, (query_embedding, relevant_asin) in enumerate(zip(all_query_embeddings, all_asins)):
        retrieved = colbertv2_search_from_cached_embedding(searcher, doc_ids, query_embedding, max(k_values))
        retrieved_asins = [r[0] for r in retrieved]
        all_metrics.append(compute_metrics(relevant_asin, retrieved_asins, k_values))
        if (index + 1) % 500 == 0:
            log(f"    ColBERTv2 搜索进度: {index+1}/{len(all_query_embeddings)}")

    return build_retriever_result('colbertv2', all_metrics, all_categories, k_values)

# ============ 评估函数 ============
def evaluate_correct_queries(retriever_name: str, queries: List[Dict], k_values: List[int]) -> Dict:
    """评估 correct 查询（使用 correct cache，按 user_id 索引）"""
    log(f"\n{'='*60}")
    log(f"评估 CORRECT - {retriever_name.upper()}")
    log(f"{'='*60}")

    if retriever_name == 'bm25':
        # 加载 ACL 和 CCOMP 两种缓存
        acl_cache = load_bm25_correct_cache('acl')
        ccomp_cache = load_bm25_correct_cache('ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if acl_cache is None or ccomp_cache is None:
            raise FileNotFoundError("BM25 correct query cache is incomplete")

        log(f"  使用 BM25 correct 查询缓存...")
        all_results = []
        for query_text, category in zip(all_query_texts, all_categories):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if query_text not in cache:
                raise KeyError(f"BM25 correct cache missing query ({category}): {query_text}")
            all_results.append(cache[query_text])

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)
        evaluated_categories = all_categories

    elif retriever_name == 'colbertv2':
        return evaluate_colbertv2_queries(queries, k_values, 'correct')

    elif retriever_name == 'splade':
        # SPLADE 特殊处理：使用矩阵乘法
        from utils.retrievers import SPLADERetriever
        from scipy import sparse

        log(f"  加载 SPLADE 检索器...")
        retriever = load_splade_retriever()

        # 确保索引已构建（触发 lazy initialization of inverted index）
        retriever.search(["dummy"], top_k=1)

        # 加载 SPLADE correct 查询缓存（特殊格式：{user_id: {query_text: sparse_vec}})
        def load_splade_correct_cache(query_category: str) -> Dict:
            cache_path = os.path.join(
                QUERY_CACHE_BASE_DIR,
                f'{query_category}_correct_query',
                f'splade__{query_category}_correct_cache.pkl'
            )
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        acl_cache = load_splade_correct_cache('acl')
        ccomp_cache = load_splade_correct_cache('ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_user_ids = [q['user_id'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if not (acl_cache or ccomp_cache):
            log(f"  错误: SPLADE Correct 查询缓存不存在")
            return None

        log(f"  使用 SPLADE correct 查询缓存...")

        # 收集所有查询的 sparse vectors
        all_q_data = []
        for user_id, query_text, category, asin in zip(all_user_ids, all_query_texts, all_categories, all_asins):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if cache and user_id in cache and query_text in cache[user_id]:
                all_q_data.append({
                    'query': query_text,
                    'q_vec': cache[user_id][query_text],
                    'relevant_asin': asin,
                    'query_category': category,
                })

        log(f"  有效查询: {len(all_q_data)}")

        if not all_q_data:
            return None

        # 从 inverted index 构建文档矩阵 (term × doc)
        inverted_index = retriever._inverted_index
        n_docs = len(retriever.doc_ids)

        log(f"  从倒排索引构建文档矩阵...")

        # 构建 term×doc 的稀疏矩阵
        row_indices = []
        col_indices = []
        data_values = []
        for term_id, doc_list in inverted_index.items():
            for doc_idx, d_weight in doc_list:
                row_indices.append(term_id)
                col_indices.append(doc_idx)
                data_values.append(d_weight)

        max_term_id = max(inverted_index.keys()) if inverted_index else 0
        n_terms = max_term_id + 1

        doc_matrix = sparse.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_terms, n_docs),
            dtype=np.float32
        )
        log(f"  文档矩阵构建完成: {n_terms} terms × {n_docs} docs")

        # 构建查询矩阵 (n_queries × n_terms)
        q_rows = []
        q_cols = []
        q_data = []

        for q_idx, qd in enumerate(all_q_data):
            for term_id, q_weight in qd['q_vec'].items():
                q_rows.append(q_idx)
                q_cols.append(term_id)
                q_data.append(q_weight)

        n_queries = len(all_q_data)
        query_matrix = sparse.csr_matrix(
            (q_data, (q_rows, q_cols)),
            shape=(n_queries, n_terms),
            dtype=np.float32
        )
        log(f"  查询矩阵构建完成: {query_matrix.shape}")

        # 矩阵乘法: (n_queries × n_terms) @ (n_terms × n_docs) = (n_queries × n_docs)
        log(f"  执行矩阵乘法...")
        score_matrix = query_matrix @ doc_matrix
        log(f"  矩阵乘法完成: {score_matrix.shape}")

        # 提取 top-k 结果
        max_k = max(k_values)
        all_metrics = []
        for i, qd in enumerate(all_q_data):
            row = score_matrix.getrow(i)
            scores_vec = row.toarray().flatten()
            top_indices = np.argsort(scores_vec)[::-1][:max_k]
            retrieved_asins = [retriever.doc_ids[idx] for idx in top_indices]
            metrics = compute_metrics(qd['relevant_asin'], retrieved_asins, k_values)
            all_metrics.append(metrics)

        evaluated_categories = [qd['query_category'] for qd in all_q_data]
        return build_retriever_result(retriever_name, all_metrics, evaluated_categories, k_values)

    else:
        embeddings, doc_ids, dim = load_dense_retriever(retriever_name)
        searcher = DenseSearcher(embeddings, doc_ids)
        # 加载 ACL 和 CCOMP 两种缓存
        acl_cache = load_correct_query_cache(retriever_name, 'acl')
        ccomp_cache = load_correct_query_cache(retriever_name, 'ccomp')

        all_user_ids = [q['user_id'] for q in queries]
        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        log(f"  使用查询缓存...")
        query_embeddings = []
        for user_id, query_text, category in zip(all_user_ids, all_query_texts, all_categories):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if cache and user_id in cache and query_text in cache[user_id]:
                query_embeddings.append(cache[user_id][query_text])
            else:
                query_embeddings.append(None)

        valid_indices = [i for i, emb in enumerate(query_embeddings) if emb is not None]
        valid_embeddings = [query_embeddings[i] for i in valid_indices]
        valid_asins = [all_asins[i] for i in valid_indices]
        valid_categories = [all_categories[i] for i in valid_indices]

        log(f"  有效查询: {len(valid_embeddings)}")

        if not valid_embeddings:
            return None

        results = searcher.search_batch(valid_embeddings, top_k=max(k_values))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(results, valid_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)
        evaluated_categories = valid_categories

        del embeddings
        del searcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return build_retriever_result(retriever_name, all_metrics, evaluated_categories, k_values)

def evaluate_noisy_queries(retriever_name: str, queries: List[Dict], k_values: List[int]) -> Dict:
    """评估 noisy 查询（使用 noisy cache，按 user_id 索引）"""
    log(f"\n{'='*60}")
    log(f"评估 NOISY - {retriever_name.upper()}")
    log(f"{'='*60}")

    if retriever_name == 'bm25':
        # 加载 ACL 和 CCOMP 两种 noisy 缓存
        acl_cache = load_bm25_noisy_cache('acl')
        ccomp_cache = load_bm25_noisy_cache('ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if acl_cache is None or ccomp_cache is None:
            raise FileNotFoundError("BM25 noisy query cache is incomplete")

        log(f"  使用 BM25 noisy 查询缓存...")
        all_results = []
        for query_text, category in zip(all_query_texts, all_categories):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if query_text not in cache:
                raise KeyError(f"BM25 noisy cache missing query ({category}): {query_text}")
            all_results.append(cache[query_text])

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)
        evaluated_categories = all_categories

    elif retriever_name == 'colbertv2':
        return evaluate_colbertv2_queries(queries, k_values, 'noisy')

    elif retriever_name == 'splade':
        # SPLADE 特殊处理：使用矩阵乘法
        from utils.retrievers import SPLADERetriever
        from scipy import sparse

        log(f"  加载 SPLADE 检索器...")
        retriever = load_splade_retriever()

        # 确保索引已构建（触发 lazy initialization of inverted index）
        retriever.search(["dummy"], top_k=1)

        # 加载 SPLADE noisy 查询缓存（特殊格式：{user_id: {query_text: sparse_vec}})
        def load_splade_noisy_cache(query_category: str) -> Dict:
            cache_path = os.path.join(
                QUERY_CACHE_BASE_DIR,
                f'{query_category}_noisy_query',
                f'splade__{query_category}_noisy_cache.pkl'
            )
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        acl_cache = load_splade_noisy_cache('acl')
        ccomp_cache = load_splade_noisy_cache('ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_user_ids = [q['user_id'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if not (acl_cache or ccomp_cache):
            log(f"  错误: SPLADE Noisy 查询缓存不存在")
            return None

        log(f"  使用 SPLADE noisy 查询缓存...")

        # 收集所有查询的 sparse vectors
        all_q_data = []
        for user_id, query_text, category, asin in zip(all_user_ids, all_query_texts, all_categories, all_asins):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if cache and user_id in cache and query_text in cache[user_id]:
                all_q_data.append({
                    'query': query_text,
                    'q_vec': cache[user_id][query_text],
                    'relevant_asin': asin,
                    'query_category': category,
                })

        log(f"  有效查询: {len(all_q_data)}")

        if not all_q_data:
            return None

        # 从 inverted index 构建文档矩阵 (term × doc)
        inverted_index = retriever._inverted_index
        n_docs = len(retriever.doc_ids)

        log(f"  从倒排索引构建文档矩阵...")

        # 构建 term×doc 的稀疏矩阵
        row_indices = []
        col_indices = []
        data_values = []
        for term_id, doc_list in inverted_index.items():
            for doc_idx, d_weight in doc_list:
                row_indices.append(term_id)
                col_indices.append(doc_idx)
                data_values.append(d_weight)

        max_term_id = max(inverted_index.keys()) if inverted_index else 0
        n_terms = max_term_id + 1

        doc_matrix = sparse.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_terms, n_docs),
            dtype=np.float32
        )
        log(f"  文档矩阵构建完成: {n_terms} terms × {n_docs} docs")

        # 构建查询矩阵 (n_queries × n_terms)
        q_rows = []
        q_cols = []
        q_data = []

        for q_idx, qd in enumerate(all_q_data):
            for term_id, q_weight in qd['q_vec'].items():
                q_rows.append(q_idx)
                q_cols.append(term_id)
                q_data.append(q_weight)

        n_queries = len(all_q_data)
        query_matrix = sparse.csr_matrix(
            (q_data, (q_rows, q_cols)),
            shape=(n_queries, n_terms),
            dtype=np.float32
        )
        log(f"  查询矩阵构建完成: {query_matrix.shape}")

        # 矩阵乘法: (n_queries × n_terms) @ (n_terms × n_docs) = (n_queries × n_docs)
        log(f"  执行矩阵乘法...")
        score_matrix = query_matrix @ doc_matrix
        log(f"  矩阵乘法完成: {score_matrix.shape}")

        # 提取 top-k 结果
        max_k = max(k_values)
        all_metrics = []
        for i, qd in enumerate(all_q_data):
            row = score_matrix.getrow(i)
            scores_vec = row.toarray().flatten()
            top_indices = np.argsort(scores_vec)[::-1][:max_k]
            retrieved_asins = [retriever.doc_ids[idx] for idx in top_indices]
            metrics = compute_metrics(qd['relevant_asin'], retrieved_asins, k_values)
            all_metrics.append(metrics)

        evaluated_categories = [qd['query_category'] for qd in all_q_data]
        return build_retriever_result(retriever_name, all_metrics, evaluated_categories, k_values)

    else:
        embeddings, doc_ids, dim = load_dense_retriever(retriever_name)
        searcher = DenseSearcher(embeddings, doc_ids)
        # 加载 ACL 和 CCOMP 两种 noisy 缓存
        acl_cache = load_noisy_cache(retriever_name, 'acl')
        ccomp_cache = load_noisy_cache(retriever_name, 'ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_user_ids = [q['user_id'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if acl_cache or ccomp_cache:
            log(f"  使用 noisy 查询缓存...")
            query_embeddings = []
            for user_id, query_text, category in zip(all_user_ids, all_query_texts, all_categories):
                cache = acl_cache if category == 'acl' else ccomp_cache
                if cache and user_id in cache:
                    user_cache = cache[user_id]
                    found = False
                    for item in user_cache:
                        if item['query'] == query_text:
                            query_embeddings.append(item['vector'])
                            found = True
                            break
                    if not found:
                        query_embeddings.append(None)
                else:
                    query_embeddings.append(None)
        else:
            log(f"  错误: Noisy 查询缓存不存在")
            return None

        valid_indices = [i for i, emb in enumerate(query_embeddings) if emb is not None]
        valid_embeddings = [query_embeddings[i] for i in valid_indices]
        valid_asins = [all_asins[i] for i in valid_indices]
        valid_categories = [all_categories[i] for i in valid_indices]

        log(f"  有效查询: {len(valid_embeddings)}")

        if not valid_embeddings:
            return None

        results = searcher.search_batch(valid_embeddings, top_k=max(k_values))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(results, valid_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)
        evaluated_categories = valid_categories

        del embeddings
        del searcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return build_retriever_result(retriever_name, all_metrics, evaluated_categories, k_values)

def print_results_table(all_results: List[Dict], title: str, category: str = ""):
    log(f"\n{'='*100}")
    log(f"{title} {f'[{category}]' if category else ''}")
    log("=" * 100)

    metrics_to_show = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']

    header = f"{'检索器':<12}"
    for m in metrics_to_show:
        header += f" {m:>10}"
    log(header)
    log("-" * 100)

    for r in all_results:
        if r is None:
            continue
        row = f"{r['retriever']:<12}"
        for m in metrics_to_show:
            val = r['metrics'].get(m, 0.0)
            row += f" {val:>10.4f}"
        log(row)

    log("-" * 100)

def extract_category_results(all_results: List[Dict], query_category: str) -> List[Dict]:
    category_results = []
    for result in all_results:
        if result is None:
            continue
        category_result = result['metrics_by_category'][query_category]
        if category_result['num_queries'] == 0:
            continue
        category_results.append({
            'retriever': result['retriever'],
            'num_queries': category_result['num_queries'],
            'metrics': category_result['metrics'],
        })
    return category_results

def compute_difference_table(correct_results: List[Dict], noisy_results: List[Dict], category: str):
    log(f"\n{'='*120}")
    log(f"CORRECT vs NOISY 差异分析（NOISY - CORRECT） [{category}]")
    log("=" * 120)

    metrics_to_show = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']

    correct_dict = {r['retriever']: r for r in correct_results}
    noisy_dict = {r['retriever']: r for r in noisy_results}
    common_retrievers = sorted(set(correct_dict.keys()) & set(noisy_dict.keys()))
    if not common_retrievers:
        log("没有可比较的共同检索器。")
        log("-" * 120)
        return

    header = f"{'检索器':<12} {'Correct_N':>10} {'Noisy_N':>10}"
    for m in metrics_to_show:
        header += f" {m:>10}"
    log(header)
    log("-" * 120)

    for retriever in common_retrievers:
        correct_result = correct_dict[retriever]
        noisy_result = noisy_dict[retriever]
        c_metrics = correct_result['metrics']
        n_metrics = noisy_result['metrics']
        row = f"{retriever:<12} {correct_result['num_queries']:>10} {noisy_result['num_queries']:>10}"
        for m in metrics_to_show:
            diff = n_metrics[m] - c_metrics[m]
            sign = "+" if diff > 0 else ""
            row += f" {sign:>1}{diff:>9.4f}"
        log(row)

    log("-" * 120)

def main():
    log("=" * 60)
    log(f"评估 - {CATEGORY_NAME} Correct vs Noisy 查询")
    log(f"类别: {CATEGORY_NAME}")
    log("=" * 60)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]

    log("\n构建词IDF字典...")
    word_idf = build_word_idf_dict(META_FILE, sample_size=50000)

    log("\n加载查询数据...")
    queries_noisy, user_asin_pairs = load_noisy_queries()
    queries_correct = load_correct_queries_for_pairs(user_asin_pairs, queries_noisy)

    log(f"  有 noisy 配对的 correct 查询数: {len(queries_correct)}")
    log(f"  Noisy 查询数: {len(queries_noisy)}")
    log(f"  配对用户数: {len(user_asin_pairs)}")
    for query_category in QUERY_CATEGORIES:
        correct_count = sum(1 for q in queries_correct if q['query_category'] == query_category)
        noisy_count = sum(1 for q in queries_noisy if q['query_category'] == query_category)
        log(f"  {query_category.upper()} correct/noisy 查询数: {correct_count}/{noisy_count}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    correct_results = []
    for retriever_name in RETRIEVERS:
        try:
            result = evaluate_correct_queries(retriever_name, queries_correct, k_values)
            if result:
                correct_results.append(result)
        except FileNotFoundError as e:
            log(f"  跳过 {retriever_name}: {e}")
        except Exception as e:
            log(f"  错误 {retriever_name}: {e}")

    noisy_results = []
    for retriever_name in RETRIEVERS:
        try:
            result = evaluate_noisy_queries(retriever_name, queries_noisy, k_values)
            if result:
                noisy_results.append(result)
        except FileNotFoundError as e:
            log(f"  跳过 {retriever_name}: {e}")
        except Exception as e:
            log(f"  错误 {retriever_name}: {e}")

    print_results_table(correct_results, "CORRECT 查询结果（有 noisy 配对）", CATEGORY_NAME)
    print_results_table(noisy_results, "NOISY 查询结果", CATEGORY_NAME)

    for query_category in QUERY_CATEGORIES:
        correct_category_results = extract_category_results(correct_results, query_category)
        noisy_category_results = extract_category_results(noisy_results, query_category)
        compute_difference_table(
            correct_category_results,
            noisy_category_results,
            f"{CATEGORY_NAME} / {query_category.upper()}",
        )

    output_file = os.path.join(OUTPUT_DIR, "correct_vs_noisy_results.json")
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'category': CATEGORY_NAME,
        'num_paired_users': len(user_asin_pairs),
        'num_correct_queries': len(queries_correct),
        'num_noisy_queries': len(queries_noisy),
        'correct_results': correct_results,
        'noisy_results': noisy_results,
    }
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n评估完成!")

if __name__ == "__main__":
    main()
