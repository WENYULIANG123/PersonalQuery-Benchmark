#!/usr/bin/env python3
"""
评估脚本 - 评估 Pet_Supplies 类别的 correct vs noisy 查询性能
只评估有 noisy 配对的 correct 查询
"""

import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery/08_retrieval')
from config import get_category_config, get_retriever_config

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============ 配置加载 ============
CATEGORY_NAME = "Pet_Supplies"
BASE_DIR = "/workspace/result/personal_query"

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
    META_FILE = f"/workspace/Amazon-Reviews-2023/raw/meta_categories/meta_{CATEGORY_NAME}.jsonl.gz"

OUTPUT_DIR = f"{BASE_DIR}/09_noisy_retrieval/{CATEGORY_NAME}"

RETRIEVER_CONFIG = get_retriever_config()
RETRIEVERS = RETRIEVER_CONFIG['retrievers']

# Noisy 查询文件路径
NOISY_QUERY_BASE = "/workspace/result/personal_query/07_inject_noisy"

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
    'gritlm': 'GRITLM_.pkl',
    'bm25': 'bm25_.pkl',
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
        f'bm25_.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

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

# ============ 评估函数 ============
def evaluate_correct_queries(retriever_name: str, queries: List[Dict], k_values: List[int]) -> Dict:
    """评估 correct 查询（使用 correct cache，按 user_id 索引）"""
    log(f"\n{'='*60}")
    log(f"评估 CORRECT - {retriever_name.upper()}")
    log(f"{'='*60}")

    if retriever_name == 'bm25':
        bm25 = load_bm25_retriever()
        searcher = BM25Searcher(bm25)
        # 加载 ACL 和 CCOMP 两种缓存
        acl_cache = load_bm25_correct_cache('acl')
        ccomp_cache = load_bm25_correct_cache('ccomp')

        all_user_ids = [q['user_id'] for q in queries]
        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        log(f"  使用查询缓存...")
        all_results = []
        for user_id, query_text, category in zip(all_user_ids, all_query_texts, all_categories):
            cache = acl_cache if category == 'acl' else ccomp_cache
            if cache and user_id in cache and query_text in cache[user_id]:
                all_results.append(cache[user_id][query_text])
            else:
                all_results.append(searcher.bm25.search(query_text, top_k=max(k_values)))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)

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

        log(f"  有效查询: {len(valid_embeddings)}")

        if not valid_embeddings:
            return None

        results = searcher.search_batch(valid_embeddings, top_k=max(k_values))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(results, valid_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)

        del embeddings
        del searcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_metrics = compute_average_metrics(all_metrics, k_values)

    return {
        'retriever': retriever_name,
        'num_queries': len(all_metrics),
        'metrics': overall_metrics
    }

def evaluate_noisy_queries(retriever_name: str, queries: List[Dict], k_values: List[int]) -> Dict:
    """评估 noisy 查询（使用 noisy cache，按 user_id 索引）"""
    log(f"\n{'='*60}")
    log(f"评估 NOISY - {retriever_name.upper()}")
    log(f"{'='*60}")

    if retriever_name == 'bm25':
        bm25 = load_bm25_retriever()
        searcher = BM25Searcher(bm25)
        # 加载 ACL 和 CCOMP 两种 noisy 缓存
        acl_cache = load_bm25_noisy_cache('acl')
        ccomp_cache = load_bm25_noisy_cache('ccomp')

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]
        all_user_ids = [q['user_id'] for q in queries]
        all_categories = [q['query_category'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        if acl_cache or ccomp_cache:
            log(f"  使用 noisy 查询缓存...")
            all_results = []
            for user_id, query_text, category in zip(all_user_ids, all_query_texts, all_categories):
                cache = acl_cache if category == 'acl' else ccomp_cache
                if cache and user_id in cache:
                    user_cache = cache[user_id]
                    found = False
                    for item in user_cache:
                        if item['query'] == query_text:
                            all_results.append(item['results'])
                            found = True
                            break
                    if not found:
                        all_results.append(searcher.bm25.search(query_text, top_k=max(k_values)))
                else:
                    all_results.append(searcher.bm25.search(query_text, top_k=max(k_values)))
        else:
            log(f"  开始批量搜索...")
            all_results = searcher.search_batch(all_query_texts, top_k=max(k_values))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)

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

        log(f"  有效查询: {len(valid_embeddings)}")

        if not valid_embeddings:
            return None

        results = searcher.search_batch(valid_embeddings, top_k=max(k_values))

        all_metrics = []
        for i, (retrieved, relevant_asin) in enumerate(zip(results, valid_asins)):
            retrieved_asins = [r[0] for r in retrieved]
            metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
            all_metrics.append(metrics)

        del embeddings
        del searcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_metrics = compute_average_metrics(all_metrics, k_values)

    return {
        'retriever': retriever_name,
        'num_queries': len(all_metrics),
        'metrics': overall_metrics
    }

def print_results_table(all_results: List[Dict], title: str):
    log(f"\n{'='*100}")
    log(title)
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

def main():
    log("=" * 60)
    log(f"评估 - Pet_Supplies Correct vs Noisy 查询")
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

    print_results_table(correct_results, "CORRECT 查询结果（有 noisy 配对）")
    print_results_table(noisy_results, "NOISY 查询结果")

    # 计算 CORRECT vs NOISY 差异分析
    def compute_difference_table(correct_results: List[Dict], noisy_results: List[Dict]):
        log(f"\n{'='*100}")
        log("CORRECT vs NOISY 差异分析（NOISY - CORRECT）")
        log("=" * 100)

        metrics_to_show = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']

        # 构建检索器索引
        correct_dict = {r['retriever']: r['metrics'] for r in correct_results}
        noisy_dict = {r['retriever']: r['metrics'] for r in noisy_results}

        # 获取共同的检索器
        common_retrievers = sorted(set(correct_dict.keys()) & set(noisy_dict.keys()))

        header = f"{'检索器':<12}"
        for m in metrics_to_show:
            header += f" {m:>10}"
        log(header)
        log("-" * 100)

        for retriever in common_retrievers:
            c_metrics = correct_dict[retriever]
            n_metrics = noisy_dict[retriever]
            row = f"{retriever:<12}"
            for m in metrics_to_show:
                diff = n_metrics.get(m, 0.0) - c_metrics.get(m, 0.0)
                # 用颜色标记（+ 绿色，- 红色）但日志中只用符号
                sign = "+" if diff > 0 else ""
                row += f" {sign:>1}{diff:>9.4f}"
            log(row)

        log("-" * 100)

    compute_difference_table(correct_results, noisy_results)

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