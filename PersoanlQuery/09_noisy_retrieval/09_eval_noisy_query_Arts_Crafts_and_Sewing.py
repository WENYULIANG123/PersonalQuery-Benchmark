#!/usr/bin/env python3
"""
评估脚本 - 评估 Arts_Crafts_and_Sewing 类别的 correct vs noisy 查询性能
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
from typing import List, Dict, Tuple

# 设置路径
sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery/08_retrieval')
from config import get_category_config, get_retriever_config

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============ 配置加载 ============
CATEGORY_NAME = "Arts_Crafts_and_Sewing"
BASE_DIR = "/workspace/result/personal_query"

# 尝试从 08_retrieval 配置获取路径
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
DENSE_RETRIEVERS = RETRIEVER_CONFIG['dense_retrievers']

# Noisy 查询文件路径
NOISY_QUERY_BASE = "/workspace/result/personal_query/07_inject_noisy"
ACL_NOISY_QUERY_FILE = f"{NOISY_QUERY_BASE}/{CATEGORY_NAME}/acl_noisy_query.json"
CCOMP_NOISY_QUERY_FILE = f"{NOISY_QUERY_BASE}/{CATEGORY_NAME}/ccomp_noisy_query.json"

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
        if f.startswith(f'{retriever_name}_') and f.endswith('_embeddings.npy'):
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

def load_query_cache(retriever_name: str, query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'{retriever_name.lower()}__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_bm25_query_cache(query_type: str = 'correct', query_category: str = 'acl') -> Dict:
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'bm25__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_queries(query_type: str = 'correct') -> Tuple[List[Dict], Dict]:
    queries = []
    doc_ids_set = set()

    if query_type == 'correct':
        with open(QUERY_FILE, 'r') as f:
            data = json.load(f)

        for item in data:
            user_id = item.get('user_id', '')
            asin = item.get('asin', '')
            doc_ids_set.add(asin)

            acl_query = item.get('acl_query', {})
            if acl_query:
                query_text = acl_query.get('query', '')
                level = acl_query.get('level', 0)
                word_count = acl_query.get('word_count', 0)
                if query_text:
                    queries.append({
                        'user_id': user_id,
                        'asin': asin,
                        'query': query_text,
                        'query_category': 'acl',
                        'level': level,
                        'word_count': word_count,
                        'query_type': query_type
                    })

            ccomp_query = item.get('ccomp_query', {})
            if ccomp_query:
                query_text = ccomp_query.get('query', '')
                level = ccomp_query.get('level', 0)
                word_count = ccomp_query.get('word_count', 0)
                if query_text:
                    queries.append({
                        'user_id': user_id,
                        'asin': asin,
                        'query': query_text,
                        'query_category': 'ccomp',
                        'level': level,
                        'word_count': word_count,
                        'query_type': query_type
                    })

    elif query_type == 'noisy':
        # 从合并的 noisy_query.json 加载
        noisy_file = f"{NOISY_QUERY_BASE}/{CATEGORY_NAME}/noisy_query.json"
        if not os.path.exists(noisy_file):
            log(f"  警告: Noisy 查询文件不存在: {noisy_file}")
        else:
            with open(noisy_file, 'r') as f:
                content = f.read().strip()

            if content.startswith('['):
                data = json.loads(content)
            else:
                # JSON Lines 格式
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
                doc_ids_set.add(asin)

                query_text = item.get('noisy_query', '')
                query_category = item.get('query_category', 'acl')
                level = item.get('level', 0)
                word_count = item.get('word_count', 0)

                if query_text:
                    queries.append({
                        'user_id': user_id,
                        'asin': asin,
                        'query': query_text,
                        'query_category': query_category,
                        'level': level,
                        'word_count': word_count,
                        'query_type': query_type
                    })

    return queries, doc_ids_set

def build_word_idf_dict(meta_file: str, sample_size: int = 50000) -> Dict[str, float]:
    from collections import defaultdict
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

def compute_query_idf(query_text: str, word_idf: Dict[str, float]) -> float:
    words = query_text.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]
    return np.mean(idf_values)

# ============ 搜索器 ============
class DenseSearcher:
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str], retriever_name: str):
        self.doc_ids = doc_ids
        self.retriever_name = retriever_name
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
def evaluate_retriever(retriever_name: str, queries: List[Dict], word_idf: Dict[str, float], k_values: List[int]) -> Dict:
    log(f"\n{'='*60}")
    log(f"评估 {retriever_name.upper()}")
    log(f"{'='*60}")

    if retriever_name == 'bm25':
        bm25 = load_bm25_retriever()
        searcher = BM25Searcher(bm25)

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        query_cache = load_bm25_query_cache('correct', 'acl')

        if query_cache is not None:
            log(f"  使用查询缓存...")
            all_results = []
            for query_text in all_query_texts:
                if query_text in query_cache:
                    all_results.append(query_cache[query_text])
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
        searcher = DenseSearcher(embeddings, doc_ids, retriever_name)

        all_query_texts = [q['query'] for q in queries]
        all_asins = [q['asin'] for q in queries]

        log(f"  查询数: {len(all_query_texts)}")

        query_cache = load_query_cache(retriever_name, 'correct', 'acl')

        if query_cache is not None:
            log(f"  使用查询缓存...")
            query_embeddings = []
            for qt in all_query_texts:
                if qt in query_cache:
                    query_embeddings.append(query_cache[qt])
                else:
                    query_embeddings.append(None)
        else:
            log(f"  错误: 查询缓存不存在")
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
        'metrics': overall_metrics,
        'all_metrics': all_metrics
    }

def print_results_table(all_results: List[Dict], k_values: List[int]):
    log("\n" + "=" * 100)
    log("评估结果汇总")
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
    log(f"评估 - Arts_Crafts_and_Sewing Correct vs Noisy 查询")
    log(f"类别: {CATEGORY_NAME}")
    log("=" * 60)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]

    log("\n构建词IDF字典...")
    word_idf = build_word_idf_dict(META_FILE, sample_size=50000)

    log("\n加载查询数据...")
    queries_correct, doc_ids_set = load_queries('correct')
    queries_noisy, _ = load_queries('noisy')

    log(f"  Correct 查询数: {len(queries_correct)}")
    log(f"  Noisy 查询数: {len(queries_noisy)}")
    log(f"  相关文档数: {len(doc_ids_set)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("\n" + "=" * 60)
    log("评估 CORRECT 查询")
    log("=" * 60)

    correct_results = []
    for retriever_name in RETRIEVERS:
        try:
            result = evaluate_retriever(retriever_name, queries_correct, word_idf, k_values)
            if result:
                correct_results.append(result)
        except FileNotFoundError as e:
            log(f"  跳过 {retriever_name}: {e}")
        except Exception as e:
            log(f"  错误 {retriever_name}: {e}")

    log("\n" + "=" * 60)
    log("评估 NOISY 查询")
    log("=" * 60)

    noisy_results = []
    for retriever_name in RETRIEVERS:
        try:
            result = evaluate_retriever(retriever_name, queries_noisy, word_idf, k_values)
            if result:
                noisy_results.append(result)
        except FileNotFoundError as e:
            log(f"  跳过 {retriever_name}: {e}")
        except Exception as e:
            log(f"  错误 {retriever_name}: {e}")

    log("\n" + "=" * 60)
    log("CORRECT 查询结果")
    log("=" * 60)
    print_results_table(correct_results, k_values)

    log("\n" + "=" * 60)
    log("NOISY 查询结果")
    log("=" * 60)
    print_results_table(noisy_results, k_values)

    # 计算 CORRECT vs NOISY 差异分析
    def compute_difference_table(correct_results: List[Dict], noisy_results: List[Dict], k_values: List[int]):
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
                sign = "+" if diff > 0 else ""
                row += f" {sign:>1}{diff:>9.4f}"
            log(row)

        log("-" * 100)

    compute_difference_table(correct_results, noisy_results, k_values)

    output_file = os.path.join(OUTPUT_DIR, "correct_vs_noisy_results.json")
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'category': CATEGORY_NAME,
        'correct_results': [
            {k: v for k, v in r.items() if k != 'all_metrics' and k != 'all_raw_metrics'}
            for r in correct_results
        ] if correct_results else [],
        'noisy_results': [
            {k: v for k, v in r.items() if k != 'all_metrics' and k != 'all_raw_metrics'}
            for r in noisy_results
        ] if noisy_results else [],
    }
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n评估完成!")

if __name__ == "__main__":
    main()