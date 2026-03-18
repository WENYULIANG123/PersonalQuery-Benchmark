#!/usr/bin/env python3
"""
性能基准测试：BGERetriever 池化优化 vs DenseRetriever
验证优化是否带来预期的性能改进 (目标: 7.5x加速)
"""

import os
import sys
import time
import json
import torch
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import utils, retrievers

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
load_preprocessed_products = utils.load_preprocessed_products


def load_test_data(limit_docs: int = None) -> Tuple[List[Dict], List[str], Dict]:
    """Load test documents and default queries for benchmarking"""
    log_with_timestamp("Loading product metadata...")
    all_metadata = load_product_metadata()
    
    log_with_timestamp("Loading preprocessed products...")
    products = load_preprocessed_products()
    
    if limit_docs:
        products = products[:limit_docs]
    
    documents = [
        {
            'asin': product.get('asin', ''),
            'title': product.get('title', ''),
            'description': product.get('description', ''),
            'price': product.get('price', ''),
            'rating': product.get('rating', ''),
        }
        for product in products
    ]
    
    queries = [
        "I need a good book for reading",
        "Looking for a comfortable chair",
        "What's a good camera for photography",
        "I want a laptop for programming",
        "Need kitchen tools for cooking",
    ]
    
    log_with_timestamp(f"Loaded {len(documents)} documents and {len(queries)} queries")
    return documents, queries, all_metadata


def benchmark_retriever(retriever, queries: List[str], name: str) -> Dict:
    """Benchmark a retriever on queries and return performance statistics"""
    log_with_timestamp(f"\nBenchmarking {name}...")
    
    latencies = []
    results_all = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        try:
            results = retriever.search(query, top_k=10)
            latency = time.time() - start_time
            latencies.append(latency)
            results_all.append(results)
            log_with_timestamp(f"  Query {i+1}/{len(queries)}: {latency:.3f}s - {query[:50]}...")
        except Exception as e:
            log_with_timestamp(f"  Query {i+1}/{len(queries)}: ERROR - {str(e)[:100]}")
            latencies.append(float('inf'))
    
    valid_latencies = [l for l in latencies if l != float('inf')]
    
    if valid_latencies:
        return {
            'name': name,
            'avg_latency': np.mean(valid_latencies),
            'median_latency': np.median(valid_latencies),
            'min_latency': np.min(valid_latencies),
            'max_latency': np.max(valid_latencies),
            'std_latency': np.std(valid_latencies),
            'total_queries': len(queries),
            'successful_queries': len(valid_latencies),
            'total_time': sum(valid_latencies),
            'results': results_all,
        }
    else:
        return {
            'name': name,
            'error': 'All queries failed',
            'total_queries': len(queries),
        }


def compare_results(results1: List[List[Tuple[str, float]]], 
                   results2: List[List[Tuple[str, float]]],
                   name1: str, name2: str) -> Dict:
    """Compare result accuracy between two retrievers"""
    log_with_timestamp(f"\nComparing results: {name1} vs {name2}...")
    
    comparison = {
        'name1': name1,
        'name2': name2,
        'total_queries': len(results1),
        'top1_match': 0,
        'top10_overlap': [],
    }
    
    for r1, r2 in zip(results1, results2):
        if not r1 or not r2:
            continue
        
        if r1[0][0] == r2[0][0]:
            comparison['top1_match'] += 1
        
        ids1 = set(doc_id for doc_id, _ in r1[:10])
        ids2 = set(doc_id for doc_id, _ in r2[:10])
        overlap = len(ids1 & ids2) / 10.0
        comparison['top10_overlap'].append(overlap)
    
    if comparison['top10_overlap']:
        comparison['avg_top10_overlap'] = np.mean(comparison['top10_overlap'])
        comparison['min_top10_overlap'] = np.min(comparison['top10_overlap'])
    
    log_with_timestamp(f"  Top-1 match rate: {comparison['top1_match']}/{len(results1)}")
    if comparison['top10_overlap']:
        log_with_timestamp(f"  Avg top-10 overlap: {comparison['avg_top10_overlap']:.2%}")
    
    return comparison


def main():
    """Run benchmark"""
    log_with_timestamp("=== BGE Optimization Benchmark ===")
    
    log_with_timestamp("\n[1/4] Loading test data...")
    try:
        documents, queries, all_metadata = load_test_data(limit_docs=10000)
    except Exception as e:
        log_with_timestamp(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    log_with_timestamp("\n[2/4] Initializing retrievers...")
    
    dense_retriever = None
    bge_retriever = None
    
    try:
        dense_retriever = retrievers.DenseRetriever()
        log_with_timestamp("  ✓ DenseRetriever initialized")
    except Exception as e:
        log_with_timestamp(f"  ✗ DenseRetriever init failed: {e}")
    
    try:
        bge_retriever = retrievers.BGERetriever()
        log_with_timestamp("  ✓ BGERetriever initialized")
    except Exception as e:
        log_with_timestamp(f"  ✗ BGERetriever init failed: {e}")
    
    if not dense_retriever or not bge_retriever:
        log_with_timestamp("ERROR: Failed to initialize retrievers")
        return
    
    log_with_timestamp("\n[3/4] Building indices...")
    
    try:
        start_time = time.time()
        dense_retriever.fit(documents, all_metadata)
        dense_fit_time = time.time() - start_time
        log_with_timestamp(f"  ✓ DenseRetriever.fit() completed in {dense_fit_time:.2f}s")
    except Exception as e:
        log_with_timestamp(f"  ✗ DenseRetriever.fit() failed: {e}")
        import traceback
        traceback.print_exc()
        dense_retriever = None
    
    try:
        start_time = time.time()
        bge_retriever.fit(documents, all_metadata)
        bge_fit_time = time.time() - start_time
        log_with_timestamp(f"  ✓ BGERetriever.fit() completed in {bge_fit_time:.2f}s")
    except Exception as e:
        log_with_timestamp(f"  ✗ BGERetriever.fit() failed: {e}")
        import traceback
        traceback.print_exc()
        bge_retriever = None
    
    if not dense_retriever or not bge_retriever:
        log_with_timestamp("ERROR: Failed to build indices")
        return
    
    log_with_timestamp("\n[4/4] Running queries benchmark...")
    
    dense_stats = benchmark_retriever(dense_retriever, queries, "DenseRetriever")
    bge_stats = benchmark_retriever(bge_retriever, queries, "BGERetriever (pooled)")
    
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("BENCHMARK RESULTS")
    log_with_timestamp("="*60)
    
    if 'error' not in dense_stats:
        log_with_timestamp(f"\n{dense_stats['name']}:")
        log_with_timestamp(f"  Avg latency per query: {dense_stats['avg_latency']:.3f}s")
        log_with_timestamp(f"  Total queries: {dense_stats['successful_queries']}/{dense_stats['total_queries']}")
        log_with_timestamp(f"  Total time: {dense_stats['total_time']:.2f}s")
    else:
        log_with_timestamp(f"\n{dense_stats['name']}: {dense_stats['error']}")
    
    if 'error' not in bge_stats:
        log_with_timestamp(f"\n{bge_stats['name']}:")
        log_with_timestamp(f"  Avg latency per query: {bge_stats['avg_latency']:.3f}s")
        log_with_timestamp(f"  Total queries: {bge_stats['successful_queries']}/{bge_stats['total_queries']}")
        log_with_timestamp(f"  Total time: {bge_stats['total_time']:.2f}s")
        
        if 'error' not in dense_stats:
            speedup = dense_stats['avg_latency'] / bge_stats['avg_latency']
            log_with_timestamp(f"\n  Expected speedup (7.5x), actual: {speedup:.2f}x")
    else:
        log_with_timestamp(f"\n{bge_stats['name']}: {bge_stats['error']}")
    
    if 'error' not in dense_stats and 'error' not in bge_stats:
        comparison = compare_results(
            dense_stats['results'],
            bge_stats['results'],
            dense_stats['name'],
            bge_stats['name']
        )
        
        log_with_timestamp(f"\n{comparison['name1']} vs {comparison['name2']}:")
        log_with_timestamp(f"  Top-1 match rate: {comparison['top1_match']}/{comparison['total_queries']}")
        if 'avg_top10_overlap' in comparison:
            log_with_timestamp(f"  Avg top-10 overlap: {comparison['avg_top10_overlap']:.2%}")
    
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("Benchmark completed!")
    
    return {
        'dense': dense_stats,
        'bge': bge_stats,
    }


if __name__ == '__main__':
    main()
