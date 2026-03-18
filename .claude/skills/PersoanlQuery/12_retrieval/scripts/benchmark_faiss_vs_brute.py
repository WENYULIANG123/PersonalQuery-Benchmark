#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import utils, retrievers

log_with_timestamp = utils.log_with_timestamp


def create_test_documents(num_docs=10000):
    documents = []
    for i in range(num_docs):
        doc = {
            'asin': f'B{i:08d}',
            'title': f'Product {i} - Quality Item',
            'description': f'This is product {i} with comprehensive description. ' * 3,
            'price': f'${10 + (i % 100)}.99',
            'rating': f'{(i % 5) + 1}.0',
        }
        documents.append(doc)
    return documents


def benchmark_retriever(retriever_class, name, documents, queries=None):
    if queries is None:
        queries = [
            "I am looking for quality craft supplies",
            "Need a good tool for crafting projects",
            "Looking for decorative items",
            "Want professional tools",
            "Need supplies for DIY projects",
            "Looking for quality materials",
            "Need affordable options",
            "Want durable products",
        ]
    
    try:
        log_with_timestamp(f"\n{'='*70}")
        log_with_timestamp(f"Testing: {name}")
        log_with_timestamp('='*70)
        
        retriever = retriever_class()
        
        log_with_timestamp(f"\n[1/3] Building index ({len(documents)} documents)...")
        fit_start = time.time()
        retriever.fit(documents, {})
        fit_time = time.time() - fit_start
        log_with_timestamp(f"✓ Index built in {fit_time:.2f}s")
        
        log_with_timestamp(f"\n[2/3] Running queries ({len(queries)} queries)...")
        
        latencies = []
        for i, query in enumerate(queries):
            search_start = time.time()
            results = retriever.search(query, top_k=10)
            latency = time.time() - search_start
            latencies.append(latency)
            
            log_with_timestamp(f"  Query {i+1}/{len(queries)}: {latency*1000:.1f}ms - {query[:40]}...")
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        log_with_timestamp(f"\n[3/3] Performance Summary:")
        log_with_timestamp(f"  Average latency: {avg_latency*1000:.1f}ms")
        log_with_timestamp(f"  Median latency:  {median_latency*1000:.1f}ms")
        log_with_timestamp(f"  P99 latency:     {p99_latency*1000:.1f}ms")
        log_with_timestamp(f"  Total time:      {sum(latencies):.2f}s")
        
        return {
            'name': name,
            'fit_time': fit_time,
            'avg_latency': avg_latency,
            'median_latency': median_latency,
            'p99_latency': p99_latency,
            'latencies': latencies,
        }
    except Exception as e:
        log_with_timestamp(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("FAISS Performance Benchmark")
    log_with_timestamp("="*70)
    
    log_with_timestamp("\nTest 1: 10K Documents")
    documents_10k = create_test_documents(num_docs=10000)
    
    results_10k = []
    
    result = benchmark_retriever(retrievers.DenseRetriever, 
                                  "DenseRetriever (Brute Force)", 
                                  documents_10k)
    if result:
        results_10k.append(result)
    
    result = benchmark_retriever(retrievers.FAISSRetriever, 
                                  "FAISSRetriever (IVFFlat, GPU)", 
                                  documents_10k)
    if result:
        results_10k.append(result)
    
    if len(results_10k) == 2:
        brute_latency = results_10k[0]['avg_latency']
        faiss_latency = results_10k[1]['avg_latency']
        speedup = brute_latency / faiss_latency
        
        log_with_timestamp(f"\n{'='*70}")
        log_with_timestamp(f"Results (10K documents):")
        log_with_timestamp('='*70)
        log_with_timestamp(f"Brute Force (DenseRetriever): {brute_latency*1000:.1f}ms")
        log_with_timestamp(f"FAISS (IVFFlat, GPU):        {faiss_latency*1000:.1f}ms")
        log_with_timestamp(f"Speedup:                      {speedup:.2f}x")
        
        if speedup > 3:
            log_with_timestamp("✓ FAISS provides significant speedup!")
        elif speedup > 1.5:
            log_with_timestamp("✓ FAISS provides moderate speedup")
        else:
            log_with_timestamp("⚠ FAISS speedup is smaller than expected (may be I/O bound)")
    
    log_with_timestamp(f"\n{'='*70}")
    log_with_timestamp("Benchmark Complete!")
    log_with_timestamp("="*70)


if __name__ == '__main__':
    main()
