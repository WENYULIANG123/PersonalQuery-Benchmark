#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import utils, retrievers

log_with_timestamp = utils.log_with_timestamp


def create_test_dataset(num_docs=5000):
    """Create a simple test dataset"""
    documents = []
    for i in range(num_docs):
        doc = {
            'asin': f'B{i:08d}',
            'title': f'Product {i} - High Quality Item for Crafting',
            'description': f'Description for product {i}. This is a test product with some description text. '
                          f'It has various features and benefits. Lorem ipsum dolor sit amet. ' * 2,
            'price': f'${10 + (i % 100)}.99',
            'rating': f'{(i % 5) + 1}.0',
        }
        documents.append(doc)
    return documents


def benchmark_retriever(retriever_class, name, documents, queries=None):
    """Benchmark a retriever"""
    log_with_timestamp(f"\n{'='*60}")
    log_with_timestamp(f"Testing {name}")
    log_with_timestamp('='*60)
    
    if queries is None:
        queries = [
            "I am looking for a nice die-cutting product",
            "Looking for craft supplies for decoration",
            "Need a quality tool for paper crafting",
            "Want embossing supplies",
            "Looking for dies and cutting tools"
        ]
    
    try:
        retriever = retriever_class()
        
        log_with_timestamp(f"\n[1/3] Building index ({len(documents)} documents)...")
        fit_start = time.time()
        retriever.fit(documents, {})
        fit_time = time.time() - fit_start
        log_with_timestamp(f"✓ Index built in {fit_time:.2f}s")
        
        if hasattr(retriever, 'doc_embeddings'):
            if torch.is_tensor(retriever.doc_embeddings):
                log_with_timestamp(f"  Embeddings device: {retriever.doc_embeddings.device}")
                log_with_timestamp(f"  Embeddings shape: {retriever.doc_embeddings.shape}")
            elif isinstance(retriever.doc_embeddings, np.ndarray):
                log_with_timestamp(f"  Embeddings: numpy array, shape {retriever.doc_embeddings.shape}")
            else:
                log_with_timestamp(f"  Embeddings: list of {len(retriever.doc_embeddings)} tensors")
        
        log_with_timestamp(f"\n[2/3] Running search queries ({len(queries)} queries)...")
        
        latencies = []
        for i, query in enumerate(queries):
            search_start = time.time()
            results = retriever.search(query, top_k=10)
            latency = time.time() - search_start
            latencies.append(latency)
            
            log_with_timestamp(f"  Query {i+1}/{len(queries)}: {latency*1000:.1f}ms - {query[:40]}...")
        
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        log_with_timestamp(f"\n[3/3] Performance Summary:")
        log_with_timestamp(f"  Average latency: {avg_latency*1000:.1f}ms")
        log_with_timestamp(f"  Median latency:  {median_latency*1000:.1f}ms")
        log_with_timestamp(f"  Min latency:     {min_latency*1000:.1f}ms")
        log_with_timestamp(f"  Max latency:     {max_latency*1000:.1f}ms")
        log_with_timestamp(f"  Total time:      {sum(latencies):.2f}s")
        
        return {
            'name': name,
            'fit_time': fit_time,
            'avg_latency': avg_latency,
            'latencies': latencies,
        }
    except Exception as e:
        log_with_timestamp(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("Dense Retriever Optimization Test")
    log_with_timestamp("="*60)
    
    log_with_timestamp("Creating test dataset (5000 documents)...")
    documents = create_test_dataset(num_docs=5000)
    
    retrievers_to_test = [
        (retrievers.DenseRetriever, "DenseRetriever (Optimized)"),
        (retrievers.BGERetriever, "BGERetriever (Optimized)"),
    ]
    
    results = []
    for retriever_class, name in retrievers_to_test:
        result = benchmark_retriever(retriever_class, name, documents)
        if result:
            results.append(result)
    
    if len(results) >= 2:
        log_with_timestamp(f"\n" + "="*60)
        log_with_timestamp("Comparison")
        log_with_timestamp("="*60)
        
        dense_latency = results[0]['avg_latency']
        bge_latency = results[1]['avg_latency']
        
        log_with_timestamp(f"DenseRetriever avg: {dense_latency*1000:.1f}ms")
        log_with_timestamp(f"BGERetriever avg:   {bge_latency*1000:.1f}ms")
        log_with_timestamp(f"Speedup: {dense_latency/bge_latency:.2f}x")
    
    log_with_timestamp(f"\n" + "="*60)
    log_with_timestamp("Test Complete!")
    log_with_timestamp("="*60)


if __name__ == '__main__':
    main()
