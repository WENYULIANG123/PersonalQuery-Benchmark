#!/usr/bin/env python3
"""
性能剖析：分解BGERetriever.search()的各个步骤耗时
"""

import os
import sys
import time
import json
import torch
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import utils, retrievers

log_with_timestamp = utils.log_with_timestamp


def profile_bge_search():
    log_with_timestamp("=== BGE Search Performance Profiling ===")
    
    log_with_timestamp("\n[1] Building test index with 302K documents...")
    
    try:
        all_asins = utils.load_all_asins()
        log_with_timestamp(f"  Loaded {len(all_asins)} ASINs")
    except Exception as e:
        log_with_timestamp(f"  Warning: Could not load all ASINs: {e}")
        all_asins = list(range(302380))
    
    try:
        all_metadata = utils.load_metadata_for_asins(all_asins)
    except:
        all_metadata = {}
    
    bge = retrievers.BGERetriever()
    
    start_fit = time.time()
    
    try:
        documents = [{'asin': str(i), 'title': f'Product {i}', 'description': f'Description {i}'} 
                     for i in range(min(100, len(all_asins)))]
        bge.fit(documents, all_metadata)
        fit_time = time.time() - start_fit
        log_with_timestamp(f"  ✓ Index built in {fit_time:.2f}s")
    except Exception as e:
        log_with_timestamp(f"  ✗ Failed to build index: {e}")
        return
    
    log_with_timestamp("\n[2] Profiling single query search...")
    
    query = "I am looking for a nice book to read"
    
    timings = {
        'total': 0,
        'model_inference': 0,
        'cosine_similarity': 0,
        'result_sorting': 0,
        'other': 0,
    }
    
    for iteration in range(3):
        log_with_timestamp(f"\n  Iteration {iteration + 1}/3:")
        
        total_start = time.time()
        
        model = bge._get_model()
        
        inference_start = time.time()
        with retrievers._model_inference_lock:
            query_with_prefix = bge._add_instruction(query, is_query=True)
            query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)[0]
        timings['model_inference'] = time.time() - inference_start
        log_with_timestamp(f"    Model inference: {timings['model_inference']:.3f}s")
        
        doc_embeddings = bge.doc_embeddings
        if isinstance(doc_embeddings, list):
            doc_embeddings = torch.stack(doc_embeddings)
        
        similarity_start = time.time()
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        timings['cosine_similarity'] = time.time() - similarity_start
        log_with_timestamp(f"    Cosine similarity: {timings['cosine_similarity']:.3f}s")
        
        sort_start = time.time()
        results = [(bge.doc_ids[i], scores[i].item()) for i in range(len(bge.doc_ids))]
        results.sort(key=lambda x: -x[1])
        results = results[:10]
        timings['result_sorting'] = time.time() - sort_start
        log_with_timestamp(f"    Result sorting: {timings['result_sorting']:.3f}s")
        
        timings['total'] = time.time() - total_start
        log_with_timestamp(f"    Total: {timings['total']:.3f}s")
    
    log_with_timestamp("\n[3] Summary")
    log_with_timestamp(f"  Model inference:   {timings['model_inference']:.3f}s ({timings['model_inference']/timings['total']*100:.1f}%)")
    log_with_timestamp(f"  Cosine similarity: {timings['cosine_similarity']:.3f}s ({timings['cosine_similarity']/timings['total']*100:.1f}%)")
    log_with_timestamp(f"  Result sorting:    {timings['result_sorting']:.3f}s ({timings['result_sorting']/timings['total']*100:.1f}%)")
    log_with_timestamp(f"  Total:             {timings['total']:.3f}s")


if __name__ == '__main__':
    profile_bge_search()
