#!/usr/bin/env python3
"""
Re-evaluate missing retrievers for user A2MNB77YGJ3CN0

This script re-runs evaluation for the 5 missing models:
- BGE, E5, MiniLM, MPNet, Star

For user A2MNB77YGJ3CN0 which had incomplete evaluation.
"""

import argparse
import json
import os
import sys
import pickle
from datetime import datetime
from typing import List, Dict, Set
import concurrent.futures

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from document_manager import get_document_manager
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever


STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"

DEFAULT_K_VALUES = [1, 3, 5, 10]

# Only the missing models for this user
MISSING_MODELS = ['bge', 'e5', 'minilm', 'mpnet', 'star']


def _print_metrics_summary(retriever_name: str, user_id: str, mode: str, metrics: Dict, num_queries: int):
    """Print detailed metrics for completed evaluation"""
    log_with_timestamp(f"  ✓ {retriever_name.upper()} ({user_id}, {mode}) - {num_queries} queries")
    log_with_timestamp(f"    Precision:  P@1={metrics.get('P@1', 0):.4f}  P@3={metrics.get('P@3', 0):.4f}  P@5={metrics.get('P@5', 0):.4f}  P@10={metrics.get('P@10', 0):.4f}")
    log_with_timestamp(f"    Recall:     R@1={metrics.get('R@1', 0):.4f}  R@3={metrics.get('R@3', 0):.4f}  R@5={metrics.get('R@5', 0):.4f}  R@10={metrics.get('R@10', 0):.4f}")
    log_with_timestamp(f"    MAP:        M@1={metrics.get('MAP@1', 0):.4f}  M@3={metrics.get('MAP@3', 0):.4f}  M@5={metrics.get('MAP@5', 0):.4f}  M@10={metrics.get('MAP@10', 0):.4f}")
    log_with_timestamp(f"    NDCG:      ND@1={metrics.get('NDCG@1', 0):.4f} ND@3={metrics.get('NDCG@3', 0):.4f} ND@5={metrics.get('NDCG@5', 0):.4f} ND@10={metrics.get('NDCG@10', 0):.4f}")
    log_with_timestamp(f"    MRR:       MR@1={metrics.get('MRR@1', 0):.4f} MR@3={metrics.get('MRR@3', 0):.4f} MR@5={metrics.get('MRR@5', 0):.4f} MR@10={metrics.get('MRR@10', 0):.4f}")


def load_user_queries(user_id: str, query_mode: str = 'both') -> Dict[str, List[Dict]]:
    """Load user queries from Stage 9"""
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    
    if not os.path.exists(query_file):
        log_with_timestamp(f"Error: Query file not found for {user_id}: {query_file}")
        return {}
    
    with open(query_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    result = {'clean': [], 'noisy': []}
    
    for q in queries:
        asin = q.get('asin', '')
        if not asin:
            continue
        
        pq = q.get('personalized_query', {})
        
        clean_query_text = pq.get('original', '')
        if clean_query_text:
            result['clean'].append({
                'asin': asin,
                'query': clean_query_text,
                'type': 'target',
            })
        
        noisy_query_text = pq.get('noisy', pq.get('original', ''))
        if noisy_query_text:
            result['noisy'].append({
                'asin': asin,
                'query': noisy_query_text,
                'type': 'target',
            })
        
        if query_mode in ['both', 'noisy'] and pq.get('modified', False):
            noisy_query_text = pq.get('noisy', pq.get('original', ''))
            if noisy_query_text:
                result['noisy'].append({
                    'asin': asin,
                    'query': noisy_query_text,
                    'type': 'target',
                    'category': '',
                    'selected_attributes': [],
                    'is_noisy': True
                })
    
    return result


def load_fullscale_asins(metadata_file: str) -> Set[str]:
    """Load all 302k ASIN keys from metadata"""
    log_with_timestamp(f"Loading full metadata from {metadata_file}...")
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    asins = set(metadata.keys())
    log_with_timestamp(f"✓ Loaded {len(asins)} unique ASINs from metadata")
    
    return asins


def _evaluate_single_mode(retriever, retriever_name: str, user_id: str, mode: str, queries: List[Dict], all_asins: List[str], output_dir: str, k_values: List[int]):
    """Evaluate a single mode (clean or noisy) and save results."""
    if not queries:
        return mode, {}
    
    log_with_timestamp(f"Evaluating {retriever_name} for {user_id} ({mode} mode)...")
    
    try:
        metrics = evaluate_retriever(retriever, queries, all_asins, k_values)
        
        _print_metrics_summary(retriever_name, user_id, mode, metrics, len(queries))
        
        output_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'num_documents': len(all_asins),
            'evaluation_scale': 'fullscale (302,380 products)',
            'k_values': k_values,
            'retriever': retriever_name,
            'query_type': 'target_user',
            'metrics': metrics
        }
        
        user_output_dir = os.path.join(output_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)
        
        output_file = os.path.join(user_output_dir, f"retrieval_{retriever_name}_{mode}_fullscale.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        log_with_timestamp(f"✓ Saved results to {output_file}")
        
        return mode, metrics
        
    except Exception as e:
        log_with_timestamp(f"✗ Error evaluating {retriever_name} for {user_id} ({mode}): {e}")
        return mode, {}


def evaluate_user_with_retriever(
    retriever,
    retriever_name: str,
    user_id: str,
    user_queries: Dict[str, List[Dict]],
    all_asins: List[str],
    output_dir: str,
    k_values: List[int]
):
    """Evaluate both clean and noisy modes in parallel."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for mode, queries in user_queries.items():
            if queries:
                futures[mode] = executor.submit(
                    _evaluate_single_mode,
                    retriever, retriever_name, user_id, mode, queries,
                    all_asins, output_dir, k_values
                )
        
        for mode, future in futures.items():
            try:
                mode_result, metrics = future.result(timeout=600)
                results[mode_result] = metrics
            except Exception as e:
                log_with_timestamp(f"  ✗ Error evaluating {mode} mode for {retriever_name}/{user_id}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate missing retrievers for incomplete user')
    parser.add_argument('--user-id', default='A2MNB77YGJ3CN0', help='User ID to re-evaluate')
    parser.add_argument('--mode', default='both', choices=['clean', 'noisy', 'both'])
    parser.add_argument('--models', nargs='+', default=MISSING_MODELS, help='Models to evaluate')
    
    args = parser.parse_args()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"RE-EVALUATION OF MISSING MODELS FOR USER: {args.user_id}")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Models to evaluate: {', '.join(args.models)}")
    log_with_timestamp(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load queries
    log_with_timestamp(f"\nLoading queries for {args.user_id}...")
    user_queries = load_user_queries(args.user_id, args.mode)
    
    total_queries = sum(len(q) for q in user_queries.values())
    if total_queries == 0:
        log_with_timestamp(f"Error: No queries found for user {args.user_id}")
        return
    
    log_with_timestamp(f"✓ Loaded {total_queries} total queries")
    log_with_timestamp(f"  - Clean mode: {len(user_queries.get('clean', []))} queries")
    log_with_timestamp(f"  - Noisy mode: {len(user_queries.get('noisy', []))} queries")
    
    # Load full metadata
    log_with_timestamp("\nLoading full metadata (302,380 products)...")
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    
    all_asins = load_fullscale_asins(metadata_file)
    all_asins_list = sorted(list(all_asins))
    
    # Load metadata for document building
    log_with_timestamp("Loading metadata for document building...")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    log_with_timestamp(f"Converting {len(all_asins)} metadata entries to document format...")
    documents = []
    for i, asin in enumerate(all_asins_list):
        if i % 50000 == 0:
            log_with_timestamp(f"  Processed {i}/{len(all_asins_list)}")
        
        if asin in metadata:
            doc = metadata[asin].copy()
            doc['asin'] = asin
            documents.append(doc)
    
    log_with_timestamp(f"Built document list: {len(documents)} documents")
    
    # Get retriever manager
    log_with_timestamp("\nInitializing retriever manager...")
    rm = get_retriever_manager()
    
    # Build retrievers
    log_with_timestamp(f"\nBuilding {len(args.models)} retrievers...")
    retrievers = {}
    for retriever_name in args.models:
        try:
            log_with_timestamp(f"Building {retriever_name}...")
            retrievers[retriever_name] = rm.get_retriever(
                retriever_name, 
                documents, 
                metadata
            )
            log_with_timestamp(f"✓ Successfully built {retriever_name}")
        except Exception as e:
            log_with_timestamp(f"✗ Failed to build {retriever_name}: {e}")
    
    log_with_timestamp(f"\nSuccessfully built {len(retrievers)}/{len(args.models)} retrievers")
    
    # Evaluate retrievers
    log_with_timestamp(f"\nStarting evaluation of {len(retrievers)} retrievers...")
    
    total_evaluations = len(retrievers)
    completed = 0
    succeeded = []
    failed = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        
        for retriever_name in args.models:
            if retriever_name in retrievers:
                retriever = retrievers[retriever_name]
                future = executor.submit(
                    evaluate_user_with_retriever,
                    retriever,
                    retriever_name,
                    args.user_id,
                    user_queries,
                    all_asins_list,
                    OUTPUT_DIR,
                    DEFAULT_K_VALUES
                )
                futures.append((future, retriever_name))
        
        for future, retriever_name in futures:
            try:
                user_results = future.result(timeout=1200)
                succeeded.append(retriever_name)
                completed += 1
                log_with_timestamp(f"✓ Completed {retriever_name} ({completed}/{total_evaluations})")
            except Exception as e:
                log_with_timestamp(f"✗ Failed {retriever_name}: {e}")
                failed.append(retriever_name)
                completed += 1
    
    # Summary
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("RE-EVALUATION SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"User: {args.user_id}")
    log_with_timestamp(f"Total evaluations: {completed}/{total_evaluations}")
    log_with_timestamp(f"Succeeded: {len(succeeded)} - {', '.join(succeeded)}")
    if failed:
        log_with_timestamp(f"Failed: {len(failed)} - {', '.join(failed)}")
    log_with_timestamp(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_with_timestamp("=" * 80)


if __name__ == '__main__':
    main()
