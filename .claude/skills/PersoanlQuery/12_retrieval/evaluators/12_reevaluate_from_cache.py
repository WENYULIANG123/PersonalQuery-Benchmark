#!/usr/bin/env python3
"""
快速重新评估缺失模型 - 从缓存加载已构建的索引
"""

import argparse
import json
import os
import sys
import pickle
from datetime import datetime
from typing import List, Dict
import concurrent.futures

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever

STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"
DEFAULT_K_VALUES = [1, 3, 5, 10]

MISSING_MODELS = ['bge', 'e5', 'minilm', 'mpnet', 'star']


def load_user_queries(user_id: str) -> Dict[str, List[Dict]]:
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    
    if not os.path.exists(query_file):
        log_with_timestamp(f"Error: Query file not found: {query_file}")
        return {}
    
    with open(query_file) as f:
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
    
    return result


def load_metadata_asins(metadata_file: str) -> List[str]:
    log_with_timestamp(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    asins = sorted(list(metadata.keys()))
    log_with_timestamp(f"✓ Loaded {len(asins)} ASINs")
    return asins


def _print_metrics_summary(retriever_name: str, user_id: str, mode: str, metrics: Dict, num_queries: int):
    log_with_timestamp(f"  ✓ {retriever_name.upper()} ({user_id}, {mode}) - {num_queries} queries")
    log_with_timestamp(f"    P@1={metrics.get('P@1', 0):.4f}  P@5={metrics.get('P@5', 0):.4f}  P@10={metrics.get('P@10', 0):.4f}")
    log_with_timestamp(f"    NDCG@1={metrics.get('NDCG@1', 0):.4f}  NDCG@5={metrics.get('NDCG@5', 0):.4f}  NDCG@10={metrics.get('NDCG@10', 0):.4f}")


def evaluate_retriever_for_user(retriever, retriever_name: str, user_id: str, user_queries: Dict, all_asins: List[str], output_dir: str):
    results = {}
    
    for mode, queries in user_queries.items():
        if not queries:
            continue
            
        log_with_timestamp(f"Evaluating {retriever_name} for {user_id} ({mode} mode)...")
        
        try:
            metrics = evaluate_retriever(retriever, queries, all_asins, DEFAULT_K_VALUES)
            _print_metrics_summary(retriever_name, user_id, mode, metrics, len(queries))
            
            output_data = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(queries),
                'num_documents': len(all_asins),
                'evaluation_scale': 'fullscale (302,380 products)',
                'k_values': DEFAULT_K_VALUES,
                'retriever': retriever_name,
                'query_type': 'target_user',
                'metrics': metrics
            }
            
            user_output_dir = os.path.join(output_dir, user_id)
            os.makedirs(user_output_dir, exist_ok=True)
            
            output_file = os.path.join(user_output_dir, f"retrieval_{retriever_name}_{mode}_fullscale.json")
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            log_with_timestamp(f"  ✓ Saved: {output_file}")
            results[mode] = metrics
            
        except Exception as e:
            log_with_timestamp(f"  ✗ Error: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate using cached retrievers')
    parser.add_argument('--user-id', default='A2MNB77YGJ3CN0')
    parser.add_argument('--models', nargs='+', default=MISSING_MODELS)
    
    args = parser.parse_args()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"FAST RE-EVALUATION (using cached indices)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"User: {args.user_id}")
    log_with_timestamp(f"Models: {', '.join(args.models)}")
    log_with_timestamp(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Load queries
    log_with_timestamp("Loading user queries...")
    user_queries = load_user_queries(args.user_id)
    total_queries = sum(len(q) for q in user_queries.values())
    if total_queries == 0:
        log_with_timestamp(f"ERROR: No queries found for user {args.user_id}")
        return
    log_with_timestamp(f"✓ Loaded {total_queries} queries (clean: {len(user_queries.get('clean', []))}, noisy: {len(user_queries.get('noisy', []))})\n")
    
    # 2. Load ASIN list
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    all_asins = load_metadata_asins(metadata_file)
    
    log_with_timestamp("Loading cached retrievers...")
    
    retrievers = {}
    for model in args.models:
        try:
            log_with_timestamp(f"  Loading {model}...")
            cache_path = f"/fs04/ar57/wenyu/result/personal_query/12_retrieval/retriever_cache/{model}_457d1871f380782c05a5d94e656fef2c.pkl"
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    retriever = pickle.load(f)
                retrievers[model] = retriever
                log_with_timestamp(f"  ✓ {model} loaded ({os.path.getsize(cache_path) / 1e9:.2f}GB)")
            else:
                log_with_timestamp(f"  ✗ Cache not found: {cache_path}")
        except Exception as e:
            log_with_timestamp(f"  ✗ Error loading {model}: {e}")
    
    log_with_timestamp(f"\nSuccessfully loaded {len(retrievers)}/{len(args.models)} retrievers\n")
    
    # 4. Evaluate
    log_with_timestamp(f"Starting evaluation of {len(retrievers)} models...")
    succeeded = []
    failed = []
    
    for model in args.models:
        if model in retrievers:
            try:
                log_with_timestamp(f"\nEvaluating {model}...")
                evaluate_retriever_for_user(
                    retrievers[model],
                    model,
                    args.user_id,
                    user_queries,
                    all_asins,
                    OUTPUT_DIR
                )
                succeeded.append(model)
            except Exception as e:
                log_with_timestamp(f"✗ Failed {model}: {e}")
                failed.append(model)
    
    # 5. Summary
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("SUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Succeeded: {len(succeeded)}/{len(args.models)} - {', '.join(succeeded)}")
    if failed:
        log_with_timestamp(f"Failed: {len(failed)} - {', '.join(failed)}")
    log_with_timestamp(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_with_timestamp("=" * 80)


if __name__ == '__main__':
    main()
