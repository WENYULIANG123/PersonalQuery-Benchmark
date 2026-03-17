#!/usr/bin/env python3
"""
Optimized Batch Retrieval Evaluation for All Users

This script efficiently evaluates multiple users by:
1. Loading documents once and caching them
2. Building retriever indices once and reusing them
3. Processing users in batches to maximize efficiency
"""

import argparse
import json
import os
import sys
import glob
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import concurrent.futures
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from document_manager import get_document_manager
from retriever_manager import get_retriever_manager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
try:
    from compare_results_updated import compare_all_users
    from full_comparison_updated import compare_all_users_full
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("Warning: Analysis scripts not found. Analysis will be skipped.")

log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever

STAGE6_DIR = "/fs04/ar57/wenyu/result/personal_query/06_query"
STAGE9_DIR = "/fs04/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/12_retrieval"
LOG_FILE = "/home/wlia0047/ar57/wenyu/stage12_optimized_batch.log"

RETRIEVER_NAMES = [
    'bm25', 'tfidf', 'dirichlet',  # Sparse
    'dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star',  # Dense
    'colbert'  # Late interaction
]

RETRIEVER_TYPES = {
    'sparse': ['bm25', 'tfidf', 'dirichlet'],
    'dense': ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star'],
    'late': []  # ['colbert']  ← 暂时禁用 ColBERT 评估
}

RETRIEVER_ORDER = ['sparse', 'dense']  # ['sparse', 'dense', 'late']  ← 移除 'late'

DEFAULT_K_VALUES = [1, 3, 5, 10]


def _clean_user_old_results(user_id: str, output_dir: str):
    """Delete old evaluation results for a user before starting new evaluation"""
    user_output_dir = os.path.join(output_dir, user_id)
    if os.path.exists(user_output_dir):
        try:
            shutil.rmtree(user_output_dir)
            log_with_timestamp(f"  🗑 Cleaned old results for {user_id}")
        except Exception as e:
            log_with_timestamp(f"  ⚠ Warning: Failed to clean old results for {user_id}: {e}")


def setup_logging():
    """Setup logging to both file and console"""
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def find_users_with_queries() -> List[str]:
    """Find all users who have completed Stage 6 query generation"""
    users = []
    
    pattern = os.path.join(STAGE6_DIR, "dual_queries_*.json")
    query_files = glob.glob(pattern)
    
    for file_path in query_files:
        filename = os.path.basename(file_path)
        if filename.startswith("dual_queries_") and filename.endswith(".json"):
            user_id = filename[13:-5]
            users.append(user_id)
    
    return sorted(users)


def load_user_queries(user_id: str, query_mode: str = 'both') -> Dict[str, List[Dict]]:
    """Load queries for a user in specified mode(s)"""
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    
    if not os.path.exists(query_file):
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
        if clean_query_text and query_mode in ['both', 'clean']:
            result['clean'].append({
                'asin': asin,
                'query': clean_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': False
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


def collect_all_asins(user_ids: List[str]) -> Set[str]:
    """Collect all ASINs across all users"""
    all_asins = set()
    
    for user_id in user_ids:
        query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
        if os.path.exists(query_file):
            with open(query_file, 'r') as f:
                data = json.load(f)
            
            for q in data.get('queries', []):
                asin = q.get('asin', '')
                if asin:
                    all_asins.add(asin)
    
    return all_asins


def evaluate_user_with_retriever(
    retriever,
    retriever_name: str,
    user_id: str,
    user_queries: Dict[str, List[Dict]],
    all_asins: List[str],
    output_dir: str,
    k_values: List[int]
) -> Dict[str, Dict]:
    """Evaluate a single user with a pre-built retriever"""
    results = {}
    
    for mode, queries in user_queries.items():
        if not queries:
            continue
        
        log_with_timestamp(f"Evaluating {retriever_name} for {user_id} ({mode} mode)...")
        
        metrics = evaluate_retriever(retriever, queries, all_asins, k_values)
        
        output_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'num_documents': len(all_asins),
            'k_values': k_values,
            'retriever': retriever_name,
            'query_type': 'target_user',
            'metrics': metrics
        }
        
        user_output_dir = os.path.join(output_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)
        
        output_file = os.path.join(user_output_dir, f"retrieval_{retriever_name}_{mode}.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        results[mode] = metrics
    
    return results


def evaluate_batch(
    user_ids: List[str],
    mode: str = 'both',
    category: str = "Arts_Crafts_and_Sewing",
    parallel_retrievers: int = 4,
    logger = None,
    skip_analysis: bool = False
) -> Dict:
    """
    Evaluate a batch of users efficiently.
    
    Args:
        user_ids: List of user IDs to evaluate
        mode: 'both', 'clean', or 'noisy'
        category: Product category
        parallel_retrievers: Number of retrievers to run in parallel
        logger: Logger instance
        
    Returns:
        Dictionary of results
    """
    if logger is None:
        logger = setup_logging()
    
    dm = get_document_manager()
    rm = get_retriever_manager()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("="*80)
    logger.info("OPTIMIZED BATCH RETRIEVAL EVALUATION")
    logger.info("="*80)
    logger.info(f"Users to process: {len(user_ids)}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Category: {category}")
    logger.info(f"Parallel retrievers: {parallel_retrievers}")
    
    logger.info("\nCollecting all ASINs across users...")
    all_asins = collect_all_asins(user_ids)
    logger.info(f"Total unique ASINs: {len(all_asins)}")
    
    logger.info("\nLoading documents (cached across all evaluations)...")
    documents, metadata = dm.load_documents(category, all_asins)
    all_asins_list = list(all_asins)
    
    logger.info("\nLoading user queries...")
    user_queries_map = {}
    valid_users = []
    
    for user_id in user_ids:
        queries = load_user_queries(user_id, mode)
        if any(queries.values()):
            user_queries_map[user_id] = queries
            valid_users.append(user_id)
        else:
            logger.warning(f"Skipping {user_id} - no valid queries found")
    
    logger.info(f"Valid users with queries: {len(valid_users)}")
    
    logger.info("\nBuilding retrievers (cached for reuse)...")
    retrievers = {}
    enabled_retrievers = []
    for retriever_type in RETRIEVER_ORDER:
        enabled_retrievers.extend(RETRIEVER_TYPES[retriever_type])
    
    for retriever_name in enabled_retrievers:
        try:
            logger.info(f"Building {retriever_name}...")
            retrievers[retriever_name] = rm.get_retriever(retriever_name, documents, metadata)
        except Exception as e:
            logger.error(f"Failed to build {retriever_name}: {e}")
    
    logger.info(f"\nSuccessfully built {len(retrievers)} retrievers")
    
    logger.info("\nCleaning old results for all users...")
    for user_id in valid_users:
        _clean_user_old_results(user_id, OUTPUT_DIR)
    
    results = {
        'succeeded': defaultdict(list),
        'failed': defaultdict(list),
        'skip_analysis': skip_analysis
    }
    
    logger.info(f"\nRetriever evaluation order: {' → '.join(RETRIEVER_ORDER)}")
    logger.info(f"  Sparse: {RETRIEVER_TYPES['sparse']}")
    logger.info(f"  Dense:  {RETRIEVER_TYPES['dense']}")
    logger.info(f"  Late:   {RETRIEVER_TYPES['late']}")
    
    total_evaluations = len(valid_users) * sum(len(RETRIEVER_TYPES[t]) for t in RETRIEVER_ORDER)
    completed = 0
    
    logger.info(f"\nStarting evaluations ({total_evaluations} total)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_retrievers) as executor:
        futures = []
        
        for user_id in valid_users:
            for retriever_type in RETRIEVER_ORDER:
                for retriever_name in RETRIEVER_TYPES[retriever_type]:
                    if retriever_name in retrievers:
                        retriever = retrievers[retriever_name]
                        future = executor.submit(
                            evaluate_user_with_retriever,
                            retriever,
                            retriever_name,
                            user_id,
                            user_queries_map[user_id],
                            all_asins_list,
                            OUTPUT_DIR,
                            DEFAULT_K_VALUES
                        )
                        futures.append((future, user_id, retriever_name))
        
        for future, user_id, retriever_name in futures:
            try:
                user_results = future.result(timeout=300)
                results['succeeded'][user_id].append(retriever_name)
                completed += 1
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total_evaluations} ({100*completed/total_evaluations:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Failed {retriever_name} for {user_id}: {e}")
                results['failed'][user_id].append(retriever_name)
                completed += 1
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total users processed: {len(valid_users)}")
    logger.info(f"Total evaluations: {completed}/{total_evaluations}")
    logger.info(f"Document cache stats: {dm.get_cache_stats()}")
    logger.info(f"Retriever cache stats: {rm.get_cache_stats()}")
    
    if mode == 'both':
        for user_id in valid_users:
            try:
                generate_user_impact_analysis(user_id, OUTPUT_DIR, logger)
            except Exception as e:
                logger.error(f"Failed to generate impact analysis for {user_id}: {e}")
    
    # Run analysis if available and all evaluations completed
    if ANALYSIS_AVAILABLE and completed == total_evaluations and not results.get('skip_analysis', False):
        logger.info("\n" + "="*80)
        logger.info("RUNNING PERFORMANCE ANALYSIS")
        logger.info("="*80)
        
        try:
            # Redirect stdout to capture analysis output
            import io
            from contextlib import redirect_stdout
            
            # Simple comparison analysis
            logger.info("\nRunning simple comparison analysis...")
            simple_output = io.StringIO()
            with redirect_stdout(simple_output):
                compare_all_users(OUTPUT_DIR)
            
            for line in simple_output.getvalue().split('\n'):
                if line.strip():
                    logger.info(line)
            
            # Full metrics comparison
            logger.info("\nRunning full metrics comparison...")
            full_output = io.StringIO()
            with redirect_stdout(full_output):
                compare_all_users_full(OUTPUT_DIR)
            
            # Log only the summary sections
            output_lines = full_output.getvalue().split('\n')
            in_summary = False
            for line in output_lines:
                if '跨用户平均性能' in line or 'Stage 12' in line:
                    in_summary = True
                if in_summary and line.strip():
                    logger.info(line)
            
            logger.info("\nAnalysis complete. Full details logged above.")
            
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
    
    return results


def generate_user_impact_analysis(user_id: str, output_dir: str, logger):
    """Generate impact analysis for a single user"""
    results = {}
    pattern = os.path.join(output_dir, f"retrieval_*_{user_id}.json")
    result_files = glob.glob(pattern)
    
    for file in result_files:
        try:
            with open(file) as f:
                data = json.load(f)
                basename = os.path.basename(file)
                parts = basename.replace('.json', '').split('_')
                method = parts[1]
                mode = parts[2]
                key = f"{method}_{mode}"
                results[key] = data['metrics']
        except Exception as e:
            continue
    
    impacts = []
    methods = list(set([k.split('_')[0] for k in results.keys()]))
    
    for method in methods:
        clean = results.get(f"{method}_clean", {})
        noisy = results.get(f"{method}_noisy", {})
        
        if not clean or not noisy:
            continue
        
        avg_impact = sum([
            noisy.get('P@1', 0) - clean.get('P@1', 0),
            noisy.get('MAP@10', 0) - clean.get('MAP@10', 0),
            noisy.get('NDCG@10', 0) - clean.get('NDCG@10', 0),
            noisy.get('MRR@10', 0) - clean.get('MRR@10', 0)
        ]) / 4
        
        impacts.append({
            'method': method,
            'avg_impact': avg_impact,
            'clean_ndcg10': clean.get('NDCG@10', 0),
            'noisy_ndcg10': noisy.get('NDCG@10', 0)
        })
    
    if impacts:
        impacts.sort(key=lambda x: x['avg_impact'])
        
        user_output_dir = os.path.join(output_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)
        impact_file = os.path.join(user_output_dir, "impact_ranking_optimized.txt")
        with open(impact_file, 'w') as f:
            f.write(f"Impact Analysis for User {user_id}\n")
            f.write("="*50 + "\n")
            for i, impact in enumerate(impacts, 1):
                f.write(f"{i}. {impact['method']}: {impact['avg_impact']:+.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized batch retrieval evaluation for multiple users",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--mode",
                       choices=["both", "clean", "noisy"],
                       default="both",
                       help="Query mode: both, clean, or noisy (default: both)")
    parser.add_argument("--user-ids",
                       nargs="+",
                       help="Specific user IDs to process (default: all users)")
    parser.add_argument("--parallel",
                       type=int,
                       default=4,
                       help="Number of retrievers to run in parallel (default: 4)")
    parser.add_argument("--clear-cache",
                       action="store_true",
                       help="Clear all caches before starting")
    parser.add_argument("--skip-analysis",
                       action="store_true",
                       help="Skip automatic analysis after evaluation completes")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    start_time = datetime.now()
    
    if args.clear_cache:
        logger.info("Clearing caches...")
        dm = get_document_manager()
        rm = get_retriever_manager()
        dm.clear_cache()
        rm.clear_cache()
    
    if args.user_ids:
        users_to_process = args.user_ids
    else:
        users_to_process = find_users_with_queries()
    
    if not users_to_process:
        logger.error("No users found to process!")
        sys.exit(1)
    
    results = evaluate_batch(
        users_to_process,
        mode=args.mode,
        parallel_retrievers=args.parallel,
        logger=logger,
        skip_analysis=args.skip_analysis
    )
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    logger.info(f"\nTotal execution time: {total_time/60:.2f} minutes")
    logger.info(f"Log saved to: {LOG_FILE}")
    
    if results['failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()