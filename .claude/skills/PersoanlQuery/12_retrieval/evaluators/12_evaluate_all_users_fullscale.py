#!/usr/bin/env python3
"""
FULL-SCALE Retrieval Evaluation for All Users (302k products)

Evaluates all retrievers on the complete product catalog (302,380 products)
instead of just the 535 products that appear in user queries.

This provides a more realistic evaluation of model performance on the full corpus.
"""

import argparse
import json
import os
import sys
import glob
import pickle
import shutil
import traceback
import psutil
import torch
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import concurrent.futures
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from document_manager import get_document_manager
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever


def _print_metrics_summary(retriever_name: str, user_id: str, mode: str, metrics: Dict, num_queries: int):
    """Print detailed metrics for completed evaluation"""
    log_with_timestamp(f"  ✓ {retriever_name.upper()} ({user_id}, {mode}) - {num_queries} queries")
    log_with_timestamp(f"    Precision:  P@1={metrics.get('P@1', 0):.4f}  P@3={metrics.get('P@3', 0):.4f}  P@5={metrics.get('P@5', 0):.4f}  P@10={metrics.get('P@10', 0):.4f}")
    log_with_timestamp(f"    Recall:     R@1={metrics.get('R@1', 0):.4f}  R@3={metrics.get('R@3', 0):.4f}  R@5={metrics.get('R@5', 0):.4f}  R@10={metrics.get('R@10', 0):.4f}")
    log_with_timestamp(f"    MAP:        M@1={metrics.get('MAP@1', 0):.4f}  M@3={metrics.get('MAP@3', 0):.4f}  M@5={metrics.get('MAP@5', 0):.4f}  M@10={metrics.get('MAP@10', 0):.4f}")
    log_with_timestamp(f"    NDCG:      ND@1={metrics.get('NDCG@1', 0):.4f} ND@3={metrics.get('NDCG@3', 0):.4f} ND@5={metrics.get('NDCG@5', 0):.4f} ND@10={metrics.get('NDCG@10', 0):.4f}")
    log_with_timestamp(f"    MRR:       MR@1={metrics.get('MRR@1', 0):.4f} MR@3={metrics.get('MRR@3', 0):.4f} MR@5={metrics.get('MRR@5', 0):.4f} MR@10={metrics.get('MRR@10', 0):.4f}")


def _clean_user_old_results(user_id: str, output_dir: str):
    """Delete old evaluation results for a user before starting new evaluation"""
    user_output_dir = os.path.join(output_dir, user_id)
    if os.path.exists(user_output_dir):
        try:
            shutil.rmtree(user_output_dir)
            log_with_timestamp(f"  🗑 Cleaned old results for {user_id}")
        except Exception as e:
            log_with_timestamp(f"  ⚠ Warning: Failed to clean old results for {user_id}: {e}")


STAGE6_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query"
STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"
LOG_FILE = "/home/wlia0047/ar57/wenyu/stage12_fullscale_evaluation.log"

RETRIEVER_NAMES = [
    'bm25', 'tfidf', 'dirichlet',
    'dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star',
    'colbert'
]

DEFAULT_K_VALUES = [1, 3, 5, 10]

RETRIEVER_TYPES = {
    'sparse': [],  # TEMPORARILY DISABLED FOR DEBUGGING DENSE TIMEOUT ISSUE
    'dense': ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star'],
    'late': []  # ['colbert']  ← 暂时禁用 ColBERT 评估
}

RETRIEVER_ORDER = ['dense']  # FOCUS: Dense retriever debugging only (skipping sparse)


def setup_logging():
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
        
        if query_mode in ['both', 'noisy']:
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


def _evaluate_single_mode(retriever, retriever_name: str, user_id: str, mode: str, queries: List[Dict], all_asins: List[str], output_dir: str, k_values: List[int]) -> Tuple[str, Dict]:
    """Evaluate a single mode (clean or noisy) and save results."""
    if not queries:
        return mode, {}
    
    try:
        log_with_timestamp(f"[EVAL_MODE_START] {retriever_name}/{user_id} ({mode}): {len(queries)} queries, {len(all_asins)} products")
        
        # Log system resources before evaluation
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_START] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU: {gpu_mem:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
        metrics = evaluate_retriever(retriever, queries, all_asins, k_values, mode=mode)
        
        _print_metrics_summary(retriever_name, user_id, mode, metrics, len(queries))
        
        # Log system resources after evaluation
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_END] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU: {gpu_mem:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
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
        
        log_with_timestamp(f"[EVAL_MODE_SUCCESS] {retriever_name}/{user_id} ({mode}) completed")
        return mode, metrics
        
    except Exception as e:
        log_with_timestamp(f"[EVAL_MODE_ERROR] {retriever_name}/{user_id} ({mode}) FAILED")
        log_with_timestamp(f"  Exception Type: {type(e).__name__}")
        log_with_timestamp(f"  Exception Message: {str(e)}")
        log_with_timestamp(f"  Exception Args: {e.args}")
        
        # Log full traceback
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            if line.strip():
                log_with_timestamp(f"  Traceback: {line}")
        
        # Log resource state at error time
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_AT_ERROR] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU Allocated: {gpu_mem:.2f}GB | Reserved: {gpu_reserved:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_AT_ERROR_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
        raise


def evaluate_user_with_retriever(
    retriever,
    retriever_name: str,
    user_id: str,
    user_queries: Dict[str, List[Dict]],
    all_asins: List[str],
    output_dir: str,
    k_values: List[int]
) -> Dict[str, Dict]:
    """并发评估 clean 和 noisy 模式，串行处理不同检索器和用户。"""
    results = {}
    
    try:
        log_with_timestamp(f"[RETRIEVER_EVAL_START] {retriever_name}/{user_id} - Submitting clean/noisy evaluation tasks")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for mode, queries in user_queries.items():
                if queries:
                    log_with_timestamp(f"[THREAD_SUBMIT] {retriever_name}/{user_id} ({mode}): {len(queries)} queries submitted to thread pool")
                    futures[mode] = executor.submit(
                        _evaluate_single_mode,
                        retriever, retriever_name, user_id, mode, queries,
                        all_asins, output_dir, k_values
                    )
            
            log_with_timestamp(f"[WAITING_RESULTS] {retriever_name}/{user_id}: Waiting for {len(futures)} mode(s) to complete (timeout: 1800s)")
            
            for mode, future in futures.items():
                try:
                    log_with_timestamp(f"[THREAD_WAITING] {retriever_name}/{user_id} ({mode}): Waiting for result...")
                    mode_result, metrics = future.result(timeout=1800)
                    results[mode_result] = metrics
                    log_with_timestamp(f"[THREAD_COMPLETED] {retriever_name}/{user_id} ({mode}): Successfully retrieved results")
                    
                except concurrent.futures.TimeoutError as te:
                    log_with_timestamp(f"[TIMEOUT_ERROR] {retriever_name}/{user_id} ({mode}): TIMEOUT after 1800 seconds")
                    log_with_timestamp(f"  Exception Type: TimeoutError")
                    log_with_timestamp(f"  Exception Message: {str(te)}")
                    log_with_timestamp(f"  [RESOURCE_AT_TIMEOUT] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                    if torch.cuda.is_available():
                        log_with_timestamp(f"  [GPU_AT_TIMEOUT] GPU Memory: {torch.cuda.memory_allocated()/(1024**3):.2f}GB")
                    
                    # Log thread dump to diagnose deadlock
                    import sys
                    log_with_timestamp(f"  [THREAD_DUMP] Stuck thread(s):")
                    for thread_id, frame in sys._current_frames().items():
                        log_with_timestamp(f"    Thread {thread_id}:")
                        for line in traceback.format_stack(frame):
                            if line.strip():
                                log_with_timestamp(f"      {line.strip()}")
                
                except Exception as e:
                    log_with_timestamp(f"[THREAD_ERROR] {retriever_name}/{user_id} ({mode}): EXCEPTION occurred")
                    log_with_timestamp(f"  Exception Type: {type(e).__name__}")
                    log_with_timestamp(f"  Exception Message: {str(e)}")
                    log_with_timestamp(f"  Exception Args: {e.args}")
                    
                    tb_lines = traceback.format_exc().split('\n')
                    log_with_timestamp(f"  [FULL_TRACEBACK] ({len(tb_lines)} lines):")
                    for i, line in enumerate(tb_lines):
                        if line.strip():
                            log_with_timestamp(f"    Line {i}: {line}")
                    
                    log_with_timestamp(f"  [RESOURCE_AT_THREAD_ERROR] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                    if torch.cuda.is_available():
                        log_with_timestamp(f"  [GPU_AT_THREAD_ERROR] Allocated: {torch.cuda.memory_allocated()/(1024**3):.2f}GB | Reserved: {torch.cuda.memory_reserved()/(1024**3):.2f}GB")
        
        log_with_timestamp(f"[RETRIEVER_EVAL_DONE] {retriever_name}/{user_id}: Completed with {len(results)} mode(s)")
        
    except Exception as e:
        log_with_timestamp(f"[RETRIEVER_EVAL_CRITICAL_ERROR] {retriever_name}/{user_id}: CRITICAL ERROR in thread pool management")
        log_with_timestamp(f"  Exception Type: {type(e).__name__}")
        log_with_timestamp(f"  Exception Message: {str(e)}")
        
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            if line.strip():
                log_with_timestamp(f"  Traceback: {line}")
    
    return results


def evaluate_batch_fullscale(
    user_ids: List[str],
    mode: str = 'both',
    category: str = "Arts_Crafts_and_Sewing",
    parallel_retrievers: int = 2,
    logger = None
) -> Dict:
    
    if logger is None:
        logger = setup_logging()
    
    dm = get_document_manager()
    rm = get_retriever_manager()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("FULL-SCALE RETRIEVAL EVALUATION (302,380 products)")
    logger.info("=" * 80)
    logger.info(f"Users to process: {len(user_ids)}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Category: {category}")
    
    logger.info("\nLoading full metadata (302,380 products)...")
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    
    all_asins = load_fullscale_asins(metadata_file)
    all_asins_list = sorted(list(all_asins))
    
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
    
    logger.info("\nLoading metadata for document building...")
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Converting {len(all_asins)} metadata entries to document format...")
    documents = []
    for i, asin in enumerate(all_asins_list):
        if i % 50000 == 0:
            logger.info(f"  Processed {i}/{len(all_asins_list)}")
        
        if asin in metadata:
            doc = metadata[asin].copy()
            doc['asin'] = asin
            documents.append(doc)
    
    logger.info(f"Built document list: {len(documents)} documents")
    
    retrievers = {}
    enabled_retrievers = []
    for retriever_type in RETRIEVER_ORDER:
        enabled_retrievers.extend(RETRIEVER_TYPES[retriever_type])
    
    logger.info(f"[LAZY_INIT_START] Creating lazy proxies for {len(enabled_retrievers)} retrievers...")
    for retriever_name in enabled_retrievers:
        try:
            logger.info(f"[LAZY_PROXY_CREATE] {retriever_name}")
            retrievers[retriever_name] = rm.create_lazy_proxy(
                retriever_name, 
                documents, 
                metadata,
                use_lazy_loading=True
            )
        except Exception as e:
            logger.error(f"Failed to create proxy for {retriever_name}: {e}")
    
    logger.info(f"[LAZY_INIT_DONE] Created proxies for {len(retrievers)} retrievers (actual loading deferred)")
    
    print(f"\n{'='*80}\n[LAZY_LOADING_STATUS]\n{'='*80}", flush=True)
    for retriever_name, proxy in retrievers.items():
        is_loaded, retriever_type = proxy.get_loaded_status()
        status = f"LOADED ({retriever_type})" if is_loaded else "NOT LOADED YET"
        logger.info(f"  {retriever_name}: {status}")
    print(f"{'='*80}\n", flush=True)
    
    logger.info(f"\nCleaning old results for {len(valid_users)} users...")
    for user_id in valid_users:
        _clean_user_old_results(user_id, OUTPUT_DIR)
    
    results = {
        'succeeded': defaultdict(list),
        'failed': defaultdict(list),
        'scale': 'fullscale'
    }
    
    total_retriever_names = sum(len(names) for names in RETRIEVER_TYPES.values())
    total_evaluations = len(valid_users) * total_retriever_names
    completed = 0
    
    logger.info(f"\nStarting evaluations ({total_evaluations} total)...")
    logger.info(f"Retriever order: {' → '.join(RETRIEVER_ORDER)}")
    logger.info(f"  Sparse ({len(RETRIEVER_TYPES['sparse'])}): {', '.join(RETRIEVER_TYPES['sparse'])}")
    logger.info(f"  Dense ({len(RETRIEVER_TYPES['dense'])}): {', '.join(RETRIEVER_TYPES['dense'])}")
    logger.info(f"  Late ({len(RETRIEVER_TYPES['late'])}): {', '.join(RETRIEVER_TYPES['late'])}")
    
    logger.info("\n[SERIAL_USERS] Users processed serially (one user completes before next starts)")
    logger.info("[SELECTIVE_CONCURRENCY] Within user: Sparse: 8 workers → Dense: 1 worker (CPU vs GPU-bound)")
    
    sparse_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix='sparse')
    dense_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='dense')
    
    try:
        for user_idx, user_id in enumerate(valid_users, 1):
            logger.info(f"\n[USER_START] Processing user {user_idx}/{len(valid_users)}: {user_id}")
            
            for retriever_type in RETRIEVER_ORDER:
                executor = sparse_executor if retriever_type == 'sparse' else dense_executor
                phase_name = retriever_type.upper()
                
                logger.info(f"  [{phase_name}_START] Evaluating {len(RETRIEVER_TYPES[retriever_type])} {retriever_type} retrievers...")
                
                futures_for_phase = []
                
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
                        futures_for_phase.append((future, user_id, retriever_name, retriever_type))
                
                for future, user_id_inner, retriever_name, retriever_type_inner in futures_for_phase:
                    try:
                        logger.info(f"  [FUTURE_WAITING] {retriever_name} ({retriever_type_inner}) for {user_id_inner}: Waiting for result (timeout: 1800s)...")
                        user_results = future.result(timeout=1800)
                        results['succeeded'][user_id_inner].append(retriever_name)
                        completed += 1
                        logger.info(f"  [FUTURE_COMPLETED] {retriever_name} ({retriever_type_inner}) for {user_id_inner}: ✓ SUCCESS")
                        
                        if completed % 5 == 0:
                            logger.info(f"  Progress: {completed}/{total_evaluations} ({100*completed/total_evaluations:.1f}%)")
                            
                    except concurrent.futures.TimeoutError as te:
                        logger.error(f"  ✗ TIMEOUT: {retriever_name} ({retriever_type_inner}) for {user_id_inner} - exceeded 1800 seconds")
                        logger.error(f"    Exception Type: TimeoutError")
                        logger.error(f"    [RESOURCE_AT_TIMEOUT] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                        if torch.cuda.is_available():
                            logger.error(f"    [GPU_AT_TIMEOUT] GPU Memory: {torch.cuda.memory_allocated()/(1024**3):.2f}GB / Reserved: {torch.cuda.memory_reserved()/(1024**3):.2f}GB")
                        results['failed'][user_id_inner].append(retriever_name)
                        completed += 1
                        
                    except Exception as e:
                        logger.error(f"  ✗ FAILED: {retriever_name} ({retriever_type_inner}) for {user_id_inner}")
                        logger.error(f"    Exception Type: {type(e).__name__}")
                        logger.error(f"    Exception Message: {str(e)}")
                        logger.error(f"    Exception Args: {e.args}")
                        
                        tb_list = traceback.format_exc().split('\n')
                        logger.error(f"    [FULL_TRACEBACK] ({len(tb_list)} lines):")
                        for i, tb_line in enumerate(tb_list):
                            if tb_line.strip():
                                logger.error(f"      [{i}] {tb_line}")
                        
                        try:
                            cpu_p = psutil.cpu_percent(interval=0.1)
                            mem = psutil.virtual_memory()
                            logger.error(f"    [RESOURCE_AT_ERROR] CPU: {cpu_p}% | RAM: {mem.percent}% ({mem.used//(1024**3)}GB/{mem.total//(1024**3)}GB)")
                            
                            if torch.cuda.is_available():
                                gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
                                gpu_resv = torch.cuda.memory_reserved() / (1024**3)
                                try:
                                    gpu_cached = torch.cuda.memory_cached() / (1024**3)
                                    logger.error(f"    [GPU_AT_ERROR] Allocated: {gpu_alloc:.2f}GB | Reserved: {gpu_resv:.2f}GB | Cached: {gpu_cached:.2f}GB")
                                except AttributeError:
                                    logger.error(f"    [GPU_AT_ERROR] Allocated: {gpu_alloc:.2f}GB | Reserved: {gpu_resv:.2f}GB")
                        except Exception as res_e:
                            logger.error(f"    [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
                        
                        results['failed'][user_id_inner].append(retriever_name)
                        completed += 1
                
                logger.info(f"  [{phase_name}_DONE] {retriever_type.capitalize()} phase completed for {user_id}")
            
            logger.info(f"[USER_DONE] ✓ User {user_idx}/{len(valid_users)} ({user_id}) completed all evaluations")
    
    finally:
        sparse_executor.shutdown(wait=True)
        dense_executor.shutdown(wait=True)
        logger.info("[EXECUTOR_SHUTDOWN] Both sparse and dense executors shut down")
    
    logger.info("\n" + "=" * 80)
    logger.info("FULL-SCALE EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total users processed: {len(valid_users)}")
    logger.info(f"Total evaluations: {completed}/{total_evaluations}")
    logger.info(f"Evaluation scale: {len(all_asins)} products")
    logger.info(f"Document corpus size: {len(metadata)} products")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Full-scale Retrieval Evaluation')
    parser.add_argument('--mode', default='both', choices=['clean', 'noisy', 'both'])
    parser.add_argument('--parallel', type=int, default=1, help='Parallel (user, retriever) pairs (default: 1 = serial across retrievers/users, but parallel clean/noisy)')
    parser.add_argument('--users', type=int, default=11, help='Number of users to evaluate')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    user_ids = find_users_with_queries()[:args.users]
    logger.info(f"Found {len(user_ids)} users")
    
    results = evaluate_batch_fullscale(
        user_ids,
        mode=args.mode,
        parallel_retrievers=args.parallel,
        logger=logger
    )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
