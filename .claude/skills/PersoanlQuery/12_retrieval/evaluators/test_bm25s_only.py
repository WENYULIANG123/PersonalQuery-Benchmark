#!/usr/bin/env python3
"""
快速评估新BM25s实现 - 仅BM25，不评估其他检索器
使用与12_evaluate_all_users_completed_only.py相同的逻辑
"""

import json
import os
import sys
import pickle
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import concurrent.futures
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils import utils
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever

STAGE6_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query"
STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"
LOG_FILE = "/home/wlia0047/ar57/wenyu/test_bm25s_only.log"

DEFAULT_K_VALUES = [1, 3, 5, 10]


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


def load_user_queries(user_id: str) -> Dict[str, List[Dict]]:
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
        if clean_query_text:
            result['clean'].append({
                'asin': asin,
                'query': clean_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': False
            })
        
        if pq.get('modified', False):
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


def load_fullscale_asins(metadata_file: str) -> set:
    if not os.path.exists(metadata_file):
        log_with_timestamp(f"ERROR: Metadata file not found: {metadata_file}")
        return set()
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return set(metadata.keys())


def main():
    logger = setup_logging()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("BM25s优化版本快速评估")
    logger.info("=" * 80)
    
    user_ids = find_users_with_queries()
    logger.info(f"Found {len(user_ids)} users with queries")
    
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    
    logger.info("\nLoading full metadata (302,380 products)...")
    all_asins = load_fullscale_asins(metadata_file)
    all_asins_list = sorted(list(all_asins))
    
    logger.info("\nLoading user queries...")
    user_queries_map = {}
    valid_users = []
    
    for user_id in user_ids:
        queries = load_user_queries(user_id)
        if any(queries.values()):
            user_queries_map[user_id] = queries
            valid_users.append(user_id)
    
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
    
    # 直接创建新BM25实例，跳过缓存
    logger.info(f"\nCreating new BM25 instance (avoiding cached old version)...")
    from utils.retrievers import BM25
    bm25 = BM25()
    logger.info(f"  Building BM25 index...")
    bm25.fit(documents, metadata)
    
    logger.info(f"\nSuccessfully loaded BM25")
    
    results = {
        'succeeded': defaultdict(list),
        'failed': defaultdict(list),
        'scale': 'fullscale',
        'retrievers_evaluated': ['bm25']
    }
    
    total_evaluations = len(valid_users) * 2  # clean + noisy
    completed = 0
    
    logger.info(f"\nStarting evaluations ({total_evaluations} total)...")
    logger.info(f"  Users: {len(valid_users)}")
    logger.info(f"  Modes: clean, noisy")
    
    for user_id in valid_users:
        user_queries = user_queries_map[user_id]
        
        for mode in ['clean', 'noisy']:
            queries = user_queries.get(mode, [])
            if not queries:
                continue
            
            try:
                logger.info(f"Evaluating BM25 for {user_id} ({mode} mode)...")
                
                metrics = evaluate_retriever(bm25, queries, all_asins_list, DEFAULT_K_VALUES)
                
                output_data = {
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'num_queries': len(queries),
                    'num_documents': len(all_asins_list),
                    'evaluation_scale': 'fullscale (302,380 products)',
                    'k_values': DEFAULT_K_VALUES,
                    'retriever': 'bm25',
                    'query_type': 'target_user',
                    'metrics': metrics
                }
                
                user_output_dir = os.path.join(OUTPUT_DIR, user_id)
                os.makedirs(user_output_dir, exist_ok=True)
                
                output_file = os.path.join(user_output_dir, f"retrieval_bm25_{mode}_fullscale_NEW.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                logger.info(f"  ✓ Saved to {output_file}")
                
                results['succeeded']['bm25'].append({
                    'user_id': user_id,
                    'mode': mode,
                    'metrics': metrics
                })
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                results['failed']['bm25'].append({'user_id': user_id, 'mode': mode, 'error': str(e)})
            
            completed += 1
            logger.info(f"Progress: {completed}/{total_evaluations} ({100*completed/total_evaluations:.1f}%)")
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("BM25s评估完成")
    logger.info("=" * 80)
    
    success_count = len(results['succeeded']['bm25'])
    fail_count = len(results['failed']['bm25'])
    
    logger.info(f"\nSucceeded: {success_count}")
    logger.info(f"Failed: {fail_count}")
    
    if success_count > 0:
        logger.info("\n新BM25s结果统计:")
        all_p1 = []
        all_p10 = []
        
        for result in results['succeeded']['bm25']:
            metrics = result['metrics']
            all_p1.append(metrics.get('P@1', 0))
            all_p10.append(metrics.get('P@10', 0))
        
        logger.info(f"  P@1 - 平均: {sum(all_p1)/len(all_p1):.6f}, 范围: {min(all_p1):.6f}-{max(all_p1):.6f}")
        logger.info(f"  P@10- 平均: {sum(all_p10)/len(all_p10):.6f}, 范围: {min(all_p10):.6f}-{max(all_p10):.6f}")


if __name__ == "__main__":
    main()
