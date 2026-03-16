#!/usr/bin/env python3
"""Stage 13: BM25 Retrieval Evaluation (Standalone)"""

import argparse
import json
import os
import sys
import hashlib
from datetime import datetime

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/13_retrieval")

from utils.utils import (
    log_with_timestamp,
    load_product_metadata,
    load_reviews_for_products,
    load_qa_for_products,
    build_document_text,
    evaluate_retriever,
    load_preprocessed_products
)
from utils.retrievers import BM25


def get_doc_fingerprint(doc, all_metadata):
    """Generate MD5 fingerprint of document text for verification"""
    text = build_document_text(doc, all_metadata)
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_data_source_info(use_preprocessed, cache_dir):
    """Get data source information and timestamp"""
    if use_preprocessed:
        products_file = os.path.join(cache_dir, "products_Arts_Crafts_and_Sewing.pkl")
        if os.path.exists(products_file):
            timestamp = datetime.fromtimestamp(os.path.getmtime(products_file)).isoformat()
            return {'source': 'cached', 'timestamp': timestamp, 'cache_file': products_file}
    return {'source': 'raw', 'timestamp': datetime.now().isoformat()}

# Default configuration
DEFAULT_USER_ID = "A13OFOB1394G31"
DEFAULT_BASE_DIR = "/home/wlia0047/ar57/wenyu"
DEFAULT_QUERY_FILE = os.path.join(DEFAULT_BASE_DIR, "result/personal_query/10_targeted_noisy_query/noisy_queries_A13OFOB1394G31.json")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "result/personal_query/13_retrieval")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_BASE_DIR, "result/personal_query/13_retrieval/cache")
DEFAULT_MATCH_FILE = os.path.join(DEFAULT_BASE_DIR, "result/personal_query/02_matching/match_A13OFOB1394G31.json")
DEFAULT_META_FILE = os.path.join(DEFAULT_BASE_DIR, "data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz")
DEFAULT_REVIEW_FILE = os.path.join(DEFAULT_BASE_DIR, "data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz")
DEFAULT_QA_FILE = os.path.join(DEFAULT_BASE_DIR, "data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz")
DEFAULT_CATEGORY = "Arts_Crafts_and_Sewing"
DEFAULT_USE_PREPROCESSED = True
DEFAULT_K_VALUES = [1, 3, 5, 10]

def main():
    parser = argparse.ArgumentParser(description="BM25 Retrieval Evaluation")
    parser.add_argument("--query-mode",
                        choices=['noisy', 'clean'],
                        default='noisy',
                        help="Query mode: 'noisy' uses noisy queries when modified=true, 'clean' always uses original queries (default: noisy)")
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE, help="Stage 10 query file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID")
    parser.add_argument("--force-raw-data", action="store_true",
                       help="Force loading raw data instead of cache (ensures consistency)")
    parser.add_argument("--verify-fingerprints", action="store_true", default=True,
                       help="Verify document fingerprints for consistency (default: True)")
    args = parser.parse_args()
    
    # Use argparse values or defaults
    QUERY_FILE = args.query_file
    OUTPUT_DIR = args.output_dir
    USER_ID = args.user_id
    QUERY_MODE = args.query_mode
    FORCE_RAW_DATA = args.force_raw_data
    VERIFY_FINGERPRINTS = args.verify_fingerprints

    # Fixed configuration
    CACHE_DIR = DEFAULT_CACHE_DIR
    MATCH_FILE = DEFAULT_MATCH_FILE
    META_FILE = DEFAULT_META_FILE
    REVIEW_FILE = DEFAULT_REVIEW_FILE
    QA_FILE = DEFAULT_QA_FILE
    CATEGORY = DEFAULT_CATEGORY
    USE_PREPROCESSED = DEFAULT_USE_PREPROCESSED and not FORCE_RAW_DATA
    K_VALUES = DEFAULT_K_VALUES
    
    log_with_timestamp("=" * 60)
    log_with_timestamp("BM25 Retrieval Evaluation (Standalone)")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"User ID: {USER_ID}")
    log_with_timestamp(f"Query file: {QUERY_FILE}")
    log_with_timestamp(f"Query mode: {QUERY_MODE}")
    log_with_timestamp(f"Match file: {MATCH_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_with_timestamp(f"Loading queries from {QUERY_FILE}")
    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)
    
    # Stage 10 format: 'queries' array with 'personalized_query'
    queries = data.get('queries', [])
    log_with_timestamp(f"Loaded {len(queries)} queries")
    log_with_timestamp(f"Modified: {data.get('modified_queries', 0)}, Unmodified: {data.get('unmodified_queries', 0)}")
    
    all_asins = set()
    target_queries = []
    
    for q in queries:
        asin = q.get('asin', '')
        if not asin:
            continue
        
        all_asins.add(asin)
        pq = q.get('personalized_query', {})
        
        # Select query text based on query mode
        if QUERY_MODE == 'clean':
            # Always use original (clean) query
            target_query_text = pq.get('original', '')
        else:
            # Use noisy if modified, otherwise original
            if pq.get('modified', False):
                target_query_text = pq.get('noisy', pq.get('original', ''))
            else:
                target_query_text = pq.get('original', '')
        
        if target_query_text:
            target_queries.append({
                'asin': asin,
                'query': target_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': pq.get('modified', False)
            })
    
    log_with_timestamp(f"Total unique ASINs: {len(all_asins)}")
    log_with_timestamp(f"Target queries: {len(target_queries)}")
    
    documents = []
    all_metadata: dict = {}
    products = {}

    use_preprocessed = USE_PREPROCESSED and not FORCE_RAW_DATA

    # Data source tracking
    if FORCE_RAW_DATA:
        log_with_timestamp("Force loading raw data (--force-raw-data flag set)")
    elif use_preprocessed:
        products_file = os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl")
        if os.path.exists(products_file):
            log_with_timestamp(f"Loading preprocessed data from cache...")
            products, loaded_metadata = load_preprocessed_products(CACHE_DIR, CATEGORY, all_asins)
            if loaded_metadata:
                all_metadata = loaded_metadata
        else:
            log_with_timestamp("Preprocessed data not found, using raw data...")
            use_preprocessed = False
    else:
        log_with_timestamp("Loading raw data...")

    if not use_preprocessed:
        if os.path.exists(META_FILE):
            products, loaded_metadata = load_product_metadata(META_FILE, all_asins)
            if loaded_metadata:
                all_metadata = loaded_metadata

            if os.path.exists(REVIEW_FILE):
                products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=25)

            if os.path.exists(QA_FILE):
                products = load_qa_for_products(QA_FILE, products)

    # Get data source information
    data_source_info = get_data_source_info(use_preprocessed, CACHE_DIR)
    log_with_timestamp(f"Data source: {data_source_info['source']}")
    log_with_timestamp(f"Data timestamp: {data_source_info['timestamp']}")
    
    for asin in all_asins:
        if asin in products:
            documents.append(products[asin])
        else:
            documents.append({
                'asin': asin,
                'title': '',
                'brand': '',
                'category': [],
                'feature': [],
                'description': [],
                'rank': '',
                'also_buy': [],
                'also_view': [],
                'reviews': []
            })

    log_with_timestamp(f"Loaded {len(documents)} documents for retrieval")

    # Generate document fingerprints for verification
    doc_fingerprints = []
    if VERIFY_FINGERPRINTS:
        log_with_timestamp("Generating document fingerprints...")
        for i, doc in enumerate(documents):
            fp = get_doc_fingerprint(doc, all_metadata)
            doc_fingerprints.append({
                'asin': doc.get('asin', ''),
                'fingerprint': fp,
                'doc_length': len(build_document_text(doc, all_metadata))
            })
            if i < 3:  # Log first 3 for verification
                log_with_timestamp(f"  {doc.get('asin', '')}: {fp[:16]}...")
    
    log_with_timestamp("=" * 50)
    log_with_timestamp("Building BM25...")
    
    if documents:
        first_doc = documents[0]
        first_text = build_document_text(first_doc, all_metadata)
        first_tokens = first_text.lower().split()
        log_with_timestamp(f"DEBUG: First doc has {len(first_doc.get('feature', []))} features, {len(first_doc.get('reviews', []))} reviews")
        log_with_timestamp(f"DEBUG: First doc text: {len(first_text)} chars, {len(first_tokens)} tokens")
    
    bm25 = BM25()
    bm25.fit(documents, all_metadata)
    
    common_metadata = {
        'user_id': USER_ID,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries),
        'num_documents': len(documents),
        'k_values': K_VALUES,
        'retriever': 'bm25',
        'query_mode': QUERY_MODE,
        'data_source': data_source_info['source'],
        'data_timestamp': data_source_info['timestamp'],
        'force_raw_data': FORCE_RAW_DATA
    }
    
    log_with_timestamp("=" * 50)
    log_with_timestamp(f"Evaluating target user queries ({QUERY_MODE} mode)...")
    bm25_candidates_target_file = os.path.join(OUTPUT_DIR, f"bm25_candidates_{USER_ID}_{QUERY_MODE}.json")
    bm25_target = evaluate_retriever(
        bm25, target_queries, list(all_asins), K_VALUES,
        save_candidates_path=bm25_candidates_target_file
    )
    
    output_data = {
        **common_metadata,
        'query_type': 'target_user',
        'metrics': bm25_target
    }

    # Add fingerprint verification data
    if VERIFY_FINGERPRINTS:
        output_data['doc_fingerprints_sample'] = doc_fingerprints[:5]  # First 5 for verification
        output_data['fingerprint_verification_enabled'] = True

    output_file = os.path.join(OUTPUT_DIR, f"retrieval_bm25_{QUERY_MODE}_{USER_ID}.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log_with_timestamp(f"Saved: {output_file}")
    
    log_with_timestamp("=" * 50)
    log_with_timestamp("EVALUATION SUMMARY")
    log_with_timestamp("=" * 50)
    
    log_with_timestamp("\nTarget User Queries (BM25):")
    for k, v in bm25_target.items():
        log_with_timestamp(f"  {k}: {v}")
    
    log_with_timestamp("\nDone!")
    

if __name__ == "__main__":
    main()
