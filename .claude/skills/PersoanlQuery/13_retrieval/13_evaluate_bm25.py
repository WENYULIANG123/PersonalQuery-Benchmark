#!/usr/bin/env python3
"""
Stage 13: BM25 Retrieval Evaluation (Standalone)

BM25 retrieval evaluation without command-line parameters.
Configuration is hardcoded in this script.

Input: result/personal_query/07_query/dual_queries_{user_id}.json
       result/personal_query/02_matching/match_{user_id}.json (for product info)
Output: result/personal_query/13_retrieval/retrieval_bm25_{user_id}.json
"""

import json
import os
import sys
from datetime import datetime

# Add current directory to sys.path for relative imports to work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from sibling modules
import utils
import retrievers

# Alias for convenience
log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
load_reviews_for_products = utils.load_reviews_for_products
load_qa_for_products = utils.load_qa_for_products
load_preprocessed_products = utils.load_preprocessed_products
build_document_text = utils.build_document_text
build_stark_document = utils.build_stark_document
evaluate_retriever = utils.evaluate_retriever

BM25 = retrievers.BM25

# =============================================================================
# CONFIGURATION - Hardcoded paths and settings
# =============================================================================

USER_ID = "A13OFOB1394G31"

BASE_DIR = "/home/wlia0047/ar57/wenyu"
OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval")
CACHE_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval/cache")
QUERY_FILE = os.path.join(BASE_DIR, "result/personal_query/07_query/dual_queries_A13OFOB1394G31.json")
MATCH_FILE = os.path.join(BASE_DIR, "result/personal_query/02_matching/match_A13OFOB1394G31.json")
META_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json")
REVIEW_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz")
QA_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz")
CATEGORY = "Arts_Crafts_and_Sewing"

USE_PREPROCESSED = True

K_VALUES = [1, 3, 5, 10]


def main():
    log_with_timestamp("=" * 60)
    log_with_timestamp("BM25 Retrieval Evaluation (Standalone)")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"User ID: {USER_ID}")
    log_with_timestamp(f"Query file: {QUERY_FILE}")
    log_with_timestamp(f"Match file: {MATCH_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load queries
    log_with_timestamp(f"Loading queries from {QUERY_FILE}")
    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    log_with_timestamp(f"Loaded {len(results)} query pairs")
    
    # Collect all ASINs
    all_asins = set()
    target_queries = []
    mass_queries = []
    
    for r in results:
        asin = r.get('asin', '')
        if asin:
            all_asins.add(asin)

            tq = r.get('target_user_query', {})
            if tq.get('query'):
                target_queries.append({
                    'asin': asin,
                    'query': tq['query'],
                    'type': 'target',
                    'category': r.get('category', ''),
                    'selected_attributes': tq.get('selected_attributes', [])
                })

            mq = r.get('mass_market_query', {})
            if mq.get('query'):
                mass_queries.append({
                    'asin': asin,
                    'query': mq['query'],
                    'type': 'mass',
                    'category': r.get('category', ''),
                    'selected_attributes': mq.get('selected_attributes', [])
                })
    
    log_with_timestamp(f"Total unique ASINs: {len(all_asins)}")
    log_with_timestamp(f"Target queries: {len(target_queries)}, Mass queries: {len(mass_queries)}")
    
    documents = []
    all_metadata = None
    
    use_preprocessed = USE_PREPROCESSED
    
    if use_preprocessed:
        products_file = os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl")
        if os.path.exists(products_file):
            log_with_timestamp(f"Loading preprocessed data from cache...")
            products, all_metadata = load_preprocessed_products(CACHE_DIR, CATEGORY, all_asins)
        else:
            log_with_timestamp("Preprocessed data not found, using raw data...")
            use_preprocessed = False
    
    if not use_preprocessed:
        if os.path.exists(META_FILE):
            products, all_metadata = load_product_metadata(META_FILE, all_asins)

            if os.path.exists(REVIEW_FILE):
                products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=25)

    # Build documents list
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

    # Build BM25 index
    log_with_timestamp("=" * 50)
    log_with_timestamp("Building BM25...")
    
    # Debug: check first document
    if documents:
        first_doc = documents[0]
        first_text = build_document_text(first_doc, all_metadata)
        first_tokens = first_text.lower().split()
        log_with_timestamp(f"DEBUG: First doc has {len(first_doc.get('feature', []))} features, {len(first_doc.get('reviews', []))} reviews")
        log_with_timestamp(f"DEBUG: First doc text: {len(first_text)} chars, {len(first_tokens)} tokens")
    
    bm25 = BM25()
    bm25.fit(documents, all_metadata)

    # Common metadata for output files
    common_metadata = {
        'user_id': USER_ID,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries) + len(mass_queries),
        'num_documents': len(documents),
        'k_values': K_VALUES,
        'retriever': 'bm25'
    }

    # Evaluate target queries
    log_with_timestamp("=" * 50)
    log_with_timestamp("Evaluating target user queries...")
    bm25_candidates_target_file = os.path.join(OUTPUT_DIR, f"bm25_candidates_{USER_ID}_target.json")
    bm25_target = evaluate_retriever(
        bm25, target_queries, list(all_asins), K_VALUES,
        save_candidates_path=bm25_candidates_target_file
    )
    
    # Save target results
    output_data = {
        **common_metadata,
        'query_type': 'target_user',
        'metrics': bm25_target
    }
    output_file = os.path.join(OUTPUT_DIR, f"retrieval_bm25_target_{USER_ID}.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log_with_timestamp(f"Saved: {output_file}")

    # Evaluate mass queries
    log_with_timestamp("=" * 50)
    log_with_timestamp("Evaluating mass market queries...")
    bm25_candidates_mass_file = os.path.join(OUTPUT_DIR, f"bm25_candidates_{USER_ID}_mass.json")
    bm25_mass = evaluate_retriever(
        bm25, mass_queries, list(all_asins), K_VALUES,
        save_candidates_path=bm25_candidates_mass_file
    )
    
    # Save mass results
    output_data = {
        **common_metadata,
        'query_type': 'mass_market',
        'metrics': bm25_mass
    }
    output_file = os.path.join(OUTPUT_DIR, f"retrieval_bm25_mass_{USER_ID}.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log_with_timestamp(f"Saved: {output_file}")

    # Print summary
    log_with_timestamp("=" * 50)
    log_with_timestamp("EVALUATION SUMMARY")
    log_with_timestamp("=" * 50)
    
    log_with_timestamp("\nTarget User Queries (BM25):")
    for k, v in bm25_target.items():
        log_with_timestamp(f"  {k}: {v}")
    
    log_with_timestamp("\nMass Market Queries (BM25):")
    for k, v in bm25_mass.items():
        log_with_timestamp(f"  {k}: {v}")
    
    log_with_timestamp("\nDone!")


if __name__ == "__main__":
    main()
