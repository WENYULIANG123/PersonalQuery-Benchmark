#!/usr/bin/env python3
"""
Stage 13: Dirichlet Prior (Language Model) Retrieval Evaluation (Standalone)

Dirichlet Prior retrieval evaluation without command-line parameters.
Configuration is hardcoded in this script.

Input: result/personal_query/07_query/dual_queries_{user_id}.json
       result/personal_query/02_matching/match_{user_id}.json (for product info)
Output: result/personal_query/13_retrieval/retrieval_dirichlet_{user_id}.json
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import retrievers

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
load_reviews_for_products = utils.load_reviews_for_products
load_qa_for_products = utils.load_qa_for_products
load_preprocessed_products = utils.load_preprocessed_products
build_document_text = utils.build_document_text
evaluate_retriever = utils.evaluate_retriever

DirichletPriorRetriever = retrievers.DirichletPriorRetriever

USER_ID = "A13OFOB1394G31"

BASE_DIR = "/home/wlia0047/ar57/wenyu"
OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval")
QUERY_FILE = os.path.join(BASE_DIR, "result/personal_query/07_query/dual_queries_A13OFOB1394G31.json")
MATCH_FILE = os.path.join(BASE_DIR, "result/personal_query/02_matching/match_A13OFOB1394G31.json")
META_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json")
REVIEW_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz")
QA_FILE = os.path.join(BASE_DIR, "data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz")
CACHE_DIR = os.path.join(BASE_DIR, "result/personal_query/13_retrieval/cache")
CATEGORY = "Arts_Crafts_and_Sewing"
USE_PREPROCESSED = True

K_VALUES = [1, 3, 5, 10]
DIRICHLET_MU = 2000


def main():
    log_with_timestamp("=" * 60)
    log_with_timestamp("Dirichlet Prior Retrieval Evaluation (Standalone)")
    log_with_timestamp("=" * 60)
    log_with_timestamp(f"User ID: {USER_ID}")
    log_with_timestamp(f"Query file: {QUERY_FILE}")
    log_with_timestamp(f"Match file: {MATCH_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Dirichlet mu: {DIRICHLET_MU}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_with_timestamp(f"Loading queries from {QUERY_FILE}")
    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    log_with_timestamp(f"Loaded {len(results)} query pairs")
    
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
    
    if USE_PREPROCESSED and os.path.exists(os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl")):
        log_with_timestamp("Loading preprocessed products from cache...")
        products, all_metadata = load_preprocessed_products(CACHE_DIR, CATEGORY, all_asins)
    elif os.path.exists(MATCH_FILE):
        products, all_metadata = load_product_metadata(MATCH_FILE, all_asins)
        if os.path.exists(REVIEW_FILE):
            products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=10)
        if os.path.exists(QA_FILE):
            products = load_qa_for_products(QA_FILE, products, max_qa_per_product=25)
    else:
        products, all_metadata = load_product_metadata(META_FILE, all_asins)
        if os.path.exists(REVIEW_FILE):
            products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=10)
        if os.path.exists(QA_FILE):
            products = load_qa_for_products(QA_FILE, products, max_qa_per_product=25)
    
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
    
    log_with_timestamp(f"Loaded {len(documents)} documents")
    
    log_with_timestamp("=" * 50)
    log_with_timestamp("Building Dirichlet Prior retriever...")
    dirichlet = DirichletPriorRetriever(mu=DIRICHLET_MU)
    dirichlet.fit(documents, all_metadata)

    common_metadata = {
        'user_id': USER_ID,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries) + len(mass_queries),
        'num_documents': len(documents),
        'k_values': K_VALUES,
        'retriever': 'dirichlet_prior',
        'mu': DIRICHLET_MU
    }

    log_with_timestamp("=" * 50)
    log_with_timestamp("Evaluating target user queries...")
    dirichlet_target = evaluate_retriever(dirichlet, target_queries, list(all_asins), K_VALUES)
    
    output_data = {
        **common_metadata,
        'query_type': 'target_user',
        'metrics': dirichlet_target
    }
    output_file = os.path.join(OUTPUT_DIR, f"retrieval_dirichlet_target_{USER_ID}.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log_with_timestamp(f"Saved: {output_file}")

    log_with_timestamp("=" * 50)
    log_with_timestamp("Evaluating mass market queries...")
    dirichlet_mass = evaluate_retriever(dirichlet, mass_queries, list(all_asins), K_VALUES)
    
    output_data = {
        **common_metadata,
        'query_type': 'mass_market',
        'metrics': dirichlet_mass
    }
    output_file = os.path.join(OUTPUT_DIR, f"retrieval_dirichlet_mass_{USER_ID}.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log_with_timestamp(f"Saved: {output_file}")

    log_with_timestamp("=" * 50)
    log_with_timestamp("EVALUATION SUMMARY")
    log_with_timestamp("=" * 50)
    
    log_with_timestamp("\nTarget User Queries (Dirichlet Prior):")
    for k, v in dirichlet_target.items():
        log_with_timestamp(f"  {k}: {v}")
    
    log_with_timestamp("\nMass Market Queries (Dirichlet Prior):")
    for k, v in dirichlet_mass.items():
        log_with_timestamp(f"  {k}: {v}")
    
    log_with_timestamp("\nDone!")


if __name__ == "__main__":
    main()
