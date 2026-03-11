#!/usr/bin/env python3
"""
Stage 13: Retrieval-based Evaluation (GLM API Version)

使用 BM25, ANCE (Dense Retrieval), ColBERTv2, 和 GLM-4.5V API 对查询进行检索评估。

Input: result/personal_query/07_query/dual_queries_{user_id}.json
       result/personal_query/02_matching/match_{user_id}.json (for product info)
Output: result/personal_query/13_retrieval/retrieval_evaluation_{user_id}.json

Metrics:
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
"""

import json
import os
import sys
import argparse
from datetime import datetime

# Add current directory to sys.path for relative imports to work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from sibling modules (following Stage 09 pattern)
import utils
import retrievers
import hybrid
import reranker_bert
import reranker_glm

# Alias for convenience
log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
load_reviews_for_products = utils.load_reviews_for_products
build_document_text = utils.build_document_text
evaluate_retriever = utils.evaluate_retriever

BM25 = retrievers.BM25
DenseRetriever = retrievers.DenseRetriever
E5Retriever = retrievers.E5Retriever
BGERetriever = retrievers.BGERetriever
ColBERTRetriever = retrievers.ColBERTRetriever
TFIDFRetriever = retrievers.TFIDFRetriever
DirichletPriorRetriever = retrievers.DirichletPriorRetriever

HybridRetriever = hybrid.HybridRetriever
BERTReRanker = reranker_bert.BERTReRanker

GLMReRanker = reranker_glm.GLMReRanker
PersonalizedGLMReRanker = reranker_glm.PersonalizedGLMReRanker


def main():
    parser = argparse.ArgumentParser(description="Retrieval-based evaluation")
    parser.add_argument("--query-file", required=True, help="Stage 7 dual queries file")
    parser.add_argument("--match-file", required=True, help="Stage 2 match file for product info")
    parser.add_argument("--meta-file", help="Product metadata file (optional)")
    parser.add_argument("--review-file", help="Product reviews file (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--user-id", required=True, help="User ID")
    parser.add_argument("--k-values", default="1,3,5,10", help="K values for metrics")
    parser.add_argument("--retriever", 
                        choices=['bm25', 'tfidf', 'dirichlet', 'dense', 'e5', 'bge', 'colbert', 
                                'hybrid_bm25_e5', 'hybrid_bm25_bge', 'bert_reranker', 
                                'glm', 'glm_personalized', 'all'],
                        default='all', help="Which retriever to run (default: all)")
    parser.add_argument("--query-type", 
                        choices=['target', 'mass', 'both'],
                        default='both', help="Which query type to evaluate (default: both)")
    parser.add_argument("--glm-model", default="GLM-4.5V", 
                        help="GLM model to use for GLM reranker (default: GLM-4.5V)")
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(',')]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load queries
    log_with_timestamp(f"Loading queries from {args.query_file}")
    with open(args.query_file, 'r') as f:
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
    
    # Load product metadata
    documents = []
    all_metadata = None  # 用于 also_buy/also_view 查找
    if args.meta_file and os.path.exists(args.meta_file):
        products, all_metadata = load_product_metadata(args.meta_file, all_asins)

        # Load reviews if review file is provided
        if args.review_file and os.path.exists(args.review_file):
            products = load_reviews_for_products(args.review_file, products, max_reviews_per_product=10)

        # Add products that match our queries
        for asin in all_asins:
            if asin in products:
                documents.append(products[asin])
            else:
                # Use placeholder with empty fields
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
    else:
        # Use category as text for now
        for r in results:
            asin = r.get('asin', '')
            cat = r.get('category', '')
            documents.append({
                'asin': asin,
                'title': cat,
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

    # 统计文档信息
    doc_lengths = []
    for doc in documents:
        doc_text = build_document_text(doc, all_metadata)
        doc_lengths.append(len(doc_text.split()))

    if doc_lengths:
        log_with_timestamp(f"  Average document length: {sum(doc_lengths)/len(doc_lengths):.1f} words")
        log_with_timestamp(f"  Min document length: {min(doc_lengths)} words")
        log_with_timestamp(f"  Max document length: {max(doc_lengths)} words")
    
    # Determine which retrievers to run
    run_bm25 = args.retriever in ['bm25', 'all']
    run_tfidf = args.retriever in ['tfidf', 'all']
    run_dirichlet = args.retriever in ['dirichlet', 'all']
    run_dense = args.retriever in ['dense', 'all']
    run_e5 = args.retriever in ['e5', 'all']
    run_bge = args.retriever in ['bge', 'all']
    run_colbert = args.retriever in ['colbert', 'all']
    run_hybrid_e5 = args.retriever in ['hybrid_bm25_e5', 'all']
    run_hybrid_bge = args.retriever in ['hybrid_bm25_bge', 'all']
    run_bert = args.retriever in ['bert_reranker', 'all']
    run_glm = args.retriever in ['glm', 'glm_personalized', 'all']
    run_glm_personalized = args.retriever in ['glm_personalized', 'all']
    
    run_target = args.query_type in ['target', 'both']
    run_mass = args.query_type in ['mass', 'both']
    
    # Build indices
    # BM25 (needed by many others)
    if run_bm25 or run_hybrid_e5 or run_hybrid_bge or run_bert or run_glm or run_glm_personalized:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building BM25...")
        bm25 = BM25()
        bm25.fit(documents, all_metadata)

    # Dense Retriever
    if run_dense:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building Dense Retriever (ANCE/all-MiniLM-L6-v2)...")
        dense = DenseRetriever()
        dense.fit(documents, all_metadata)

    # E5 Retriever
    if run_e5 or run_hybrid_e5:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building E5-large-v2...")
        e5_retriever = E5Retriever()
        e5_retriever.fit(documents, all_metadata)

    # BGE Retriever
    if run_bge or run_hybrid_bge:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building BGE-large-en...")
        bge_retriever = BGERetriever()
        bge_retriever.fit(documents, all_metadata)

    # ColBERT Retriever
    if run_colbert:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building ColBERT Retriever...")
        colbert = ColBERTRetriever()
        colbert.fit(documents, all_metadata)

    # Hybrid (BM25 + E5)
    if run_hybrid_e5:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building Hybrid (BM25 + E5) with RRF...")
        hybrid_bm25_e5 = HybridRetriever([bm25, e5_retriever], k=60)
        hybrid_bm25_e5.fit(documents, all_metadata)

    # Hybrid (BM25 + BGE)
    if run_hybrid_bge:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building Hybrid (BM25 + BGE) with RRF...")
        hybrid_bm25_bge = HybridRetriever([bm25, bge_retriever], k=60)
        hybrid_bm25_bge.fit(documents, all_metadata)

    # TF-IDF
    if run_tfidf:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building TF-IDF (baseline)...")
        tfidf = TFIDFRetriever()
        tfidf.fit(documents, all_metadata)

    # Dirichlet Prior
    if run_dirichlet:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building Dirichlet Prior (Language Model)...")
        dirichlet = DirichletPriorRetriever(mu=2000)
        dirichlet.fit(documents, all_metadata)

    # BERT Reranker
    if run_bert:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Building BERT Reranker (BM25 + BERT cross-encoder)...")
        bm25_for_bert_reranker = BM25()
        bert_reranker = BERTReRanker(bm25_for_bert_reranker, top_k=50)
        bert_reranker.fit(documents, all_metadata)

    # GLM Rerankers
    if run_glm or run_glm_personalized:
        glm_models = [args.glm_model]  # Use specified model
        glm_rerankers = {}
        glm_rerankers_personalized = {}

        for model_name in glm_models:
            log_with_timestamp("=" * 50)
            log_with_timestamp(f"Building GLM Reranker: {model_name}")
            
            if run_glm:
                bm25_for_glm = BM25()
                glm_reranker = GLMReRanker(bm25_for_glm, top_k=30, model=model_name)
                glm_reranker.fit(documents, all_metadata)
                glm_rerankers[model_name] = glm_reranker

            if run_glm_personalized:
                log_with_timestamp(f"Building Personalized GLM Reranker: {model_name}")
                log_with_timestamp("  Innovation: Add user persona context to Stage 2 reranking")
                bm25_for_personalized = BM25()
                persona_dir = os.path.join(os.path.dirname(args.query_file), "../04_persona")
                persona_dir = os.path.abspath(persona_dir)
                glm_reranker_personalized = PersonalizedGLMReRanker(
                    bm25_for_personalized,
                    top_k=30,
                    persona_dir=persona_dir,
                    model=model_name
                )
                glm_reranker_personalized.fit(documents, all_metadata, queries=target_queries, user_id=args.user_id)
                glm_rerankers_personalized[model_name] = glm_reranker_personalized

    # Initialize all result variables to None
    bm25_target = dense_target = e5_target = bge_target = colbert_target = None
    hybrid_bm25_e5_target = hybrid_bm25_bge_target = tfidf_target = dirichlet_target = bert_reranker_target = None
    bm25_mass = dense_mass = e5_mass = bge_mass = colbert_mass = None
    hybrid_bm25_e5_mass = hybrid_bm25_bge_mass = tfidf_mass = dirichlet_mass = bert_reranker_mass = None
    glm_target_results = glm_mass_results = {}
    glm_personalized_target_results = glm_personalized_mass_results = {}

    # Evaluate selected retrievers
    bm25_candidates_file = os.path.join(args.output_dir, f"bm25_candidates_{args.user_id}.json")
    
    if run_target:
        log_with_timestamp("=" * 50)
        log_with_timestamp("Evaluating target user queries...")
        
        if run_bm25:
            bm25_target = evaluate_retriever(bm25, target_queries, list(all_asins), k_values,
                                            save_candidates_path=bm25_candidates_file.replace('.json', '_target.json'))
        if run_dense:
            dense_target = evaluate_retriever(dense, target_queries, list(all_asins), k_values)
        if run_e5:
            e5_target = evaluate_retriever(e5_retriever, target_queries, list(all_asins), k_values)
        if run_bge:
            bge_target = evaluate_retriever(bge_retriever, target_queries, list(all_asins), k_values)
        if run_colbert:
            colbert_target = evaluate_retriever(colbert, target_queries, list(all_asins), k_values)
        if run_hybrid_e5:
            hybrid_bm25_e5_target = evaluate_retriever(hybrid_bm25_e5, target_queries, list(all_asins), k_values)
        if run_hybrid_bge:
            hybrid_bm25_bge_target = evaluate_retriever(hybrid_bm25_bge, target_queries, list(all_asins), k_values)
        if run_tfidf:
            tfidf_target = evaluate_retriever(tfidf, target_queries, list(all_asins), k_values)
        if run_dirichlet:
            dirichlet_target = evaluate_retriever(dirichlet, target_queries, list(all_asins), k_values)
        if run_bert:
            bert_reranker_target = evaluate_retriever(bert_reranker, target_queries, list(all_asins), k_values)
        
        # GLM evaluations
        if run_glm or run_glm_personalized:
            for model_name in glm_models:
                log_with_timestamp(f"Evaluating {model_name} (target queries)...")
                if run_glm:
                    glm_target_results[model_name] = evaluate_retriever(
                        glm_rerankers[model_name], target_queries, list(all_asins), k_values
                    )
                if run_glm_personalized:
                    glm_personalized_target_results[model_name] = evaluate_retriever(
                        glm_rerankers_personalized[model_name], target_queries, list(all_asins), k_values
                    )

    if run_mass:
        log_with_timestamp("Evaluating mass market queries...")
        
        if run_bm25:
            bm25_mass = evaluate_retriever(bm25, mass_queries, list(all_asins), k_values,
                                          save_candidates_path=bm25_candidates_file.replace('.json', '_mass.json'))
        if run_dense:
            dense_mass = evaluate_retriever(dense, mass_queries, list(all_asins), k_values)
        if run_e5:
            e5_mass = evaluate_retriever(e5_retriever, mass_queries, list(all_asins), k_values)
        if run_bge:
            bge_mass = evaluate_retriever(bge_retriever, mass_queries, list(all_asins), k_values)
        if run_colbert:
            colbert_mass = evaluate_retriever(colbert, mass_queries, list(all_asins), k_values)
        if run_hybrid_e5:
            hybrid_bm25_e5_mass = evaluate_retriever(hybrid_bm25_e5, mass_queries, list(all_asins), k_values)
        if run_hybrid_bge:
            hybrid_bm25_bge_mass = evaluate_retriever(hybrid_bm25_bge, mass_queries, list(all_asins), k_values)
        if run_tfidf:
            tfidf_mass = evaluate_retriever(tfidf, mass_queries, list(all_asins), k_values)
        if run_dirichlet:
            dirichlet_mass = evaluate_retriever(dirichlet, mass_queries, list(all_asins), k_values)
        if run_bert:
            bert_reranker_mass = evaluate_retriever(bert_reranker, mass_queries, list(all_asins), k_values)
        
        # GLM evaluations for mass
        if run_glm or run_glm_personalized:
            for model_name in glm_models:
                log_with_timestamp(f"Evaluating {model_name} (mass market queries)...")
                if run_glm:
                    glm_mass_results[model_name] = evaluate_retriever(
                        glm_rerankers[model_name], mass_queries, list(all_asins), k_values
                    )
                if run_glm_personalized:
                    glm_personalized_mass_results[model_name] = evaluate_retriever(
                        glm_rerankers_personalized[model_name], mass_queries, list(all_asins), k_values
                    )

    # Save each retriever result to separate files
    log_with_timestamp("=" * 50)
    log_with_timestamp("Saving results to individual files...")
    
    # Common metadata for all files
    common_metadata = {
        'user_id': args.user_id,
        'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries) + len(mass_queries),
        'num_documents': len(documents),
        'k_values': k_values,
    }
    
    # Save target user queries results (only if evaluated)
    if run_target:
        retrievers_target = {
            'bm25': ('bm25', bm25_target),
            'tfidf': ('tfidf', tfidf_target),
            'dirichlet_prior': ('dirichlet', dirichlet_target),
            'ance_minilm': ('dense', dense_target),
            'e5_large_v2': ('e5', e5_target),
            'bge_large_en': ('bge', bge_target),
            'colbertv2': ('colbert', colbert_target),
            'hybrid_bm25_e5': ('hybrid_bm25_e5', hybrid_bm25_e5_target),
            'hybrid_bm25_bge': ('hybrid_bm25_bge', hybrid_bm25_bge_target),
            'bert_reranker_bm25': ('bert_reranker', bert_reranker_target),
        }
        
        for key, (filename_suffix, data) in retrievers_target.items():
            if data is not None:
                output_data = {
                    **common_metadata,
                    'query_type': 'target_user',
                    'retriever': key,
                    'metrics': data
                }
                output_file = os.path.join(args.output_dir, f"retrieval_{filename_suffix}_target_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
    
    # Save mass market queries results (only if evaluated)
    if run_mass:
        retrievers_mass = {
            'bm25': ('bm25', bm25_mass),
            'tfidf': ('tfidf', tfidf_mass),
            'dirichlet_prior': ('dirichlet', dirichlet_mass),
            'ance_minilm': ('dense', dense_mass),
            'e5_large_v2': ('e5', e5_mass),
            'bge_large_en': ('bge', bge_mass),
            'colbertv2': ('colbert', colbert_mass),
            'hybrid_bm25_e5': ('hybrid_bm25_e5', hybrid_bm25_e5_mass),
            'hybrid_bm25_bge': ('hybrid_bm25_bge', hybrid_bm25_bge_mass),
            'bert_reranker_bm25': ('bert_reranker', bert_reranker_mass),
        }
        
        for key, (filename_suffix, data) in retrievers_mass.items():
            if data is not None:
                output_data = {
                    **common_metadata,
                    'query_type': 'mass_market',
                    'retriever': key,
                    'metrics': data
                }
                output_file = os.path.join(args.output_dir, f"retrieval_{filename_suffix}_mass_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
    
    # Save GLM reranker results (each model separate file)
    if run_glm or run_glm_personalized:
        for model_name in glm_models:
            # GLM target
            if run_target and run_glm and model_name in glm_target_results:
                output_data = {
                    **common_metadata,
                    'query_type': 'target_user',
                    'retriever': f'glm_reranker',
                    'model': model_name,
                    'personalized': False,
                    'metrics': glm_target_results[model_name]
                }
                output_file = os.path.join(args.output_dir, f"retrieval_glm_{model_name}_target_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
            
            # GLM personalized target
            if run_target and run_glm_personalized and model_name in glm_personalized_target_results:
                output_data = {
                    **common_metadata,
                    'query_type': 'target_user',
                    'retriever': 'glm_reranker',
                    'model': model_name,
                    'personalized': True,
                    'metrics': glm_personalized_target_results[model_name]
                }
                output_file = os.path.join(args.output_dir, f"retrieval_glm_{model_name}_personalized_target_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
            
            # GLM mass
            if run_mass and run_glm and model_name in glm_mass_results:
                output_data = {
                    **common_metadata,
                    'query_type': 'mass_market',
                    'retriever': 'glm_reranker',
                    'model': model_name,
                    'personalized': False,
                    'metrics': glm_mass_results[model_name]
                }
                output_file = os.path.join(args.output_dir, f"retrieval_glm_{model_name}_mass_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
            
            # GLM personalized mass
            if run_mass and run_glm_personalized and model_name in glm_personalized_mass_results:
                output_data = {
                    **common_metadata,
                    'query_type': 'mass_market',
                    'retriever': 'glm_reranker',
                    'model': model_name,
                    'personalized': True,
                    'metrics': glm_personalized_mass_results[model_name]
                }
                output_file = os.path.join(args.output_dir, f"retrieval_glm_{model_name}_personalized_mass_{args.user_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                log_with_timestamp(f"  Saved: {output_file}")
        
        # GLM personalized mass
        output_data = {
            **common_metadata,
            'query_type': 'mass_market',
            'retriever': 'glm_reranker',
            'model': model_name,
            'personalized': True,
            'metrics': glm_personalized_mass_results[model_name]
        }
        output_file = os.path.join(args.output_dir, f"retrieval_glm_{model_name}_personalized_mass_{args.user_id}.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        log_with_timestamp(f"  Saved: {output_file}")

    log_with_timestamp("All results saved!")

    # Print summary
    log_with_timestamp("=" * 50)
    log_with_timestamp("EVALUATION SUMMARY")
    log_with_timestamp("=" * 50)

    # Target queries summary
    log_with_timestamp("\nTarget User Queries:")
    log_with_timestamp("  BM25:")
    for k, v in bm25_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  TF-IDF:")
    for k, v in tfidf_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  Dirichlet Prior:")
    for k, v in dirichlet_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  ANCE (MiniLM):")
    for k, v in dense_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  E5-large-v2:")
    for k, v in e5_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  BGE-large-en:")
    for k, v in bge_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  ColBERTv2:")
    for k, v in colbert_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  Hybrid (BM25 + E5):")
    for k, v in hybrid_bm25_e5_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  Hybrid (BM25 + BGE):")
    for k, v in hybrid_bm25_bge_target.items():
        log_with_timestamp(f"    {k}: {v}")
    log_with_timestamp("  BERT Reranker (BM25 + BERT):")
    for k, v in bert_reranker_target.items():
        log_with_timestamp(f"    {k}: {v}")
    # GLM Rerankers Summary
    for model_name in glm_models:
        log_with_timestamp(f"  GLM Reranker (BM25 + {model_name} API):")
        for k, v in glm_reranker_target_results[model_name].items():
            log_with_timestamp(f"    {k}: {v}")
        log_with_timestamp(f"  Personalized GLM Reranker (BM25 + {model_name} + Persona):")
        for k, v in glm_reranker_personalized_target_results[model_name].items():
            log_with_timestamp(f"    {k}: {v}")

    # Mass queries summary
    log_with_timestamp("\nMass Market Queries:")
    log_with_timestamp("  BM25:")
    for k, v in bm25_mass.items():
        log_with_timestamp(f"    {k}: {v}")

    # GLM Rerankers Summary for Mass Market
    for model_name in glm_models:
        log_with_timestamp(f"  GLM Reranker (BM25 + {model_name} API):")
        for k, v in glm_reranker_mass_results[model_name].items():
            log_with_timestamp(f"    {k}: {v}")
        log_with_timestamp(f"  Personalized GLM Reranker (BM25 + {model_name} + Persona):")
        for k, v in glm_reranker_personalized_mass_results[model_name].items():
            log_with_timestamp(f"    {k}: {v}")

    log_with_timestamp("\nDone!")


if __name__ == "__main__":
    main()
