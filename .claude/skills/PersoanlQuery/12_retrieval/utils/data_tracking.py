#!/usr/bin/env python3
"""
Universal Data Source Tracking for Stage 13 Retrieval Evaluations

This module provides standardized data source tracking and fingerprint verification
for all retrieval evaluation scripts to ensure consistency between clean and noisy modes.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import build_document_text


def get_doc_fingerprint(doc, all_metadata):
    """Generate MD5 fingerprint of document text for verification"""
    text = build_document_text(doc, all_metadata)
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_data_source_info(use_preprocessed, cache_dir, category="Arts_Crafts_and_Sewing"):
    """Get data source information and timestamp"""
    if use_preprocessed:
        products_file = os.path.join(cache_dir, f"products_{category}.pkl")
        if os.path.exists(products_file):
            timestamp = datetime.fromtimestamp(os.path.getmtime(products_file)).isoformat()
            return {
                'source': 'cached',
                'timestamp': timestamp,
                'cache_file': products_file,
                'cache_exists': True
            }
    return {
        'source': 'raw',
        'timestamp': datetime.now().isoformat(),
        'cache_exists': False
    }


def generate_doc_fingerprints(documents, all_metadata, log_sample=3):
    """Generate document fingerprints for verification"""
    fingerprints = []
    for i, doc in enumerate(documents):
        fp = get_doc_fingerprint(doc, all_metadata)
        text = build_document_text(doc, all_metadata)
        fingerprints.append({
            'asin': doc.get('asin', ''),
            'fingerprint': fp,
            'doc_length': len(text)
        })
    return fingerprints


def add_data_tracking_metadata(
    output_data: dict,
    query_mode: str,
    user_id: str,
    num_queries: int,
    num_documents: int,
    retriever_name: str,
    k_values: List[int],
    data_source_info: dict,
    force_raw_data: bool,
    doc_fingerprints: Optional[List[dict]] = None,
    fingerprint_sample_size: int = 5
) -> dict:
    """
    Add standardized data tracking metadata to output

    Args:
        output_data: Existing output dictionary
        query_mode: 'clean' or 'noisy'
        user_id: User ID
        num_queries: Number of queries
        num_documents: Number of documents
        retriever_name: Name of the retriever
        k_values: K values for evaluation
        data_source_info: Data source information from get_data_source_info()
        force_raw_data: Whether raw data was forced
        doc_fingerprints: Optional list of document fingerprints
        fingerprint_sample_size: Number of fingerprints to include in output

    Returns:
        Updated output dictionary with data tracking metadata
    """
    # Add common metadata
    output_data.update({
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'num_queries': num_queries,
        'num_documents': num_documents,
        'k_values': k_values,
        'retriever': retriever_name,
        'query_mode': query_mode,
        'data_source': data_source_info['source'],
        'data_timestamp': data_source_info['timestamp'],
        'force_raw_data': force_raw_data
    })

    # Add fingerprint verification if available
    if doc_fingerprints:
        output_data['doc_fingerprints_sample'] = doc_fingerprints[:fingerprint_sample_size]
        output_data['fingerprint_verification_enabled'] = True
        output_data['total_documents_fingerprinted'] = len(doc_fingerprints)

    return output_data


def verify_data_consistency(clean_file: str, noisy_file: str) -> dict:
    """
    Verify data consistency between clean and noisy evaluation results

    Args:
        clean_file: Path to clean evaluation results
        noisy_file: Path to noisy evaluation results

    Returns:
        Dictionary with verification results
    """
    if not os.path.exists(clean_file):
        return {'error': f'Clean file not found: {clean_file}'}
    if not os.path.exists(noisy_file):
        return {'error': f'Noisy file not found: {noisy_file}'}

    try:
        with open(clean_file, 'r') as f:
            clean_data = json.load(f)
        with open(noisy_file, 'r') as f:
            noisy_data = json.load(f)

        # Check data source consistency
        clean_source = clean_data.get('data_source', 'unknown')
        noisy_source = noisy_data.get('data_source', 'unknown')
        source_consistent = clean_source == noisy_source

        # Check fingerprint consistency
        fingerprint_match = None
        if 'doc_fingerprints_sample' in clean_data and 'doc_fingerprints_sample' in noisy_data:
            clean_fps = {fp['asin']: fp['fingerprint'] for fp in clean_data['doc_fingerprints_sample']}
            noisy_fps = {fp['asin']: fp['fingerprint'] for fp in noisy_data['doc_fingerprints_sample']}

            # Find common ASINs
            common_asins = set(clean_fps.keys()) & set(noisy_fps.keys())
            matches = sum(1 for asin in common_asins if clean_fps[asin] == noisy_fps[asin])
            fingerprint_match = {
                'common_asins': len(common_asins),
                'matches': matches,
                'consistent': matches == len(common_asins)
            }

        return {
            'data_source_consistent': source_consistent,
            'clean_source': clean_source,
            'noisy_source': noisy_source,
            'fingerprint_match': fingerprint_match,
            'overall_consistent': source_consistent and (
                fingerprint_match is None or fingerprint_match['consistent']
            )
        }

    except Exception as e:
        return {'error': str(e)}


def log_data_tracking_info(log_func, data_source_info, doc_fingerprints=None):
    """Log data tracking information"""
    log_func(f"Data source: {data_source_info['source']}")
    log_func(f"Data timestamp: {data_source_info['timestamp']}")

    if data_source_info['source'] == 'cached':
        log_func(f"Cache file: {data_source_info.get('cache_file', 'unknown')}")

    if doc_fingerprints:
        log_func(f"Generated {len(doc_fingerprints)} document fingerprints")
        # Log first few for verification
        for i, fp in enumerate(doc_fingerprints[:3]):
            log_func(f"  {fp['asin']}: {fp['fingerprint'][:16]}...")


# Predefined metadata templates for common retrievers
def get_retriever_metadata_template(retriever_name: str) -> dict:
    """Get metadata template for specific retriever"""
    return {
        'retriever': retriever_name,
        'evaluation_stage': '13_retrieval',
        'tracking_version': '1.0'
    }


if __name__ == "__main__":
    # Test the module
    print("Data Tracking Module - Stage 13")
    print("=" * 50)

    # Test data source info
    cache_dir = "/home/wlia0047/ar57/wenyu/result/personal_query/13_retrieval/cache"
    info = get_data_source_info(True, cache_dir)
    print(f"Data source info: {info}")

    # Test verification
    clean_file = "/home/wlia0047/ar57/wenyu/result/personal_query/13_retrieval/retrieval_bm25_clean_A13OFOB1394G31.json"
    noisy_file = "/home/wlia0047/ar57/wenyu/result/personal_query/13_retrieval/retrieval_bm25_noisy_A13OFOB1394G31.json"

    result = verify_data_consistency(clean_file, noisy_file)
    print(f"\nVerification result: {json.dumps(result, indent=2)}")
