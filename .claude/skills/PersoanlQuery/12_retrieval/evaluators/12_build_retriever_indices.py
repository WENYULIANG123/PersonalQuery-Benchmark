#!/usr/bin/env python3
"""
Independent Retriever Index Builder

This script ONLY builds retriever indices for missing or incomplete retrievers.
Does NOT run evaluations - pure index construction.

Outputs are saved to the same cache directory as the main evaluation script,
so they can be immediately used by 12_evaluate_all_users_fullscale.py
"""

import argparse
import json
import os
import sys
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/home/wlia0047/ar57/wenyu"
OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/12_retrieval")
DOCUMENT_CACHE_DIR = os.path.join(OUTPUT_DIR, "document_cache")
RETRIEVER_CACHE_DIR = os.path.join(OUTPUT_DIR, "retriever_cache")

METADATA_FILE = os.path.join(DOCUMENT_CACHE_DIR, "Arts_Crafts_and_Sewing_metadata.pkl")

# All available retrievers in the system
ALL_RETRIEVERS = [
    'bm25', 'tfidf', 'dirichlet',
    'dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star',
    'colbert'
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_fullscale_metadata() -> Dict:
    """Load the 302k product metadata"""
    log_with_timestamp(f"Loading metadata from {METADATA_FILE}...")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    
    log_with_timestamp(f"✓ Loaded {len(metadata)} products from metadata")
    return metadata


def build_document_list(metadata: Dict) -> List[Dict]:
    """Convert metadata to document format (same as main script)"""
    log_with_timestamp(f"Converting {len(metadata)} products to document format...")
    
    documents = []
    asins_list = sorted(list(metadata.keys()))
    
    for i, asin in enumerate(asins_list):
        if i % 50000 == 0:
            log_with_timestamp(f"  Processed {i}/{len(asins_list)}")
        
        doc = metadata[asin].copy()
        doc['asin'] = asin
        documents.append(doc)
    
    log_with_timestamp(f"✓ Built document list: {len(documents)} documents")
    return documents


def compute_document_hash(documents: List[Dict]) -> str:
    """Compute hash of document set (matches retriever_manager logic)"""
    doc_ids = sorted([doc.get('asin', '') for doc in documents])
    hash_input = '|'.join(doc_ids)
    doc_hash = hashlib.md5(hash_input.encode()).hexdigest()
    return doc_hash


def check_retriever_cache(retriever_name: str, doc_hash: str) -> bool:
    """Check if retriever cache already exists"""
    cache_path = os.path.join(RETRIEVER_CACHE_DIR, f"{retriever_name}_{doc_hash}.pkl")
    exists = os.path.exists(cache_path)
    
    if exists:
        size = os.path.getsize(cache_path) / (1024**2)  # MB
        log_with_timestamp(f"  [{retriever_name}] Cache exists ({size:.1f}MB)")
    else:
        log_with_timestamp(f"  [{retriever_name}] Cache MISSING")
    
    return exists


def build_retriever_indices(
    documents: List[Dict],
    metadata: Dict,
    doc_hash: str,
    retriever_names: Optional[List[str]] = None,
    skip_existing: bool = True
) -> Dict[str, str]:
    """
    Build retriever indices.
    
    Args:
        documents: List of document dicts
        metadata: Product metadata dict
        doc_hash: Hash of document set
        retriever_names: List of retrievers to build (None = all)
        skip_existing: Skip if cache already exists
    
    Returns:
        Dict with success/failure status for each retriever
    """
    
    if retriever_names is None:
        retriever_names = ALL_RETRIEVERS
    
    rm = get_retriever_manager()
    results = {}
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("BUILDING RETRIEVER INDICES")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Document hash: {doc_hash}")
    log_with_timestamp(f"Number of documents: {len(documents)}")
    
    # Check which caches already exist
    log_with_timestamp("\nCache Status:")
    missing_retrievers = []
    for retriever_name in retriever_names:
        if check_retriever_cache(retriever_name, doc_hash):
            results[retriever_name] = 'cached'
        else:
            missing_retrievers.append(retriever_name)
    
    if skip_existing and not missing_retrievers:
        log_with_timestamp("\n✓ All requested retrievers are cached")
        log_with_timestamp("=" * 80)
        return results
    
    if missing_retrievers:
        log_with_timestamp(f"\nBuilding {len(missing_retrievers)} missing retrievers:")
        for retriever_name in missing_retrievers:
            log_with_timestamp(f"  - {retriever_name}")
    
    # Build missing retrievers
    log_with_timestamp("\n" + "-" * 80)
    for retriever_name in missing_retrievers:
        try:
            log_with_timestamp(f"\n[{retriever_name}] Building index...")
            start_time = datetime.now()
            
            # Get retriever from manager (which handles caching automatically)
            retriever = rm.get_retriever(retriever_name, documents, metadata)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            log_with_timestamp(f"[{retriever_name}] ✓ Built in {elapsed:.2f}s")
            
            results[retriever_name] = 'built'
            
        except Exception as e:
            log_with_timestamp(f"[{retriever_name}] ✗ ERROR: {e}")
            log_with_timestamp(f"[{retriever_name}] Traceback:\n{traceback.format_exc()}")
            results[retriever_name] = f'error: {str(e)}'
    
    log_with_timestamp("\n" + "=" * 80)
    return results


def print_summary(results: Dict[str, str], doc_hash: str):
    """Print summary of build results"""
    built = [k for k, v in results.items() if v == 'built']
    cached = [k for k, v in results.items() if v == 'cached']
    errors = [k for k, v in results.items() if v.startswith('error')]
    
    log_with_timestamp("\nSUMMARY")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Document hash: {doc_hash}")
    log_with_timestamp(f"Cache directory: {RETRIEVER_CACHE_DIR}\n")
    
    if built:
        log_with_timestamp(f"✓ Built ({len(built)}):")
        for r in sorted(built):
            cache_file = os.path.join(RETRIEVER_CACHE_DIR, f"{r}_{doc_hash}.pkl")
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file) / (1024**2)  # MB
                log_with_timestamp(f"    {r}: {size:.1f}MB")
            else:
                log_with_timestamp(f"    {r}")
    
    if cached:
        log_with_timestamp(f"\n↻ Already cached ({len(cached)}):")
        for r in sorted(cached):
            cache_file = os.path.join(RETRIEVER_CACHE_DIR, f"{r}_{doc_hash}.pkl")
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file) / (1024**2)  # MB
                log_with_timestamp(f"    {r}: {size:.1f}MB")
            else:
                log_with_timestamp(f"    {r}")
    
    if errors:
        log_with_timestamp(f"\n✗ Errors ({len(errors)}):")
        for r in sorted(errors):
            log_with_timestamp(f"    {r}: {results[r]}")
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp(f"Total: {len(built)} built, {len(cached)} cached, {len(errors)} errors")
    log_with_timestamp("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build retriever indices for fullscale evaluation'
    )
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=None,
        choices=ALL_RETRIEVERS,
        help='Specific retrievers to build (default: all missing ones)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip retrievers that already have cache (default: True)'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        default=False,
        help='Rebuild all retrievers even if cached'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup
        os.makedirs(RETRIEVER_CACHE_DIR, exist_ok=True)
        
        # Load metadata and build documents
        metadata = load_fullscale_metadata()
        documents = build_document_list(metadata)
        doc_hash = compute_document_hash(documents)
        
        # Build indices
        skip_existing = not args.rebuild
        results = build_retriever_indices(
            documents,
            metadata,
            doc_hash,
            retriever_names=args.retrievers,
            skip_existing=skip_existing
        )
        
        # Print results
        print_summary(results, doc_hash)
        
        # Return appropriate exit code
        errors = [r for r in results.values() if isinstance(r, str) and r.startswith('error')]
        if errors:
            log_with_timestamp(f"\n⚠ {len(errors)} retriever(s) failed to build")
            sys.exit(1)
        else:
            log_with_timestamp("\n✓ All retrievers processed successfully")
            sys.exit(0)
            
    except Exception as e:
        log_with_timestamp(f"\n✗ FATAL ERROR: {e}")
        log_with_timestamp(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
