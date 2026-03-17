#!/usr/bin/env python3
"""
Build all retriever indices for full-scale evaluation.
Extracts index-building logic from main evaluation script.
Only builds indices, does NOT evaluate.
"""

import os
import sys
import pickle
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

# Add utils path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
build_document_text = utils.build_document_text

# Import retriever utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import retrievers


# ========== DocumentManager (minimal) ==========
class DocumentManager:
    """Minimal document manager for building indices"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DocumentManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._documents_cache = {}
        self._metadata_cache = {}
        self._cache_lock = threading.Lock()
        
        self.base_dir = "/home/wlia0047/ar57/wenyu"
        self.cache_dir = os.path.join(self.base_dir, "result/personal_query/12_retrieval/document_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_documents(self, category: str) -> Tuple[List[Dict], Set[str]]:
        """Load documents for a category"""
        with self._cache_lock:
            if category in self._documents_cache:
                log_with_timestamp(f"Using cached documents for {category}")
                return self._documents_cache[category], self._metadata_cache[category]
        
        log_with_timestamp(f"Loading documents for {category}...")
        metadata_file = os.path.join(self.base_dir, "data/Amazon-Reviews-2023/metafiles2/meta_{category}.json.gz")
        metadata = load_product_metadata(metadata_file)
        
        documents = []
        asins = set()
        for asin, meta in metadata.items():
            doc = build_document_text(asin, meta)
            documents.append({'id': asin, 'text': doc})
            asins.add(asin)
        
        with self._cache_lock:
            self._documents_cache[category] = documents
            self._metadata_cache[category] = asins
        
        return documents, asins


def load_fullscale_metadata(metadata_file: str) -> Dict:
    """Load full metadata"""
    log_with_timestamp(f"Loading metadata from {metadata_file}...")
    metadata = load_product_metadata(metadata_file)
    return metadata


def build_fullscale_documents(category: str, metadata: Dict) -> Tuple[List[Dict], Set[str]]:
    """Build full-scale document set"""
    log_with_timestamp(f"Building {len(metadata)} documents from metadata...")
    
    documents = []
    asins = set()
    
    for idx, (asin, meta) in enumerate(metadata.items()):
        if (idx + 1) % 50000 == 0:
            log_with_timestamp(f"  Processed {idx + 1}/{len(metadata)}")
        
        doc = build_document_text(asin, meta)
        documents.append({'id': asin, 'text': doc})
        asins.add(asin)
    
    log_with_timestamp(f"Built document list: {len(documents)} documents")
    return documents, asins


def get_retriever_manager():
    """Get lazy singleton instance of RetrieverManager"""
    from utils import retrievers as ret_module
    manager = getattr(ret_module, '_RETRIEVER_MANAGER', None)
    
    if manager is None:
        log_with_timestamp("[LAZY_MANAGER_INIT] Creating new RetrieverManager instance...")
        manager = retriever_manager = ret_module.RetrieverManager()
        ret_module._RETRIEVER_MANAGER = manager
        log_with_timestamp(f"[LAZY_MANAGER_READY] RetrieverManager created")
    else:
        log_with_timestamp("[LAZY_MANAGER_CACHE_HIT] Using cached RetrieverManager instance")
    
    return manager


def main():
    """Main: Build all retriever indices"""
    setup_logging()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD ALL RETRIEVER INDICES")
    log_with_timestamp("=" * 80)
    
    category = "Arts_Crafts_and_Sewing"
    
    # Load full-scale metadata
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    
    if os.path.exists(metadata_file):
        log_with_timestamp(f"Loading metadata from cache: {metadata_file}")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        log_with_timestamp("Metadata cache not found, loading from raw data...")
        raw_metadata_file = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/metafiles2/meta_Arts_Crafts_and_Sewing.json.gz"
        metadata = load_fullscale_metadata(raw_metadata_file)
    
    # Build documents
    documents, asins = build_fullscale_documents(category, metadata)
    log_with_timestamp(f"Total documents: {len(documents)}, Total ASINs: {len(asins)}")
    
    # Get retriever manager
    manager = get_retriever_manager()
    
    # Define retrievers to build
    DENSE_RETRIEVERS = ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star']
    SPARSE_RETRIEVERS = ['bm25', 'tfidf']
    ALL_RETRIEVERS = DENSE_RETRIEVERS + SPARSE_RETRIEVERS
    
    log_with_timestamp(f"\nBuilding {len(ALL_RETRIEVERS)} retrievers:")
    log_with_timestamp(f"  Dense: {DENSE_RETRIEVERS}")
    log_with_timestamp(f"  Sparse: {SPARSE_RETRIEVERS}")
    
    # Build each retriever
    start_time = datetime.now()
    results = {}
    
    for retriever_name in ALL_RETRIEVERS:
        log_with_timestamp(f"\n[BUILD] {retriever_name}")
        try:
            retriever_start = datetime.now()
            retriever = manager.get_retriever(retriever_name, documents, use_lazy_loading=True)
            elapsed = (datetime.now() - retriever_start).total_seconds()
            
            log_with_timestamp(f"[BUILD_SUCCESS] {retriever_name} built in {elapsed:.1f}s")
            results[retriever_name] = {'status': 'success', 'time': elapsed}
        except Exception as e:
            log_with_timestamp(f"[BUILD_FAILED] {retriever_name}: {e}")
            results[retriever_name] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("BUILD SUMMARY")
    log_with_timestamp("=" * 80)
    
    for retriever_name, result in results.items():
        status = result['status']
        if status == 'success':
            time = result['time']
            log_with_timestamp(f"  ✓ {retriever_name:15} - {time:7.1f}s")
        else:
            error = result['error']
            log_with_timestamp(f"  ✗ {retriever_name:15} - ERROR: {error}")
    
    log_with_timestamp(f"\nTotal time: {total_time:.1f}s")
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    log_with_timestamp(f"Success: {successful}/{len(ALL_RETRIEVERS)}")
    
    log_with_timestamp("=" * 80)


def setup_logging():
    """Setup logging directory"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)


if __name__ == '__main__':
    main()
    log_with_timestamp("当前任务已完成，请做下一个任务的指示。")
