#!/usr/bin/env python3
"""
Centralized Retriever Manager for Retrieval Evaluations

This module manages retriever instances and their indices, building them once
and reusing them across multiple user evaluations.
"""

import os
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import retrievers
from document_manager import get_document_manager
from lazy_retriever_wrapper import LazyRetrieverWrapper, BatchedLazyRetrieverWrapper, PreloadedBatchedDenseRetriever
from lazy_cache_manager import LazyEmbeddingCache
from lazy_retriever_proxy import LazyRetrieverProxy

DENSE_RETRIEVERS = ['e5', 'bge', 'dense', 'ance', 'minilm', 'mpnet', 'star']

log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class RetrieverManager:
    """
    Manages retriever instances and their indices.
    
    Features:
    - Builds each retriever index once for a given document set
    - Thread-safe for parallel evaluation
    - Disk caching for persistence across runs
    - Automatic invalidation when documents change
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RetrieverManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._retrievers = {}
        self._retriever_hashes = {}
        self._cache_lock = threading.Lock()
        
        self.base_dir = "/home/wlia0047/ar57/wenyu"
        self.cache_dir = os.path.join(self.base_dir, "result/personal_query/12_retrieval/retriever_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.lazy_cache = LazyEmbeddingCache(self.cache_dir)
        
        self._available_retrievers = {
            'bm25': retrievers.BM25,
            'tfidf': retrievers.TFIDFRetriever,
            'dirichlet': retrievers.DirichletPriorRetriever,
            'dense': retrievers.DenseRetriever,
            'ance': retrievers.ANCERetriever,
            'bge': retrievers.BGERetriever,
            'e5': retrievers.E5Retriever,
            'minilm': retrievers.MiniLMRetriever,
            'mpnet': retrievers.MPNetRetriever,
            'star': retrievers.STARRetriever,
            'colbert': retrievers.ColBERTRetriever
        }
    
    def _compute_document_hash(self, documents: List[Dict]) -> str:
        """Compute hash of document set to detect changes"""
        doc_ids = sorted([doc.get('asin', '') for doc in documents])
        hash_input = '|'.join(doc_ids)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, retriever_name: str, doc_hash: str) -> str:
        """Get cache file path for a retriever"""
        return os.path.join(self.cache_dir, f"{retriever_name}_{doc_hash}.pkl")
    
    def _load_from_cache(self, retriever_name: str, doc_hash: str) -> Optional[Any]:
        """Load retriever from disk cache if available"""
        if retriever_name in DENSE_RETRIEVERS:
            log_with_timestamp(f"[CACHE_LOAD_START] Loading {retriever_name} from cache...")
            log_with_timestamp(f"  Using LazyEmbeddingCache for {retriever_name}...")
            retriever = self.lazy_cache.load_retriever(retriever_name, doc_hash)
            if retriever:
                log_with_timestamp(f"[CACHE_LOAD_SUCCESS] Loaded {retriever_name}, embeddings deferred")
                if hasattr(retriever, '_embeddings_path'):
                    log_with_timestamp(f"  → Embeddings reference: {retriever._embeddings_path}")
            return retriever
        
        cache_path = self._get_cache_path(retriever_name, doc_hash)
        if os.path.exists(cache_path):
            try:
                log_with_timestamp(f"[CACHE_LOAD_START] Loading {retriever_name} from cache...")
                cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                log_with_timestamp(f"  Cache file size: {cache_size_mb:.1f} MB")
                with open(cache_path, 'rb') as f:
                    retriever = pickle.load(f)
                
                log_with_timestamp(f"[CACHE_LOAD_SUCCESS] Loaded {retriever_name}, type: {type(retriever).__name__}")
                return retriever
            except Exception as e:
                log_with_timestamp(f"Error loading cache for {retriever_name}: {e}")
        
        return None
    
    def _save_to_cache(self, retriever_name: str, doc_hash: str, retriever: Any):
        """Save retriever to disk cache"""
        cache_path = self._get_cache_path(retriever_name, doc_hash)
        
        try:
            if retriever_name in DENSE_RETRIEVERS:
                log_with_timestamp(f"[CACHE_SAVE_DUMP] Dumping {retriever_name} with LazyEmbeddingCache (separating embeddings)...")
                self.lazy_cache.save_retriever(retriever_name, doc_hash, retriever)
                log_with_timestamp(f"[CACHE_SAVE_DONE] Saved {retriever_name} with separated embeddings")
            else:
                log_with_timestamp(f"[CACHE_SAVE_DUMP] Dumping {retriever_name} (type={type(retriever).__name__})...")
                with open(cache_path, 'wb') as f:
                    pickle.dump(retriever, f)
                
                cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                log_with_timestamp(f"[CACHE_SAVE_DONE] Saved {retriever_name} to cache ({cache_size_mb:.1f} MB)")
        except Exception as e:
            log_with_timestamp(f"Error saving cache for {retriever_name}: {e}")
    
    def create_lazy_proxy(self, retriever_name: str, documents: List[Dict], 
                        metadata: Optional[Dict] = None, use_lazy_loading: bool = True) -> LazyRetrieverProxy:
        """
        Create a lazy proxy that defers retriever loading until actual use
        
        Args:
            retriever_name: Name of the retriever
            documents: List of document dictionaries
            metadata: Optional metadata dictionary
            use_lazy_loading: If True, use lazy loading wrappers for dense retrievers
            
        Returns:
            LazyRetrieverProxy that loads the actual retriever on first use
        """
        return LazyRetrieverProxy(retriever_name, self, documents, metadata, use_lazy_loading)
    
    def get_retriever(self, retriever_name: str, documents: List[Dict], 
                     metadata: Optional[Dict] = None, use_lazy_loading: bool = True) -> Any:
        """
        Get or build a retriever for the given document set.
        
        Args:
            retriever_name: Name of the retriever (e.g., 'bm25', 'dense')
            documents: List of document dictionaries
            metadata: Optional metadata dictionary
            use_lazy_loading: If True, wrap Dense retrievers with LazyRetrieverWrapper
            
        Returns:
            Fitted retriever instance (optionally wrapped with lazy loading)
        """
        if retriever_name not in self._available_retrievers:
            raise ValueError(f"Unknown retriever: {retriever_name}")
        
        doc_hash = self._compute_document_hash(documents)
        cache_key = f"{retriever_name}_{doc_hash}"
        
        with self._cache_lock:
            if cache_key in self._retrievers:
                log_with_timestamp(f"Using cached {retriever_name} index (memory cache)")
                return self._retrievers[cache_key]
            
            cached_retriever = self._load_from_cache(retriever_name, doc_hash)
            if cached_retriever is not None:
                if use_lazy_loading and retriever_name in DENSE_RETRIEVERS:
                    log_with_timestamp(f"[CACHE_WRAP_START] Wrapping cached {retriever_name} with PreloadedBatchedDenseRetriever...")
                    before_type = type(cached_retriever).__name__
                    cached_retriever = PreloadedBatchedDenseRetriever(cached_retriever)
                    after_type = type(cached_retriever).__name__
                    log_with_timestamp(f"[CACHE_WRAP_SUCCESS] {retriever_name}: {before_type} → {after_type}")
                else:
                    log_with_timestamp(f"[CACHE_NO_WRAP] {retriever_name}: use_lazy_loading={use_lazy_loading}, skipping wrapper")
                
                self._retrievers[cache_key] = cached_retriever
                return cached_retriever
            
            log_with_timestamp(f"[BUILD_START] Building new {retriever_name} index on {len(documents)} documents...")
            start_time = datetime.now()
            
            retriever_class = self._available_retrievers[retriever_name]
            retriever = retriever_class()
            log_with_timestamp(f"[BUILD_FIT_START] Fitting {retriever_name}...")
            
            if metadata and hasattr(retriever, 'fit'):
                retriever.fit(documents, metadata)
            else:
                retriever.fit(documents)
            
            build_time = (datetime.now() - start_time).total_seconds()
            log_with_timestamp(f"[BUILD_FIT_SUCCESS] Built {retriever_name} index in {build_time:.2f}s")
            
            if hasattr(retriever, 'doc_embeddings'):
                if retriever.doc_embeddings is not None:
                    log_with_timestamp(f"  → {retriever_name} has doc_embeddings: {type(retriever.doc_embeddings).__name__} with {len(retriever.doc_embeddings)} items")
                else:
                    log_with_timestamp(f"  → {retriever_name} doc_embeddings is None")
            
            if use_lazy_loading and retriever_name in DENSE_RETRIEVERS:
                log_with_timestamp(f"[WRAP_START] Wrapping {retriever_name} with PreloadedBatchedDenseRetriever...")
                before_type = type(retriever).__name__
                retriever = PreloadedBatchedDenseRetriever(retriever)
                after_type = type(retriever).__name__
                log_with_timestamp(f"[WRAP_SUCCESS] {retriever_name}: {before_type} → {after_type}")
            else:
                log_with_timestamp(f"[NO_WRAP] {retriever_name}: use_lazy_loading={use_lazy_loading}, skipping wrapper")
            
            self._retrievers[cache_key] = retriever
            self._retriever_hashes[cache_key] = doc_hash
            
            log_with_timestamp(f"[CACHE_SAVE_START] Saving {retriever_name} to cache...")
            self._save_to_cache(retriever_name, doc_hash, retriever)
            
            if retriever_name in DENSE_RETRIEVERS:
                embeddings_path = os.path.join(
                    self.lazy_cache.cache_dir,
                    f"{retriever_name}_{doc_hash}_embeddings.npy"
                )
                if hasattr(retriever, '_embeddings_preload_path'):
                    retriever._embeddings_preload_path = embeddings_path
                    log_with_timestamp(f"[SET_PRELOAD_PATH] Set wrapper embeddings path: {embeddings_path}")
            
            return retriever
    
    def clear_cache(self, retriever_name: Optional[str] = None):
        """Clear cached retrievers"""
        with self._cache_lock:
            if retriever_name:
                keys_to_remove = [k for k in self._retrievers.keys() if k.startswith(f"{retriever_name}_")]
                for key in keys_to_remove:
                    self._retrievers.pop(key, None)
                    self._retriever_hashes.pop(key, None)
                log_with_timestamp(f"Cleared cache for {retriever_name}")
            else:
                self._retrievers.clear()
                self._retriever_hashes.clear()
                log_with_timestamp("Cleared all retriever caches")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                'cached_retrievers': list(self._retrievers.keys()),
                'total_cached': len(self._retrievers),
                'memory_cache_size': sum(sys.getsizeof(r) for r in self._retrievers.values()),
                'disk_cache_files': len(list(Path(self.cache_dir).glob("*.pkl")))
            }


def get_retriever_manager() -> RetrieverManager:
    """Get the singleton RetrieverManager instance"""
    return RetrieverManager()


if __name__ == "__main__":
    from document_manager import get_document_manager
    
    dm = get_document_manager()
    rm = get_retriever_manager()
    
    test_asins = {"B07MQRTKXH", "B07N2RQFJL", "B07P6Y7954"}
    docs, metadata = dm.load_documents("Arts_Crafts_and_Sewing", test_asins)
    
    bm25 = rm.get_retriever("bm25", docs)
    print(f"Built BM25 retriever")
    
    bm25_cached = rm.get_retriever("bm25", docs)
    print(f"Got cached BM25 retriever")
    
    print(f"Cache stats: {rm.get_cache_stats()}")