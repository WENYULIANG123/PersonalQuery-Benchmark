#!/usr/bin/env python3
"""
Centralized Document Manager for Retrieval Evaluations

This module provides a singleton document manager that loads and caches documents
(product metadata, reviews, Q&A) once and shares them across all retrievers and users.

This eliminates repeated loading of the same data for each retrieval method.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
import threading
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata
load_reviews_for_products = utils.load_reviews_for_products
load_qa_for_products = utils.load_qa_for_products
load_preprocessed_products = utils.load_preprocessed_products
build_document_text = utils.build_document_text


class DocumentManager:
    """
    Singleton document manager that loads and caches documents for all retrievers.
    
    Features:
    - Loads documents once per category
    - Thread-safe for parallel retriever execution
    - Memory-efficient caching with optional disk persistence
    - Supports incremental loading for specific ASINs
    """
    
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
        self._loaded_asins = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'loads': 0
        }
        self._cache_lock = threading.Lock()
        
        self.base_dir = "/home/wlia0047/ar57/wenyu"
        self.cache_dir = os.path.join(self.base_dir, "result/personal_query/12_retrieval/document_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.category_files = {
            "Arts_Crafts_and_Sewing": {
                "meta": os.path.join(self.base_dir, "data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"),
                "review": os.path.join(self.base_dir, "data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"),
                "qa": os.path.join(self.base_dir, "data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz")
            }
        }
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self._cache_lock:
            total = self._cache_stats['hits'] + self._cache_stats['misses']
            hit_rate = self._cache_stats['hits'] / total if total > 0 else 0
            return {
                **self._cache_stats,
                'hit_rate': hit_rate,
                'categories_loaded': list(self._documents_cache.keys()),
                'total_documents': sum(len(docs) for docs in self._documents_cache.values())
            }
    
    def clear_cache(self, category: Optional[str] = None):
        """Clear cached documents for a specific category or all categories"""
        with self._cache_lock:
            if category:
                self._documents_cache.pop(category, None)
                self._metadata_cache.pop(category, None)
                self._loaded_asins.pop(category, None)
                log_with_timestamp(f"Cleared cache for category: {category}")
            else:
                self._documents_cache.clear()
                self._metadata_cache.clear()
                self._loaded_asins.clear()
                log_with_timestamp("Cleared all document caches")
    
    def _get_cache_file_path(self, category: str, cache_type: str) -> str:
        """Get path for cache file"""
        return os.path.join(self.cache_dir, f"{category}_{cache_type}.pkl")
    
    def _load_from_disk_cache(self, category: str) -> Optional[Tuple[Dict, Dict]]:
        """Try to load documents from disk cache"""
        docs_cache_file = self._get_cache_file_path(category, "documents")
        meta_cache_file = self._get_cache_file_path(category, "metadata")
        
        if os.path.exists(docs_cache_file) and os.path.exists(meta_cache_file):
            try:
                cache_age = datetime.now().timestamp() - os.path.getmtime(docs_cache_file)
                if cache_age < 86400:
                    log_with_timestamp(f"Loading {category} documents from disk cache...")
                    with open(docs_cache_file, 'rb') as f:
                        documents = pickle.load(f)
                    with open(meta_cache_file, 'rb') as f:
                        metadata = pickle.load(f)
                    return documents, metadata
                else:
                    log_with_timestamp(f"Disk cache for {category} is stale (>{24}h old)")
            except Exception as e:
                log_with_timestamp(f"Error loading disk cache: {e}")
        return None
    
    def _save_to_disk_cache(self, category: str, documents: Dict, metadata: Dict):
        """Save documents to disk cache"""
        try:
            docs_cache_file = self._get_cache_file_path(category, "documents")
            meta_cache_file = self._get_cache_file_path(category, "metadata")
            
            with open(docs_cache_file, 'wb') as f:
                pickle.dump(documents, f)
            with open(meta_cache_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            log_with_timestamp(f"Saved {category} documents to disk cache")
        except Exception as e:
            log_with_timestamp(f"Error saving disk cache: {e}")
    
    def load_documents(self, category: str, required_asins: Set[str], 
                      use_preprocessed: bool = True,
                      max_reviews_per_product: int = 10,
                      max_qa_per_product: int = 25) -> Tuple[List[Dict], Dict]:
        """
        Load documents for specified ASINs in a category.
        
        Returns:
        - List of document dictionaries for the requested ASINs
        - Full metadata dictionary for all products
        """
        with self._cache_lock:
            if category in self._loaded_asins:
                loaded = self._loaded_asins[category]
                if required_asins.issubset(loaded):
                    self._cache_stats['hits'] += 1
                    log_with_timestamp(f"Cache hit: All {len(required_asins)} ASINs already loaded for {category}")
                    
                    requested_docs = []
                    for asin in required_asins:
                        if asin in self._documents_cache[category]:
                            requested_docs.append(self._documents_cache[category][asin])
                        else:
                            requested_docs.append({
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
                    
                    return requested_docs, self._metadata_cache[category]
            
            self._cache_stats['misses'] += 1
            self._cache_stats['loads'] += 1
            
            if category not in self.category_files:
                raise ValueError(f"Unknown category: {category}")
            
            files = self.category_files[category]
            
            if category not in self._documents_cache:
                cache_result = self._load_from_disk_cache(category)
                if cache_result:
                    documents, metadata = cache_result
                    self._documents_cache[category] = documents
                    self._metadata_cache[category] = metadata
                    self._loaded_asins[category] = set(documents.keys())
                    
                    if required_asins.issubset(self._loaded_asins[category]):
                        requested_docs = [
                            documents.get(asin, {
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
                            }) for asin in required_asins
                        ]
                        return requested_docs, metadata
            
            if category in self._loaded_asins:
                missing_asins = required_asins - self._loaded_asins[category]
                if missing_asins:
                    log_with_timestamp(f"Loading {len(missing_asins)} additional ASINs for {category}")
                    
                    products, _ = load_product_metadata(files["meta"], missing_asins)
                    
                    if os.path.exists(files["review"]):
                        products = load_reviews_for_products(
                            files["review"], products, 
                            max_reviews_per_product=max_reviews_per_product
                        )
                    if os.path.exists(files["qa"]):
                        products = load_qa_for_products(
                            files["qa"], products,
                            max_qa_per_product=max_qa_per_product
                        )
                    
                    self._documents_cache[category].update(products)
                    self._loaded_asins[category].update(missing_asins)
            else:
                log_with_timestamp(f"Initial load of {len(required_asins)} ASINs for {category}")
                
                preprocessed_cache_dir = os.path.join(
                    self.base_dir, "result/personal_query/12_retrieval/cache"
                )
                
                if use_preprocessed and os.path.exists(
                    os.path.join(preprocessed_cache_dir, f"products_{category}.pkl")
                ):
                    log_with_timestamp("Loading from preprocessed cache...")
                    products, metadata = load_preprocessed_products(
                        preprocessed_cache_dir, category, required_asins
                    )
                else:
                    log_with_timestamp("Loading from raw files...")
                    products, metadata = load_product_metadata(files["meta"], required_asins)
                    
                    if os.path.exists(files["review"]):
                        products = load_reviews_for_products(
                            files["review"], products,
                            max_reviews_per_product=max_reviews_per_product
                        )
                    if os.path.exists(files["qa"]):
                        products = load_qa_for_products(
                            files["qa"], products,
                            max_qa_per_product=max_qa_per_product
                        )
                
                self._documents_cache[category] = products
                self._metadata_cache[category] = metadata
                self._loaded_asins[category] = set(products.keys())
                
                self._save_to_disk_cache(category, products, metadata)
            
            requested_docs = []
            for asin in required_asins:
                if asin in self._documents_cache[category]:
                    requested_docs.append(self._documents_cache[category][asin])
                else:
                    requested_docs.append({
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
            
            log_with_timestamp(
                f"Loaded {len(requested_docs)} documents for {category} "
                f"(cache now contains {len(self._documents_cache[category])} documents)"
            )
            
            return requested_docs, self._metadata_cache[category]


def get_document_manager() -> DocumentManager:
    """Get the singleton DocumentManager instance"""
    return DocumentManager()


if __name__ == "__main__":
    dm = get_document_manager()
    
    test_asins = {"B07MQRTKXH", "B07N2RQFJL", "B07P6Y7954"}
    docs, metadata = dm.load_documents("Arts_Crafts_and_Sewing", test_asins)
    
    print(f"Loaded {len(docs)} documents")
    print(f"Cache stats: {dm.get_cache_stats()}")
    
    docs2, metadata2 = dm.load_documents("Arts_Crafts_and_Sewing", test_asins)
    print(f"Cache stats after second load: {dm.get_cache_stats()}")