#!/usr/bin/env python3
"""
Full-scale dense retrieval evaluation with all modules integrated.
"""

#!/usr/bin/env python3
"""
Full-scale dense retrieval evaluation with all modules integrated.
"""

# ===== Standard Library Imports =====
import sys
import os
import pickle
import json
import hashlib
import threading
import logging
import time
import gc
import glob
import shutil
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ===== Third-party Imports =====
import numpy as np
import torch
try:
    import psutil
except ImportError:
    psutil = None

# ===== Type Hints =====
from typing import List, Dict, Tuple, Set, Optional, Any

# ========== Setup Path and Imports ==========
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from utils import retrievers

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

# ========== lazy_cache_manager.py ==========
log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class LazyEmbeddingCache:
    """
    分离式embedding缓存管理
    
    存储结构：
    cache_dir/
    ├─ e5_457d1871f380782c05a5d94e656fef2c.pkl  (retriever配置，不含embeddings)
    ├─ e5_embeddings_457d1871f380782c05a5d94e656fef2c.npy  (302k embeddings, mmap方式)
    ├─ e5_doc_ids_457d1871f380782c05a5d94e656fef2c.pkl
    └─ e5_metadata_457d1871f380782c05a5d94e656fef2c.pkl
    """
    
    def __init__(self, cache_dir: str = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/retriever_cache"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_paths(self, retriever_name: str, doc_hash: str) -> Dict[str, str]:
        """获取retriever的所有cache文件路径"""
        base_path = os.path.join(self.cache_dir, f"{retriever_name}_{doc_hash}")
        return {
            'config': f"{base_path}_config.pkl",        # retriever配置（无embeddings）
            'embeddings': f"{base_path}_embeddings.npy", # embeddings矩阵
            'doc_ids': f"{base_path}_doc_ids.pkl",      # doc ID列表
            'metadata': f"{base_path}_metadata.pkl",    # other metadata
        }
    
    def save_retriever(self, retriever_name: str, doc_hash: str, retriever: Any):
        """
        保存retriever，分离embeddings
        """
        cache_paths = self._get_cache_paths(retriever_name, doc_hash)
        
        # 1. 保存embeddings为numpy（可mmap）
        if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
            embeddings = retriever.doc_embeddings
            
            # 转换为numpy格式
            if isinstance(embeddings, list):
                embeddings_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else (e if isinstance(e, np.ndarray) else e.numpy())
                                         for e in embeddings], dtype=np.float32)
            else:
                if isinstance(embeddings, np.ndarray):
                    embeddings_np = embeddings.astype(np.float32)
                elif hasattr(embeddings, 'cpu'):
                    embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                else:
                    embeddings_np = embeddings.numpy().astype(np.float32)
            
            # 保存为.npy（支持mmap）
            np.save(cache_paths['embeddings'], embeddings_np)
            log_with_timestamp(f"  Saved embeddings: {cache_paths['embeddings']} ({embeddings_np.nbytes / 1024**3:.2f}GB)")
            
            # 清除embeddings，保存清洁的retriever
            retriever.doc_embeddings = None
        
        # 2. 保存retriever配置（不含embeddings）
        with open(cache_paths['config'], 'wb') as f:
            pickle.dump(retriever, f)
        log_with_timestamp(f"  Saved retriever config: {cache_paths['config']}")
        
        # 3. 保存doc_ids
        if hasattr(retriever, 'doc_ids'):
            with open(cache_paths['doc_ids'], 'wb') as f:
                pickle.dump(retriever.doc_ids, f)
        
        # 4. 保存metadata
        if hasattr(retriever, 'all_metadata'):
            with open(cache_paths['metadata'], 'wb') as f:
                pickle.dump(retriever.all_metadata, f)
    
    def load_retriever(self, retriever_name: str, doc_hash: str) -> Optional[Any]:
        """
        加载retriever（不加载embeddings）
        
        支持两种格式：
        1. 新格式：{retriever}_{hash}_config.pkl + {retriever}_{hash}_embeddings.npy
        2. 旧格式：{retriever}_{hash}.pkl（向后兼容，会自动转换）
        
        Returns:
            Retriever对象，其中doc_embeddings=None
        """
        cache_paths = self._get_cache_paths(retriever_name, doc_hash)
        old_cache_path = os.path.join(self.cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        
        # 优先尝试新格式
        if os.path.exists(cache_paths['config']):
            try:
                # 1. 加载retriever配置（快速，不涉及embeddings）
                with open(cache_paths['config'], 'rb') as f:
                    retriever = pickle.load(f)
                
                log_with_timestamp(f"  Loaded retriever config (embeddings deferred)")
                
                # 2. 如果embeddings文件存在，创建mmap引用
                if os.path.exists(cache_paths['embeddings']):
                    retriever._embeddings_path = cache_paths['embeddings']
                    retriever._embeddings_mmap = None  # 延迟加载
                    log_with_timestamp(f"  Embeddings available at: {cache_paths['embeddings']}")
                
                return retriever
            
            except Exception as e:
                log_with_timestamp(f"Error loading from new cache format: {e}")
        
        # 回退到旧格式（向后兼容）
        if os.path.exists(old_cache_path):
            try:
                log_with_timestamp(f"  Old cache format detected, loading and converting...")
                with open(old_cache_path, 'rb') as f:
                    retriever = pickle.load(f)
                
                log_with_timestamp(f"  Loaded old format cache")
                
                # 旧格式已包含embeddings，立即分离并保存为新格式
                if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                    num_embeddings = len(retriever.doc_embeddings) if isinstance(retriever.doc_embeddings, list) else retriever.doc_embeddings.shape[0]
                    log_with_timestamp(f"  Extracting {num_embeddings} embeddings from old cache...")
                    
                    # 保存为新格式（自动分离embeddings）
                    self.save_retriever(retriever_name, doc_hash, retriever)
                    log_with_timestamp(f"  Converted to new format with separated embeddings")
                    
                    # 重新加载，这次会使用新格式
                    if os.path.exists(cache_paths['config']):
                        with open(cache_paths['config'], 'rb') as f:
                            retriever = pickle.load(f)
                        
                        if os.path.exists(cache_paths['embeddings']):
                            retriever._embeddings_path = cache_paths['embeddings']
                            retriever._embeddings_mmap = None
                            log_with_timestamp(f"  Embeddings extracted to: {cache_paths['embeddings']}")
                
                return retriever
            
            except Exception as e:
                log_with_timestamp(f"Error loading from old cache format: {e}")
        
        return None
    
    def load_embeddings_on_demand(self, retriever: Any):
        """
        按需加载embeddings到GPU/内存
        """
        if not hasattr(retriever, '_embeddings_path'):
            return
        
        if hasattr(retriever, '_embeddings_mmap') and retriever._embeddings_mmap is not None:
            # 已加载
            return
        
        try:
            # 使用mmap加载（不占用内存）
            embeddings_mmap = np.load(retriever._embeddings_path, mmap_mode='r')
            retriever._embeddings_mmap = embeddings_mmap
            
            log_with_timestamp(f"  Loaded embeddings via mmap: {embeddings_mmap.shape}")
        
        except Exception as e:
            log_with_timestamp(f"Error loading embeddings: {e}")
    
    def get_embeddings(self, retriever: Any) -> Optional[np.ndarray]:
        """
        获取embeddings（如果尚未加载则加载）
        """
        if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
            return retriever.doc_embeddings
        
        if hasattr(retriever, '_embeddings_mmap') and retriever._embeddings_mmap is not None:
            return retriever._embeddings_mmap
        
        if hasattr(retriever, '_embeddings_path'):
            self.load_embeddings_on_demand(retriever)
            return retriever._embeddings_mmap
        
        return None

# ========== lazy_retriever_proxy.py ==========
log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class LazyRetrieverProxy:
    """
    Lazy代理：延迟加载retriever直到实际使用
    
    使用场景：
    - 初始化时只创建proxy，不加载任何retriever
    - 第一次调用search()时，自动加载retriever
    - 后续调用直接使用已加载的retriever，无额外开销
    """
    
    def __init__(self, retriever_name: str, retriever_manager: 'RetrieverManager', 
                 documents: List[Dict], metadata: Optional[Dict] = None, 
                 use_lazy_loading: bool = True):
        self.retriever_name = retriever_name
        self.retriever_manager = retriever_manager
        self.documents = documents
        self.metadata = metadata
        self.use_lazy_loading = use_lazy_loading
        
        self._actual_retriever = None
        self._loaded = False
        
        log_with_timestamp(f"[PROXY_CREATE] Created lazy proxy for {retriever_name}")
    
    def _load_actual_retriever(self):
        """真正加载retriever的地方"""
        if self._loaded:
            return
        
        log_with_timestamp(f"[PROXY_LOAD_START] Loading actual retriever: {self.retriever_name}")
        
        self._actual_retriever = self.retriever_manager.get_retriever(
            self.retriever_name,
            self.documents,
            self.metadata,
            use_lazy_loading=self.use_lazy_loading
        )
        
        self._loaded = True
        log_with_timestamp(f"[PROXY_LOAD_DONE] {self.retriever_name} loaded (type={type(self._actual_retriever).__name__})")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        执行搜索 - 包装lazy loading逻辑以处理None embeddings
        """
        if not self._loaded:
            self._load_actual_retriever()
        
        retriever = self._actual_retriever
        
        # 仅对有 doc_embeddings 的检索器（Dense）进行检查
        if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is None:
            raise RuntimeError(f"{self.retriever_name} embeddings not loaded - got None")
        
        # 检查 embeddings 列表中的 None 值
        if hasattr(retriever, 'doc_embeddings') and isinstance(retriever.doc_embeddings, list) and any(e is None for e in retriever.doc_embeddings):
            non_none_count = sum(1 for e in retriever.doc_embeddings if e is not None)
            log_with_timestamp(f"[PROXY_WARN] {self.retriever_name}: {non_none_count}/{len(retriever.doc_embeddings)} embeddings are valid")
        
        return retriever.search(query, top_k)
    
    def __getattr__(self, name: str) -> Any:
        """
        代理所有其他属性访问
        
        任何对proxy的方法/属性访问都会触发加载
        """
        if not self._loaded:
            self._load_actual_retriever()
        
        return getattr(self._actual_retriever, name)
    
    def __repr__(self) -> str:
        if self._loaded:
            return f"LazyRetrieverProxy({self.retriever_name} → loaded {type(self._actual_retriever).__name__})"
        else:
            return f"LazyRetrieverProxy({self.retriever_name} → not loaded yet)"
    
    def get_loaded_status(self) -> tuple:
        if self._loaded:
            return (True, type(self._actual_retriever).__name__)
        else:
            return (False, None)

# ========== lazy_retriever_wrapper.py ==========
log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class LazyRetrieverWrapper:
    """
    包装任何Retriever，实现按需加载GPU embeddings
    
    工作原理：
    1. 在wrap时，从retriever中提取embeddings信息
    2. 将embeddings保存到磁盘（如果还没保存）
    3. 清除retriever中的GPU embeddings
    4. 在search()时，只加载所需的embeddings到GPU
    5. 搜索完立即释放GPU内存
    """
    
    def __init__(self, retriever: Any, cache_dir: str = None):
        self.retriever = retriever
        self.retriever_name = self._infer_retriever_name(retriever)
        self.cache_dir = cache_dir or "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/embedding_cache"
        
        # 创建embeddings索引（轻量级，不涉及GPU）
        self._build_embedding_index()
        
        # ✅ 关键：清除retriever中的GPU embeddings
        self._clear_gpu_embeddings()
        
        log_with_timestamp(f"[LazyRetriever] Wrapped {self.retriever_name} retriever")
        log_with_timestamp(f"  - Total embeddings: {len(self.doc_ids)}")
        log_with_timestamp(f"  - GPU embeddings cleared: embeddings will be loaded on-demand")
    
    def _infer_retriever_name(self, retriever: Any) -> str:
        """推断retriever类型"""
        class_name = retriever.__class__.__name__
        return class_name.replace('Retriever', '').lower()
    
    def _build_embedding_index(self):
        """构建轻量级的embedding索引（仅记录形状和类型，不加载数据）"""
        self.doc_ids = self.retriever.doc_ids
        self.all_metadata = self.retriever.all_metadata
        self.doc_embeddings_info = []
        
        if not hasattr(self.retriever, 'doc_embeddings') or self.retriever.doc_embeddings is None:
            if hasattr(self.retriever, '_embeddings_path'):
                log_with_timestamp(f"[BUILD_IDX] Deferred embeddings at: {self.retriever._embeddings_path}")
                try:
                    embeddings_shape = np.load(self.retriever._embeddings_path, mmap_mode='r').shape
                    num_docs = embeddings_shape[0]
                    for i in range(num_docs):
                        self.doc_embeddings_info.append({'type': 'single', 'index': i})
                    log_with_timestamp(f"[BUILD_IDX] Built index for {num_docs} deferred embeddings")
                except Exception as e:
                    log_with_timestamp(f"[BUILD_IDX] Error building index from mmap: {e}")
            return
        
        doc_embeddings = self.retriever.doc_embeddings
        
        for i, emb in enumerate(doc_embeddings):
            if isinstance(emb, list):
                shapes = [e.shape if hasattr(e, 'shape') else None for e in emb]
                self.doc_embeddings_info.append({
                    'type': 'multi_window',
                    'shapes': shapes,
                    'dtype': emb[0].dtype if emb and hasattr(emb[0], 'dtype') else torch.float32,
                    'index': i
                })
            else:
                shape = emb.shape if hasattr(emb, 'shape') else None
                self.doc_embeddings_info.append({
                    'type': 'single',
                    'shape': shape,
                    'dtype': emb.dtype if hasattr(emb, 'dtype') else torch.float32,
                    'index': i
                })
    
    def _clear_gpu_embeddings(self):
        """清除retriever中的GPU embeddings"""
        if not hasattr(self.retriever, 'doc_embeddings'):
            log_with_timestamp(f"  [CLEAR_EMB_DEBUG] No doc_embeddings attribute found")
            return
        
        # 对于Dense Retrievers（E5, BGE, MiniLM等）
        if self.retriever_name in ['e5', 'bge', 'dense', 'ance', 'minilm', 'mpnet', 'star']:
            log_with_timestamp(f"  [CLEAR_EMB_START] Clearing embeddings from {self.retriever_name}...")
            # 把embeddings转移到CPU
            if self.retriever.doc_embeddings is not None:
                doc_embeddings = self.retriever.doc_embeddings
                log_with_timestamp(f"    Embeddings type: {type(doc_embeddings).__name__}")
                
                # 如果是list，逐个移到CPU
                if isinstance(doc_embeddings, list):
                    log_with_timestamp(f"    Processing list of {len(doc_embeddings)} embeddings...")
                    cpu_embeddings = []
                    for i, emb in enumerate(doc_embeddings):
                        if hasattr(emb, 'cpu'):
                            cpu_embeddings.append(emb.cpu())
                        elif isinstance(emb, list):
                            cpu_embeddings.append([e.cpu() if hasattr(e, 'cpu') else e for e in emb])
                        else:
                            cpu_embeddings.append(emb)
                        if (i + 1) % 1000 == 0 or i == 0:
                            log_with_timestamp(f"      Processed {i + 1}/{len(doc_embeddings)}")
                    self.retriever.doc_embeddings = cpu_embeddings
                    log_with_timestamp(f"    ✓ All {len(cpu_embeddings)} embeddings moved to CPU")
                else:
                    # 如果是单个tensor，移到CPU
                    log_with_timestamp(f"    Moving tensor {doc_embeddings.shape} to CPU...")
                    if hasattr(doc_embeddings, 'cpu'):
                        self.retriever.doc_embeddings = doc_embeddings.cpu()
                        log_with_timestamp(f"    ✓ Tensor moved to CPU")
            else:
                log_with_timestamp(f"    Embeddings already None/cleared")
            
            log_with_timestamp(f"  [CLEAR_EMB_DONE] GPU memory cleanup for {self.retriever_name}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_with_timestamp(f"  [GPU_CACHE_EMPTY] torch.cuda.empty_cache() called")
    
    def _load_embeddings_to_gpu(self, indices: Optional[List[int]] = None):
        """
        按需加载embeddings到GPU
        
        Args:
            indices: 要加载的embedding索引。如果为None，加载全部。
        """
        import numpy as np
        
        doc_embeddings = None
        device = self._get_device()
        
        if hasattr(self.retriever, 'doc_embeddings') and self.retriever.doc_embeddings is not None:
            doc_embeddings = self.retriever.doc_embeddings
            log_with_timestamp(f"  [LOAD_EMB_FAST] Loading from doc_embeddings in memory")
        elif hasattr(self.retriever, '_embeddings_path') and self.retriever._embeddings_path:
            if self._mmap_cache is None:
                log_with_timestamp(f"  [LOAD_EMB_MMAP_INIT] Initializing persistent mmap cache: {self.retriever._embeddings_path}")
                try:
                    self._mmap_cache = np.load(self.retriever._embeddings_path, mmap_mode='r')
                    log_with_timestamp(f"    ✓ Cached mmap shape: {self._mmap_cache.shape}")
                except Exception as e:
                    log_with_timestamp(f"    Error loading embeddings: {e}")
                    return {}
            else:
                log_with_timestamp(f"  [LOAD_EMB_MMAP_CACHED] Reusing cached mmap (shape: {self._mmap_cache.shape})")
            
            doc_embeddings = self._mmap_cache
        else:
            log_with_timestamp(f"  [LOAD_EMB_NONE] No embeddings found (doc_embeddings={getattr(self.retriever, 'doc_embeddings', 'missing')}, _embeddings_path={getattr(self.retriever, '_embeddings_path', 'missing')})")
            return {}
        
        if doc_embeddings is None:
            return {}
        
        if indices is None:
            indices = list(range(len(doc_embeddings)))
        
        gpu_embeddings = {}
        
        for idx in indices:
            if idx < len(doc_embeddings):
                emb = doc_embeddings[idx]
                
                if isinstance(emb, list):
                    gpu_embeddings[idx] = [torch.tensor(e).to(device) if isinstance(e, np.ndarray) else e.to(device) if hasattr(e, 'to') else e for e in emb]
                else:
                    if isinstance(emb, np.ndarray):
                        gpu_embeddings[idx] = torch.tensor(emb).to(device)
                    else:
                        gpu_embeddings[idx] = emb.to(device) if hasattr(emb, 'to') else emb
        
        return gpu_embeddings
    
    def _release_gpu_embeddings(self, gpu_embeddings: Dict):
        """释放GPU上的embeddings"""
        if not gpu_embeddings:
            return
        
        del gpu_embeddings
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        按需加载式搜索 - 安全的embeddings重建
        """
        try:
            all_indices = list(range(len(self.doc_embeddings_info)))
            gpu_embeddings = self._load_embeddings_to_gpu(all_indices)
            
            if not gpu_embeddings:
                log_with_timestamp(f"[SEARCH_FAIL] No embeddings loaded for {self.retriever_name}")
                raise RuntimeError(f"Failed to load embeddings for {self.retriever_name}")
            
            expected_count = len(self.doc_embeddings_info)
            loaded_count = len(gpu_embeddings)
            if loaded_count != expected_count:
                log_with_timestamp(f"[SEARCH_WARN] Expected {expected_count} embeddings but got {loaded_count}")
            
            embeddings_array = []
            for i in range(expected_count):
                if i in gpu_embeddings:
                    embeddings_array.append(gpu_embeddings[i])
                else:
                    log_with_timestamp(f"[SEARCH_WARN] Missing embedding at index {i}")
                    embeddings_array.append(None)
            
            original_embeddings = self.retriever.doc_embeddings
            
            try:
                self.retriever.doc_embeddings = embeddings_array
                results = self.retriever.search(query, top_k)
            finally:
                self.retriever.doc_embeddings = original_embeddings
                self._release_gpu_embeddings(gpu_embeddings)
            
            return results
        
        except Exception as e:
            log_with_timestamp(f"[SEARCH_ERROR] {self.retriever_name}: {str(e)[:200]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    # 代理其他属性和方法
    def __getattr__(self, name):
        """代理所有未找到的属性到原retriever"""
        return getattr(self.retriever, name)
    
    def _get_device(self):
        """获取设备，兼容各种检索器"""
        if hasattr(self.retriever, 'device'):
            return self.retriever.device
        elif hasattr(self.retriever, '_model') and self.retriever._model is not None:
            if hasattr(self.retriever._model, 'device'):
                return self.retriever._model.device
        elif hasattr(self.retriever, '_get_model'):
            try:
                model = self.retriever._get_model()
                if hasattr(model, 'device'):
                    return model.device
            except:
                pass
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class BatchedLazyRetrieverWrapper(LazyRetrieverWrapper):
    """
    改进版本：批处理加载embeddings，避免一次性加载302k个
    """
    
    def __init__(self, retriever: Any, batch_size: int = 2500, cache_dir: str = None):
        super().__init__(retriever, cache_dir)
        self.batch_size = batch_size
        self._mmap_cache = None
        log_with_timestamp(f"[BatchedLazyRetriever] Batch size: {batch_size}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        批处理式搜索：分批加载embeddings，避免一次性加载所有
        
        工作流程：
        1. 编码query
        2. 分批加载document embeddings
        3. 批量计算相似度
        4. 立即释放该批GPU内存
        5. 继续下一批
        6. 合并结果并返回
        """
        try:
            search_start = time.time()
            
            model = self.retriever._get_model() if hasattr(self.retriever, '_get_model') else None
            device = self._get_device()
            
            # 编码query
            if hasattr(self.retriever, '_add_instruction'):
                query_with_prefix = self.retriever._add_instruction(query, is_query=True)
            else:
                query_with_prefix = f"query: {query}"
            
            query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)[0].to(device)
            
            log_with_timestamp(f"  [BATCHED_SEARCH_START] Query: '{query[:40]}...' | batch_size={self.batch_size} | total_docs={len(self.doc_embeddings_info)}")
            
            all_scores = []
            num_batches = (len(self.doc_embeddings_info) + self.batch_size - 1) // self.batch_size
            log_with_timestamp(f"    → {num_batches} batches to process")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.doc_embeddings_info))
                
                batch_start = time.time()
                
                # ✅ 只加载这个批次的embeddings到GPU
                batch_indices = list(range(start_idx, end_idx))
                gpu_embeddings = self._load_embeddings_to_gpu(batch_indices)
                
                # 计算这个批次的分数
                batch_scores = []
                for idx, doc_idx in enumerate(batch_indices):
                    doc_emb = gpu_embeddings[doc_idx]
                    
                    if isinstance(doc_emb, list):
                        # 多窗口
                        window_scores = util.cos_sim(query_embedding, torch.stack(doc_emb))[0]
                        score = window_scores.max().item()
                    else:
                        # 单窗口
                        score = util.cos_sim(query_embedding, doc_emb.unsqueeze(0))[0][0].item()
                    
                    batch_scores.append((self.doc_ids[doc_idx], score))
                
                all_scores.extend(batch_scores)
                
                # ✅ 立即释放这个批次的GPU内存
                self._release_gpu_embeddings(gpu_embeddings)
                
                batch_time = time.time() - batch_start
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    log_with_timestamp(f"    Batch {batch_idx + 1}/{num_batches}: {end_idx}/{len(self.doc_ids)} docs ({batch_time:.2f}s)")
            
            # 排序并返回top-k
            all_scores.sort(key=lambda x: -x[1])
            
            search_time = time.time() - search_start
            log_with_timestamp(f"  [BATCHED_SEARCH_DONE] Completed in {search_time:.2f}s, returning top-{min(top_k, len(all_scores))} results")
            
            return all_scores[:top_k]
        
        except Exception as e:
            log_with_timestamp(f"[BATCHED_SEARCH_ERROR] Error in batched lazy search: {e}")
            torch.cuda.empty_cache()
            raise


# ========== retriever_manager.py ==========
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

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
    
    def cache_exists(self, retriever_name: str, doc_hash: str) -> bool:
        """Check if cache files exist for a retriever"""
        if retriever_name in DENSE_RETRIEVERS:
            # Dense retrievers use separated storage format
            base_path = os.path.join(self.cache_dir, f"{retriever_name}_{doc_hash}")
            config_path = f"{base_path}_config.pkl"
            embeddings_path = f"{base_path}_embeddings.npy"
            return os.path.exists(config_path) and os.path.exists(embeddings_path)
        else:
            # Sparse retrievers use single file format
            cache_path = self._get_cache_path(retriever_name, doc_hash)
            return os.path.exists(cache_path)
    
    def _load_from_cache(self, retriever_name: str, doc_hash: str) -> Optional[Any]:
        """Load retriever from disk cache if available"""
        log_with_timestamp(f"[LOAD_CACHE_ATTEMPT] Attempting to load {retriever_name} cache (hash={doc_hash[:8]}...)")
        
        if retriever_name in DENSE_RETRIEVERS:
            log_with_timestamp(f"[LOAD_CACHE_DENSE] {retriever_name} is DENSE_RETRIEVER, using LazyEmbeddingCache")
            log_with_timestamp(f"  Calling lazy_cache.load_retriever({retriever_name}, {doc_hash[:8]}...)")
            retriever = self.lazy_cache.load_retriever(retriever_name, doc_hash)
            log_with_timestamp(f"[LOAD_CACHE_DENSE_RESULT] lazy_cache.load_retriever returned: {type(retriever).__name__ if retriever else 'None'}")
            if retriever:
                log_with_timestamp(f"[CACHE_LOAD_SUCCESS] Loaded {retriever_name}, embeddings deferred")
                if hasattr(retriever, '_embeddings_path'):
                    log_with_timestamp(f"  → Embeddings reference: {retriever._embeddings_path}")
                else:
                    log_with_timestamp(f"  → WARNING: No _embeddings_path attribute!")
            else:
                log_with_timestamp(f"[LOAD_CACHE_DENSE_FAILED] lazy_cache.load_retriever returned None - will rebuild")
            return retriever
        
        log_with_timestamp(f"[LOAD_CACHE_SPARSE] {retriever_name} is NOT dense, checking old format cache")
        cache_path = self._get_cache_path(retriever_name, doc_hash)
        log_with_timestamp(f"  Old format cache path: {cache_path}")
        log_with_timestamp(f"  File exists: {os.path.exists(cache_path)}")
        
        if os.path.exists(cache_path):
            try:
                log_with_timestamp(f"[CACHE_LOAD_START] Loading {retriever_name} from old format cache...")
                cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                log_with_timestamp(f"  Cache file size: {cache_size_mb:.1f} MB")
                with open(cache_path, 'rb') as f:
                    retriever = pickle.load(f)
                
                log_with_timestamp(f"[CACHE_LOAD_SUCCESS] Loaded {retriever_name}, type: {type(retriever).__name__}")
                return retriever
            except Exception as e:
                log_with_timestamp(f"[CACHE_LOAD_ERROR] Error loading cache for {retriever_name}: {e}")
        else:
            log_with_timestamp(f"[CACHE_NOT_FOUND] Old format cache file does not exist")
        
        log_with_timestamp(f"[LOAD_CACHE_RETURN_NONE] No cache found for {retriever_name}")
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
        
        log_with_timestamp(f"[GET_RETRIEVER] Getting {retriever_name} with {len(documents)} documents")
        doc_hash = self._compute_document_hash(documents)
        log_with_timestamp(f"[GET_RETRIEVER_HASH] Computed doc_hash: {doc_hash}")
        cache_key = f"{retriever_name}_{doc_hash}"
        log_with_timestamp(f"[GET_RETRIEVER_KEY] Cache key: {cache_key}")
        
        with self._cache_lock:
            log_with_timestamp(f"[GET_RETRIEVER_LOCK] Acquired cache lock")
            if cache_key in self._retrievers:
                log_with_timestamp(f"[GET_RETRIEVER_MEMORY_HIT] Using {retriever_name} from memory cache")
                return self._retrievers[cache_key]
            
            log_with_timestamp(f"[GET_RETRIEVER_MEMORY_MISS] Not in memory cache, checking disk...")
            cached_retriever = self._load_from_cache(retriever_name, doc_hash)
            log_with_timestamp(f"[GET_RETRIEVER_DISK_RESULT] _load_from_cache returned: {type(cached_retriever).__name__ if cached_retriever else 'None'}")
            if cached_retriever is not None:
                log_with_timestamp(f"[CACHE_LOADED] Using cached {retriever_name}")
                
                if hasattr(cached_retriever, '_embeddings_path') and cached_retriever._embeddings_path:
                    if cached_retriever.doc_embeddings is None:
                        log_with_timestamp(f"[EMBEDDINGS_LOAD] {retriever_name} loading from mmap, making writable copy")
                        try:
                            embeddings_mmap = np.load(cached_retriever._embeddings_path, mmap_mode='r')
                            log_with_timestamp(f"[EMBEDDINGS_SIZE] Shape: {embeddings_mmap.shape}")
                            embeddings_tensor = torch.from_numpy(embeddings_mmap).float().clone()
                            
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            embeddings_tensor = embeddings_tensor.to(device)
                            cached_retriever.doc_embeddings = embeddings_tensor
                            
                            log_with_timestamp(f"[EMBEDDINGS_READY] Shape: {embeddings_tensor.shape}, device: {device}")
                        except Exception as e:
                            log_with_timestamp(f"[EMBEDDINGS_FAIL] {str(e)[:200]}")
                            raise RuntimeError(f"Failed to load embeddings for {retriever_name}: {e}")
                
                self._retrievers[cache_key] = cached_retriever
                return cached_retriever
            
            log_with_timestamp(f"[ERROR] No cached index found for {retriever_name}")
            raise RuntimeError(f"Retriever {retriever_name} not in cache. Script only uses pre-built cached indices.")
    
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


# ========== Main Script ==========
"""
FULL-SCALE Retrieval Evaluation for All Users (302k products)

Evaluates all retrievers on the complete product catalog (302,380 products)
instead of just the 535 products that appear in user queries.

This provides a more realistic evaluation of model performance on the full corpus.
"""

import argparse
import json
import os
import sys
import glob
import pickle
import shutil
import traceback
import psutil
import torch
import threading
from datetime import datetime
log_with_timestamp = utils.log_with_timestamp
evaluate_retriever = utils.evaluate_retriever


def _print_metrics_summary(retriever_name: str, user_id: str, mode: str, metrics: Dict, num_queries: int, clean_metrics: Dict = None):
    """Print detailed metrics for completed evaluation with optional noise robustness comparison"""
    log_with_timestamp(f"  ✓ {retriever_name.upper()} ({user_id}, {mode}) - {num_queries} queries")
    log_with_timestamp(f"    Precision:  P@1={metrics.get('P@1', 0):.4f}  P@3={metrics.get('P@3', 0):.4f}  P@5={metrics.get('P@5', 0):.4f}  P@10={metrics.get('P@10', 0):.4f}")
    log_with_timestamp(f"    Recall:     R@1={metrics.get('R@1', 0):.4f}  R@3={metrics.get('R@3', 0):.4f}  R@5={metrics.get('R@5', 0):.4f}  R@10={metrics.get('R@10', 0):.4f}")
    log_with_timestamp(f"    MAP:        M@1={metrics.get('MAP@1', 0):.4f}  M@3={metrics.get('MAP@3', 0):.4f}  M@5={metrics.get('MAP@5', 0):.4f}  M@10={metrics.get('MAP@10', 0):.4f}")
    log_with_timestamp(f"    NDCG:      ND@1={metrics.get('NDCG@1', 0):.4f} ND@3={metrics.get('NDCG@3', 0):.4f} ND@5={metrics.get('NDCG@5', 0):.4f} ND@10={metrics.get('NDCG@10', 0):.4f}")
    log_with_timestamp(f"    MRR:       MR@1={metrics.get('MRR@1', 0):.4f} MR@3={metrics.get('MRR@3', 0):.4f} MR@5={metrics.get('MRR@5', 0):.4f} MR@10={metrics.get('MRR@10', 0):.4f}")
    
    if 'F1@1' in metrics:
        log_with_timestamp(f"    F1-Score:   F@1={metrics.get('F1@1', 0):.4f}  F@3={metrics.get('F1@3', 0):.4f}  F@5={metrics.get('F1@5', 0):.4f}  F@10={metrics.get('F1@10', 0):.4f}")
        log_with_timestamp(f"    Hit Rate:   H@1={metrics.get('Hit@1', 0):.4f}  H@3={metrics.get('Hit@3', 0):.4f}  H@5={metrics.get('Hit@5', 0):.4f}  H@10={metrics.get('Hit@10', 0):.4f}")
        log_with_timestamp(f"    Avg Rank:   AR@1={metrics.get('AvgRank@1', 0):.1f}  AR@3={metrics.get('AvgRank@3', 0):.1f}  AR@5={metrics.get('AvgRank@5', 0):.1f}  AR@10={metrics.get('AvgRank@10', 0):.1f}")
    
    if 'DCG@1' in metrics:
        log_with_timestamp(f"    DCG:       DCG@1={metrics.get('DCG@1', 0):.4f} DCG@3={metrics.get('DCG@3', 0):.4f} DCG@5={metrics.get('DCG@5', 0):.4f} DCG@10={metrics.get('DCG@10', 0):.4f}")
        log_with_timestamp(f"    CG:        CG@1={metrics.get('CG@1', 0):.4f}  CG@3={metrics.get('CG@3', 0):.4f}  CG@5={metrics.get('CG@5', 0):.4f}  CG@10={metrics.get('CG@10', 0):.4f}")
        log_with_timestamp(f"    ERR:       ERR@1={metrics.get('ERR@1', 0):.4f} ERR@3={metrics.get('ERR@3', 0):.4f} ERR@5={metrics.get('ERR@5', 0):.4f} ERR@10={metrics.get('ERR@10', 0):.4f}")
        log_with_timestamp(f"    RBP:       RBP@1={metrics.get('RBP@1', 0):.4f} RBP@3={metrics.get('RBP@3', 0):.4f} RBP@5={metrics.get('RBP@5', 0):.4f} RBP@10={metrics.get('RBP@10', 0):.4f}")
    
    if 'NDCG@10_stats' in metrics:
        stats = metrics['NDCG@10_stats']
        log_with_timestamp(f"    NDCG@10 Distribution: median={stats.get('median', 0):.4f} p90={stats.get('p90', 0):.4f} p95={stats.get('p95', 0):.4f} std={stats.get('std', 0):.4f}")
    
    if 'Performance_Distribution@10' in metrics:
        dist = metrics['Performance_Distribution@10']
        log_with_timestamp(f"    Query Classification: High{dist.get('high', 0):.1f}% Medium{dist.get('medium', 0):.1f}% Low{dist.get('low', 0):.1f}%")
    
    if clean_metrics and mode == 'noisy':
        ndcg_clean = clean_metrics.get('NDCG@10', 0)
        ndcg_noisy = metrics.get('NDCG@10', 0)
        ndcg_delta = ndcg_noisy - ndcg_clean
        ndcg_rel = (ndcg_delta / ndcg_clean * 100) if ndcg_clean > 0 else 0
        
        hit_clean = clean_metrics.get('Hit@10', 0)
        hit_noisy = metrics.get('Hit@10', 0)
        hit_delta = hit_noisy - hit_clean
        
        robustness = 1 - abs(ndcg_delta) / ndcg_clean if ndcg_clean > 0 else 0
        
        log_with_timestamp(f"    Noise Robustness (vs Clean):")
        log_with_timestamp(f"      NDCG@10: {ndcg_clean:.4f} → {ndcg_noisy:.4f} ({ndcg_delta:+.4f}, {ndcg_rel:+.1f}%)")
        log_with_timestamp(f"      Hit@10:  {hit_clean:.4f} → {hit_noisy:.4f} ({hit_delta:+.4f})")
        log_with_timestamp(f"      Robustness: {robustness:.1%}")


def _clean_user_old_results(user_id: str, output_dir: str):
    """Delete old evaluation results for a user before starting new evaluation"""
    user_output_dir = os.path.join(output_dir, user_id)
    if os.path.exists(user_output_dir):
        try:
            shutil.rmtree(user_output_dir)
            log_with_timestamp(f"  🗑 Cleaned old results for {user_id}")
        except Exception as e:
            log_with_timestamp(f"  ⚠ Warning: Failed to clean old results for {user_id}: {e}")


STAGE6_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query"
STAGE9_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/09_targeted_noisy_query"
STAGE7_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/07_iterative_refinement"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval"
LOG_FILE = "/home/wlia0047/ar57/wenyu/stage12_fullscale_evaluation.log"

RETRIEVER_NAMES = [
    'bm25', 'tfidf', 'dirichlet',
    'dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star',
    'colbert'
]

DEFAULT_K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]

RETRIEVER_TYPES = {
    'sparse': ['bm25', 'tfidf'],
    'dense': ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star'],
    'late': []  # ['colbert']  ← 暂时禁用 ColBERT 评估
}

RETRIEVER_ORDER = ['sparse', 'dense']


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

    stage7_pattern = os.path.join(STAGE7_DIR, "*_interative_query.json")
    for file_path in glob.glob(stage7_pattern):
        filename = os.path.basename(file_path)
        user_id = filename.replace("_interative_query.json", "")
        if user_id:
            users.append(user_id)

    iterative_noisy_pattern = os.path.join(STAGE9_DIR, "iterative_noisy_query_*.json")
    for file_path in glob.glob(iterative_noisy_pattern):
        filename = os.path.basename(file_path)
        user_id = filename.replace("iterative_noisy_query_", "").replace(".json", "")
        if user_id and user_id not in {"summary", "all_users_summary"}:
            users.append(user_id)
    
    return sorted(set(users))


def load_stage7_queries(user_id: str) -> List[Dict]:
    query_file = os.path.join(STAGE7_DIR, f"{user_id}_interative_query.json")
    if not os.path.exists(query_file):
        return []

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    stage7_queries = []
    for item in data:
        if not isinstance(item, dict):
            continue

        asin = item.get('asin', '')
        query_text = item.get('final_query', '') or item.get('target_query', '') or item.get('base_query', '')
        if not asin or not query_text:
            continue

        stage7_queries.append({
            'asin': asin,
            'query': query_text,
            'type': 'target',
            'category': '',
            'selected_attributes': [],
            'is_noisy': False
        })

    return stage7_queries


def load_stage7_clean_noisy_queries(user_id: str) -> Dict[str, List[Dict]]:
    query_file = os.path.join(STAGE9_DIR, f"iterative_noisy_query_{user_id}.json")
    if not os.path.exists(query_file):
        return {'stage7_clean': [], 'stage7_noisy': []}

    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {'stage7_clean': [], 'stage7_noisy': []}

    result = {'stage7_clean': [], 'stage7_noisy': []}
    for q in data.get('queries', []):
        if not isinstance(q, dict):
            continue
        asin = q.get('asin', '')
        if not asin:
            continue
        pq = q.get('personalized_query', {}) if isinstance(q.get('personalized_query', {}), dict) else {}

        clean_query_text = pq.get('original', '')
        if clean_query_text:
            result['stage7_clean'].append({
                'asin': asin,
                'query': clean_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': False
            })

        noisy_query_text = pq.get('noisy', '') or clean_query_text
        if noisy_query_text:
            result['stage7_noisy'].append({
                'asin': asin,
                'query': noisy_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': True
            })

    return result


def load_user_queries(user_id: str, query_mode: str = 'both') -> Dict[str, List[Dict]]:
    query_file = os.path.join(STAGE9_DIR, f"noisy_queries_{user_id}.json")
    
    if not os.path.exists(query_file):
        return {}
    
    with open(query_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    result = {'clean': [], 'noisy': [], 'stage7': [], 'stage7_clean': [], 'stage7_noisy': []}
    
    for q in queries:
        asin = q.get('asin', '')
        if not asin:
            continue
        
        pq = q.get('personalized_query', {})
        
        clean_query_text = pq.get('original', '')
        if clean_query_text and query_mode in ['both', 'clean', 'all']:
            result['clean'].append({
                'asin': asin,
                'query': clean_query_text,
                'type': 'target',
                'category': '',
                'selected_attributes': [],
                'is_noisy': False
            })
        
        if query_mode in ['both', 'noisy', 'all']:
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
    
    if query_mode in ['stage7', 'all']:
        result['stage7'] = load_stage7_queries(user_id)

    if query_mode in ['stage7_clean', 'stage7_noisy', 'stage7_both', 'all']:
        stage7_pair = load_stage7_clean_noisy_queries(user_id)
        if query_mode in ['stage7_clean', 'stage7_both', 'all']:
            result['stage7_clean'] = stage7_pair.get('stage7_clean', [])
        if query_mode in ['stage7_noisy', 'stage7_both', 'all']:
            result['stage7_noisy'] = stage7_pair.get('stage7_noisy', [])

    if query_mode == 'clean':
        result['noisy'] = []
        result['stage7'] = []
    elif query_mode == 'noisy':
        result['clean'] = []
        result['stage7'] = []
    elif query_mode == 'stage7':
        result['clean'] = []
        result['noisy'] = []
        result['stage7_clean'] = []
        result['stage7_noisy'] = []
    elif query_mode == 'stage7_clean':
        result['clean'] = []
        result['noisy'] = []
        result['stage7'] = []
        result['stage7_noisy'] = []
    elif query_mode == 'stage7_noisy':
        result['clean'] = []
        result['noisy'] = []
        result['stage7'] = []
        result['stage7_clean'] = []
    elif query_mode == 'stage7_both':
        result['clean'] = []
        result['noisy'] = []
        result['stage7'] = []

    return result


def _cache_path_for_mode(query_cache_root: str, retriever_name: str, user_id: str, mode: str) -> str:
    if mode == 'clean':
        subdir = 'stage6_clean_query'
        suffix = 'clean'
    elif mode == 'noisy':
        subdir = 'stage6_noisy_query'
        suffix = 'noisy'
    elif mode == 'stage7_clean':
        subdir = 'stage7_clean_query'
        suffix = 'clean'
    elif mode == 'stage7_noisy':
        subdir = 'stage7_noisy_query'
        suffix = 'noisy'
    elif mode == 'stage7':
        subdir = 'stage7_clean_query'
        suffix = 'clean'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return os.path.join(
        query_cache_root,
        subdir,
        f"{retriever_name.lower()}_{user_id}_{suffix}_cache.pkl"
    )


def load_fullscale_asins(metadata_file: str) -> Set[str]:
    """Load all 302k ASIN keys from metadata"""
    log_with_timestamp(f"Loading full metadata from {metadata_file}...")
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    asins = set(metadata.keys())
    log_with_timestamp(f"✓ Loaded {len(asins)} unique ASINs from metadata")
    
    return asins


def _evaluate_single_mode(retriever, retriever_name: str, user_id: str, mode: str, queries: List[Dict], all_asins: List[str], output_dir: str, k_values: List[int]) -> Tuple[str, Dict]:
    """Evaluate a single mode (clean or noisy) and save results."""
    if not queries:
        return mode, {}
    
    try:
        log_with_timestamp(f"[EVAL_MODE_START] {retriever_name}/{user_id} ({mode}): {len(queries)} queries, {len(all_asins)} products")
        
        # Load query cache if available (禁止编码，直接使用预生成缓存)
        query_cache_root = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/query_cache"
        query_cache_file = _cache_path_for_mode(query_cache_root, retriever_name, user_id, mode)

        if not os.path.exists(query_cache_file):
            legacy_cache_file = os.path.join(
                query_cache_root,
                f"{retriever_name.lower()}_{user_id}_{mode}_cache.pkl"
            )
            if os.path.exists(legacy_cache_file):
                query_cache_file = legacy_cache_file
        
        if os.path.exists(query_cache_file):
            log_with_timestamp(f"  ✓ 加载查询缓存: {query_cache_file}")
            retriever = retrievers.CachedRetriever(retriever, query_cache_file)
            log_with_timestamp(f"  ✓ 已用 CachedRetriever 包装检索器 (禁止编码)")
        else:
            log_with_timestamp(f"  ⚠️  查询缓存不存在: {query_cache_file}")
            log_with_timestamp(f"  ⚠️  将使用原始检索器 (可能涉及编码)")
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_START] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU: {gpu_mem:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
        # 调用修改后的evaluate_retriever，返回metrics和query_results
        metrics, query_results = evaluate_retriever(retriever, queries, all_asins, k_values, mode=mode, return_query_results=True)
        
        _print_metrics_summary(retriever_name, user_id, mode, metrics, len(queries))
        
        # Log system resources after evaluation
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_END] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU: {gpu_mem:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
        output_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'num_documents': len(all_asins),
            'evaluation_scale': 'fullscale (302,380 products)',
            'k_values': k_values,
            'retriever': retriever_name,
            'query_type': 'target_user',
            'metrics': metrics
        }
        
        user_output_dir = os.path.join(output_dir, user_id)
        os.makedirs(user_output_dir, exist_ok=True)
        
        # 保存metrics文件（已有）
        output_file = os.path.join(user_output_dir, f"retrieval_{retriever_name}_{mode}_fullscale.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # 保存top10结果文件（新增）
        if query_results:
            top10_output_file = os.path.join(user_output_dir, f"retrieval_{retriever_name}_{mode}_top10_results.json")
            top10_data = {
                'user_id': user_id,
                'retriever': retriever_name,
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(query_results),
                'query_results': query_results
            }
            with open(top10_output_file, 'w') as f:
                json.dump(top10_data, f, indent=2, ensure_ascii=False)
            log_with_timestamp(f"  ✓ 已保存top10结果: {top10_output_file}")
        
        log_with_timestamp(f"[EVAL_MODE_SUCCESS] {retriever_name}/{user_id} ({mode}) completed")
        log_with_timestamp(f"  ✓ 已保存metrics: {output_file}")
        return mode, metrics
        
    except Exception as e:
        log_with_timestamp(f"[EVAL_MODE_ERROR] {retriever_name}/{user_id} ({mode}) FAILED")
        log_with_timestamp(f"  Exception Type: {type(e).__name__}")
        log_with_timestamp(f"  Exception Message: {str(e)}")
        log_with_timestamp(f"  Exception Args: {e.args}")
        
        # Log full traceback
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            if line.strip():
                log_with_timestamp(f"  Traceback: {line}")
        
        # Log resource state at error time
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
            log_with_timestamp(f"  [RESOURCE_AT_ERROR] CPU: {cpu_percent}% | RAM: {mem_info.percent}% ({mem_info.used//(1024**3)}GB/{mem_info.total//(1024**3)}GB) | GPU Allocated: {gpu_mem:.2f}GB | Reserved: {gpu_reserved:.2f}GB")
        except Exception as res_e:
            log_with_timestamp(f"  [RESOURCE_AT_ERROR_FAILED] {type(res_e).__name__}: {str(res_e)}")
        
        raise


def evaluate_user_with_retriever(
    retriever,
    retriever_name: str,
    user_id: str,
    user_queries: Dict[str, List[Dict]],
    all_asins: List[str],
    output_dir: str,
    k_values: List[int]
) -> Dict[str, Dict]:
    """并发评估多种查询模式，串行处理不同检索器和用户。"""
    results = {}
    
    try:
        log_with_timestamp(f"[RETRIEVER_EVAL_START] {retriever_name}/{user_id} - Submitting multi-mode evaluation tasks")
        
        non_empty_modes = [(mode, queries) for mode, queries in user_queries.items() if queries]
        max_workers = max(1, min(3, len(non_empty_modes)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for mode, queries in non_empty_modes:
                log_with_timestamp(f"[THREAD_SUBMIT] {retriever_name}/{user_id} ({mode}): {len(queries)} queries submitted to thread pool")
                futures[mode] = executor.submit(
                    _evaluate_single_mode,
                    retriever, retriever_name, user_id, mode, queries,
                    all_asins, output_dir, k_values
                )
            
            log_with_timestamp(f"[WAITING_RESULTS] {retriever_name}/{user_id}: Waiting for {len(futures)} mode(s) to complete (timeout: 1800s)")
            
            for mode, future in futures.items():
                try:
                    log_with_timestamp(f"[THREAD_WAITING] {retriever_name}/{user_id} ({mode}): Waiting for result...")
                    mode_result, metrics = future.result(timeout=1800)
                    results[mode_result] = metrics
                    log_with_timestamp(f"[THREAD_COMPLETED] {retriever_name}/{user_id} ({mode}): Successfully retrieved results")
                    
                except concurrent.futures.TimeoutError as te:
                    log_with_timestamp(f"[TIMEOUT_ERROR] {retriever_name}/{user_id} ({mode}): TIMEOUT after 1800 seconds")
                    log_with_timestamp(f"  Exception Type: TimeoutError")
                    log_with_timestamp(f"  Exception Message: {str(te)}")
                    log_with_timestamp(f"  [RESOURCE_AT_TIMEOUT] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                    if torch.cuda.is_available():
                        log_with_timestamp(f"  [GPU_AT_TIMEOUT] GPU Memory: {torch.cuda.memory_allocated()/(1024**3):.2f}GB")
                    
                    # Log thread dump to diagnose deadlock
                    import sys
                    log_with_timestamp(f"  [THREAD_DUMP] Stuck thread(s):")
                    for thread_id, frame in sys._current_frames().items():
                        log_with_timestamp(f"    Thread {thread_id}:")
                        for line in traceback.format_stack(frame):
                            if line.strip():
                                log_with_timestamp(f"      {line.strip()}")
                
                except Exception as e:
                    log_with_timestamp(f"[THREAD_ERROR] {retriever_name}/{user_id} ({mode}): EXCEPTION occurred")
                    log_with_timestamp(f"  Exception Type: {type(e).__name__}")
                    log_with_timestamp(f"  Exception Message: {str(e)}")
                    log_with_timestamp(f"  Exception Args: {e.args}")
                    
                    tb_lines = traceback.format_exc().split('\n')
                    log_with_timestamp(f"  [FULL_TRACEBACK] ({len(tb_lines)} lines):")
                    for i, line in enumerate(tb_lines):
                        if line.strip():
                            log_with_timestamp(f"    Line {i}: {line}")
                    
                    log_with_timestamp(f"  [RESOURCE_AT_THREAD_ERROR] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                    if torch.cuda.is_available():
                        log_with_timestamp(f"  [GPU_AT_THREAD_ERROR] Allocated: {torch.cuda.memory_allocated()/(1024**3):.2f}GB | Reserved: {torch.cuda.memory_reserved()/(1024**3):.2f}GB")
        
        log_with_timestamp(f"[RETRIEVER_EVAL_DONE] {retriever_name}/{user_id}: Completed with {len(results)} mode(s)")
        
        if 'clean' in results and 'noisy' in results:
            clean_metrics = results.get('clean')
            noisy_metrics = results.get('noisy')
            if clean_metrics and noisy_metrics:
                _print_metrics_summary(retriever_name, user_id, 'noisy', noisy_metrics, len(user_queries.get('noisy', [])), clean_metrics=clean_metrics)

        if 'stage7_clean' in results and 'stage7_noisy' in results:
            stage7_clean_metrics = results.get('stage7_clean')
            stage7_noisy_metrics = results.get('stage7_noisy')
            if stage7_clean_metrics and stage7_noisy_metrics:
                _print_metrics_summary(
                    retriever_name,
                    user_id,
                    'noisy',
                    stage7_noisy_metrics,
                    len(user_queries.get('stage7_noisy', [])),
                    clean_metrics=stage7_clean_metrics
                )
        
    except Exception as e:
        log_with_timestamp(f"[RETRIEVER_EVAL_CRITICAL_ERROR] {retriever_name}/{user_id}: CRITICAL ERROR in thread pool management")
        log_with_timestamp(f"  Exception Type: {type(e).__name__}")
        log_with_timestamp(f"  Exception Message: {str(e)}")
        
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            if line.strip():
                log_with_timestamp(f"  Traceback: {line}")
    
    return results


def evaluate_batch_fullscale(
    user_ids: List[str],
    mode: str = 'both',
    category: str = "Arts_Crafts_and_Sewing",
    parallel_retrievers: int = 2,
    logger = None
) -> Dict:
    
    if logger is None:
        logger = setup_logging()
    
    dm = get_document_manager()
    rm = get_retriever_manager()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("FULL-SCALE RETRIEVAL EVALUATION (302,380 products)")
    logger.info("=" * 80)
    logger.info(f"Users to process: {len(user_ids)}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Category: {category}")
    
    logger.info("\nLoading full metadata (302,380 products)...")
    metadata_file = "/home/wlia0047/ar57/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    
    all_asins = load_fullscale_asins(metadata_file)
    all_asins_list = sorted(list(all_asins))
    
    logger.info("\nLoading user queries...")
    user_queries_map = {}
    valid_users = []
    
    for user_id in user_ids:
        queries = load_user_queries(user_id, mode)
        if any(queries.values()):
            user_queries_map[user_id] = queries
            valid_users.append(user_id)
        else:
            logger.warning(f"Skipping {user_id} - no valid queries found")
    
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
    
    retrievers = {}
    enabled_retrievers = []
    for retriever_type in RETRIEVER_ORDER:
        enabled_retrievers.extend(RETRIEVER_TYPES[retriever_type])
    
    # Compute document hash to check cache existence
    doc_hash = dm._compute_document_hash(documents) if hasattr(dm, '_compute_document_hash') else hashlib.md5('|'.join(sorted([doc.get('asin', '') for doc in documents])).encode()).hexdigest()
    
    logger.info(f"[LAZY_INIT_START] Creating lazy proxies for {len(enabled_retrievers)} retrievers...")
    logger.info(f"[CACHE_CHECK] Document hash: {doc_hash[:16]}...")
    
    for retriever_name in enabled_retrievers:
        try:
            # Check if cache exists before creating proxy
            if not rm.cache_exists(retriever_name, doc_hash):
                logger.warning(f"[SKIP_RETRIEVER] {retriever_name} - cache not found, will skip evaluation")
                continue
            
            logger.info(f"[LAZY_PROXY_CREATE] {retriever_name}")
            retrievers[retriever_name] = rm.create_lazy_proxy(
                retriever_name, 
                documents, 
                metadata,
                use_lazy_loading=True
            )
        except Exception as e:
            logger.error(f"Failed to create proxy for {retriever_name}: {e}")
    
    logger.info(f"[LAZY_INIT_DONE] Created proxies for {len(retrievers)} retrievers (actual loading deferred)")
    
    print(f"\n{'='*80}\n[LAZY_LOADING_STATUS]\n{'='*80}", flush=True)
    for retriever_name, proxy in retrievers.items():
        is_loaded, retriever_type = proxy.get_loaded_status()
        status = f"LOADED ({retriever_type})" if is_loaded else "NOT LOADED YET"
        logger.info(f"  {retriever_name}: {status}")
    print(f"{'='*80}\n", flush=True)
    
    logger.info(f"\nCleaning old results for {len(valid_users)} users...")
    for user_id in valid_users:
        _clean_user_old_results(user_id, OUTPUT_DIR)
    
    results = {
        'succeeded': defaultdict(list),
        'failed': defaultdict(list),
        'scale': 'fullscale'
    }
    
    total_retriever_names = sum(len(names) for names in RETRIEVER_TYPES.values())
    total_evaluations = len(valid_users) * total_retriever_names
    completed = 0
    
    logger.info(f"\nStarting evaluations ({total_evaluations} total)...")
    logger.info(f"Retriever order: {' → '.join(RETRIEVER_ORDER)}")
    logger.info(f"  Sparse ({len(RETRIEVER_TYPES['sparse'])}): {', '.join(RETRIEVER_TYPES['sparse'])}")
    logger.info(f"  Dense ({len(RETRIEVER_TYPES['dense'])}): {', '.join(RETRIEVER_TYPES['dense'])}")
    logger.info(f"  Late ({len(RETRIEVER_TYPES['late'])}): {', '.join(RETRIEVER_TYPES['late'])}")
    
    logger.info("\n[SERIAL_USERS] Users processed serially (one user completes before next starts)")
    logger.info("[SELECTIVE_CONCURRENCY] Within user: Sparse: 8 workers → Dense: 1 worker (CPU vs GPU-bound)")
    
    sparse_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix='sparse')
    dense_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='dense')
    
    try:
        for user_idx, user_id in enumerate(valid_users, 1):
            logger.info(f"\n[USER_START] Processing user {user_idx}/{len(valid_users)}: {user_id}")
            
            for retriever_type in RETRIEVER_ORDER:
                executor = sparse_executor if retriever_type == 'sparse' else dense_executor
                phase_name = retriever_type.upper()
                
                logger.info(f"  [{phase_name}_START] Evaluating {len(RETRIEVER_TYPES[retriever_type])} {retriever_type} retrievers...")
                
                futures_for_phase = []
                
                for retriever_name in RETRIEVER_TYPES[retriever_type]:
                    if retriever_name in retrievers:
                        retriever = retrievers[retriever_name]
                        future = executor.submit(
                            evaluate_user_with_retriever,
                            retriever,
                            retriever_name,
                            user_id,
                            user_queries_map[user_id],
                            all_asins_list,
                            OUTPUT_DIR,
                            DEFAULT_K_VALUES
                        )
                        futures_for_phase.append((future, user_id, retriever_name, retriever_type))
                
                for future, user_id_inner, retriever_name, retriever_type_inner in futures_for_phase:
                    try:
                        logger.info(f"  [FUTURE_WAITING] {retriever_name} ({retriever_type_inner}) for {user_id_inner}: Waiting for result (timeout: 1800s)...")
                        user_results = future.result(timeout=1800)
                        results['succeeded'][user_id_inner].append(retriever_name)
                        completed += 1
                        logger.info(f"  [FUTURE_COMPLETED] {retriever_name} ({retriever_type_inner}) for {user_id_inner}: ✓ SUCCESS")
                        
                        if completed % 5 == 0:
                            logger.info(f"  Progress: {completed}/{total_evaluations} ({100*completed/total_evaluations:.1f}%)")
                            
                    except concurrent.futures.TimeoutError as te:
                        logger.error(f"  ✗ TIMEOUT: {retriever_name} ({retriever_type_inner}) for {user_id_inner} - exceeded 1800 seconds")
                        logger.error(f"    Exception Type: TimeoutError")
                        logger.error(f"    [RESOURCE_AT_TIMEOUT] CPU: {psutil.cpu_percent(interval=0.1)}% | RAM: {psutil.virtual_memory().percent}%")
                        if torch.cuda.is_available():
                            logger.error(f"    [GPU_AT_TIMEOUT] GPU Memory: {torch.cuda.memory_allocated()/(1024**3):.2f}GB / Reserved: {torch.cuda.memory_reserved()/(1024**3):.2f}GB")
                        results['failed'][user_id_inner].append(retriever_name)
                        completed += 1
                        
                    except Exception as e:
                        logger.error(f"  ✗ FAILED: {retriever_name} ({retriever_type_inner}) for {user_id_inner}")
                        logger.error(f"    Exception Type: {type(e).__name__}")
                        logger.error(f"    Exception Message: {str(e)}")
                        logger.error(f"    Exception Args: {e.args}")
                        
                        tb_list = traceback.format_exc().split('\n')
                        logger.error(f"    [FULL_TRACEBACK] ({len(tb_list)} lines):")
                        for i, tb_line in enumerate(tb_list):
                            if tb_line.strip():
                                logger.error(f"      [{i}] {tb_line}")
                        
                        try:
                            cpu_p = psutil.cpu_percent(interval=0.1)
                            mem = psutil.virtual_memory()
                            logger.error(f"    [RESOURCE_AT_ERROR] CPU: {cpu_p}% | RAM: {mem.percent}% ({mem.used//(1024**3)}GB/{mem.total//(1024**3)}GB)")
                            
                            if torch.cuda.is_available():
                                gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
                                gpu_resv = torch.cuda.memory_reserved() / (1024**3)
                                try:
                                    gpu_cached = torch.cuda.memory_cached() / (1024**3)
                                    logger.error(f"    [GPU_AT_ERROR] Allocated: {gpu_alloc:.2f}GB | Reserved: {gpu_resv:.2f}GB | Cached: {gpu_cached:.2f}GB")
                                except AttributeError:
                                    logger.error(f"    [GPU_AT_ERROR] Allocated: {gpu_alloc:.2f}GB | Reserved: {gpu_resv:.2f}GB")
                        except Exception as res_e:
                            logger.error(f"    [RESOURCE_LOG_FAILED] {type(res_e).__name__}: {str(res_e)}")
                        
                        results['failed'][user_id_inner].append(retriever_name)
                        completed += 1
                
                logger.info(f"  [{phase_name}_DONE] {retriever_type.capitalize()} phase completed for {user_id}")
            
            logger.info(f"[USER_DONE] ✓ User {user_idx}/{len(valid_users)} ({user_id}) completed all evaluations")
    
    finally:
        sparse_executor.shutdown(wait=True)
        dense_executor.shutdown(wait=True)
        logger.info("[EXECUTOR_SHUTDOWN] Both sparse and dense executors shut down")
    
    logger.info("\n" + "=" * 80)
    logger.info("FULL-SCALE EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total users processed: {len(valid_users)}")
    logger.info(f"Total evaluations: {completed}/{total_evaluations}")
    logger.info(f"Evaluation scale: {len(all_asins)} products")
    logger.info(f"Document corpus size: {len(metadata)} products")
    
    return results


def load_all_results(output_dir, user_ids):
    """Load all evaluation result files and aggregate metrics."""
    logger = logging.getLogger(__name__)
    
    # Result structure: {retriever: {mode: {metric: value}}}
    aggregated = defaultdict(lambda: defaultdict(dict))
    file_count = 0
    
    for user_id in user_ids:
        user_dir = os.path.join(output_dir, user_id)
        if not os.path.exists(user_dir):
            continue
        
        for result_file in glob.glob(os.path.join(user_dir, "retrieval_*.json")):
            try:
                filename = os.path.basename(result_file)
                parts = filename.replace("retrieval_", "").replace("_fullscale.json", "").split("_")
                
                if len(parts) >= 2:
                    mode = parts[-1]
                    retriever = "_".join(parts[:-1])
                    
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'metrics' in data:
                        metrics = data['metrics']
                        for metric_key, metric_val in metrics.items():
                            if isinstance(metric_val, (int, float)):
                                key = f"{metric_key}"
                                if key not in aggregated[retriever][mode]:
                                    aggregated[retriever][mode][key] = []
                                aggregated[retriever][mode][key].append(metric_val)
                    
                    file_count += 1
            except Exception as e:
                logger.error(f"Error loading {result_file}: {e}")
    
    return aggregated, file_count


def compute_comparison_metrics(aggregated):
    """Compute metrics comparison (clean/noisy with stage7 fallback)."""
    logger = logging.getLogger(__name__)
    
    comparison = {}
    has_both_modes = False
    
    for retriever, modes_data in aggregated.items():
        has_clean = 'clean' in modes_data
        has_noisy = 'noisy' in modes_data
        has_stage7 = 'stage7' in modes_data
        
        if not (has_clean or has_noisy or has_stage7):
            continue
        
        clean_metrics = {}
        noisy_metrics = {}
        stage7_metrics = {}
        
        if has_clean:
            for metric, values in modes_data['clean'].items():
                if values:
                    clean_metrics[metric] = np.mean(values)
        
        if has_noisy:
            for metric, values in modes_data['noisy'].items():
                if values:
                    noisy_metrics[metric] = np.mean(values)

        if has_stage7:
            for metric, values in modes_data['stage7'].items():
                if values:
                    stage7_metrics[metric] = np.mean(values)

        if not clean_metrics and stage7_metrics:
            clean_metrics = stage7_metrics.copy()
        
        if has_clean and has_noisy:
            has_both_modes = True
        
        comparison[retriever] = {
            'clean': clean_metrics,
            'noisy': noisy_metrics,
            'stage7': stage7_metrics,
            'degradation': {}
        }
        
        if has_both_modes:
            for metric in clean_metrics.keys():
                if metric in noisy_metrics:
                    clean_val = clean_metrics[metric]
                    noisy_val = noisy_metrics[metric]
                    if clean_val > 0:
                        degradation = (noisy_val - clean_val) / clean_val * 100
                    else:
                        degradation = 0
                    comparison[retriever]['degradation'][metric] = degradation
    
    return comparison, has_both_modes


def print_comparison_report(comparison, has_both_modes=False):
    """Print retriever metrics report (clean vs noisy if available, else clean only)."""
    logger = logging.getLogger(__name__)

    print("\n" + "="*100)
    if has_both_modes:
        print("🔥 CLEAN vs NOISY 性能对比分析 - 全量评估结果".center(100))
    else:
        print("📊 检索器性能评估结果（Clean Mode）".center(100))
    print("="*100)

    # Add debug information to verify calculations
    if has_both_modes:
        print("\n🔍 [DEBUG] 验证关键指标计算")
        print("─" * 80)
        for retriever in sorted(comparison.keys()):
            data = comparison[retriever]
            clean_ndcg = data['clean'].get('NDCG@10', 0)
            noisy_ndcg = data['noisy'].get('NDCG@10', 0) if data['noisy'] else 0

            if clean_ndcg > 0 and noisy_ndcg > 0:
                change = (noisy_ndcg - clean_ndcg) / clean_ndcg * 100
            else:
                change = 0

            print(f"{retriever}: Clean={clean_ndcg:.4f}, Noisy={noisy_ndcg:.4f}, 变化={change:+.2f}%")

            # Check for any model that shows improvement (should be none)
            if change > 0:
                print(f"  ⚠️  WARNING: {retriever} shows improvement with noise! This is unexpected.")
    
    normalized_comparison = {}
    for retriever, data in comparison.items():
        normalized_comparison[retriever] = {
            'clean': {k.lower(): v for k, v in data['clean'].items()},
            'noisy': {k.lower(): v for k, v in data['noisy'].items()} if data['noisy'] else {},
        }
    
    ndcg_ranking = []
    for retriever, data in normalized_comparison.items():
        ndcg_clean = data['clean'].get('ndcg@10', 0)
        ndcg_noisy = data['noisy'].get('ndcg@10', 0) if data['noisy'] else 0
        if ndcg_clean > 0 and ndcg_noisy > 0:
            ndcg_deg = (ndcg_noisy - ndcg_clean) / ndcg_clean * 100
        else:
            ndcg_deg = 0
        ndcg_ranking.append((retriever, ndcg_clean, ndcg_noisy, ndcg_deg))
    
    ndcg_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 检索器性能排序 (NDCG@10)")
    print("─" * 100)
    
    if has_both_modes:
        print(f"{'排名':<6} {'模型':<12} {'Clean':<12} {'Noisy':<12} {'变化':<12} {'评级':<12}")
    else:
        print(f"{'排名':<6} {'模型':<12} {'NDCG@10':<12} {'P@1':<12} {'P@10':<12} {'MAP@10':<12}")
    
    print("─" * 100)
    for idx, (retriever, clean, noisy, deg) in enumerate(ndcg_ranking, 1):
        if has_both_modes:
            if deg > 0:
                rating = "✅ 提升!" if deg > 1 else "✅ 稳定"
                change = f"↑ +{deg:.2f}%"
            else:
                if deg > -2:
                    rating = "✅ 优秀"
                elif deg > -5:
                    rating = "✓ 良好"
                else:
                    rating = "⚠️ 一般"
                change = f"↓ {deg:.2f}%"
            print(f"{idx:<6} {retriever:<12} {clean:<12.4f} {noisy:<12.4f} {change:<12} {rating:<12}")
        else:
            p1 = normalized_comparison[retriever]['clean'].get('p@1', 0)
            p10 = normalized_comparison[retriever]['clean'].get('p@10', 0)
            map10 = normalized_comparison[retriever]['clean'].get('map@10', 0)
            print(f"{idx:<6} {retriever:<12} {clean:<12.4f} {p1:<12.4f} {p10:<12.4f} {map10:<12.4f}")
    
    if has_both_modes:
        print("\n" + "="*100)
        print("📊 多k值性能排序对比")
        print("="*100)
        
        k_values_rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]
        metric_keys = ['ndcg', 'p', 'map', 'r']
        
        for metric_key in metric_keys:
            print(f"\n{metric_key.upper()}值排序 (Clean Mode)")
            print("─" * 100)
            
            for k in k_values_rank:
                metric = f'{metric_key}@{k}'
                ranking_list = []
                
                for retriever, data in normalized_comparison.items():
                    clean_val = data['clean'].get(metric, None)
                    if clean_val is None:
                        continue
                    noisy_val = data['noisy'].get(metric, 0) if data['noisy'] else 0
                    deg = (noisy_val - clean_val) / clean_val * 100 if clean_val > 0 else 0
                    ranking_list.append((retriever, clean_val, noisy_val, deg))
                
                ranking_list.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n{metric}:")
                for idx, (retriever, clean, noisy, deg) in enumerate(ranking_list, 1):
                    if deg > 0:
                        indicator = "↑"
                    else:
                        indicator = "↓"
                    print(f"  {idx}. {retriever:<12} Clean: {clean:.4f} | Noisy: {noisy:.4f} | 变化: {indicator} {deg:+.2f}%")
    
    print("\n" + "="*100)
    print("📈 全指标对比 (Clean Mode)")
    print("="*100)
    
    for retriever in sorted(normalized_comparison.keys()):
        data = normalized_comparison[retriever]
        print(f"\n{retriever.upper()}")
        print("─" * 80)
        
        k_values_to_show = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]
        metric_types = ['p', 'ndcg', 'map', 'r']
        
        for metric_type in metric_types:
            print(f"  {metric_type.upper()}:")
            for k in k_values_to_show:
                metric = f'{metric_type}@{k}'
                clean = data['clean'].get(metric, None)
                if clean is None:
                    continue
                noisy = data['noisy'].get(metric, 0) if data['noisy'] else 0
                
                if has_both_modes and noisy > 0:
                    deg = (noisy - clean) / clean * 100 if clean > 0 else 0
                    indicator = "↑" if deg > 0 else "↓"
                    print(f"    {metric:<10} Clean: {clean:.4f}  Noisy: {noisy:.4f}  变化: {indicator} {deg:+.2f}%")
                elif clean > 0:
                    print(f"    {metric:<10} {clean:.4f}")
    
    if has_both_modes:
        print("\n" + "="*100)
        print("🛡️ 噪声鲁棒性排名 (NDCG@10 降幅)")
        print("="*100)
        
        robustness_ranking = []
        for idx, (retriever, clean, noisy, deg) in enumerate(ndcg_ranking):
            robustness_ranking.append((retriever, deg))
        
        robustness_ranking.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'排名':<6} {'模型':<12} {'降幅':<12} {'评级':<20} {'特征':<30}")
        print("─" * 80)
        
        for idx, (retriever, deg) in enumerate(robustness_ranking, 1):
            if deg > 0:
                rating = "🟢 免疫"
                feature = "唯一提升的模型"
            elif deg > -2:
                rating = "✅ 极稳定"
                feature = "性能最稳定"
            elif deg > -5:
                rating = "✓ 稳定"
                feature = "容错性好"
            elif deg > -10:
                rating = "⚠️ 一般"
                feature = "中等敏感"
            else:
                rating = "❌ 极差"
                feature = "噪声敏感"
            
            print(f"{idx:<6} {retriever:<12} {deg:+.2f}%{' ':<6} {rating:<20} {feature:<30}")
    
    print("\n" + "="*100)
    print("✅ 评估完成".center(100))
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Full-scale Retrieval Evaluation')
    parser.add_argument('--mode', default='both', choices=['clean', 'noisy', 'stage7', 'stage7_clean', 'stage7_noisy', 'stage7_both', 'both', 'all'])
    parser.add_argument('--parallel', type=int, default=1, help='Parallel (user, retriever) pairs (default: 1 = serial across retrievers/users, but parallel clean/noisy)')
    parser.add_argument('--users', type=int, default=11, help='Number of users to evaluate')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    user_ids = find_users_with_queries()[:args.users]
    logger.info(f"Found {len(user_ids)} users")
    
    results = evaluate_batch_fullscale(
        user_ids,
        mode=args.mode,
        parallel_retrievers=args.parallel,
        logger=logger
    )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Load all results and print comparison report
    logger.info("\nLoading all results for comparison analysis...")
    aggregated, file_count = load_all_results(OUTPUT_DIR, user_ids)
    logger.info(f"Loaded {file_count} result files")
    
    if aggregated:
        comparison, has_both_modes = compute_comparison_metrics(aggregated)
        print_comparison_report(comparison, has_both_modes)
    else:
        logger.warning("No results found for comparison analysis")


if __name__ == '__main__':
    main()
