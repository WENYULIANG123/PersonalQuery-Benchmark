#!/usr/bin/env python3
"""
Lazy Cache Manager - 分离embeddings和retriever对象，实现真正的按需加载

问题：当前cache存储整个retriever对象（包括所有embeddings），
导致pickle.load()时一次性加载302k个embeddings到内存。

解决方案：
1. 只在cache中保存retriever的配置和模型，不保存embeddings
2. Embeddings单独存储为mmap（内存映射）或硬盘文件
3. 加载时只加载配置，embeddings需要时再动态加载
"""

import pickle
import numpy as np
import torch
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

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
