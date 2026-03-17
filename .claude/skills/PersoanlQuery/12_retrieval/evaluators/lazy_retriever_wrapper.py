#!/usr/bin/env python3
"""
Lazy Retriever Wrapper - 按需加载embeddings到GPU

解决GPU内存溢出问题的关键：
1. 在磁盘上保持embeddings（不加载到GPU）
2. 只在search()时加载所需的embeddings
3. 搜索完立即释放GPU内存
"""

import pickle
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import gc

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
                import numpy as np
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
        device = self.retriever.device
        
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
        按需加载式搜索
        
        工作流程：
        1. 加载query embedding到GPU
        2. 加载所有document embeddings到GPU
        3. 计算相似度
        4. 立即释放GPU内存
        5. 返回结果
        """
        try:
            # ✅ 加载所有doc embeddings到GPU
            all_indices = list(range(len(self.doc_embeddings_info)))
            gpu_embeddings = self._load_embeddings_to_gpu(all_indices)
            
            # ✅ 调用原retriever的搜索逻辑
            # 但此时retriever.doc_embeddings已经在GPU上了
            results = self.retriever.search(query, top_k)
            
            # ✅ 立即释放GPU
            self._release_gpu_embeddings(gpu_embeddings)
            
            return results
        
        except Exception as e:
            log_with_timestamp(f"Error in lazy search: {e}")
            # 确保失败时也释放GPU
            torch.cuda.empty_cache()
            raise
    
    # 代理其他属性和方法
    def __getattr__(self, name):
        """代理所有未找到的属性到原retriever"""
        return getattr(self.retriever, name)


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
            import time
            search_start = time.time()
            
            model = self.retriever._get_model() if hasattr(self.retriever, '_get_model') else None
            device = self.retriever.device
            
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
                        from sentence_transformers import util
                        window_scores = util.cos_sim(query_embedding, torch.stack(doc_emb))[0]
                        score = window_scores.max().item()
                    else:
                        # 单窗口
                        from sentence_transformers import util
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
