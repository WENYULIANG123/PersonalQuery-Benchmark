#!/usr/bin/env python3
"""
优化版密集检索器 - 预加载所有向量到GPU一次

核心改进：
- 不再分批处理，预加载所有302k向量到GPU
- 每个query只需要编码query向量，然后矩阵乘法
- 期望性能提升: 450倍 (450s → 1s/query)
"""

import torch
import numpy as np
from typing import List, Tuple
from datetime import datetime
import time

log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class PreloadedDenseRetriever:
    """
    预加载版本：一次性加载所有向量到GPU，然后快速查询
    """
    
    def __init__(self, base_retriever, embeddings_path: str = None):
        self.base_retriever = base_retriever
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.doc_ids = base_retriever.doc_ids
        self.all_metadata = base_retriever.all_metadata
        
        # 预加载所有向量到GPU
        self._preload_embeddings(embeddings_path)
    
    def _preload_embeddings(self, embeddings_path: str = None):
        """一次性加载所有向量到GPU"""
        start_time = time.time()
        
        if embeddings_path is None:
            embeddings_path = getattr(self.base_retriever, '_embeddings_path', None)
        
        if not embeddings_path:
            raise ValueError("Cannot find embeddings path")
        
        log_with_timestamp(f"[PRELOAD_START] Loading all embeddings to {self.device.type.upper()}...")
        log_with_timestamp(f"  Path: {embeddings_path}")
        
        # 从文件加载
        doc_embeddings = np.load(embeddings_path, mmap_mode='r')
        log_with_timestamp(f"  Loaded shape: {doc_embeddings.shape}")
        
        # 转换为GPU tensor
        self.embeddings = torch.from_numpy(np.array(doc_embeddings)).float().to(self.device)
        
        load_time = time.time() - start_time
        log_with_timestamp(f"[PRELOAD_DONE] All {len(self.embeddings):,} embeddings on {self.device.type.upper()}")
        log_with_timestamp(f"  Memory: {self.embeddings.element_size() * self.embeddings.nelement() / 1e9:.2f} GB")
        log_with_timestamp(f"  Time: {load_time:.2f}s")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        超快速搜索：直接矩阵乘法，无需batch处理
        """
        search_start = time.time()
        
        # 编码query
        query_embedding = self.base_retriever._get_model().encode(
            [query], convert_to_tensor=True
        )[0].to(self.device)
        
        # 计算所有相似度（一次性矩阵乘法）
        # query_embedding: (384,)
        # self.embeddings: (302380, 384)
        # result: (302380,)
        similarities = torch.mm(
            query_embedding.unsqueeze(0),
            self.embeddings.T
        )[0]
        
        # 获取top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        results = [
            (self.doc_ids[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
        
        search_time = time.time() - search_start
        log_with_timestamp(f"  Query: '{query[:40]}...' → {top_k} results in {search_time:.3f}s")
        
        return results


if __name__ == '__main__':
    print("Optimized Dense Retriever loaded successfully")
