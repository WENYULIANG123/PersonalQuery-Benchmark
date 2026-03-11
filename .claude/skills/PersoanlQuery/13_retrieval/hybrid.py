#!/usr/bin/env python3
"""
Stage 13: Hybrid Retrieval

Contains hybrid retriever class:
- HybridRetriever: Hybrid retrieval using RRF (Reciprocal Rank Fusion)
"""

from collections import defaultdict
from typing import List, Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp


class HybridRetriever:
    """Hybrid retrieval using RRF (Reciprocal Rank Fusion)"""
    def __init__(self, retrievers: List, k: int = 60):
        """
        Args:
            retrievers: 检索器列表，如 [bm25, dense_retriever]
            k: RRF 参数，默认60
        """
        self.retrievers = retrievers
        self.k = k
        self.doc_ids = []  # 从第一个检索器获取

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """训练所有检索器"""
        log_with_timestamp("  Building Hybrid Retrieval index (RRF Fusion)...")
        for retriever in self.retrievers:
            # 只对 DenseRetriever 及其子类传递 all_metadata
            if hasattr(retriever, 'all_metadata'):
                retriever.fit(documents, all_metadata)
            else:
                retriever.fit(documents)

        # 从第一个检索器获取 doc_ids
        if hasattr(self.retrievers[0], 'doc_ids'):
            self.doc_ids = self.retrievers[0].doc_ids
        elif hasattr(self.retrievers[0], 'doc_asins'):
            self.doc_ids = self.retrievers[0].doc_asins

        log_with_timestamp(f"  Hybrid index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List:
        """使用 RRF 融合多个检索器的结果"""
        # 收集所有检索器的排名
        all_scores = defaultdict(float)

        for retriever in self.retrievers:
            results = retriever.search(query, top_k=100)  # 获取更多候选

            for rank, (doc_id, score) in enumerate(results, start=1):
                # RRF 公式: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                all_scores[doc_id] += rrf_score

        # 按分数降序排序
        results = sorted(all_scores.items(), key=lambda x: -x[1])
        return results[:top_k]
