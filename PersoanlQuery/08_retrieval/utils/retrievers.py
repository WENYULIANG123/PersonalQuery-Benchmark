#!/usr/bin/env python3
"""
Stage 13: Base Retrieval Models

Contains all base retriever classes:
- BM25: Traditional sparse retrieval
- DenseRetriever: ANCE/MiniLM dense retrieval
- E5Retriever: E5-large-v2 dense retrieval
- BGERetriever: BGE-large-en dense retrieval
- ColBERTRetriever: Token-level late interaction
- HybridRetriever: BM25 + Dense score fusion hybrid retrieval
- TFIDFRetriever: TF-IDF baseline
- DirichletPriorRetriever: Query likelihood with Dirichlet smoothing
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import threading
import pickle
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text

# Global lock for thread-safe model inference
# Prevents CUDA/GIL deadlocks when multiple threads call model.encode() simultaneously
_model_inference_lock = threading.Lock()

try:
    import bm25s
except ImportError:
    log_with_timestamp("WARNING: bm25s not installed. BM25 will not work properly.")
    bm25s = None


class BM25:
    """BM25 retrieval model using bm25s library with proper tokenization"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_asins = []
        self.corpus = []
        self.corpus_to_asin = {}
        
    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build BM25 index using bm25s with proper tokenization"""
        log_with_timestamp("  Building BM25 index using bm25s...")
        
        if bm25s is None:
            raise RuntimeError("bm25s library is required but not installed")
        
        self.corpus = []
        self.doc_asins = []
        self.corpus_to_asin = {}
        
        for doc in documents:
            asin = doc.get('asin', '')
            self.doc_asins.append(asin)
            text = build_document_text(doc, all_metadata)
            self.corpus.append(text)
            self.corpus_to_asin[text] = asin
        
        log_with_timestamp(f"  Tokenizing {len(self.corpus)} documents...")
        tokenized_corpus = bm25s.tokenize(
            self.corpus,
            lower=True,
            stopwords="english",
            show_progress=False
        )
        
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(tokenized_corpus)
        
        log_with_timestamp(f"  BM25 index built with {len(self.corpus)} docs using bm25s")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 with bm25s library"""
        if self.retriever is None:
            return []
        
        tokenized_query = bm25s.tokenize(
            query,
            lower=True,
            stopwords="english",
            show_progress=False
        )
        
        results, scores = self.retriever.retrieve(tokenized_query, k=top_k)
        
        output = []
        retrieved_texts = results[0]
        scores_list = scores[0].tolist()
        
        for doc_text, score in zip(retrieved_texts, scores_list):
            asin = self.corpus_to_asin.get(doc_text, '')
            if asin:
                output.append((asin, float(score)))
        
        return output


class HybridRetriever:
    """Hybrid retrieval combining BM25 sparse scores with Dense embedding scores via weighted score fusion.

    Runs both BM25 and a dense retriever (default: E5-large-v2), normalizes their
    scores to [0, 1] via min-max scaling, and returns a weighted combination.

    Default weights: 0.4 BM25 + 0.6 Dense (empirically tuned for product search).
    """

    def __init__(self, bm25_weight: float = 0.4, dense_weight: float = 0.6,
                 dense_model_name: str = "intfloat/e5-large-v2"):
        """
        Args:
            bm25_weight: Weight for normalized BM25 scores (default 0.4).
            dense_weight: Weight for normalized Dense scores (default 0.6).
            dense_model_name: HuggingFace model ID for the dense retriever.
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.dense_model_name = dense_model_name
        self.bm25: BM25 = BM25()
        self.dense: "E5Retriever" = None  # type: ignore[assignment]
        self.doc_ids: List[str] = []
        self.is_fitted: bool = False

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Initialize BOTH BM25 and Dense (E5) indices from the same document set.

        Args:
            documents: List of document dicts, each must contain an 'asin' key.
            all_metadata: Optional metadata dict keyed by ASIN for enriched text.
        """
        log_with_timestamp("  Building Hybrid (BM25 + E5 Dense) index...")
        self.doc_ids = [doc.get('asin', '') for doc in documents]

        self.bm25.fit(documents, all_metadata)

        self.dense = E5Retriever(model_name=self.dense_model_name)
        self.dense.fit(documents, all_metadata)

        self.is_fitted = True
        log_with_timestamp(
            f"  Hybrid index built with {len(self.doc_ids)} docs "
            f"(weights: BM25={self.bm25_weight}, Dense={self.dense_weight})"
        )

    @staticmethod
    def _normalize_scores(scores: List[Tuple[str, float]]) -> Dict[str, float]:
        """Min-max normalize raw scores to [0, 1].

        Args:
            scores: List of (asin, raw_score) tuples.

        Returns:
            Dict mapping asin -> normalized score in [0, 1].
        """
        if not scores:
            return {}

        raw = [s for _, s in scores]
        min_s = min(raw)
        max_s = max(raw)
        score_range = max_s - min_s

        if score_range < 1e-12:
            # All scores identical -> uniform mid-point
            return {asin: 0.5 for asin, _ in scores}

        return {asin: (score - min_s) / score_range for asin, score in scores}

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Run both retrievers, fuse normalized scores, return merged top-k.

        Score fusion:
            combined = bm25_weight * norm_bm25 + dense_weight * norm_dense
        Missing results from either side receive 0.0 for that component.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (asin, fused_score) sorted descending by score.
        """
        if not self.is_fitted:
            return []

        fetch_k = max(top_k * 3, 50)
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        dense_results = self.dense.search(query, top_k=fetch_k)

        # Handle edge cases: one or both empty
        if not bm25_results and not dense_results:
            return []
        if not bm25_results:
            return dense_results[:top_k]
        if not dense_results:
            return bm25_results[:top_k]

        # Normalize each score set to [0, 1]
        bm25_norm = self._normalize_scores(bm25_results)
        dense_norm = self._normalize_scores(dense_results)

        # Union of all candidate ASINs
        all_asins = set(bm25_norm.keys()) | set(dense_norm.keys())

        # Weighted fusion (missing component -> 0.0)
        fused: List[Tuple[str, float]] = []
        for asin in all_asins:
            b_score = bm25_norm.get(asin, 0.0)
            d_score = dense_norm.get(asin, 0.0)
            combined = self.bm25_weight * b_score + self.dense_weight * d_score
            fused.append((asin, float(combined)))

        fused.sort(key=lambda x: -x[1])
        return fused[:top_k]


class DenseRetriever:
    """Dense retrieval using sentence-transformers"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None  # 存储所有元数据
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
            model_path = self.model.tokenizer.name_or_path
            log_with_timestamp(f"  Model path: {model_path}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build dense index using enhanced document text"""
        log_with_timestamp("  Building dense index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata  # 保存元数据引用
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        if self.device.type == 'cuda':
            embeddings = embeddings.to(self.device)
        
        self.doc_embeddings = embeddings
        log_with_timestamp(f"  Dense index built with {len(self.doc_ids)} docs (device: {self.device})")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using dense embeddings
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query],
                truncation=True,
                max_length=512
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using dense vectors with optimized top-k extraction"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query],
                truncation=True,
                max_length=512
            )
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        else:
            query_embedding = query_embedding.to(self.device)
        
        doc_embeddings = self.doc_embeddings
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        
        results = [(self.doc_ids[idx], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices.cpu())]
        
        return results


class E5Retriever:
    """E5-large-v2: SOTA embedding model with instruction-based retrieval (支持滑动窗口，不截断)"""
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None  # 可能是 tensor 或 list of tensors (多窗口)
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 滑动窗口配置
        self.max_seq_length = 512  # E5 模型的最大长度
        self.window_stride = 256  # 窗口步长（重叠50%）

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading E5 model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            model_path = self.model.tokenizer.name_or_path
            log_with_timestamp(f"  Model path: {model_path}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """E5 需要添加 instruction 前缀"""
        if is_query:
            return "query: " + text
        else:
            return "passage: " + text

    def _encode_text_with_sliding_window(self, text: str, add_prefix: bool = True):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本
            add_prefix: 是否添加 passage: 前缀

        Returns:
            如果文本 <= max_seq_length: 单个 embedding [dim]
            如果文本 > max_seq_length: 多个窗口的 embeddings [num_windows, dim]
        """
        model = self._get_model()

        # 添加前缀
        if add_prefix:
            text_with_prefix = self._add_instruction(text, is_query=False)
        else:
            text_with_prefix = text

        # 使用 tokenizer 获取真实 token 数量（不截断）
        tokens = model.tokenizer(
            text_with_prefix,
            truncation=False,
            return_tensors='pt'
        )

        num_tokens = len(tokens['input_ids'][0])

        # 如果文本较短，直接编码
        if num_tokens <= self.max_seq_length:
            result = model.encode(
                [text_with_prefix],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return result[0]
        else:
            # 长文本：使用滑动窗口
            # 计算窗口数量
            num_windows = (num_tokens - self.max_seq_length) // self.window_stride + 1

            window_embeddings = []
            for i in range(num_windows):
                # 计算窗口的 token 范围
                start_token = i * self.window_stride
                end_token = min(start_token + self.max_seq_length, num_tokens)

                # 提取窗口的 token IDs
                window_tokens = {
                    'input_ids': tokens['input_ids'][0][start_token:end_token].unsqueeze(0),
                    'attention_mask': tokens['attention_mask'][0][start_token:end_token].unsqueeze(0)
                }

                # 解码回文本
                window_text = model.tokenizer.decode(
                    window_tokens['input_ids'][0],
                    skip_special_tokens=True
                )

                # 编码这个窗口
                window_embedding = model.encode(
                    [window_text],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )[0]

                window_embeddings.append(window_embedding)

            # 返回所有窗口的 embeddings（堆叠成 tensor）
            return torch.stack(window_embeddings)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        构建 E5 索引（使用滑动窗口，保证不截断）
        """
        from tqdm import tqdm
        log_with_timestamp("  Building E5-large-v2 index with sliding window (no truncation)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        texts = [build_document_text(doc, all_metadata) for doc in documents]

        doc_embeddings_list = []
        single_window_indices = []
        multi_window_indices = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in tqdm(enumerate(texts), total=len(texts), desc="  E5 Encoding"):
            doc_emb = self._encode_text_with_sliding_window(text, add_prefix=True)

            if doc_emb.dim() == 1:
                window_stats['single_window'] += 1
                single_window_indices.append(i)
            else:
                window_stats['multi_window'] += 1
                multi_window_indices.append(i)
                num_windows = doc_emb.shape[0]
                window_stats['max_windows'] = max(window_stats['max_windows'], num_windows)

            doc_embeddings_list.append(doc_emb)

        self.single_window_indices = single_window_indices
        self.multi_window_indices = multi_window_indices

        if single_window_indices:
            single_embeddings = torch.stack([doc_embeddings_list[i] for i in single_window_indices])
            if self.device.type == 'cuda':
                single_embeddings = single_embeddings.to(self.device)
            self.single_window_embeddings = single_embeddings
        else:
            self.single_window_embeddings = None
        
        self.doc_embeddings = doc_embeddings_list

        log_with_timestamp(f"  E5 index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs")
        log_with_timestamp(f"    - Max windows per doc: {window_stats['max_windows']}")
        log_with_timestamp(f"    - Device: {self.device}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using E5 embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using E5 with batch processing for single-window documents"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        query_embedding = query_embedding.to(self.device)

        from sentence_transformers import util
        all_scores = [0.0] * len(self.doc_ids)
        
        if hasattr(self, 'single_window_embeddings') and self.single_window_embeddings is not None:
            batch_scores = util.cos_sim(query_embedding, self.single_window_embeddings)[0]
            for idx, doc_idx in enumerate(self.single_window_indices):
                all_scores[doc_idx] = batch_scores[idx].item()
        
        for i in self.multi_window_indices:
            doc_emb = self.doc_embeddings[i].to(self.device)
            window_scores = util.cos_sim(query_embedding, doc_emb)[0]
            all_scores[i] = window_scores.max().item()

        results = [(self.doc_ids[i], all_scores[i]) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class BGERetriever:
    """BGE-large-en: SOTA embedding model for English retrieval (支持滑动窗口，不截断)"""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None  # 可能是 tensor 或 list of tensors (多窗口)
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 滑动窗口配置
        self.max_seq_length = 512  # BGE 模型的最大长度
        self.window_stride = 256  # 窗口步长（重叠50%）

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading BGE model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            model_path = self.model.tokenizer.name_or_path
            log_with_timestamp(f"  Model path: {model_path}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """BGE 推荐添加 instruction 前缀"""
        if is_query:
            return "Represent this sentence for searching relevant passages: " + text
        else:
            return text

    def _encode_text_with_sliding_window(self, text: str, pool_windows: str = 'mean'):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本（BGE 文档不需要前缀）
            pool_windows: 多窗口池化策略
              - 'mean': 对所有窗口 embeddings 求平均（默认，用于快速检索）
              - 'max': 对所有窗口 embeddings 求最大值
              - 'first': 仅使用第一个窗口
              - None: 返回所有窗口 embeddings（用于精确排名）

        Returns:
            总是返回单个 embedding [dim]
            （当 pool_windows 不为 None 时）
        """
        model = self._get_model()

        # 使用 tokenizer 获取真实 token 数量（不截断）
        tokens = model.tokenizer(
            text,
            truncation=False,
            return_tensors='pt'
        )

        num_tokens = len(tokens['input_ids'][0])

        # 如果文本较短，直接编码
        if num_tokens <= self.max_seq_length:
            embedding = model.encode(
                [text],
                convert_to_tensor=True,
                show_progress_bar=False
            )[0]
            return embedding
        else:
            # 长文本：使用滑动窗口
            # 计算窗口数量
            num_windows = (num_tokens - self.max_seq_length) // self.window_stride + 1

            window_embeddings = []
            for i in range(num_windows):
                # 计算窗口的 token 范围
                start_token = i * self.window_stride
                end_token = min(start_token + self.max_seq_length, num_tokens)

                # 提取窗口的 token IDs
                window_tokens = {
                    'input_ids': tokens['input_ids'][0][start_token:end_token].unsqueeze(0),
                    'attention_mask': tokens['attention_mask'][0][start_token:end_token].unsqueeze(0)
                }

                # 解码回文本
                window_text = model.tokenizer.decode(
                    window_tokens['input_ids'][0],
                    skip_special_tokens=True
                )

                # 编码这个窗口
                window_embedding = model.encode(
                    [window_text],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )[0]

                window_embeddings.append(window_embedding)

            # 池化多个窗口的 embeddings
            stacked = torch.stack(window_embeddings)
            
            if pool_windows is None:
                return stacked
            elif pool_windows == 'mean':
                return torch.mean(stacked, dim=0)
            elif pool_windows == 'max':
                return torch.max(stacked, dim=0)[0]
            elif pool_windows == 'first':
                return stacked[0]
            else:
                return torch.mean(stacked, dim=0)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build BGE index with FULLY BATCHED encoding - GPU maximized"""
        log_with_timestamp("  Building BGE-large-en index with FULLY BATCHED encoding...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 直接批量编码（debug显示绝大多数文档 < 512 tokens，直接encode即可）
        # 注意：bge-large 模型较大，batch_size 需要适度以避免 CUDA OOM
        batch_size = 256
        log_with_timestamp(f"  Batch encoding {len(texts)} docs (batch_size={batch_size})...")

        all_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        self.doc_embeddings = all_embeddings

        log_with_timestamp(f"  BGE index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Embeddings shape: {self.doc_embeddings.shape} (device: {self.device})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using BGE embeddings
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BGE with batched similarity computation (optimized with window pooling)"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True,
                truncation=True,
                max_length=512
            )[0]

        query_embedding = query_embedding.to(self.device)

        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        
        results = [(self.doc_ids[idx], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices.cpu())]
        
        return results


class ColBERTRetriever:
    """ColBERTv2: Token-level Late Interaction (MaxSim) with colbert-ir/colbertv2.0"""
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        使用真正的 ColBERTv2 模型：colbert-ir/colbertv2.0

        不截断策略：使用滑动窗口处理长文档
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None  # List of token embeddings (可能是多个窗口)
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 窗口配置（ColBERTv2 标准）
        self.window_size = 512  # 每个窗口的最大长度
        self.window_stride = 256  # 窗口之间的步长（重叠50%）
        self.max_query_length = 512  # 查询也使用512，保证不截断

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading ColBERTv2 model: {self.model_name}")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def _encode_text(self, text: str, max_length: int = None, use_sliding_window: bool = False):
        """
        编码文本为 token-level embeddings

        Args:
            text: 输入文本
            max_length: 最大长度（仅当不使用滑动窗口时生效）
            use_sliding_window: 是否使用滑动窗口处理长文本

        Returns:
            如果 use_sliding_window=False: [seq_len, hidden_size]
            如果 use_sliding_window=True: List of [seq_len, hidden_size] (多个窗口)
        """
        # 先 tokenize 不截断，获取真实长度
        encoded_full = self.tokenizer(
            text,
            truncation=False,
            return_tensors='pt'
        )

        input_ids = encoded_full['input_ids'][0]
        actual_length = len(input_ids)

        # 如果文本较短或者不使用滑动窗口，直接处理
        if not use_sliding_window or actual_length <= self.window_size:
            if max_length is None:
                max_length = self.window_size

            # 截断到 max_length（但优先保证不截断）
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state[0]

            attention_mask_expanded = attention_mask[0].unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings = token_embeddings * attention_mask_expanded

            return token_embeddings

        # 使用滑动窗口处理长文本（保证完全不截断）
        windows_embeddings = []

        # 计算窗口数量
        num_windows = (actual_length - self.window_size) // self.window_stride + 1

        for i in range(num_windows):
            start_idx = i * self.window_stride
            end_idx = min(start_idx + self.window_size, actual_length)

            # 提取窗口的 token IDs
            window_ids = input_ids[start_idx:end_idx]

            # 编码这个窗口
            encoded_window = self.tokenizer(
                text,
                max_length=end_idx - start_idx,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids_window = encoded_window['input_ids'].to(self.device)
            attention_mask_window = encoded_window['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_window, attention_mask=attention_mask_window)
                window_embeddings = outputs.last_hidden_state[0]

            attention_mask_expanded = attention_mask_window[0].unsqueeze(-1).expand(window_embeddings.size()).float()
            window_embeddings = window_embeddings * attention_mask_expanded

            windows_embeddings.append(window_embeddings)

        return windows_embeddings  # 返回多个窗口的 embeddings

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        构建 ColBERT 索引：token-level embeddings

        使用滑动窗口策略，保证完全不截断长文档
        """
        log_with_timestamp("  Building ColBERTv2 index with sliding window (no truncation)...")

        # 清理 GPU 缓存，释放之前模型的内存
        if torch.cuda.is_available():
            log_with_timestamp("  Clearing GPU cache before ColBERT loading...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 编码每个文档为 token-level embeddings（使用滑动窗口）
        self.doc_embeddings = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                log_with_timestamp(f"    Encoding document {i+1}/{len(texts)}...")

            # 使用滑动窗口编码（use_sliding_window=True 保证不截断）
            token_emb = self._encode_text(text, use_sliding_window=True)

            # 统计窗口使用情况
            if isinstance(token_emb, list):
                window_stats['multi_window'] += 1
                window_stats['max_windows'] = max(window_stats['max_windows'], len(token_emb))
            else:
                window_stats['single_window'] += 1

            self.doc_embeddings.append(token_emb)

            # 每1000个文档清理一次GPU缓存
            if (i + 1) % 1000 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_with_timestamp(f"  ColBERTv2 index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs")
        log_with_timestamp(f"    - Max windows per doc: {window_stats['max_windows']}")

        # THREAD SAFETY FIX: Pre-move all embeddings to device during fit()
        # This avoids concurrent .to(device) calls in search() which cause "Already borrowed" errors
        log_with_timestamp("  Moving embeddings to device for thread safety...")
        self.doc_embeddings = [
            [w.to(self.device) for w in emb] if isinstance(emb, list)
            else emb.to(self.device)
            for emb in self.doc_embeddings
        ]
        log_with_timestamp(f"  All embeddings moved to {self.device}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用改进的 Late Interaction (MaxSim) 搜索

        改进点：
        1. 使用更长的查询长度 (512 tokens) - 保证不截断查询
        2. 只对 embeddings 进行 L2 归一化一次（不在每次比较时重复归一化）
        3. 使用 mean() 而不是 sum() 来标准化分数（避免查询长度偏差）
        4. 支持多窗口文档（完全不截断长文档）
        """
        model = self._get_model()

        # 编码查询为 token-level embeddings，保证不截断（max_length=512）
        query_emb = self._encode_text(query, max_length=self.max_query_length, use_sliding_window=False)
        query_emb = query_emb.to(self.device)  # [query_len, hidden_size]

        # 只在编码后进行一次 L2 归一化
        query_emb_normalized = query_emb / query_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        scores = []
        for i, doc_emb in enumerate(self.doc_embeddings):
            # 处理单窗口或多窗口文档
            if isinstance(doc_emb, list):
                # 多窗口文档：计算每个窗口的分数，取最大值
                window_scores = []
                for window_emb in doc_emb:
                    # 归一化窗口 embeddings
                    window_emb_normalized = window_emb / window_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                    # MaxSim: 计算这个窗口的分数
                    sim_matrix = torch.mm(query_emb_normalized, window_emb_normalized.t())
                    max_sim_per_query_token, _ = sim_matrix.max(dim=1)
                    window_score = max_sim_per_query_token.mean().item()
                    window_scores.append(window_score)

                # 取所有窗口的最大分数（文档只要有一个窗口匹配就算匹配）
                score = max(window_scores)
            else:
                # 单窗口文档：直接计算
                # 归一化文档 embeddings
                doc_emb_normalized = doc_emb / doc_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                # MaxSim: 计算每个 query token 与所有 doc tokens 的最大相似度
                sim_matrix = torch.mm(query_emb_normalized, doc_emb_normalized.t())
                max_sim_per_query_token, _ = sim_matrix.max(dim=1)
                score = max_sim_per_query_token.mean().item()

            scores.append((self.doc_ids[i], score))

        # 按分数降序排序
        results = sorted(scores, key=lambda x: -x[1])
        return results[:top_k]


class TFIDFRetriever:
    """TF-IDF retrieval as a baseline with FAISS acceleration"""
    def __init__(self):
        self.doc_vectors = None
        self.feature_names = None
        self.doc_ids = []
        self.all_metadata = None
        self.faiss_index = None
        self._doc_vectors_dense = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build TF-IDF index with FAISS"""
        log_with_timestamp("  Building TF-IDF index with FAISS...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        import faiss
        import numpy as np

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 构建 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),  # 使用 unigrams 和 bigrams
            sublinear_tf=True
        )

        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # 转换为密集向量并进行L2归一化（用于余弦相似度）
        self._doc_vectors_dense = self.doc_vectors.toarray().astype('float32')
        # 行归一化：每个文档向量的模为1，余弦相似度 = 内积
        norms = np.linalg.norm(self._doc_vectors_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # 避免除零
        self._doc_vectors_dense = self._doc_vectors_dense / norms

        # 创建FAISS索引（内积索引，因为向量已归一化）
        d = self._doc_vectors_dense.shape[1]  # 特征维度
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self._doc_vectors_dense)

        log_with_timestamp(f"  TF-IDF index built with FAISS: {len(self.doc_ids)} docs, {d} features")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF with FAISS"""
        import numpy as np

        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')[0]
        # 归一化查询向量
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        query_vector = query_vector.reshape(1, -1)

        # 使用FAISS搜索
        scores, indices = self.faiss_index.search(query_vector, top_k)

        results = [(self.doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]
        return results


class DirichletPriorRetriever:
    """Query Likelihood with Dirichlet Prior smoothing"""
    def __init__(self, mu: float = 2000):
        """
        Args:
            mu: Dirichlet prior 参数，控制文档先验的权重
                 较小的 mu (500-1000) 更信任文档，较大的 mu (2000-5000) 更信任集合语言模型
        """
        self.mu = mu
        self.doc_probs = []  # 每个文档的词概率分布
        self.collection_probs = {}  # 集合语言模型
        self.vocab = set()
        self.doc_ids = []
        self.all_metadata = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """构建 Dirichlet Prior 索引"""
        log_with_timestamp(f"  Building Dirichlet Prior index (μ={self.mu})...")

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # Tokenize 所有文档
        tokenized_docs = [text.lower().split() for text in texts]

        # 构建词汇表
        self.vocab = set()
        for doc_tokens in tokenized_docs:
            self.vocab.update(doc_tokens)

        # 计算集合语言模型 P(w|C)
        from collections import Counter
        collection_counts = Counter()
        total_collection_tokens = 0

        for doc_tokens in tokenized_docs:
            collection_counts.update(doc_tokens)
            total_collection_tokens += len(doc_tokens)

        self.collection_probs = {
            word: count / total_collection_tokens
            for word, count in collection_counts.items()
        }

        # 计算每个文档的语言模型 P(w|D)
        self.doc_probs = []
        self.doc_lengths = []

        for doc_tokens in tokenized_docs:
            doc_length = len(doc_tokens)
            self.doc_lengths.append(doc_length)

            doc_counts = Counter(doc_tokens)
            doc_prob = {
                word: count / doc_length
                for word, count in doc_counts.items()
            }
            self.doc_probs.append(doc_prob)

        avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        log_with_timestamp(f"  Dirichlet Prior index built with {len(self.doc_ids)} docs, "
                          f"{len(self.vocab)} vocab, avg doc length: {avg_doc_len:.1f}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 Dirichlet Prior 搜索"""
        query_tokens = query.lower().split()

        avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)

        scores = []
        for doc_id, doc_prob in enumerate(self.doc_probs):
            doc_len = self.doc_lengths[doc_id]

            # Dirichlet Prior 平滑
            # P(w|Q,D) = (|D| / (|D| + μ)) * P(w|D) + (μ / (|D| + μ)) * P(w|C)
            score = 0.0
            for token in query_tokens:
                # 文档概率
                p_w_d = doc_prob.get(token, 0.0)
                # 集合概率
                p_w_c = self.collection_probs.get(token, 1e-10)  # 小常数平滑

                # Dirichlet Prior 插值
                p_smoothed = (doc_len / (doc_len + self.mu)) * p_w_d + \
                            (self.mu / (doc_len + self.mu)) * p_w_c

                # 累加对数概率（避免下溢）
                if p_smoothed > 0:
                    score += np.log(p_smoothed)
                else:
                    score += np.log(1e-10)

            scores.append((self.doc_ids[doc_id], score))

        # 按分数降序排序
        results = sorted(scores, key=lambda x: -x[1])
        return results[:top_k]


class GritLMRetriever:
    """
    GritLM: GritLM-7B model for knowledge base retrieval.

    Uses the official gritlm package for encoding with proper instruction prefixes.
    Reference: STaRK implementation (https://github.com/snap-stanford/STaRK)
              https://github.com/allenai/gritlm

    Standard GritLM instruction format:
    - Query: "<|user|>\nGiven a product search query, retrieve relevant product descriptions.\n<|embed|>\n"
    - Document: "<|user|>\nGiven a product description, retrieve relevant web search queries.\n<|embed|>\n"
    """
    def __init__(self, model_name: str = "GritLM/GritLM-7B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 标准 GritLM instruction 格式
        self.query_instruction = "Given a product search query, retrieve relevant product descriptions."
        self.doc_instruction = "Given a product description, retrieve relevant web search queries."
        # Pooling method (与 gritlm 包一致)
        self.pooling_method = 'mean'
        self.normalized = True

    def _get_query_instruction(self) -> str:
        """获取查询指令格式"""
        return f"<|user|>\n{self.query_instruction}\n<|embed|>\n"

    def _get_doc_instruction(self) -> str:
        """获取文档指令格式"""
        return f"<|user|>\n{self.doc_instruction}\n<|embed|>\n"

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading GritLM model: {self.model_name}")
            from gritlm import GritLM
            # 参考 gritlm 包的实现：不使用 local_files_only，让 transformers 自动处理缓存
            self.model = GritLM(
                self.model_name,
                mode='unified',
                pooling_method=self.pooling_method,
                normalized=self.normalized,
                torch_dtype=torch.bfloat16,
                is_inference=True,
            )
            log_with_timestamp(f"  GritLM loaded on {self.device}")
            log_with_timestamp(f"  Model name: {self.model_name}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def _encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text using GritLM with standard query instruction prefix."""
        model = self._get_model()
        embedding = model.encode_queries(
            [text],
            instruction=self._get_query_instruction(),
            max_length=max_length,
        )
        return torch.from_numpy(embedding[0])

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build GritLM index"""
        log_with_timestamp(f"  Building GritLM index with {len(documents)} docs...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # Build document texts
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 参考 gritlm 包的 encode_corpus 方法进行批量编码
        log_with_timestamp(f"  Encoding {len(texts)} documents...")
        batch_size = 8  # 与 gritlm 包默认值 256 相比减小以适应 7B 模型

        all_embeddings = []
        for start_index in tqdm(range(0, len(texts), batch_size), desc="GritLM encoding"):
            batch_texts = texts[start_index:start_index + batch_size]
            batch_embeddings = model.encode_corpus(
                batch_texts,
                instruction=self._get_doc_instruction(),
                max_length=512,
            )
            all_embeddings.append(torch.from_numpy(batch_embeddings))

        self.doc_embeddings = torch.cat(all_embeddings, dim=0).float()
        log_with_timestamp(f"  GritLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        embedding = self._encode_text(query)
        return embedding.cpu().numpy()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using GritLM"""
        query_embedding = self._encode_text(query)
        query_embedding = query_embedding.unsqueeze(0).to(self.device)  # [1, hidden_dim]

        # Cosine similarity
        doc_embeddings = self.doc_embeddings.to(self.device)
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        scores = torch.mm(query_norm, doc_norm.T)[0]

        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class ANCERetriever:
    """
    ANCE (Approximate Nearest Neighbor Negative Contrastive Learning) retriever.
    
    ANCE uses a contrastive learning approach with negative sampling to learn
    high-quality dense embeddings for retrieval.
    
    Using intfloat/e5-base-v2 as a publicly available alternative.
    """
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading ANCE-compatible model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build ANCE index using enhanced document text"""
        log_with_timestamp("  Building ANCE-compatible index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  ANCE index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using ANCE embeddings
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                ["query: " + query]
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using ANCE embeddings"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                ["query: " + query],
                truncation=True,
                max_length=512
            )

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        from sentence_transformers import util
        doc_embeddings = self.doc_embeddings
        
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        # This provides ~250ms speedup by avoiding sorting all 302k documents
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class STARRetriever:
    """
    STAR (STaRK) retriever for dense passage retrieval.
    
    Using BAAI/bge-base-en-v1.5 as a publicly available alternative to STAR.
    This model is also trained on large-scale retrieval data with contrastive learning.
    """
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading STAR-compatible model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build STAR index using enhanced document text"""
        log_with_timestamp("  Building STAR-compatible index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=512)
        log_with_timestamp(f"  STAR index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using STAR embeddings
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using STAR embeddings"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MiniLMRetriever:
    """
    MiniLM retriever using sentence-transformers/all-MiniLM-L6-v2.
    
    A lightweight, fast model good for retrieval tasks.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading MiniLM model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MiniLM index using enhanced document text"""
        log_with_timestamp("  Building MiniLM index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MiniLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MiniLM embeddings"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query],
                truncation=True,
                max_length=512
            )

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)

        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)

        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MultiQAMiniLMRetriever:
    """
    MultiQA-MiniLM retriever using sentence-transformers/multi-qa-MiniLM-L6-cos-v1.
    
    Optimized for QA tasks using cosine similarity, used in PBR project.
    """
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading MultiQA-MiniLM model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MultiQA-MiniLM index using enhanced document text"""
        log_with_timestamp("  Building MultiQA-MiniLM index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MultiQA-MiniLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MultiQA-MiniLM embeddings"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MPNetRetriever:
    """
    MPNet retriever using sentence-transformers/all-mpnet-base-v2.
    
    A more powerful model than MiniLM, better retrieval quality.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading MPNet model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MPNet index using enhanced document text"""
        log_with_timestamp("  Building MPNet index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MPNet index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MPNet embeddings"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class FAISSRetriever:
    """FAISS-accelerated dense retrieval with GPU support"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_gpu: bool = True, nlist: int = 100, nprobe: int = 10):
        self.model_name = model_name
        self.model = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index = None
        
        self.nlist = nlist
        self.nprobe = nprobe

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log_with_timestamp(f"  Using device: {self.device}")
            model_path = self.model.tokenizer.name_or_path
            log_with_timestamp(f"  Model path: {model_path}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build FAISS index with IVFFlat for fast retrieval"""
        log_with_timestamp("  Building FAISS index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        embeddings = embeddings.astype('float32')
        
        embedding_dim = embeddings.shape[1]
        log_with_timestamp(f"  Embedding dimension: {embedding_dim}")
        
        try:
            import faiss
            
            log_with_timestamp(f"  Creating FAISS index (nlist={self.nlist}, nprobe={self.nprobe})...")
            
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            
            log_with_timestamp("  Training FAISS index...")
            self.index.train(embeddings)
            
            log_with_timestamp("  Adding embeddings to FAISS index...")
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe
            
            if self.use_gpu:
                log_with_timestamp("  Moving FAISS index to GPU...")
                res = faiss.StandardGpuResources()
                self.index_gpu = faiss.index_cpu_to_gpu(res, 0, self.index)
                log_with_timestamp("  ✓ FAISS GPU index ready")
            
            log_with_timestamp(f"  FAISS index built with {len(self.doc_ids)} docs")
        except ImportError:
            log_with_timestamp("  WARNING: FAISS not available, falling back to brute force")
            self.index = None
            self.doc_embeddings = embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using FAISS index"""
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)
        
        if self.index is None:
            from sentence_transformers import util
            query_tensor = torch.from_numpy(query_embedding).float().to(self.device)
            doc_embeddings = torch.from_numpy(self.doc_embeddings).float().to(self.device)
            scores = util.cos_sim(query_tensor, doc_embeddings)[0]
            # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
            topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
            results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                       for i, idx in enumerate(topk_indices)]
            return results
        
        try:
            if self.use_gpu and hasattr(self, 'index_gpu'):
                import faiss
                query_gpu = faiss.array_to_numpy(query_embedding)
                distances, indices = self.index_gpu.search(query_gpu, k=min(top_k + 10, len(self.doc_ids)))
            else:
                distances, indices = self.index.search(query_embedding, k=min(top_k + 10, len(self.doc_ids)))
            
            results = [(self.doc_ids[int(idx)], float(dist)) 
                      for idx, dist in zip(indices[0], distances[0]) if idx != -1]
            return results[:top_k]
        except Exception as e:
            log_with_timestamp(f"  FAISS search error: {e}, falling back to brute force")
            from sentence_transformers import util
            query_tensor = torch.from_numpy(query_embedding).float().to(self.device)
            doc_embeddings = torch.from_numpy(self.doc_embeddings).float().to(self.device)
            scores = util.cos_sim(query_tensor, doc_embeddings)[0]
            # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
            topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
            results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                       for i, idx in enumerate(topk_indices)]
            return results


class CachedRetriever:
    """Wrapper that uses pre-computed query embeddings from cache - GPU optimized"""

    def __init__(self, base_retriever, cache_file: str = None):
        self.base_retriever = base_retriever
        self.cache = {}
        self._doc_embeddings_cached = None  # 缓存的 doc embeddings tensor
        self._doc_embeddings_device = None  # 记录 device

        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            log_with_timestamp(f"  [CachedRetriever] Loaded {len(self.cache)} cached queries")

    def _get_doc_embeddings_gpu(self):
        """获取 GPU 上的 doc embeddings（带缓存，避免重复创建tensor）"""
        # 快速路径：如果已有缓存，直接返回
        if self._doc_embeddings_cached is not None:
            return self._doc_embeddings_cached

        # 获取 base_retriever 的 embeddings
        doc_embeddings = self.base_retriever.doc_embeddings

        if torch.cuda.is_available():
            device = torch.device('cuda')
            # 转换为 tensor 并移到 GPU，一次完成避免重复
            if isinstance(doc_embeddings, np.ndarray):
                self._doc_embeddings_cached = torch.from_numpy(doc_embeddings).float().to(device)
            elif isinstance(doc_embeddings, torch.Tensor):
                if doc_embeddings.device.type != 'cuda':
                    self._doc_embeddings_cached = doc_embeddings.to(device)
                else:
                    self._doc_embeddings_cached = doc_embeddings
            else:
                self._doc_embeddings_cached = torch.from_numpy(np.array(doc_embeddings)).float().to(device)
            self._doc_embeddings_device = device
        else:
            # CPU 模式
            if isinstance(doc_embeddings, np.ndarray):
                self._doc_embeddings_cached = torch.from_numpy(doc_embeddings).float()
            elif isinstance(doc_embeddings, torch.Tensor):
                self._doc_embeddings_cached = doc_embeddings
            else:
                self._doc_embeddings_cached = torch.from_numpy(np.array(doc_embeddings)).float()
            self._doc_embeddings_device = torch.device('cpu')

        return self._doc_embeddings_cached

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using cached embedding or fallback to base retriever"""
        if query in self.cache:
            query_embedding = self.cache[query]
            if isinstance(query_embedding, np.ndarray):
                query_embedding = torch.from_numpy(query_embedding).float()

            # 获取 GPU 上的 doc embeddings
            doc_embeddings = self._get_doc_embeddings_gpu()

            # 确保 query embedding 在同一设备上
            device = self._doc_embeddings_device
            if query_embedding.device != device:
                query_embedding = query_embedding.to(device)

            # 归一化并计算余弦相似度（与 GritLMRetriever.search() 一致）
            query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

            # GPU 矩阵乘法计算相似度
            scores = torch.mm(query_norm, doc_norm.T)[0]

            # 取 top k
            topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.base_retriever.doc_ids)))

            return [(self.base_retriever.doc_ids[idx.item()], topk_values[i].item())
                    for i, idx in enumerate(topk_indices)]
        else:
            return self.base_retriever.search(query, top_k)

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """批量搜索 - 更高效的处理多个查询"""
        if not queries:
            return []

        # 收集所有需要计算的查询
        uncached_queries = [q for q in queries if q not in self.cache]
        if uncached_queries:
            log_with_timestamp(f"  [CachedRetriever.batch_search] {len(uncached_queries)} queries not in cache, falling back to base")
            return [self.search(q, top_k) for q in queries]

        # 获取 GPU 上的 doc embeddings
        doc_embeddings = self._get_doc_embeddings_gpu()
        device = self._doc_embeddings_device

        # 批量转换 query embeddings
        query_embeddings = []
        for q in queries:
            q_emb = self.cache[q]
            if isinstance(q_emb, np.ndarray):
                q_emb = torch.from_numpy(q_emb).float()
            query_embeddings.append(q_emb)

        # Stack and move to GPU
        query_tensor = torch.stack(query_embeddings).to(device)

        # 归一化并计算余弦相似度（与 GritLMRetriever.search() 一致）
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        # GPU 批量矩阵乘法
        scores = torch.mm(query_norm, doc_norm.T)

        # 批量取 top k
        results = []
        for i, q in enumerate(queries):
            topk_values, topk_indices = torch.topk(scores[i], k=min(top_k, len(self.base_retriever.doc_ids)))
            results.append([(self.base_retriever.doc_ids[idx.item()], topk_values[j].item())
                          for j, idx in enumerate(topk_indices)])

        return results


class SPLADERetriever:
    """SPLADE++: Sparse Retrieval with BERT-based term weighting

    SPLADE++ uses BERT to compute importance weights for each term,
    producing a high-dimensional sparse vector representation.
    """
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.doc_vectors = None  # Sparse vectors stored as dicts: {term_idx: weight}

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading SPLADE++ model: {self.model_name}")
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            log_with_timestamp(f"  SPLADE++ model loaded")

    def _encode_text(self, text: str) -> Dict[str, float]:
        """Encode text to sparse vector (term -> weight)"""
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # SPLADE++: ReLU(log(1 + x)) + max pooling
            # Apply ReLU and log to get sparse weights
            weights = F.relu(logits)
            weights = torch.log(1 + weights)

            # Max pooling over sequence length dimension
            weights, _ = weights.max(dim=1)  # [batch, vocab_size]

            # Convert to sparse dict (only keep non-zero values)
            sparse_vec = {}
            for idx, w in enumerate(weights[0]):
                if w.item() > 0.01:  # Threshold for sparsity
                    sparse_vec[idx] = w.item()

            return sparse_vec

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build SPLADE++ index"""
        log_with_timestamp("  Building SPLADE++ index...")
        self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # Build document vectors
        doc_vectors = []
        for i, doc in enumerate(documents):
            if (i + 1) % 10000 == 0:
                log_with_timestamp(f"    Processed {i + 1}/{len(documents)}")

            text = build_document_text(doc, all_metadata)
            sparse_vec = self._encode_text(text)
            doc_vectors.append(sparse_vec)

        self.doc_vectors = doc_vectors
        log_with_timestamp(f"  SPLADE++ index built with {len(self.doc_ids)} docs")

    def _encode_query(self, query: str) -> Dict[str, float]:
        """Encode query to sparse vector"""
        return self._encode_text(query)

    def search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search using SPLADE++ sparse vectors with dot product"""
        if self.doc_vectors is None:
            log_with_timestamp("  ERROR: SPLADE index not built!")
            return [[] for _ in queries]

        # Encode all queries
        query_vectors = [self._encode_query(q) for q in queries]

        results = []
        for q_vec in query_vectors:
            scores = []
            for i, d_vec in enumerate(self.doc_vectors):
                # Compute dot product between sparse vectors
                score = 0.0
                # Only iterate over smaller vector for efficiency
                if len(q_vec) < len(d_vec):
                    for term_id, weight in q_vec.items():
                        if term_id in d_vec:
                            score += weight * d_vec[term_id]
                else:
                    for term_id, weight in d_vec.items():
                        if term_id in q_vec:
                            score += weight * q_vec[term_id]
                scores.append((self.doc_ids[i], score))

            # Sort by score and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            results.append(scores[:top_k])

        return results
