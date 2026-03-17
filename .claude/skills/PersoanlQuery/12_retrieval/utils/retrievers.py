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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text

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
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build dense index using enhanced document text"""
        log_with_timestamp("  Building dense index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata  # 保存元数据引用
        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        log_with_timestamp(f"  Dense index built with {len(self.doc_ids)} docs")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using dense vectors"""
        model = self._get_model()
        query_embedding = model.encode([query])
        
        # Cosine similarity
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


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
            log_with_timestamp(f"    [E5_ENC_DEBUG] text len={len(text_with_prefix)}, tokens={num_tokens}, result shape={result.shape}")
            embedding = result[0]
            log_with_timestamp(f"    [E5_ENC_DEBUG] embedding shape={embedding.shape}, norm={embedding.norm().item():.4f}")
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

            # 返回所有窗口的 embeddings（堆叠成 tensor）
            return torch.stack(window_embeddings)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        构建 E5 索引（使用滑动窗口，保证不截断）
        """
        log_with_timestamp("  Building E5-large-v2 index with sliding window (no truncation)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 使用滑动窗口编码每个文档
        doc_embeddings_list = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                log_with_timestamp(f"    Encoding document {i+1}/{len(texts)}...")

            # 使用滑动窗口编码
            doc_emb = self._encode_text_with_sliding_window(text, add_prefix=True)

            # 统计窗口使用情况
            if doc_emb.dim() == 1:
                # 单窗口
                window_stats['single_window'] += 1
            else:
                # 多窗口
                window_stats['multi_window'] += 1
                num_windows = doc_emb.shape[0]
                window_stats['max_windows'] = max(window_stats['max_windows'], num_windows)

            doc_embeddings_list.append(doc_emb)
            
            # [DEBUG] Check if embeddings are all the same (only for single-window docs)
            if i == 0:
                first_emb_norm = doc_emb.norm().item() if doc_emb.dim() == 1 else doc_emb[0].norm().item()
                log_with_timestamp(f"    [E5_FIT_DEBUG] First doc embedding norm: {first_emb_norm:.6f}")
            elif i < 5 and doc_emb.dim() == 1:  # Only compare single-window embeddings
                curr_emb_norm = doc_emb.norm().item()
                log_with_timestamp(f"    [E5_FIT_DEBUG] Doc {i} embedding norm: {curr_emb_norm:.6f}")
                if i == 1 and len(doc_embeddings_list) >= 2 and doc_embeddings_list[0].dim() == 1:
                    # Check if first two embeddings are identical
                    if torch.allclose(doc_embeddings_list[0], doc_embeddings_list[1], atol=1e-7):
                        log_with_timestamp(f"    [E5_FIT_DEBUG] ⚠️ WARNING: First two embeddings are IDENTICAL!")

        # 将列表转换为 tensor（对于单窗口的文档）
        # 注意：我们保持 list 格式，因为不同文档可能有不同数量的窗口
        self.doc_embeddings = doc_embeddings_list

        log_with_timestamp(f"  E5 index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs")
        log_with_timestamp(f"    - Max windows per doc: {window_stats['max_windows']}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using E5 (supports both multi-window and pooled embeddings)"""
        model = self._get_model()

        # 添加 query 前缀
        query_with_prefix = self._add_instruction(query, is_query=True)
        query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)[0]

        # 计算每个文档的分数
        from sentence_transformers import util
        scores = []

        for i, doc_emb in enumerate(self.doc_embeddings):
            # Handle both list elements (tensors) and numpy array rows
            if isinstance(doc_emb, np.ndarray):
                doc_emb = torch.from_numpy(doc_emb).to(self.device)
            
            if doc_emb.dim() == 1:
                # Single-window or pooled embedding: direct cosine similarity
                score = util.cos_sim(query_embedding, doc_emb.unsqueeze(0))[0][0].item()
            else:
                # Multi-window: max-pool across windows
                window_scores = util.cos_sim(query_embedding, doc_emb)[0]
                score = window_scores.max().item()

            scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


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
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """BGE 推荐添加 instruction 前缀"""
        if is_query:
            return "Represent this sentence for searching relevant passages: " + text
        else:
            return text

    def _encode_text_with_sliding_window(self, text: str):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本（BGE 文档不需要前缀）

        Returns:
            如果文本 <= max_seq_length: 单个 embedding [dim]
            如果文本 > max_seq_length: 多个窗口的 embeddings [num_windows, dim]
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

            # 返回所有窗口的 embeddings（堆叠成 tensor）
            return torch.stack(window_embeddings)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        构建 BGE 索引（使用滑动窗口，保证不截断）
        """
        log_with_timestamp("  Building BGE-large-en index with sliding window (no truncation)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 使用滑动窗口编码每个文档
        doc_embeddings_list = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                log_with_timestamp(f"    Encoding document {i+1}/{len(texts)}...")

            # 使用滑动窗口编码
            doc_emb = self._encode_text_with_sliding_window(text)

            # 统计窗口使用情况
            if doc_emb.dim() == 1:
                # 单窗口
                window_stats['single_window'] += 1
            else:
                # 多窗口
                window_stats['multi_window'] += 1
                num_windows = doc_emb.shape[0]
                window_stats['max_windows'] = max(window_stats['max_windows'], num_windows)

            doc_embeddings_list.append(doc_emb)

        # 将列表保持为 list 格式
        self.doc_embeddings = doc_embeddings_list

        log_with_timestamp(f"  BGE index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs")
        log_with_timestamp(f"    - Max windows per doc: {window_stats['max_windows']}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用 BGE 搜索（支持多窗口文档）
        """
        model = self._get_model()

        # 添加 instruction 前缀
        query_with_prefix = self._add_instruction(query, is_query=True)
        query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)[0]

        # 计算每个文档的分数
        from sentence_transformers import util
        scores = []

        for i, doc_emb in enumerate(self.doc_embeddings):
            if doc_emb.dim() == 1:
                # 单窗口文档：直接计算余弦相似度
                score = util.cos_sim(query_embedding, doc_emb.unsqueeze(0))[0][0].item()
            else:
                # 多窗口文档：计算每个窗口的分数，取最大值
                window_scores = util.cos_sim(query_embedding, doc_emb)[0]
                score = window_scores.max().item()

            scores.append((self.doc_ids[i], score))

        # 按分数降序排序
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class ColBERTRetriever:
    """ColBERTv2-inspired: Token-level Late Interaction (MaxSim) with proper implementation"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        使用更好的 sentence-transformer 模型作为基础
        相比 bert-base-uncased，这个模型经过优化用于语义相似度任务

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
            log_with_timestamp(f"  Loading ColBERT-inspired model: {self.model_name}")
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
    """TF-IDF retrieval as a baseline"""
    def __init__(self):
        self.doc_vectors = None
        self.feature_names = None
        self.doc_ids = []
        self.all_metadata = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build TF-IDF index"""
        log_with_timestamp("  Building TF-IDF index...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 构建 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),  # 使用 unigrams 和 bigrams
            sublinear_tf=True
        )

        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        log_with_timestamp(f"  TF-IDF index built with {len(self.doc_ids)} docs, {len(self.feature_names)} features")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF"""
        query_vector = self.vectorizer.transform([query])

        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]

        results = [(self.doc_ids[i], scores[i]) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


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
    
    Uses SGPT-7B-weightedmean-nli-bitfit embedding model.
    Reference: STaRK implementation
    """
    def __init__(self, model_name: str = "Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit",
                 hf_token: str = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading GritLM model: {self.model_name}")
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModel.from_pretrained(self.model_name, token=self.hf_token)
            self.model.eval()
            self.model.to(self.device)
            
            # Enable optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            log_with_timestamp(f"  GritLM loaded on {self.device}")
        return self.model

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using GritLM with mean pooling."""
        model = self._get_model()
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                              padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.squeeze(0)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build GritLM index"""
        log_with_timestamp(f"  Building GritLM index with {len(documents)} docs...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # Encode all documents
        texts = [build_document_text(doc, all_metadata) for doc in documents]
        
        # Batch encode
        embeddings = []
        batch_size = 8  # Small batch size for 7B model
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings_batch = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_batch.size()).float()
                sum_embeddings = torch.sum(embeddings_batch * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
            embeddings.append(batch_embeddings.cpu())
            
            if (i // batch_size) % 10 == 0:
                log_with_timestamp(f"    Encoded {min(i+batch_size, len(texts))}/{len(texts)} docs")
        
        self.doc_embeddings = torch.cat(embeddings, dim=0)
        log_with_timestamp(f"  GritLM index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using GritLM"""
        query_embedding = self._encode_text(query)
        query_embedding = query_embedding.unsqueeze(0)  # [1, hidden_dim]
        
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

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading ANCE-compatible model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build ANCE index using enhanced document text"""
        log_with_timestamp("  Building ANCE-compatible index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        log_with_timestamp(f"  ANCE index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using ANCE embeddings"""
        model = self._get_model()
        query_embedding = model.encode(["query: " + query])
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


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

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading STAR-compatible model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build STAR index using enhanced document text"""
        log_with_timestamp("  Building STAR-compatible index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        log_with_timestamp(f"  STAR index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using STAR embeddings"""
        model = self._get_model()
        query_embedding = model.encode([query])
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


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

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading MiniLM model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MiniLM index using enhanced document text"""
        log_with_timestamp("  Building MiniLM index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        log_with_timestamp(f"  MiniLM index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MiniLM embeddings"""
        model = self._get_model()
        query_embedding = model.encode([query])
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


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

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading MPNet model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MPNet index using enhanced document text"""
        log_with_timestamp("  Building MPNet index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        log_with_timestamp(f"  MPNet index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MPNet embeddings"""
        model = self._get_model()
        query_embedding = model.encode([query])
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
