#!/usr/bin/env python3
"""
Stage 13: Base Retrieval Models

Contains all base retriever classes:
- BM25: Traditional sparse retrieval
- DenseRetriever: ANCE/MiniLM dense retrieval
- E5Retriever: E5-large-v2 dense retrieval
- BGERetriever: BGE-large-en dense retrieval
- ColBERTRetriever: Token-level late interaction
- TFIDFRetriever: TF-IDF baseline
- DirichletPriorRetriever: Query likelihood with Dirichlet smoothing
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text


class BM25:
    """BM25 retrieval model"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_tokens = []
        self.doc_asins = []  # 添加 ASIN 列表
        
    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build BM25 index using enhanced document text"""
        log_with_timestamp("  Building BM25 index...")
        nd = {}
        for doc in documents:
            asin = doc.get('asin', '')
            self.doc_asins.append(asin)  # 存储 ASIN
            # 使用增强的文档文本构建
            text = build_document_text(doc, all_metadata)
            tokens = text.lower().split()
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

            frequencies = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            for token, freq in frequencies.items():
                if token not in nd:
                    nd[token] = 0
                nd[token] += 1

        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Calculate IDF
        N = len(documents)
        for token, freq in nd.items():
            self.idf[token] = np.log((N - freq + 0.5) / (freq + 0.5) + 1)

        log_with_timestamp(f"  BM25 index built with {len(self.doc_tokens)} docs, {len(self.idf)} terms")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25"""
        query_tokens = query.lower().split()
        scores = []

        for doc_id, tokens in enumerate(self.doc_tokens):
            score = 0
            doc_len = self.doc_len[doc_id]

            frequencies = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            for token in query_tokens:
                if token in frequencies:
                    freq = frequencies[token]
                    idf = self.idf.get(token, 0)
                    score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))

            # 返回 (asin, score) 而不是 (doc_id, score)
            asin = self.doc_asins[doc_id]
            scores.append((asin, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class DenseRetriever:
    """Dense retrieval using sentence-transformers"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None  # 存储所有元数据

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
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
    """E5-large-v2: SOTA embedding model with instruction-based retrieval"""
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """构建 E5 索引"""
        log_with_timestamp("  Building E5-large-v2 index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建，添加 passage 前缀
        texts = [build_document_text(doc, all_metadata) for doc in documents]
        texts_with_prefix = [self._add_instruction(text, is_query=False) for text in texts]

        self.doc_embeddings = model.encode(
            texts_with_prefix,
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True
        )
        log_with_timestamp(f"  E5 index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 E5 搜索"""
        model = self._get_model()

        # 添加 query 前缀
        query_with_prefix = self._add_instruction(query, is_query=True)
        query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)

        # Cosine similarity
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class BGERetriever:
    """BGE-large-en: SOTA embedding model for English retrieval"""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """构建 BGE 索引"""
        log_with_timestamp("  Building BGE-large-en index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建，文档不需要前缀
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True
        )
        log_with_timestamp(f"  BGE index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 BGE 搜索"""
        model = self._get_model()

        # 添加 instruction 前缀
        query_with_prefix = self._add_instruction(query, is_query=True)
        query_embedding = model.encode([query_with_prefix], convert_to_tensor=True)

        # Cosine similarity
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class ColBERTRetriever:
    """真正的 ColBERTv2: Token-level Late Interaction (MaxSim)"""
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None  # List of token embeddings
        self.doc_ids = []
        self.all_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_doc_length = 180  # ColBERTv2 default

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading ColBERT model: {self.model_name}")
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def _encode_text(self, text: str, max_length: int = None):
        """编码文本为 token-level embeddings"""
        if max_length is None:
            max_length = self.max_doc_length

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用最后一层隐藏状态
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

        # Apply attention mask to zero out padding tokens
        attention_mask_expanded = attention_mask[0].unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings * attention_mask_expanded

        return token_embeddings  # [seq_len, hidden_size]

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """构建 ColBERT 索引：token-level embeddings"""
        log_with_timestamp("  Building ColBERTv2 index (token-level)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 编码每个文档为 token-level embeddings
        self.doc_embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                log_with_timestamp(f"    Encoding document {i+1}/{len(texts)}...")

            token_emb = self._encode_text(text)
            self.doc_embeddings.append(token_emb)

        log_with_timestamp(f"  ColBERTv2 index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 Late Interaction (MaxSim) 搜索"""
        model = self._get_model()

        # 编码查询为 token-level embeddings
        query_emb = self._encode_text(query, max_length=32)  # Query 较短
        query_emb = query_emb.to(self.device)  # [query_len, hidden_size]

        # 归一化查询 embeddings
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        scores = []
        for i, doc_emb in enumerate(self.doc_embeddings):
            doc_emb = doc_emb.to(self.device)  # [doc_len, hidden_size]

            # 归一化文档 embeddings
            doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # MaxSim: 计算每个 query token 与所有 doc tokens 的最大相似度
            # sim_matrix: [query_len, doc_len]
            sim_matrix = torch.mm(query_emb, doc_emb.t())

            # 对于每个 query token，取最大相似度
            max_sim_per_query_token, _ = sim_matrix.max(dim=1)  # [query_len]

            # 所有 query token 的最大相似度求和（Late Interaction）
            score = max_sim_per_query_token.sum().item()

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
