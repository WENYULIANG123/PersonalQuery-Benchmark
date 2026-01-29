import os
import pickle
import hashlib
from typing import Any, Union, List, Dict, Optional
from collections import defaultdict

from tqdm import tqdm

from stark_qa.models.base import ModelForSTaRKQA
import bm25s


class BM25(ModelForSTaRKQA):
    """
    BM25 Model for STaRK QA.

    This model uses the BM25 algorithm for information retrieval to rank candidates
    based on their relevance to the query.
    """
    
    def __init__(self, skb: Any, cache_dir: Optional[str] = None) -> None:
        """
        Initialize the BM25 model with the given knowledge base.
        Supports caching to avoid rebuilding the index on subsequent runs.

        Args:
            skb (Any): The knowledge base containing candidate documents.
            cache_dir (Optional[str]): Directory to store cache files. 
                                       If None, uses {dataset_root}/cache/bm25
        """
        super(BM25, self).__init__(skb)
        
        # Set cache directory
        if cache_dir is None:
            # Try to get root directory from skb
            # Check for common attribute names
            root = getattr(skb, 'dataset_root', getattr(skb, 'root', None))
            
            if root:
                cache_dir = os.path.join(os.path.abspath(root), 'cache', 'bm25')
                print(f"🔍 Automatically set BM25 cache directory to: {cache_dir}")
            else:
                # Fallback to current directory
                cache_dir = os.path.join(os.getcwd(), 'cache', 'bm25')
                print(f"⚠️  No SKB root found, using fallback cache directory: {cache_dir}")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get candidate indices
        self.indices: List[int] = skb.candidate_ids
        
        # Prepare paths
        cache_key = self._generate_cache_key(self.indices)
        model_path = os.path.join(cache_dir, f'bm25_model_{cache_key}')
        meta_path = os.path.join(cache_dir, f'bm25_meta_{cache_key}.pkl')
        
        # Try to load from cache
        if os.path.exists(model_path) and os.path.exists(meta_path):
            print(f"📦 Loading BM25 index from cache: {model_path}")
            self._load_from_cache(model_path, meta_path)
            print(f"✅ BM25 index loaded from cache ({len(self.corpus)} documents)")
        else:
            print(f"🔨 Building BM25 index (first time, will be cached)...")
            self._build_index(skb)
            self._save_to_cache(model_path, meta_path)
            print(f"💾 BM25 index saved to cache: {model_path}")
    
    def _generate_cache_key(self, indices: List[int]) -> str:
        """Generate a unique cache key based on candidate indices."""
        fingerprint = f"{len(indices)}_{indices[0]}_{indices[-1]}"
        return hashlib.md5(fingerprint.encode()).hexdigest()[:16]
    
    def _build_index(self, skb: Any) -> None:
        """Build BM25 index from scratch."""
        self.corpus = [
            skb.get_doc_info(idx) for idx in tqdm(self.indices, desc="Gathering documents")
        ]
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(bm25s.tokenize(self.corpus))

        self.doc_to_candidate_ids = defaultdict(list)
        for doc, candidate_id in zip(self.corpus, self.indices):
            self.doc_to_candidate_ids[doc].append(candidate_id)
    
    def _save_to_cache(self, model_path: str, meta_path: str) -> None:
        """Save BM25 index and metadata to cache."""
        # Save model using bm25s internal method
        self.retriever.save(model_path, corpus=self.corpus)
        
        # Save mapping and indices using pickle
        meta_data = {
            'indices': self.indices,
            'doc_to_candidate_ids': dict(self.doc_to_candidate_ids),
            'corpus': self.corpus # Keep for reference
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_from_cache(self, model_path: str, meta_path: str) -> None:
        """Load BM25 index and metadata from cache."""
        # Load metadata
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        
        self.indices = meta_data['indices']
        self.doc_to_candidate_ids = defaultdict(list, meta_data['doc_to_candidate_ids'])
        self.corpus = meta_data['corpus']
        
        # Load retriever model
        # We load without corpus to ensure retrieve returns indices, which is more reliable for mapping
        self.retriever = bm25s.BM25.load(model_path, load_corpus=False)
            
    def forward(
        self, 
        query: str, 
        query_id: Optional[int] = None, 
        k: int = 100, 
        **kwargs: Any
    ) -> Dict[int, float]:
        """
        Compute similarity scores for the given query using BM25.

        Args:
            query (str): The query string.
            query_id (Optional[int], optional): The query ID. Defaults to None.
            k (int, optional): The number of top candidates to retrieve. Defaults to 100.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to their corresponding similarity scores.

        Raises:
            KeyError: If a retrieved document position is not found in the position_to_candidate_id mapping.
        """
        # Tokenize the query
        # IMPORTANT: return_ids=False to return strings, letting the retriever handle vocab mapping
        # If we return IDs (default), they are independent of corpus vocab and cause mismatch
        tokenized_query = bm25s.tokenize(query, return_ids=False)
        
        # Retrieve top k documents
        # When model has no corpus loaded, retrieve returns document indices
        results, scores = self.retriever.retrieve(tokenized_query, k=k)

        # Get retrieved document indices and their scores
        retrieved_indices = results[0]  # Indices into self.corpus
        scores_list = scores[0].tolist()  # Corresponding scores

        # Map document indices to candidate IDs
        candidate_ids = []
        scores_expanded = []
        for doc_idx, score in zip(retrieved_indices, scores_list):
            # Convert to int in case bm25s returns numpy types
            doc_idx = int(doc_idx)
            doc = self.corpus[doc_idx]
            mapped_ids = self.doc_to_candidate_ids[doc]
            candidate_ids.extend(mapped_ids)
            scores_expanded.extend([score] * len(mapped_ids))

        # Return a dictionary mapping candidate IDs to scores
        return dict(zip(candidate_ids, scores_expanded))
