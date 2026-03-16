"""
Stage 13 Retrieval Utilities

This package provides core utilities for the retrieval evaluation pipeline.
"""

from .utils import (
    log_with_timestamp,
    load_product_metadata,
    load_reviews_for_products,
    build_document_text,
    evaluate_retriever
)

from .retrievers import (
    BM25,
    ColBERTRetriever,
    DenseRetriever,
    E5Retriever,
    BGERetriever,
    TFIDFRetriever
)

# from .hybrid import HybridRetriever  # File not found
from .reranker_bert import BERTReRanker

__all__ = [
    # Core utilities
    'log_with_timestamp',
    'load_product_metadata',
    'load_reviews_for_products',
    'build_document_text',
    'evaluate_retriever',

    # Retriever classes
    'BM25',
    'ColBERTRetriever',
    'DenseRetriever',
    'E5Retriever',
    'BGERetriever',
    'TFIDFRetriever',
    # 'HybridRetriever',  # File not found
    'BERTReRanker',
]