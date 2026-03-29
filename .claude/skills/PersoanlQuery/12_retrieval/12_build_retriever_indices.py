#!/usr/bin/env python3
"""
Build all retriever indices for full-scale evaluation.
Extracts index-building logic from main evaluation script.
Only builds indices, does NOT evaluate.
"""

import os
import sys
import pickle
import hashlib
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

# Add utils path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata

# Import retriever utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import retrievers



def load_fullscale_metadata(metadata_file: str) -> Dict:
    """Load full metadata"""
    log_with_timestamp(f"Loading metadata from {metadata_file}...")
    metadata, _ = load_product_metadata(metadata_file, None)
    return metadata


def build_fullscale_documents(category: str, metadata: Dict) -> Tuple[List[Dict], Set[str]]:
    """Build full-scale document set"""
    log_with_timestamp(f"Building {len(metadata)} documents from metadata...")
    
    documents = []
    asins = set()
    
    for idx, (asin, meta) in enumerate(metadata.items()):
        if (idx + 1) % 50000 == 0:
            log_with_timestamp(f"  Processed {idx + 1}/{len(metadata)}")
        
        doc = meta.copy()
        doc['asin'] = asin
        documents.append(doc)
        asins.add(asin)
    
    log_with_timestamp(f"Built document list: {len(documents)} documents")
    return documents, asins


def compute_document_hash(documents: List[Dict]) -> str:
    """Compute hash of document set to detect changes"""
    doc_ids = sorted([doc.get('asin', '') for doc in documents])
    hash_input = '|'.join(doc_ids)
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_cache_paths(retriever_name: str, doc_hash: str, cache_dir: str) -> Dict[str, str]:
    """Get cache file paths for a retriever"""
    DENSE_RETRIEVERS = ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star', 'multi_qa_minilm']
    
    if retriever_name in DENSE_RETRIEVERS:
        base_path = os.path.join(cache_dir, f"{retriever_name}_{doc_hash}")
        return {
            'config': f"{base_path}_config.pkl",
            'embeddings': f"{base_path}_embeddings.npy",
            'doc_ids': f"{base_path}_doc_ids.pkl",
            'metadata': f"{base_path}_metadata.pkl",
        }
    else:
        return {
            'pickle': os.path.join(cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        }


def cache_exists(retriever_name: str, doc_hash: str, cache_dir: str) -> bool:
    """Check if retriever cache already exists"""
    DENSE_RETRIEVERS = ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star', 'multi_qa_minilm']
    
    if retriever_name in DENSE_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['config']) and os.path.exists(paths['embeddings'])
    else:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['pickle'])


def _normalize_embeddings_for_save(embeddings, retriever_name: str) -> np.ndarray:
    """
    Normalize embeddings for saving, handling variable-length cases (e.g., E5 multi-window).
    
    For mixed-shape embeddings (some 1D, some 2D), average-pool multi-window embeddings
    to create uniform 2D shape (n_docs, embedding_dim).
    
    Args:
        embeddings: List or array of embeddings
        retriever_name: Name of retriever (for logging)
    
    Returns:
        Normalized numpy array with shape (n_docs, embedding_dim)
    """
    log_with_timestamp(f"[DEBUG] _normalize_embeddings_for_save called for {retriever_name}")
    log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")
    if isinstance(embeddings, list):
        log_with_timestamp(f"[DEBUG] embeddings is list with {len(embeddings)} items")
    
    if not isinstance(embeddings, list):
        # Already a tensor or array
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        elif hasattr(embeddings, 'cpu'):
            return embeddings.cpu().numpy().astype(np.float32)
        else:
            return embeddings.numpy().astype(np.float32)
    
    # Handle list of embeddings (possibly with mixed shapes)
    normalized = []
    has_multiwindow = False
    
    for i, emb in enumerate(embeddings):
        # Convert to numpy
        log_with_timestamp(f"[DEBUG] Processing embedding {i}: type={type(emb)}")
        if hasattr(emb, 'cpu'):
            emb_np = emb.cpu().numpy()
        elif isinstance(emb, np.ndarray):
            emb_np = emb
        else:
            emb_np = emb.numpy()
        
        log_with_timestamp(f"[DEBUG] emb[{i}] shape={emb_np.shape}, ndim={emb_np.ndim}")
        
        # Handle multi-window embeddings (2D) vs single-window (1D)
        if emb_np.ndim == 2:
            # Multi-window: average-pool across windows
            has_multiwindow = True
            emb_pooled = emb_np.mean(axis=0)  # [num_windows, dim] -> [dim]
            log_with_timestamp(f"[DEBUG] emb[{i}] multi-window: {emb_np.shape} -> {emb_pooled.shape}")
            normalized.append(emb_pooled)
        elif emb_np.ndim == 1:
            # Single-window: keep as is
            log_with_timestamp(f"[DEBUG] emb[{i}] single-window: kept as {emb_np.shape}")
            normalized.append(emb_np)
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_np.shape}")
    
    if has_multiwindow and retriever_name == 'e5':
        log_with_timestamp(f"  [INFO] E5: Applied average pooling to multi-window embeddings for uniform shape")
    
    # Stack into (n_docs, embedding_dim)
    log_with_timestamp(f"[DEBUG] Stacking {len(normalized)} normalized embeddings...")
    embeddings_np = np.stack(normalized, axis=0).astype(np.float32)
    log_with_timestamp(f"[DEBUG] Final embeddings shape: {embeddings_np.shape}")
    return embeddings_np


def save_retriever_to_cache(retriever_name: str, doc_hash: str, retriever: object, cache_dir: str) -> bool:
    """Save retriever to disk cache. Returns True if successful."""
    DENSE_RETRIEVERS = ['dense', 'ance', 'bge', 'e5', 'minilm', 'mpnet', 'star', 'multi_qa_minilm']
    
    log_with_timestamp(f"[DEBUG] save_retriever_to_cache called for {retriever_name}")
    
    try:
        if retriever_name in DENSE_RETRIEVERS:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} with separated embeddings...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Paths: {paths}")
            
            log_with_timestamp(f"[DEBUG] Checking for doc_embeddings...")
            if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                log_with_timestamp(f"[DEBUG] doc_embeddings found, normalizing...")
                embeddings = retriever.doc_embeddings
                log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")
                
                # Use robust normalization that handles variable-length embeddings
                log_with_timestamp(f"[DEBUG] Calling _normalize_embeddings_for_save...")
                embeddings_np = _normalize_embeddings_for_save(embeddings, retriever_name)
                log_with_timestamp(f"[DEBUG] Normalization complete, shape: {embeddings_np.shape}")
                
                log_with_timestamp(f"[DEBUG] Saving embeddings to {paths['embeddings']}...")
                np.save(paths['embeddings'], embeddings_np)
                log_with_timestamp(f"[DEBUG] Embeddings saved successfully")
                size_gb = embeddings_np.nbytes / (1024**3)
                log_with_timestamp(f"  → Embeddings: {paths['embeddings']} ({size_gb:.2f}GB)")
                log_with_timestamp(f"  → Shape: {embeddings_np.shape}")
                
                log_with_timestamp(f"[DEBUG] Clearing doc_embeddings from memory...")
                retriever.doc_embeddings = None
            
            log_with_timestamp(f"[DEBUG] Saving config to {paths['config']}...")
            with open(paths['config'], 'wb') as f:
                pickle.dump(retriever, f)
            log_with_timestamp(f"  → Config: {paths['config']}")
            log_with_timestamp(f"[DEBUG] Config saved")
            
            if hasattr(retriever, 'doc_ids'):
                log_with_timestamp(f"[DEBUG] Saving doc_ids...")
                with open(paths['doc_ids'], 'wb') as f:
                    pickle.dump(retriever.doc_ids, f)
                log_with_timestamp(f"[DEBUG] doc_ids saved")
            
            if hasattr(retriever, 'all_metadata'):
                log_with_timestamp(f"[DEBUG] Saving metadata...")
                with open(paths['metadata'], 'wb') as f:
                    pickle.dump(retriever.all_metadata, f)
                log_with_timestamp(f"[DEBUG] metadata saved")
            
            log_with_timestamp(f"[DEBUG] Dense retriever save complete")
            return True
        else:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} (sparse retriever)...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Saving to {paths['pickle']}...")
            
            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)
            
            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] Sparse retriever save complete")
            return True
    except Exception as e:
        log_with_timestamp(f"[ERROR] Error saving {retriever_name}: {type(e).__name__}: {e}")
        import traceback
        log_with_timestamp(traceback.format_exc())
        return False


def build_retriever(retriever_name: str, documents: List[Dict], doc_hash: str, cache_dir: str, all_metadata: Dict = None) -> Tuple[bool, str]:
    """Build and save retriever to cache. Returns (success, cache_path_or_error)"""
    log_with_timestamp(f"[DEBUG] build_retriever: Creating {retriever_name}...")
    try:
        log_with_timestamp(f"[DEBUG] Creating retriever instance...")
        if retriever_name == 'bm25':
            retriever = retrievers.BM25()
            log_with_timestamp(f"[DEBUG] BM25 instance created")
        elif retriever_name == 'dense':
            retriever = retrievers.DenseRetriever()
            log_with_timestamp(f"[DEBUG] DenseRetriever instance created")
        elif retriever_name == 'ance':
            retriever = retrievers.ANCERetriever()
            log_with_timestamp(f"[DEBUG] ANCERetriever instance created")
        elif retriever_name == 'bge':
            retriever = retrievers.BGERetriever()
            log_with_timestamp(f"[DEBUG] BGERetriever instance created")
        elif retriever_name == 'e5':
            retriever = retrievers.E5Retriever()
            log_with_timestamp(f"[DEBUG] E5Retriever instance created")
        elif retriever_name == 'minilm':
            retriever = retrievers.MiniLMRetriever()
            log_with_timestamp(f"[DEBUG] MiniLMRetriever instance created")
        elif retriever_name == 'multi_qa_minilm':
            retriever = retrievers.MultiQAMiniLMRetriever()
            log_with_timestamp(f"[DEBUG] MultiQAMiniLMRetriever instance created")
        elif retriever_name == 'mpnet':
            retriever = retrievers.MPNetRetriever()
            log_with_timestamp(f"[DEBUG] MPNetRetriever instance created")
        elif retriever_name == 'star':
            retriever = retrievers.STARRetriever()
            log_with_timestamp(f"[DEBUG] STARRetriever instance created")
        else:
            return False, f"Unknown retriever type: {retriever_name}"
        
        log_with_timestamp(f"[DEBUG] Calling retriever.fit() on {len(documents)} documents...")
        retriever.fit(documents, all_metadata)
        log_with_timestamp(f"[DEBUG] retriever.fit() completed")
        
        log_with_timestamp(f"[DEBUG] Calling save_retriever_to_cache()...")
        if not save_retriever_to_cache(retriever_name, doc_hash, retriever, cache_dir):
            log_with_timestamp(f"[DEBUG] save_retriever_to_cache() returned False")
            return False, "Failed to save cache"
        
        log_with_timestamp(f"[DEBUG] save_retriever_to_cache() succeeded")
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        cache_path = paths.get('config') or paths.get('pickle', '')
        return True, cache_path
    except Exception as e:
        import traceback
        log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
        log_with_timestamp(traceback.format_exc())
        return False, f"{type(e).__name__}: {e}"


def main():
    """Main: Build all retriever indices"""
    setup_logging()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD ALL RETRIEVER INDICES - STARTING")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Python version: {__import__('sys').version}")
    log_with_timestamp(f"[DEBUG] Current time: {__import__('datetime').datetime.now()}")
    
    category = "Arts_Crafts_and_Sewing"
    log_with_timestamp(f"[DEBUG] Category: {category}")
    
    # Load full-scale metadata
    metadata_file = "/home/wlia0047/ar57_scratch/wenyu/result/personal_query/12_retrieval/document_cache/Arts_Crafts_and_Sewing_metadata.pkl"
    log_with_timestamp(f"[DEBUG] Metadata file path: {metadata_file}")
    log_with_timestamp(f"[DEBUG] Metadata file exists: {os.path.exists(metadata_file)}")
    
    if os.path.exists(metadata_file):
        log_with_timestamp(f"[DEBUG] Loading metadata from cache...")
        try:
            with open(metadata_file, 'rb') as f:
                log_with_timestamp(f"[DEBUG] Opened metadata file successfully")
                metadata = pickle.load(f)
                log_with_timestamp(f"[DEBUG] Metadata loaded: {len(metadata)} items")
        except Exception as e:
            log_with_timestamp(f"[ERROR] Failed to load metadata: {type(e).__name__}: {e}")
            import traceback
            log_with_timestamp(traceback.format_exc())
            raise
    else:
        log_with_timestamp("[DEBUG] Metadata cache not found, loading from raw data...")
        raw_metadata_file = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
        log_with_timestamp(f"[DEBUG] Raw metadata file: {raw_metadata_file}")
        metadata = load_fullscale_metadata(raw_metadata_file)
    
    # Build documents
    log_with_timestamp(f"[DEBUG] Starting document building...")
    documents, asins = build_fullscale_documents(category, metadata)
    log_with_timestamp(f"[DEBUG] Document building complete")
    log_with_timestamp(f"Total documents: {len(documents)}, Total ASINs: {len(asins)}")
    
    # Compute document hash for cache keys
    log_with_timestamp(f"[DEBUG] Computing document hash...")
    doc_hash = compute_document_hash(documents)
    cache_dir = "/home/wlia0047/ar57_scratch/wenyu/result/personal_query/12_retrieval/retriever_cache"
    log_with_timestamp(f"[DEBUG] Creating cache directory...")
    os.makedirs(cache_dir, exist_ok=True)
    log_with_timestamp(f"Document hash: {doc_hash}")
    log_with_timestamp(f"Cache directory: {cache_dir}")
    
    # Define retrievers to build
    DENSE_RETRIEVERS = ['dense', 'ance', 'bge', 'e5', 'minilm', 'multi_qa_minilm', 'mpnet', 'star']
    SPARSE_RETRIEVERS = ['bm25']
    ALL_RETRIEVERS = DENSE_RETRIEVERS + SPARSE_RETRIEVERS
    
    log_with_timestamp(f"\nBuilding {len(ALL_RETRIEVERS)} retrievers:")
    log_with_timestamp(f"  Dense: {DENSE_RETRIEVERS}")
    log_with_timestamp(f"  Sparse: {SPARSE_RETRIEVERS}")
    
    # Build each retriever
    log_with_timestamp(f"[DEBUG] Starting retriever build loop...")
    start_time = datetime.now()
    results = {}
    
    for retriever_name in ALL_RETRIEVERS:
        log_with_timestamp(f"[DEBUG] === Processing retriever: {retriever_name} ===")
        log_with_timestamp(f"\n[BUILD] {retriever_name}")
        log_with_timestamp(f"[DEBUG] Checking cache for {retriever_name}...")
        
        if cache_exists(retriever_name, doc_hash, cache_dir):
            log_with_timestamp(f"[CACHE_EXISTS] {retriever_name} cache already exists")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            if retriever_name in DENSE_RETRIEVERS:
                log_with_timestamp(f"  → {paths['config']}")
                log_with_timestamp(f"  → {paths['embeddings']}")
            else:
                log_with_timestamp(f"  → {paths['pickle']}")
            results[retriever_name] = {'status': 'cached', 'time': 0}
            log_with_timestamp(f"[DEBUG] {retriever_name} skipped (cached)")
        else:
            log_with_timestamp(f"[CACHE_NOT_FOUND] Building {retriever_name}...")
            log_with_timestamp(f"[DEBUG] Calling build_retriever({retriever_name})...")
            retriever_start = datetime.now()
            try:
                success, cache_path = build_retriever(retriever_name, documents, doc_hash, cache_dir, metadata)
                log_with_timestamp(f"[DEBUG] build_retriever returned: success={success}")
            except Exception as e:
                log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
                import traceback
                log_with_timestamp(traceback.format_exc())
                success = False
                cache_path = str(e)
            
            elapsed = (datetime.now() - retriever_start).total_seconds()
            log_with_timestamp(f"[DEBUG] Elapsed time: {elapsed:.1f}s")
            
            if success:
                log_with_timestamp(f"[BUILD_SUCCESS] {retriever_name} built in {elapsed:.1f}s")
                paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
                if retriever_name in DENSE_RETRIEVERS:
                    log_with_timestamp(f"  → {paths['config']}")
                    log_with_timestamp(f"  → {paths['embeddings']}")
                else:
                    log_with_timestamp(f"  → {paths['pickle']}")
                results[retriever_name] = {'status': 'success', 'time': elapsed}
            else:
                log_with_timestamp(f"[BUILD_FAILED] {retriever_name}: {cache_path}")
                results[retriever_name] = {'status': 'failed', 'error': cache_path}
    
    # Summary
    log_with_timestamp(f"[DEBUG] Build loop complete, generating summary...")
    total_time = (datetime.now() - start_time).total_seconds()
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("BUILD SUMMARY")
    log_with_timestamp("=" * 80)
    
    for retriever_name, result in results.items():
        status = result['status']
        if status == 'success':
            time = result['time']
            log_with_timestamp(f"  ✓ {retriever_name:15} - Built in {time:7.1f}s")
        elif status == 'cached':
            log_with_timestamp(f"  ⚡ {retriever_name:15} - Already cached (skipped)")
        else:
            error = result['error']
            log_with_timestamp(f"  ✗ {retriever_name:15} - ERROR: {error}")
    
    log_with_timestamp(f"\nTotal time: {total_time:.1f}s")
    successful = sum(1 for r in results.values() if r['status'] in ['success', 'cached'])
    log_with_timestamp(f"Ready: {successful}/{len(ALL_RETRIEVERS)} (built + cached)")
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    if failed > 0:
        log_with_timestamp(f"Failed: {failed}/{len(ALL_RETRIEVERS)}")
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Main function complete")


def setup_logging():
    """Setup logging directory"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)


if __name__ == '__main__':
    main()
    log_with_timestamp("当前任务已完成，请做下一个任务的指示。")
