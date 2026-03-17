#!/usr/bin/env python3
"""Build mpnet retriever index for fullscale evaluation"""
import os, sys, pickle, hashlib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils
from retriever_manager import get_retriever_manager

log_with_timestamp = utils.log_with_timestamp

BASE_DIR = "/home/wlia0047/ar57/wenyu"
OUTPUT_DIR = os.path.join(BASE_DIR, "result/personal_query/12_retrieval")
METADATA_FILE = os.path.join(OUTPUT_DIR, "document_cache/Arts_Crafts_and_Sewing_metadata.pkl")
RETRIEVER_NAME = 'mpnet'

def main():
    try:
        log_with_timestamp(f"[{RETRIEVER_NAME}] Loading metadata...")
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        
        log_with_timestamp(f"[{RETRIEVER_NAME}] Building documents from {len(metadata)} products...")
        documents = []
        for i, asin in enumerate(sorted(metadata.keys())):
            if i % 50000 == 0:
                log_with_timestamp(f"[{RETRIEVER_NAME}] Processed {i}/{len(metadata)}")
            doc = metadata[asin].copy()
            doc['asin'] = asin
            documents.append(doc)
        
        doc_ids = sorted([doc.get('asin', '') for doc in documents])
        doc_hash = hashlib.md5('|'.join(doc_ids).encode()).hexdigest()
        
        log_with_timestamp(f"[{RETRIEVER_NAME}] Document hash: {doc_hash}")
        log_with_timestamp(f"[{RETRIEVER_NAME}] Building index...")
        
        start_time = datetime.now()
        rm = get_retriever_manager()
        retriever = rm.get_retriever(RETRIEVER_NAME, documents, metadata)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        cache_dir = os.path.join(OUTPUT_DIR, "retriever_cache")
        cache_file = os.path.join(cache_dir, f"{RETRIEVER_NAME}_{doc_hash}.pkl")
        size = os.path.getsize(cache_file) / (1024**2)
        
        log_with_timestamp(f"[{RETRIEVER_NAME}] ✓ Built in {elapsed:.2f}s ({size:.1f}MB)")
        log_with_timestamp(f"[{RETRIEVER_NAME}] Saved to {cache_file}")
        
    except Exception as e:
        log_with_timestamp(f"[{RETRIEVER_NAME}] ✗ ERROR: {e}")
        import traceback
        log_with_timestamp(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
