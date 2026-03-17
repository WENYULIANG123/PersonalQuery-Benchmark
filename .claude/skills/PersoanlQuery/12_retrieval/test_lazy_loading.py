#!/usr/bin/env python3
"""
测试LazyRetriever是否正确释放GPU内存
"""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluators'))

from retriever_manager import get_retriever_manager
from document_manager import get_document_manager
from datetime import datetime

log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def check_gpu_memory():
    """检查GPU内存占用"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return {'allocated': allocated, 'reserved': reserved}


def test_lazy_loading():
    log_with_timestamp("=" * 80)
    log_with_timestamp("Testing Lazy Loader vs Direct Loading")
    log_with_timestamp("=" * 80)
    
    dm = get_document_manager()
    rm = get_retriever_manager()
    
    # 加载文档
    log_with_timestamp("\n1️⃣ Loading documents...")
    docs, metadata = dm.load_documents("Arts_Crafts_and_Sewing")
    log_with_timestamp(f"Loaded {len(docs)} documents")
    
    # 清空内存缓存，确保从磁盘加载
    rm.clear_cache()
    
    # 测试1：不使用lazy loading
    log_with_timestamp("\n2️⃣ Testing WITHOUT Lazy Loading...")
    log_with_timestamp("Initial GPU memory:")
    mem = check_gpu_memory()
    if mem:
        log_with_timestamp(f"  Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    try:
        e5_direct = rm.get_retriever('e5', docs, metadata, use_lazy_loading=False)
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After loading E5 (direct): Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
        
        # 执行一次搜索
        query = "Die-cutting dies for crafts"
        log_with_timestamp(f"\nSearching: '{query}'...")
        results = e5_direct.search(query, top_k=5)
        log_with_timestamp(f"✓ Got {len(results)} results")
        
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After search: Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    except Exception as e:
        log_with_timestamp(f"❌ Error: {e}")
    
    finally:
        torch.cuda.empty_cache()
        rm.clear_cache()
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After cleanup: Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    # 测试2：使用lazy loading
    log_with_timestamp("\n3️⃣ Testing WITH Lazy Loading...")
    log_with_timestamp("Initial GPU memory:")
    mem = check_gpu_memory()
    if mem:
        log_with_timestamp(f"  Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    try:
        e5_lazy = rm.get_retriever('e5', docs, metadata, use_lazy_loading=True)
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After loading E5 (lazy wrapped): Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
        
        # 执行一次搜索
        query = "Die-cutting dies for crafts"
        log_with_timestamp(f"\nSearching: '{query}'...")
        results = e5_lazy.search(query, top_k=5)
        log_with_timestamp(f"✓ Got {len(results)} results")
        
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After search: Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    except Exception as e:
        log_with_timestamp(f"❌ Error: {e}")
    
    finally:
        torch.cuda.empty_cache()
        rm.clear_cache()
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After cleanup: Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    # 测试3：同时运行多个Dense Retrievers
    log_with_timestamp("\n4️⃣ Testing CONCURRENT Dense Retrievers (lazy)...")
    log_with_timestamp("Initial GPU memory:")
    mem = check_gpu_memory()
    if mem:
        log_with_timestamp(f"  Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
    
    try:
        log_with_timestamp("Loading E5, BGE, MiniLM concurrently...")
        e5 = rm.get_retriever('e5', docs, metadata, use_lazy_loading=True)
        bge = rm.get_retriever('bge', docs, metadata, use_lazy_loading=True)
        minilm = rm.get_retriever('minilm', docs, metadata, use_lazy_loading=True)
        
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After loading 3 models: Allocated: {mem['allocated']:.2f} GB, Reserved: {mem['reserved']:.2f} GB")
        
        query = "Die-cutting dies for crafts"
        
        # 并发搜索
        log_with_timestamp(f"\nRunning concurrent searches...")
        e5_results = e5.search(query, top_k=5)
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After E5 search: Allocated: {mem['allocated']:.2f} GB")
        
        bge_results = bge.search(query, top_k=5)
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After BGE search: Allocated: {mem['allocated']:.2f} GB")
        
        minilm_results = minilm.search(query, top_k=5)
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After MiniLM search: Allocated: {mem['allocated']:.2f} GB")
        
        log_with_timestamp(f"✓ All searches completed")
    
    except Exception as e:
        log_with_timestamp(f"❌ Error: {e}")
    
    finally:
        torch.cuda.empty_cache()
        rm.clear_cache()
        mem = check_gpu_memory()
        if mem:
            log_with_timestamp(f"After cleanup: Allocated: {mem['allocated']:.2f} GB")
    
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("✅ Test completed!")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    test_lazy_loading()
