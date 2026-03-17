#!/usr/bin/env python3
"""
Lazy Retriever Proxy - 拦截retriever调用，实现真正的按需加载

核心思想：
- 不在初始化时加载任何retriever
- 只有调用search()或其他方法时，才真正加载retriever对象
- 完全透明代理，调用者无感知
"""

from typing import Any, Optional, List, Tuple, Dict
from datetime import datetime

log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class LazyRetrieverProxy:
    """
    Lazy代理：延迟加载retriever直到实际使用
    
    使用场景：
    - 初始化时只创建proxy，不加载任何retriever
    - 第一次调用search()时，自动加载retriever
    - 后续调用直接使用已加载的retriever，无额外开销
    """
    
    def __init__(self, retriever_name: str, retriever_manager: 'RetrieverManager', 
                 documents: List[Dict], metadata: Optional[Dict] = None, 
                 use_lazy_loading: bool = True):
        self.retriever_name = retriever_name
        self.retriever_manager = retriever_manager
        self.documents = documents
        self.metadata = metadata
        self.use_lazy_loading = use_lazy_loading
        
        self._actual_retriever = None
        self._loaded = False
        
        log_with_timestamp(f"[PROXY_CREATE] Created lazy proxy for {retriever_name}")
    
    def _load_actual_retriever(self):
        """真正加载retriever的地方"""
        if self._loaded:
            return
        
        log_with_timestamp(f"[PROXY_LOAD_START] Loading actual retriever: {self.retriever_name}")
        
        self._actual_retriever = self.retriever_manager.get_retriever(
            self.retriever_name,
            self.documents,
            self.metadata,
            use_lazy_loading=self.use_lazy_loading
        )
        
        self._loaded = True
        log_with_timestamp(f"[PROXY_LOAD_DONE] {self.retriever_name} loaded (type={type(self._actual_retriever).__name__})")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        执行搜索 - 第一次调用时触发真正的加载
        """
        if not self._loaded:
            self._load_actual_retriever()
        
        return self._actual_retriever.search(query, top_k)
    
    def __getattr__(self, name: str) -> Any:
        """
        代理所有其他属性访问
        
        任何对proxy的方法/属性访问都会触发加载
        """
        if not self._loaded:
            self._load_actual_retriever()
        
        return getattr(self._actual_retriever, name)
    
    def __repr__(self) -> str:
        if self._loaded:
            return f"LazyRetrieverProxy({self.retriever_name} → loaded {type(self._actual_retriever).__name__})"
        else:
            return f"LazyRetrieverProxy({self.retriever_name} → not loaded yet)"
    
    def get_loaded_status(self) -> tuple:
        if self._loaded:
            return (True, type(self._actual_retriever).__name__)
        else:
            return (False, None)
