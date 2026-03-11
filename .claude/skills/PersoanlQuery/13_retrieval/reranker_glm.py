#!/usr/bin/env python3
"""
Stage 13: GLM API-based Rerankers

Contains GLM-based reranking classes:
- GLMReRanker: GLM API-based reranker for relevance scoring
- PersonalizedGLMReRanker: GLM reranker with user persona context
"""

import os
import re
from typing import List, Dict, Tuple

# Import LLM Client for GLM API
import sys
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')
from llm_client import LLMClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text


class GLMReRanker:
    """GLM-4.5V API-based reranker for relevance scoring"""

    def __init__(self, base_retriever, top_k: int = 50, model: str = "GLM-4.5V"):
        """
        Args:
            base_retriever: 底层检索器（BM25, E5等）
            top_k: 从底层检索器获取多少候选用于重排序
        """
        self.base_retriever = base_retriever
        self.top_k = top_k
        self.model_name = model
        self.llm_client = LLMClient(model=model)
        self.documents = None
        self.all_metadata = None
        self.doc_ids = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """训练底层检索器"""
        log_with_timestamp(f"  Building {self.model_name} Reranker index...")
        # 训练底层检索器
        if hasattr(self.base_retriever, 'all_metadata'):
            self.base_retriever.fit(documents, all_metadata)
        else:
            self.base_retriever.fit(documents)

        # 保存文档用于重排序
        self.documents = documents
        self.all_metadata = all_metadata
        self.doc_ids = [doc.get('asin', '') for doc in documents]

        log_with_timestamp(f"  {self.model_name} Reranker index built with {len(self.doc_ids)} docs")

    def _score_by_generation(self, query: str, doc_text: str) -> float:
        """使用GLM API生成相关性分数"""
        # 构造 prompt（使用完整文档）
        prompt = f"""Query: {query}

Document: {doc_text}

Rate the relevance of this document to the query on a scale of 1 to 10, where 1 is completely irrelevant and 10 is perfectly relevant. Output only the number."""

        # 调用GLM API
        response = self.llm_client.call(prompt, max_tokens=10, temperature=0.0)

        # 提取数字
        match = re.search(r'\d+', response)
        if match:
            score = float(match.group())
            # 限制在 1-10 范围内
            score = max(1.0, min(10.0, score))
            return score / 10.0  # 归一化到 0-1
        else:
            return 0.1  # 默认低分

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 GLM-4.5V 重排序"""
        # Step 1: 使用底层检索器获取候选
        candidates = self.base_retriever.search(query, top_k=self.top_k)

        if not candidates:
            return []

        # Step 2: 使用 GLM-4.5V 打分重排序
        scores = []
        for asin, _ in candidates:
            # 找到对应的文档
            doc_idx = self.doc_ids.index(asin)
            doc_text = build_document_text(self.documents[doc_idx], self.all_metadata)

            # 打分
            score = self._score_by_generation(query, doc_text)
            scores.append((asin, score))

        # 按分数降序排序
        reranked_results = sorted(scores, key=lambda x: -x[1])

        return reranked_results[:top_k]


class PersonalizedGLMReRanker(GLMReRanker):
    """GLM-4.5V reranker with user persona context for personalized ranking"""

    def __init__(self, base_retriever, top_k: int = 50, persona_dir: str = None, model: str = "GLM-4.5V"):
        """
        Args:
            base_retriever: 底层检索器
            top_k: 从底层检索器获取多少候选
            persona_dir: 画像文件目录
        """
        GLMReRanker.__init__(self, base_retriever, top_k, model=model)
        self.persona_dir = persona_dir
        self.query_metadata = {}  # query text → {category, selected_attributes, user_id}
        self.personas = {}  # category → persona dict

    def _load_persona(self, category: str, user_id: str) -> Dict:
        """加载特定类别的用户画像"""
        if category in self.personas:
            return self.personas[category]

        if self.persona_dir is None:
            return None

        # 转换 category 名称以匹配文件名格式
        # "Stickers & Sticker Machines" -> "Stickers_and_Sticker_Machines"
        # "Die-Cuts" -> "Die-Cuts" (保留连字符)
        category_filename = category.replace(" & ", "_and_").replace(" ", "_")
        # 注意：不要替换 "-" ，因为文件名中保留了 "-"

        # 构造画像文件路径
        persona_file = os.path.join(self.persona_dir, f"persona_{category_filename}_{user_id}.json")

        if not os.path.exists(persona_file):
            log_with_timestamp(f"  Warning: Persona file not found: {persona_file}")
            return None

        try:
            with open(persona_file, 'r') as f:
                persona = json.load(f)
                self.personas[category] = persona  # 用原始 category 作为 key
                log_with_timestamp(f"  Loaded persona for {category}")
                return persona
        except Exception as e:
            log_with_timestamp(f"  Error loading persona: {e}")
            return None

    def _build_persona_context(self, category: str, selected_attributes: List[Dict]) -> str:
        """构建画像上下文文本"""
        persona = self._load_persona(category, self.user_id)

        if persona is None:
            return ""

        dimension_personas = persona.get('dimension_personas', {})

        # 提取相关维度的画像描述
        relevant_contexts = []
        for attr in selected_attributes:
            dimension = attr.get('dimension', '')
            value = attr.get('value', '')

            if dimension in dimension_personas:
                persona_desc = dimension_personas[dimension]
                relevant_contexts.append(f"  - {dimension}: {persona_desc}")

        if not relevant_contexts:
            return ""

        context = "User Preferences:\n" + "\n".join(relevant_contexts)
        return context

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None,
            queries: List[Dict] = None, user_id: str = None):
        """
        训练模型并建立查询元数据映射

        Args:
            documents: 文档列表
            all_metadata: 元数据
            queries: 查询列表（包含 category, selected_attributes）
            user_id: 用户ID
        """
        # 调用父类 fit
        super().fit(documents, all_metadata)

        # 保存 user_id
        self.user_id = user_id

        # 建立查询 → 元数据映射
        if queries is not None:
            for q in queries:
                query_text = q.get('query', '')
                if query_text:
                    self.query_metadata[query_text] = {
                        'category': q.get('category', ''),
                        'selected_attributes': q.get('selected_attributes', [])
                    }
            log_with_timestamp(f"  Built query metadata mapping for {len(self.query_metadata)} queries")

    def _score_by_generation(self, query: str, doc_text: str) -> float:
        """使用用户画像上下文进行生成式打分"""
        import json as json_module
        
        # 获取查询的元数据
        metadata = self.query_metadata.get(query, {})
        category = metadata.get('category', '')
        selected_attributes = metadata.get('selected_attributes', [])

        # 构建画像上下文
        persona_context = ""
        if category and selected_attributes:
            persona_context = self._build_persona_context(category, selected_attributes)

        # 构造 prompt（加入画像上下文，使用完整文档）
        if persona_context:
            prompt = f"""User Profile:
{persona_context}

Query: {query}

Document: {doc_text}

Rate the relevance of this document to the query on a scale of 1 to 10,
where 1 is completely irrelevant and 10 is perfectly relevant.
Consider the user's preferences when rating. Output only the number."""
        else:
            # 没有画像上下文，使用原来的 prompt
            prompt = f"""Query: {query}

Document: {doc_text}

Rate the relevance of this document to the query on a scale of 1 to 10,
where 1 is completely irrelevant and 10 is perfectly relevant. Output only the number."""

        # 调用GLM API
        response = self.llm_client.call(prompt, max_tokens=10, temperature=0.0)

        # 提取数字
        match = re.search(r'\d+', response)
        if match:
            score = float(match.group())
            score = max(1.0, min(10.0, score))
            return score / 10.0
        else:
            return 0.1


import json
