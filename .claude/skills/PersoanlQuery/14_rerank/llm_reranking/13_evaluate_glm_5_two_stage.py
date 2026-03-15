#!/usr/bin/env python3
"""
GLM-5 Two-Stage LLM Evaluation

Stage 1 LLM: 判断每个维度的值是清晰还是模糊
Stage 2 LLM: 根据Stage 1的判断，只使用模糊维度的persona进行评分
"""

import json
import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')

from utils import log_with_timestamp, load_product_metadata, load_reviews_for_products, build_document_text, evaluate_retriever, load_cached_candidates, load_qa_for_products, load_preprocessed_products
from llm_client import LLMClient

USER_ID = "A13OFOB1394G31"
GLM_MODEL = "GLM-5"

BASE_DIR = "/home/wlia0047/ar57/wenyu"
QUERY_FILE = f"{BASE_DIR}/result/personal_query/07_query/dual_queries_{USER_ID}.json"
META_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json"
REVIEW_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
QA_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz"
OUTPUT_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval"
PERSONA_DIR = f"{BASE_DIR}/result/personal_query/04_persona"
PROCESSING_DIR = f"{BASE_DIR}/result/personal_query/03_processing"
CACHE_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval/cache"
CATEGORY = "Arts_Crafts_and_Sewing"
USE_PREPROCESSED = True

K_VALUES = [1, 3, 5, 10]
TOP_K_CANDIDATES = 30
MAX_CONCURRENT = 3
SIM_WEIGHT = 0.1


class CachedE5:
    """E5 that uses cached candidates instead of recomputing"""
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.candidates = None
        self.doc_ids = None

    def fit(self, documents: List[Dict], all_metadata: Dict = None):
        self.candidates = load_cached_candidates(self.cache_file)
        if self.candidates:
            self.doc_ids = [doc.get('asin', '') for doc in documents]
            log_with_timestamp(f"  Loaded {len(self.candidates)} cached E5 candidates from {self.cache_file}")
        else:
            log_with_timestamp(f"  Warning: Cache file not found: {self.cache_file}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.candidates:
            return []
        for c in self.candidates:
            if c.get('query') == query:
                return c.get('candidates', [])[:top_k]
        return []


class TwoStageGLMReRanker:
    """
    Two-Stage LLM Personalized Re-Ranker

    Stage 1: Standard 评分（无 persona）
    Stage 2: Two-stage 评分（根据维度模糊情况加载对应 persona）
    """

    def __init__(self, base_retriever, top_k: int = 50, persona_dir: str = None, processing_dir: str = None, model: str = "GLM-5", stage_mode: str = "both"):
        """
        Args:
            stage_mode: 'stage1' | 'stage2' | 'both'
        """
        self.base_retriever = base_retriever
        self.top_k = top_k
        self.model_name = model
        self.llm_client = LLMClient(model=model)
        self.persona_dir = persona_dir
        self.processing_dir = processing_dir
        self.documents = None
        self.all_metadata = None
        self.doc_ids = None
        self.query_metadata = {}
        self.personas = {}
        self.processing_attrs = {}  # 从 03_processing 加载的属性
        self.stage_mode = stage_mode  # 'stage1', 'stage2', 'both'

    def fit(self, documents: List[Dict], all_metadata: Dict = None, queries: List[Dict] = None, user_id: str = None):
        if hasattr(self.base_retriever, 'all_metadata'):
            self.base_retriever.fit(documents, all_metadata)
        else:
            self.base_retriever.fit(documents)
        self.documents = documents
        self.all_metadata = all_metadata
        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.user_id = user_id

        if queries:
            for q in queries:
                if q.get('query'):
                    self.query_metadata[q['query']] = {
                        'category': q.get('category', ''),
                        'selected_attributes': q.get('selected_attributes', [])
                    }

    def _load_processing_attrs(self, category: str) -> List[Dict]:
        """从 03_processing 目录加载属性值"""
        if category in self.processing_attrs:
            return self.processing_attrs[category]
        if not self.processing_dir or not self.user_id:
            return []
        category_filename = category.replace(" & ", "_and_").replace(" ", "_")
        processing_file = os.path.join(self.processing_dir, f"persona_{category_filename}_{self.user_id}.json")
        if not os.path.exists(processing_file):
            return []
        try:
            with open(processing_file, 'r') as f:
                data = json.load(f)
                attrs = data.get('attributes', [])
                self.processing_attrs[category] = attrs
                return attrs
        except Exception as e:
            print(f"Warning: Failed to load processing attrs: {e}")
            return []

    def _load_persona(self, category: str) -> Dict:
        if category in self.personas:
            return self.personas[category]
        if not self.persona_dir:
            return None
        category_filename = category.replace(" & ", "_and_").replace(" ", "_")
        persona_file = os.path.join(self.persona_dir, f"persona_{category_filename}_{self.user_id}.json")
        if not os.path.exists(persona_file):
            return None
        with open(persona_file, 'r') as f:
            persona = json.load(f)
            self.personas[category] = persona
            return persona

    def stage1_analyze_dimensions(self, query: str, query_attrs: List[Dict]) -> Dict:
        """
        Stage 1: 分析每个维度的明确性
        """
        attrs_text = "\n".join([
            f"- {attr['dimension']}: {attr['value']}"
            for attr in query_attrs
        ])

        prompt = f'''You are an expert at analyzing user search queries. Your task is to determine whether each attribute value is CLEAR or AMBIGUOUS.

**Query:** "{query}"

**Attributes extracted from the query:**
{attrs_text}

**Your task:** For each attribute, classify its value as:
- **CLEAR**: The value is specific, concrete, and can be directly matched (e.g., specific brand name, exact quantity, precise size, specific machine model)
- **AMBIGUOUS**: The value is vague, subjective, or needs clarification (e.g., "good quality", "nice design", "well-made", "beautiful")
- **NEUTRAL**: Somewhat specific but could benefit from more context

**Output Format (JSON):**
{{
  "dimensions": {{
    "Brand_Preference": {{
      "clarity": "clear",
      "reasoning": "Specific brand name mentioned",
      "use_persona": false
    }}
  }}
}}

**IMPORTANT:**
- If clarity is "clear" → use_persona must be false
- If clarity is "ambiguous" → use_persona must be true
- If clarity is "neutral" → use persona only if the value is somewhat vague

Analyze and output ONLY the JSON:'''

        try:
            response = self.llm_client.call(prompt, max_tokens=500, temperature=0.0)

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())

                # 兼容多种返回格式
                use_persona_for = self._parse_stage1_result(result)

                result['use_persona_for'] = use_persona_for

                return result
        except Exception as e:
            log_with_timestamp(f"  Stage 1 error: {e}")

        # fallback
        return {'dimensions': {}, 'use_persona_for': []}

    def _parse_stage1_result(self, result: Dict) -> List[str]:
        """
        解析Stage 1结果 - 兼容多种返回格式
        
        支持的格式:
        1. {"dimensions": {"dim": {"clarity": "clear", "use_persona": false}}}
        2. {"dim": {"clarity": "clear", "use_persona": false}}
        3. {"attributes": [{"id": "dim", "clarity": "clear", "use_persona": false}]}
        4. {"attributes": [{"attribute": "dim", "clarity": "clear", "use_persona": false}]}
        """
        use_persona_for = []
        
        # 格式1: {"dimensions": {...}}
        if 'dimensions' in result and isinstance(result['dimensions'], dict):
            for dim, dim_data in result['dimensions'].items():
                if self._should_use_persona(dim_data):
                    use_persona_for.append(dim)
        
        # 格式2: 直接是 {dim: {...}}
        for key, value in result.items():
            if key == 'dimensions' or key == 'attributes':
                continue
            if isinstance(value, dict) and self._should_use_persona(value):
                use_persona_for.append(key)
        
        # 格式3/4: {"attributes": [...]}
        if 'attributes' in result and isinstance(result['attributes'], list):
            for item in result['attributes']:
                # 支持 "id" 或 "attribute" 作为维度名
                dim_name = item.get('id') or item.get('attribute') or item.get('dimension', '')
                if dim_name and self._should_use_persona(item):
                    use_persona_for.append(dim_name)
        
        return use_persona_for

    def _should_use_persona(self, dim_data: Dict) -> bool:
        """判断是否应该使用该维度的persona"""
        # 检查 use_persona 字段
        if dim_data.get('use_persona', False):
            return True
        
        # 也检查 clarity 字段
        clarity = dim_data.get('clarity', dim_data.get('classification', ''))
        if clarity.lower() in ['ambiguous', 'neutral']:
            return True
        
        return False

    def stage2_score_with_persona(self, query: str, doc_text: str,
                                  stage1_result: Dict, query_attrs: List[Dict], category: str) -> float:
        """
        Stage 2: 根据Stage 1的分析，使用模糊维度的persona进行评分
        从 03_processing 读取原始属性值
        """
        use_persona_for = stage1_result.get('use_persona_for', [])

        # 如果没有需要使用persona的维度，使用Standard prompt
        if not use_persona_for:
            prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query}"
Product Info:
{doc_text}

Please analyze what RELEVANT features this product has:
1. List all matching elements (brand, category, compatibility, item types)
2. For each query requirement, note if it's met, partially met, or not met
3. Give credit for partial matches
4. End with "Final Score: X.X" where X.X reflects overall relevance

Analysis:'''

            response = self.llm_client.call(prompt, max_tokens=768, temperature=0.0)
            return self._parse_score(response)

        # 从 03_processing 加载属性
        all_attrs = self._load_processing_attrs(category)
        
        if not all_attrs:
            # 没有加载到属性，使用Standard prompt
            prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query}"
Product Info:
{doc_text}

Please analyze what RELEVANT features this product has:
1. List all matching elements (brand, category, compatibility, item types)
2. For each query requirement, note if it's met, partially met, or not met
3. Give credit for partial matches
4. End with "Final Score: X.X" where X.X reflects overall relevance

Analysis:'''

            response = self.llm_client.call(prompt, max_tokens=768, temperature=0.0)
            return self._parse_score(response)

        # 过滤出模糊维度的属性
        relevant_attrs = [attr for attr in all_attrs if attr.get('dimension') in use_persona_for]

        if not relevant_attrs:
            prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query}"
Product Info:
{doc_text}

Constraint: Output ONLY a single floating point number between 0.0 and 1.0. Do not output text.
Relevance Score:'''

            response = self.llm_client.call(prompt, max_tokens=10, temperature=0.0)
            return self._parse_score(response)

        # 构建包含persona的prompt (使用03_processing的原始属性)
        contexts = []
        for attr in relevant_attrs:
            dim = attr.get('dimension', '')
            attr_val = attr.get('attribute', '')
            sentiment = attr.get('sentiment', 'neutral')
            contexts.append(f"  - {dim}: {attr_val} (sentiment: {sentiment})")

        persona_context = "User Preferences (for ambiguous aspects only):\n" + "\n".join(contexts)

        prompt = f'''User Profile:
{persona_context}

You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[Key Guidelines]
1. If a product matches the main intent (brand + category + compatibility), it is RELEVANT even if some details are missing.
2. For die-cut products, piece counts and specific item lists are often NOT in titles - absence is not evidence of mismatch.
3. Partial match is still a match - give credit for what IS present.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query}"
Product Info:
{doc_text}

Please analyze what RELEVANT features this product has:
1. List all matching elements (brand, category, compatibility, item types)
2. For each query requirement, note if it's met, partially met, or not met
3. Give credit for partial matches
4. End with "Final Score: X.X" where X.X reflects overall relevance

Analysis:'''

        response = self.llm_client.call(prompt, max_tokens=768, temperature=0.0)
        # Look for "Final Score: X.X" pattern first
        match = re.search(r'Final Score:\s*(0\.\d+|1\.0)', response, re.IGNORECASE)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        # Fallback
        match = re.search(r'0\.\d+|1\.0', response)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.0

    def _parse_score(self, response: str) -> float:
        """解析LLM返回的分数"""
        # Look for "Final Score: X.X" pattern first
        match = re.search(r'Final Score:\s*(0\.\d+|1\.0)', response, re.IGNORECASE)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        # Fallback
        match = re.search(r'0\.\d+|1\.0', response)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        return 0.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        candidates = self.base_retriever.search(query, top_k=self.top_k)
        if not candidates:
            return []

        cand_len = len(candidates)

        # 获取query metadata
        metadata = self.query_metadata.get(query, {})
        category = metadata.get('category', '')
        query_attrs = metadata.get('selected_attributes', [])

        # Stage 1: 分析维度明确性
        stage1_result = self.stage1_analyze_dimensions(query, query_attrs)

        # Stage 1: Standard 评分（不带 persona）
        def score_doc_stage1(asin, idx):
            doc_idx = self.doc_ids.index(asin)
            doc_text = build_document_text(self.documents[doc_idx], self.all_metadata)
            
            prompt = f'''You are an expert search relevance evaluator. Your task is to score the relevance of a product to a user query on a continuous scale from 0.0 to 1.0.
Scoring Rubric:
- 0.0: Completely Irrelevant. Does not match the query.
- 0.5: Partially Relevant. Matches some keywords but misses core intent (e.g. wrong brand or function).
- 1.0: Perfectly Relevant. Matches all constraints (brand, function, attributes) in the query.

Query: "{query}"
Product Info:
{doc_text}

Constraint: Output ONLY a single floating point number between 0.0 and 1.0. Do not output text.
Relevance Score:'''
            response = self.llm_client.call(prompt, max_tokens=10, temperature=0.0)
            llm_score = self._parse_score(response)
            
            sim_score = (cand_len - idx) / cand_len
            final_score = llm_score + SIM_WEIGHT * sim_score
            return (asin, final_score)

        # Stage 2: Two-stage 评分（根据维度模糊情况加载对应 persona，使用 03_processing 数据）
        def score_doc_stage2(asin, idx):
            doc_idx = self.doc_ids.index(asin)
            doc_text = build_document_text(self.documents[doc_idx], self.all_metadata)

            # 使用 03_processing 数据
            llm_score = self.stage2_score_with_persona(query, doc_text, stage1_result, query_attrs, category)

            # STaRK style: combine LLM score with position-based similarity score
            sim_score = (cand_len - idx) / cand_len
            final_score = llm_score + SIM_WEIGHT * sim_score
            return (asin, final_score)

        # 根据 stage_mode 执行对应的 stage
        if self.stage_mode == 'stage1':
            # 只执行 Stage 1
            scores = []
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
                futures = {executor.submit(score_doc_stage1, asin, idx): asin for idx, (asin, _) in enumerate(candidates)}
                for future in as_completed(futures):
                    asin, final_score = future.result()
                    scores.append((asin, final_score))
            return sorted(scores, key=lambda x: -x[1])[:top_k]
        
        elif self.stage_mode == 'stage2':
            # 只执行 Stage 2
            scores = []
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
                futures = {executor.submit(score_doc_stage2, asin, idx): asin for idx, (asin, _) in enumerate(candidates)}
                for future in as_completed(futures):
                    asin, final_score = future.result()
                    scores.append((asin, final_score))
            return sorted(scores, key=lambda x: -x[1])[:top_k]
        
        else:  # 'both'
            # 执行两个 stage，返回 Stage 2 的结果（用于 evaluate_retriever）
            # 但保存两个 stage 的结果用于比较
            scores = []
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
                futures = {executor.submit(score_doc_stage2, asin, idx): asin for idx, (asin, _) in enumerate(candidates)}
                for future in as_completed(futures):
                    asin, final_score = future.result()
                    scores.append((asin, final_score))
            return sorted(scores, key=lambda x: -x[1])[:top_k]


def main():
    log_with_timestamp("=" * 50)
    log_with_timestamp(f"GLM-5 Two-Stage LLM Evaluation")
    log_with_timestamp(f"User: {USER_ID}")
    log_with_timestamp("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    all_asins = set()
    target_queries = []

    for r in results:
        asin = r.get('asin', '')
        if asin:
            all_asins.add(asin)
            tq = r.get('target_user_query', {})
            if tq.get('query'):
                target_queries.append({'asin': asin, 'query': tq['query'], 'type': 'target',
                    'category': r.get('category', ''), 'selected_attributes': tq.get('selected_attributes', [])})

    log_with_timestamp(f"ASINs: {len(all_asins)}, Target queries: {len(target_queries)}")

    if USE_PREPROCESSED and os.path.exists(os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl")):
        log_with_timestamp("Loading preprocessed products from cache...")
        products, all_metadata = load_preprocessed_products(CACHE_DIR, CATEGORY, all_asins)
    else:
        products, all_metadata = load_product_metadata(META_FILE, all_asins)
        products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=10)
        if os.path.exists(QA_FILE):
            products = load_qa_for_products(QA_FILE, products, max_qa_per_product=25)

    documents = [products.get(asin, {'asin': asin, 'title': '', 'brand': '', 'category': [],
        'feature': [], 'description': [], 'rank': '', 'also_buy': [], 'also_view': [], 'reviews': []})
        for asin in all_asins]

    common_metadata = {'user_id': USER_ID, 'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries), 'num_documents': len(documents),
        'k_values': K_VALUES, 'model': GLM_MODEL}

    e5_target_cache = os.path.join(OUTPUT_DIR, f"e5_candidates_{USER_ID}_target.json")

    e5 = CachedE5(e5_target_cache)
    e5.fit(documents, all_metadata)

    # ===== Stage 1: Standard 评分 =====
    log_with_timestamp("=" * 50)
    log_with_timestamp("=== Stage 1: Standard (E5 -> GLM Standard) ===")

    glm_stage1 = TwoStageGLMReRanker(e5, top_k=TOP_K_CANDIDATES, persona_dir=PERSONA_DIR, processing_dir=PROCESSING_DIR, model=GLM_MODEL, stage_mode='stage1')
    glm_stage1.fit(documents, all_metadata, queries=target_queries, user_id=USER_ID)

    stage1_result = evaluate_retriever(glm_stage1, target_queries, list(all_asins), K_VALUES)

    out = {**common_metadata, 'query_type': 'target_user', 'retriever': 'glm_standard', 'stage': 1, 'metrics': stage1_result}
    with open(os.path.join(OUTPUT_DIR, f"retrieval_glm_{GLM_MODEL}_stage1_{USER_ID}.json"), 'w') as f:
        json.dump(out, f, indent=2)

    log_with_timestamp(f"Stage 1 (Standard): {stage1_result}")

    # ===== Stage 2: Two-stage 评分 =====
    log_with_timestamp("=" * 50)
    log_with_timestamp("=== Stage 2: Two-Stage (E5 -> GLM Two-Stage) ===")

    glm_stage2 = TwoStageGLMReRanker(e5, top_k=TOP_K_CANDIDATES, persona_dir=PERSONA_DIR, processing_dir=PROCESSING_DIR, model=GLM_MODEL, stage_mode='stage2')
    glm_stage2.fit(documents, all_metadata, queries=target_queries, user_id=USER_ID)

    stage2_result = evaluate_retriever(glm_stage2, target_queries, list(all_asins), K_VALUES)

    out = {**common_metadata, 'query_type': 'target_user', 'retriever': 'glm_two_stage', 'stage': 2, 'metrics': stage2_result}
    with open(os.path.join(OUTPUT_DIR, f"retrieval_glm_{GLM_MODEL}_stage2_{USER_ID}.json"), 'w') as f:
        json.dump(out, f, indent=2)

    log_with_timestamp(f"Stage 2 (Two-Stage): {stage2_result}")

    # ===== 对比结果 =====
    log_with_timestamp("=" * 50)
    log_with_timestamp("=== 对比结果 ===")
    for k in K_VALUES:
        p1 = stage1_result.get(f'P@{k}', 0)
        p2 = stage2_result.get(f'P@{k}', 0)
        log_with_timestamp(f"P@{k}: Stage1={p1:.4f}, Stage2={p2:.4f}, Delta={p2-p1:.4f}")

    log_with_timestamp("=" * 50)
    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
