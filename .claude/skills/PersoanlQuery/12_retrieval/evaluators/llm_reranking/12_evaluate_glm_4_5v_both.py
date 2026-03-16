#!/usr/bin/env python3
"""GLM-4.5V Reranker Evaluation (Standard + Personalized) - Using cached E5 candidates"""

import json
import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'utils')
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')

from utils import log_with_timestamp, load_product_metadata, load_reviews_for_products, load_qa_for_products, load_preprocessed_products, build_document_text, evaluate_retriever, load_cached_candidates
from llm_client import LLMClient

USER_ID = "A13OFOB1394G31"
GLM_MODEL = "GLM-4.5V"

BASE_DIR = "/home/wlia0047/ar57/wenyu"
QUERY_FILE = f"{BASE_DIR}/result/personal_query/07_query/dual_queries_{USER_ID}.json"
META_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json"
REVIEW_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
QA_FILE = f"{BASE_DIR}/data/Amazon-Reviews-2018/raw/qa_Arts_Crafts_and_Sewing.json.gz"
OUTPUT_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval"
CACHE_DIR = f"{BASE_DIR}/result/personal_query/13_retrieval/cache"
PERSONA_DIR = f"{BASE_DIR}/result/personal_query/04_persona"
PROCESSING_DIR = f"{BASE_DIR}/result/personal_query/03_processing"
CATEGORY = "Arts_Crafts_and_Sewing"
USE_PREPROCESSED = True

K_VALUES = [1, 3, 5, 10]
TOP_K_CANDIDATES = 30
MAX_CONCURRENT = 3
SIM_WEIGHT = 0.1  # STaRK style: combine LLM score with position score
FIRST_STAGE_MODEL = "E5"  # STaRK uses text-embedding-ada-002, we use E5 as first stage


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


class CachedBGE:
    """BGE that uses cached candidates instead of recomputing"""
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.candidates = None
        self.doc_ids = None

    def fit(self, documents: List[Dict], all_metadata: Dict = None):
        self.candidates = load_cached_candidates(self.cache_file)
        if self.candidates:
            self.doc_ids = [doc.get('asin', '') for doc in documents]
            log_with_timestamp(f"  Loaded {len(self.candidates)} cached BGE candidates from {self.cache_file}")
        else:
            log_with_timestamp(f"  Warning: Cache file not found: {self.cache_file}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.candidates:
            return []
        for c in self.candidates:
            if c.get('query') == query:
                return c.get('candidates', [])[:top_k]
        return []


class CachedColBERT:
    """ColBERT that uses cached candidates instead of recomputing"""
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.candidates = None
        self.doc_ids = None

    def fit(self, documents: List[Dict], all_metadata: Dict = None):
        self.candidates = load_cached_candidates(self.cache_file)
        if self.candidates:
            self.doc_ids = [doc.get('asin', '') for doc in documents]
            log_with_timestamp(f"  Loaded {len(self.candidates)} cached ColBERT candidates from {self.cache_file}")
        else:
            log_with_timestamp(f"  Warning: Cache file not found: {self.cache_file}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.candidates:
            return []
        for c in self.candidates:
            if c.get('query') == query:
                return c.get('candidates', [])[:top_k]
        return []


class GLMReRanker:
    def __init__(self, base_retriever, top_k: int = 50, model: str = "GLM-4.5V"):
        self.base_retriever = base_retriever
        self.top_k = top_k
        self.model_name = model
        self.llm_client = LLMClient(model=model)
        self.documents = None
        self.all_metadata = None
        self.doc_ids = None

    def fit(self, documents: List[Dict], all_metadata: Dict = None):
        if hasattr(self.base_retriever, 'all_metadata'):
            self.base_retriever.fit(documents, all_metadata)
        else:
            self.base_retriever.fit(documents)
        self.documents = documents
        self.all_metadata = all_metadata
        self.doc_ids = [doc.get('asin', '') for doc in documents]

    def _score_by_generation(self, query: str, doc_text: str) -> float:
        # Improved prompt with positive reinforcement
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

Output ONLY a single floating point number between 0.0 and 1.0. No explanation needed.

Score: '''
        response = self.llm_client.call(prompt, max_tokens=50, temperature=0.0)
        
        # DEBUG: Log raw response
        log_with_timestamp(f"[DEBUG] Raw LLM response: '{response}'")
        
        # Parse simple number format
        match = re.search(r'0\.\d+|1\.0', response)
        if match:
            score = float(match.group())
            log_with_timestamp(f"[DEBUG] Regex matched: '{match.group()}' → score={score}")
            return max(0.0, min(1.0, score))
        
        # DEBUG: Log parse failure
        log_with_timestamp(f"[DEBUG] ❌ PARSE FAILED - regex did not match any pattern in: '{response}'")
        return 0.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        candidates = self.base_retriever.search(query, top_k=self.top_k)
        if not candidates:
            return []
        
        cand_len = len(candidates)

        def score_doc(asin, idx):
            doc_idx = self.doc_ids.index(asin)
            doc_text = build_document_text(self.documents[doc_idx], self.all_metadata)
            try:
                llm_score = self._score_by_generation(query, doc_text)
            except Exception as e:
                # STaRK style: when LLM fails, still preserve position score
                log_with_timestamp(f"⚠️ LLM scoring failed for {asin}: {e}, using fallback score")
                llm_score = 0.0
            # STaRK style: combine LLM score with position-based similarity score
            sim_score = (cand_len - idx) / cand_len
            final_score = llm_score + SIM_WEIGHT * sim_score
            return (asin, final_score, idx)

        scores = []
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {executor.submit(score_doc, asin, idx): (asin, idx) for idx, (asin, _) in enumerate(candidates)}
            for future in as_completed(futures):
                try:
                    asin, final_score, idx = future.result()
                    scores.append((asin, final_score))
                except Exception as e:
                    asin, idx = futures[future]
                    # STaRK style: when everything fails, still compute position score
                    sim_score = (cand_len - idx) / cand_len
                    fallback_score = 0.0 + SIM_WEIGHT * sim_score
                    scores.append((asin, fallback_score))

        return sorted(scores, key=lambda x: -x[1])[:top_k]


class PersonalizedGLMReRanker(GLMReRanker):
    def __init__(self, base_retriever, top_k: int = 50, persona_dir: str = None, processing_dir: str = None, model: str = "GLM-4.5V"):
        super().__init__(base_retriever, top_k, model)
        self.persona_dir = persona_dir
        self.processing_dir = processing_dir
        self.query_metadata = {}
        self.personas = {}
        self.processing_attrs = {}  # 从 03_processing 加载的属性

    def fit(self, documents: List[Dict], all_metadata: Dict = None, queries: List[Dict] = None, user_id: str = None):
        super().fit(documents, all_metadata)
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
                # 获取 attributes 数组
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

    def _build_persona_context(self, category: str, selected_attributes: List[Dict]) -> str:
        """
        从 03_processing 目录读取属性值，构建用户偏好上下文
        使用 selected_attributes 中的维度来过滤相关属性
        """
        if not selected_attributes:
            return ""
        
        # 获取选中的维度列表
        selected_dims = set(attr.get('dimension', '') for attr in selected_attributes if attr.get('dimension'))
        
        # 从 03_processing 加载该类别的所有属性
        all_attrs = self._load_processing_attrs(category)
        
        if not all_attrs:
            # 如果没有加载到属性，回退到直接使用 selected_attributes
            contexts = [f"  - {attr.get('dimension', '')}: {attr.get('value', '')}" 
                        for attr in selected_attributes if attr.get('dimension')]
            return "User Preferences:\n" + "\n".join(contexts) if contexts else ""
        
        # 按维度分组属性
        attrs_by_dim = {}
        for attr in all_attrs:
            dim = attr.get('dimension', '')
            if dim in selected_dims:
                if dim not in attrs_by_dim:
                    attrs_by_dim[dim] = []
                # 包含 attribute, sentiment, original_text
                attrs_by_dim[dim].append(attr)
        
        # 构建上下文
        contexts = []
        for attr in selected_attributes:
            dim = attr.get('dimension', '')
            if dim in attrs_by_dim:
                # 显示该维度的所有相关属性
                for a in attrs_by_dim[dim]:
                    sentiment = a.get('sentiment', 'neutral')
                    orig_text = a.get('original_text', '')
                    attr_val = a.get('attribute', '')
                    contexts.append(f"  - {dim}: {attr_val} (sentiment: {sentiment})")
        
        return "User Preferences:\n" + "\n".join(contexts) if contexts else ""

    def _score_by_generation(self, query: str, doc_text: str) -> float:
        metadata = self.query_metadata.get(query, {})
        persona_context = self._build_persona_context(
            metadata.get('category', ''), metadata.get('selected_attributes', []))

        # Improved prompt with positive reinforcement (same as test script)
        if persona_context:
            prompt = f'''You are an expert search relevance evaluator. Your task is to score how RELEVANT a product is to a user query on a scale from 0.0 to 1.0.

IMPORTANT: Focus on what the product DOES have, not what it doesn't have. Give credit for partial matches.

[User Profile]
{persona_context}

[Key Guidelines]
1. If a product matches the main intent (brand + category + compatibility), it is RELEVANT even if some details are missing.
2. For die-cut products, piece counts and specific item lists are often NOT in titles - absence is not evidence of mismatch.
3. Partial match is still a match - give credit for what IS present.
4. Ignore conflicting preferences - they do not reduce relevance.

[Scoring Rules]
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met (at least brand + category + compatibility)
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met

Query: "{query}"
Product Info:
{doc_text}

Output ONLY a single floating point number between 0.0 and 1.0. No explanation needed.

Score:'''
        else:
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

Output ONLY a single floating point number between 0.0 and 1.0. No explanation needed.

Score:'''

        response = self.llm_client.call(prompt, max_tokens=768, temperature=0.0)
        
        # DEBUG: Log raw response
        log_with_timestamp(f"[DEBUG-PERSONALIZED] Raw LLM response: '{response}'")
        
        # Look for "Final Score: X.X" pattern first
        match = re.search(r'Final Score:\s*(0\.\d+|1\.0)', response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            log_with_timestamp(f"[DEBUG-PERSONALIZED] Final Score matched: '{match.group(1)}' → score={score}")
            return max(0.0, min(1.0, score))
        
        # Fallback: parse any number in response
        match = re.search(r'0\.\d+|1\.0', response)
        if match:
            score = float(match.group())
            log_with_timestamp(f"[DEBUG-PERSONALIZED] Regex matched: '{match.group()}' → score={score}")
            return max(0.0, min(1.0, score))
        
        # DEBUG: Log parse failure
        log_with_timestamp(f"[DEBUG-PERSONALIZED] ❌ PARSE FAILED - regex did not match any pattern in: '{response}'")
        return 0.0


def main():
    log_with_timestamp("=" * 50)
    log_with_timestamp(f"GLM-4.5V Reranker Evaluation (Standard + Personalized)")
    log_with_timestamp(f"User: {USER_ID}")
    log_with_timestamp("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(QUERY_FILE, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    all_asins = set()
    target_queries, mass_queries = [], []

    for r in results:
        asin = r.get('asin', '')
        if asin:
            all_asins.add(asin)
            tq = r.get('target_user_query', {})
            if tq.get('query'):
                target_queries.append({'asin': asin, 'query': tq['query'], 'type': 'target',
                    'category': r.get('category', ''), 'selected_attributes': tq.get('selected_attributes', [])})
            mq = r.get('mass_market_query', {})
            if mq.get('query'):
                mass_queries.append({'asin': asin, 'query': mq['query'], 'type': 'mass',
                    'category': r.get('category', ''), 'selected_attributes': mq.get('selected_attributes', [])})

    log_with_timestamp(f"ASINs: {len(all_asins)}, Target: {len(target_queries)}, Mass: {len(mass_queries)}")

    # Load products - use preprocessed data if available
    products_file = os.path.join(CACHE_DIR, f"products_{CATEGORY}.pkl")
    if USE_PREPROCESSED and os.path.exists(products_file):
        log_with_timestamp("Loading preprocessed data from cache...")
        products, all_metadata = load_preprocessed_products(CACHE_DIR, CATEGORY, all_asins)
    else:
        log_with_timestamp("Loading from raw data files...")
        products, all_metadata = load_product_metadata(META_FILE, all_asins)
        products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=10)
        products = load_reviews_for_products(REVIEW_FILE, products, max_reviews_per_product=25)
        if os.path.exists(QA_FILE):
            products = load_qa_for_products(QA_FILE, products, max_qa_per_product=25)

    documents = [products.get(asin, {'asin': asin, 'title': '', 'brand': '', 'category': [],
        'feature': [], 'description': [], 'rank': '', 'also_buy': [], 'also_view': [], 'reviews': [], 'qa': []})
        for asin in all_asins]

    common_metadata = {'user_id': USER_ID, 'timestamp': datetime.now().isoformat(),
        'num_queries': len(target_queries) + len(mass_queries), 'num_documents': len(documents),
        'k_values': K_VALUES, 'model': GLM_MODEL, 'first_stage_model': FIRST_STAGE_MODEL}

    # Cache files for different first-stage models
    bm25_target_cache = os.path.join(OUTPUT_DIR, f"bm25_candidates_{USER_ID}_target.json")
    bm25_mass_cache = os.path.join(OUTPUT_DIR, f"bm25_candidates_{USER_ID}_mass.json")
    e5_target_cache = os.path.join(OUTPUT_DIR, f"e5_candidates_{USER_ID}_target.json")
    bge_target_cache = os.path.join(OUTPUT_DIR, f"bge_candidates_{USER_ID}_target.json")
    colbert_target_cache = os.path.join(OUTPUT_DIR, f"colbert_candidates_{USER_ID}_target.json")

    # Select first-stage retriever based on configuration
    if FIRST_STAGE_MODEL == "E5":
        first_stage_cache = e5_target_cache
        FirstStageRetriever = CachedE5
    elif FIRST_STAGE_MODEL == "BGE":
        first_stage_cache = bge_target_cache
        FirstStageRetriever = CachedBGE
    elif FIRST_STAGE_MODEL == "ColBERT":
        first_stage_cache = colbert_target_cache
        FirstStageRetriever = CachedColBERT
    else:
        first_stage_cache = bm25_target_cache
        FirstStageRetriever = CachedBM25

    log_with_timestamp("=" * 50)
    log_with_timestamp(f"=== Standard ({FIRST_STAGE_MODEL} -> GLM) ===")

    first_stage_std = FirstStageRetriever(first_stage_cache)
    first_stage_std.fit(documents, all_metadata)
    glm_std = GLMReRanker(first_stage_std, top_k=TOP_K_CANDIDATES, model=GLM_MODEL)
    glm_std.fit(documents, all_metadata)

    target_std = evaluate_retriever(glm_std, target_queries, list(all_asins), K_VALUES)

    # Skip mass query reranking - only evaluate target queries
    # bm25_mass = CachedBM25(bm25_mass_cache)
    # bm25_mass.fit(documents, all_metadata)
    # glm_std_mass = GLMReRanker(bm25_mass, top_k=TOP_K_CANDIDATES, model=GLM_MODEL)
    # glm_std_mass.fit(documents, all_metadata)
    # mass_std = evaluate_retriever(glm_std_mass, mass_queries, list(all_asins), K_VALUES)

    for metrics, qtype, suffix in [(target_std, 'target_user', 'target')]:
        out = {**common_metadata, 'query_type': qtype, 'retriever': 'glm_reranker', 'personalized': False, 'metrics': metrics}
        with open(os.path.join(OUTPUT_DIR, f"retrieval_glm_{GLM_MODEL}_{FIRST_STAGE_MODEL}_{suffix}_{USER_ID}.json"), 'w') as f:
            json.dump(out, f, indent=2)
    log_with_timestamp(f"Standard Target: {target_std}")
    # log_with_timestamp(f"Standard Mass: {mass_std}")

    log_with_timestamp("=" * 50)
    log_with_timestamp(f"=== Personalized ({FIRST_STAGE_MODEL} -> GLM + Persona) ===")

    first_stage_per = FirstStageRetriever(first_stage_cache)
    first_stage_per.fit(documents, all_metadata)
    glm_per = PersonalizedGLMReRanker(first_stage_per, top_k=TOP_K_CANDIDATES, 
                                       persona_dir=PERSONA_DIR, processing_dir=PROCESSING_DIR, 
                                       model=GLM_MODEL)
    glm_per.fit(documents, all_metadata, queries=target_queries, user_id=USER_ID)

    target_per = evaluate_retriever(glm_per, target_queries, list(all_asins), K_VALUES)

    # Skip mass query reranking - only evaluate target queries
    # bm25_per_mass = CachedBM25(bm25_mass_cache)
    # bm25_per_mass.fit(documents, all_metadata)
    # glm_per_mass = PersonalizedGLMReRanker(bm25_per_mass, top_k=TOP_K_CANDIDATES, persona_dir=PERSONA_DIR, model=GLM_MODEL)
    # glm_per_mass.fit(documents, all_metadata, queries=target_queries, user_id=USER_ID)
    # mass_per = evaluate_retriever(glm_per_mass, mass_queries, list(all_asins), K_VALUES)

    for metrics, qtype, suffix in [(target_per, 'target_user', 'personalized_target')]:
        out = {**common_metadata, 'query_type': qtype, 'retriever': 'glm_reranker', 'personalized': True, 'metrics': metrics}
        with open(os.path.join(OUTPUT_DIR, f"retrieval_glm_{GLM_MODEL}_{FIRST_STAGE_MODEL}_{suffix}_{USER_ID}.json"), 'w') as f:
            json.dump(out, f, indent=2)
    log_with_timestamp(f"Personalized Target: {target_per}")
    # log_with_timestamp(f"Personalized Mass: {mass_per}")

    log_with_timestamp("=" * 50)
    log_with_timestamp("Done!")


if __name__ == "__main__":
    main()
