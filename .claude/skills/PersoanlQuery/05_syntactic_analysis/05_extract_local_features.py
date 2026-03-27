#!/usr/bin/env python3
"""
Stage 6: Linguistic Profiling with Local Extractor

Fast local feature extraction using spaCy.
Extracts ONLY the 16 style features used in Stage 8 (Iterative Refinement).

This replaces the slow ProfilingUD web service with fast local processing.
"""

import os
import sys
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

# # 16 style features used in Stage 8 (Iterative Refinement) [COMMENTED OUT]
# STYLE_ONLY_16_FEATURES = {
#     "tokens_per_sent", "char_per_tok", "ttr_lemma_chunks_100",
#     "lexical_density", "upos_dist_NOUN", "upos_dist_VERB",
#     "upos_dist_ADJ", "upos_dist_ADV", "upos_dist_PRON",
#     "upos_dist_DET", "upos_dist_AUX",
#     "upos_dist_PART", "upos_dist_SCONJ", "upos_dist_CCONJ",
#     "upos_dist_ADP", "n_tokens"
# }

# (Python doesn't allow imports starting with numbers, so we use importlib)
import importlib.util
import os

stage7_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "07_iterative_refinement")
module_path = os.path.join(stage7_dir, "07_extract_sentence_level_features.py")

spec = importlib.util.spec_from_file_location("extract_features", module_path)
if spec and spec.loader:
    extract_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_module)
    SentenceLevelFeatureExtractor = extract_module.SentenceLevelFeatureExtractor
else:
    raise ImportError("Failed to load SentenceLevelFeatureExtractor")

# Create an alias for compatibility
LocalFeatureExtractor = SentenceLevelFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_user_reviews(reviews_file: str) -> Dict[str, List[Dict]]:
    """
    Load user reviews from JSON file or directory.

    If reviews_file is a directory, scan for reviews_*.json files.
    Expected format in each file:
       {"user_id": "...", "results": [{"target_review": "...", "other_reviews": [...]}, ...]}

    Args:
        reviews_file: Path to reviews JSON file or directory

    Returns:
        Dict mapping user_id to list of reviews
    """
    def normalize_text_entry(entry) -> Optional[str]:
        if isinstance(entry, str):
            text = entry.strip()
            return text or None
        if isinstance(entry, dict):
            for key in ("reviewText", "review_text", "text"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def collect_reviews_from_results(results: List[Dict]) -> List[Dict]:
        reviews: List[Dict] = []
        for item in results:
            if not isinstance(item, dict):
                continue

            target_single = normalize_text_entry(item.get("target_review", ""))
            if target_single:
                reviews.append({"reviewText": target_single})

            target_multi = item.get("target_reviews", [])
            if isinstance(target_multi, list):
                for target_text in target_multi:
                    normalized = normalize_text_entry(target_text)
                    if normalized:
                        reviews.append({"reviewText": normalized})

            other_reviews = item.get("other_reviews", [])
            if isinstance(other_reviews, list):
                for other in other_reviews:
                    normalized = normalize_text_entry(other)
                    if normalized:
                        reviews.append({"reviewText": normalized})

        return reviews

    user_reviews = {}
    reviews_dir = Path(reviews_file)

    # If it's a directory, scan for reviews_*.json files
    if reviews_dir.is_dir():
        for json_file in sorted(reviews_dir.glob("reviews_*.json")):
            user_id = json_file.stem.replace("reviews_", "")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            reviews = collect_reviews_from_results(data.get("results", []))
            user_reviews[user_id] = reviews
    else:
        # Single file
        with open(reviews_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                # Stage 0 format
                user_id = data.get("user_id", "unknown")
                reviews = collect_reviews_from_results(data.get("results", []))
                user_reviews[user_id] = reviews
            else:
                # Dict format
                for uid, user_data in data.items():
                    if isinstance(user_data, dict):
                        reviews = []
                        for review in user_data.get("reviews", []):
                            normalized = normalize_text_entry(review)
                            if normalized:
                                reviews.append({"reviewText": normalized})
                        user_reviews[uid] = reviews
        elif isinstance(data, list):
            for user_data in data:
                user_id = user_data.get("user_id")
                if user_id:
                    reviews = []
                    for review in user_data.get("reviews", []):
                        normalized = normalize_text_entry(review)
                        if normalized:
                            reviews.append({"reviewText": normalized})
                    user_reviews[user_id] = reviews

    total_reviews = sum(len(reviews) for reviews in user_reviews.values())
    logger.info(f"Loaded {total_reviews} reviews for {len(user_reviews)} users")
    return user_reviews


# ============ Multiprocessing worker for dependency extraction ============
import re
import multiprocessing as mp

# Global variables for worker processes
_worker_nlp = None

def _init_worker():
    """Initialize spaCy model in each worker process."""
    global _worker_nlp
    import spacy
    _worker_nlp = spacy.load('en_core_web_sm')

def _process_user_mp(item):
    """Process a single user's reviews (for multiprocessing)."""
    user_id, reviews = item
    global _worker_nlp
    # Create a simple extractor-like object
    class SimpleExtractor:
        nlp = _worker_nlp
    simple_extractor = SimpleExtractor()
    profile = extract_user_profile(user_id, reviews, simple_extractor, None, spacy_n_process=1, spacy_batch_size=32)
    return user_id, profile


def filter_long_sentences(text: str, min_words: int = 25) -> str:
    """
    Pre-filter sentences by word count before spaCy parsing.
    Only keep sentences with >= min_words words.
    This avoids parsing short sentences that won't be used.
    """
    # Simple sentence splitting using regex (faster than spaCy for this task)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    long_sents = [s for s in sentences if len(s.split()) >= min_words]
    return ' '.join(long_sents)


def extract_dependency_structures_from_doc(doc, min_token_length: int = 25) -> Dict:
    """
    Extract comprehensive linguistic features from a single text.
    Only considers sentences with >= min_token_length tokens.

    Features extracted:
    1. Length features: avg sentence length, token count, review length
    2. Syntactic complexity: dependency depth, subordinate clause count/ratio
    3. Structure type: relative clause, passive, participial, infinitive, appositive, parenthetical, PP ratios
    4. Modifier density: adjective, adverb, noun modifier densities
    5. Organization: coordination, clause chaining, insertion frequency, linearity

    Returns:
        Dict with all linguistic features.
    """
    if len(doc) == 0:
        return {}

    # Token-level tokenization for length features
    n_tokens = len(doc)

    # Dependency relation distribution (for complexity_score compatibility)
    dep_counts = defaultdict(int)
    for token in doc:
        dep_counts[token.dep_] += 1
    dep_raw_counts = dict(dep_counts)

    # Dependency relation types
    RELATIVE_CLAUSE_DEPS = {"relcl", "acl:relcl", "rcmod"}
    PASSIVE_DEPS = {"nsubjpass", "auxpass", "agent"}
    PARTICIPIAL_DEPS = {"vbg", "vbn"}  # gerund and past participle
    INFINITIVE_DEPS = {"aux", "auxpass", "inf"}  # infinitive markers
    APPOSITIVE_DEPS = {"appos", "parataxis"}
    PARENTHETICAL_DEPS = {"cc", "prep"}
    PREP_PHRASE_DEPS = {"prep", "pobj", "pcomp"}
    COORDINATION_DEPS = {"cc", "conj"}
    NOUN_MODIFIER_DEPS = {"compound", "nn", "amod", "nummod"}

    # Deps for depth calculation
    CLAUSAL_DEPS = {"ccomp", "xcomp", "advcl", "acl", "acl:relcl", "relcl", "csubj", "csubjpass", "ccomp"}

    def token_depth(token):
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth

    # ========================================
    # Filter sentences by minimum token length
    # ========================================
    all_sents = list(doc.sents)
    total_sents = len(all_sents)
    long_sents = [sent for sent in all_sents if len(sent) >= min_token_length]

    if not long_sents:
        # No sentences meet the threshold - return minimal features
        return {
            "n_tokens": n_tokens,
            "n_long_sentences": 0,
            "long_sent_ratio": 0.0,
            "mean_sent_length": 0.0,
            "mean_token_count": 0.0,
            "mean_review_length": 0.0,
            "avg_dependency_depth": 0.0,
            "max_dependency_depth": 0,
            "avg_subordinate_clause_count": 0.0,
            "subordinate_ratio": 0.0,
            "relative_clause_ratio": 0.0,
            "passive_ratio": 0.0,
            "participial_ratio": 0.0,
            "infinitive_ratio": 0.0,
            "appositive_ratio": 0.0,
            "parenthetical_ratio": 0.0,
            "prep_phrase_ratio": 0.0,
            "adj_density": 0.0,
            "adv_density": 0.0,
            "noun_modifier_density": 0.0,
            "coordination_ratio": 0.0,
            "clause_chaining_ratio": 0.0,
            "insertion_frequency": 0.0,
            "linearity_degree": 0.0,
            "_dep_raw_counts": dep_raw_counts,
        }

    # ========================================
    # 1. LENGTH FEATURES (from long sentences only)
    # ========================================
    sent_lengths = [len(sent) for sent in long_sents]
    mean_sent_length = sum(sent_lengths) / len(sent_lengths)
    mean_token_count = n_tokens  # total tokens in doc
    mean_review_length = n_tokens / len(long_sents) if long_sents else 0.0

    # ========================================
    # 2. SYNTACTIC COMPLEXITY FEATURES
    # ========================================
    all_depths = []
    all_subordinate_counts = []
    total_subordinate = 0

    for sent in long_sents:
        sent_depths = [token_depth(tok) for tok in sent]
        all_depths.extend(sent_depths)

        # Count subordinate clauses in this sentence
        sub_count = sum(1 for tok in sent if tok.dep_ in CLAUSAL_DEPS)
        all_subordinate_counts.append(sub_count)
        total_subordinate += sub_count

    avg_dependency_depth = sum(all_depths) / len(all_depths) if all_depths else 0.0
    max_dependency_depth = max(all_depths) if all_depths else 0
    avg_subordinate_clause_count = sum(all_subordinate_counts) / len(all_subordinate_counts) if all_subordinate_counts else 0.0
    subordinate_ratio = total_subordinate / n_tokens if n_tokens > 0 else 0.0

    # ========================================
    # 3. STRUCTURE TYPE FEATURES (ratios)
    # ========================================
    n_relative = sum(1 for tok in doc if tok.dep_ in RELATIVE_CLAUSE_DEPS)
    n_passive = sum(1 for tok in doc if tok.dep_ in PASSIVE_DEPS)
    n_participial = sum(1 for tok in doc if tok.dep_ in PARTICIPIAL_DEPS or tok.tag_ in {"VBG", "VBN"})
    n_infinitive = sum(1 for tok in doc if tok.dep_ in INFINITIVE_DEPS or tok.tag_ in {"VB", "TO"})
    n_appositive = sum(1 for tok in doc if tok.dep_ in APPOSITIVE_DEPS)
    n_parenthetical = sum(1 for tok in doc if tok.dep_ in PARENTHETICAL_DEPS)
    n_prep_phrase = sum(1 for tok in doc if tok.dep_ in PREP_PHRASE_DEPS)

    relative_clause_ratio = n_relative / n_tokens if n_tokens > 0 else 0.0
    passive_ratio = n_passive / n_tokens if n_tokens > 0 else 0.0
    participial_ratio = n_participial / n_tokens if n_tokens > 0 else 0.0
    infinitive_ratio = n_infinitive / n_tokens if n_tokens > 0 else 0.0
    appositive_ratio = n_appositive / n_tokens if n_tokens > 0 else 0.0
    parenthetical_ratio = n_parenthetical / n_tokens if n_tokens > 0 else 0.0
    prep_phrase_ratio = n_prep_phrase / n_tokens if n_tokens > 0 else 0.0

    # ========================================
    # 4. MODIFIER DENSITY FEATURES
    # ========================================
    n_adjectives = sum(1 for tok in doc if tok.pos_ == "ADJ")
    n_adverbs = sum(1 for tok in doc if tok.pos_ == "ADV")
    n_noun_modifiers = sum(1 for tok in doc if tok.dep_ in NOUN_MODIFIER_DEPS)

    adj_density = n_adjectives / n_tokens if n_tokens > 0 else 0.0
    adv_density = n_adverbs / n_tokens if n_tokens > 0 else 0.0
    noun_modifier_density = n_noun_modifiers / n_tokens if n_tokens > 0 else 0.0

    # ========================================
    # 5. ORGANIZATION FEATURES
    # ========================================
    n_coordination = sum(1 for tok in doc if tok.dep_ in COORDINATION_DEPS)
    coordination_ratio = n_coordination / n_tokens if n_tokens > 0 else 0.0

    # Clause chaining: sequences of clauses connected by conj
    clause_chaining_count = 0
    for sent in long_sents:
        clause_tags = [tok.dep_ for tok in sent]
        for i in range(len(clause_tags) - 1):
            if clause_tags[i] == "conj" and clause_tags[i+1] == "conj":
                clause_chaining_count += 1
    clause_chaining_ratio = clause_chaining_count / len(long_sents) if long_sents else 0.0

    # Insertion frequency: adverbial clauses (advcl)
    n_insertions = sum(1 for tok in doc if tok.dep_ == "advcl")
    insertion_frequency = n_insertions / len(long_sents) if long_sents else 0.0

    # Linearity degree: average distance between related tokens / expected distance
    # High linearity = tokens are close to their heads
    dep_distances = [abs(tok.i - tok.head.i) for tok in doc if tok.dep_ != "ROOT" and tok.dep_ != "punct"]
    avg_linearity = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    # Normalize: lower distance = higher linearity
    linearity_degree = 1.0 / (1.0 + avg_linearity) if avg_linearity >= 0 else 1.0

    # ========================================
    # PRECOMPUTE for complexity_score (avoid re-traversal in complexity_score)
    # ========================================
    # Max clause nesting depth
    clausal_set = {"advcl", "acl", "acl:relcl", "relcl", "ccomp", "xcomp", "csubj"}
    clause_nesting = 0
    for tok in doc:
        if tok.dep_ not in clausal_set:
            continue
        cur = tok
        depth = 0
        while cur.head != cur:
            if cur.dep_ in clausal_set:
                depth += 1
            cur = cur.head
        if depth > clause_nesting:
            clause_nesting = depth

    # Clausal count
    clausal_count = sum(1 for t in doc if t.dep_ in clausal_set)

    # Max conj chain depth
    conj_chain = 0
    for tok in doc:
        if tok.dep_ != "conj":
            continue
        cur = tok
        depth = 0
        while cur.head != cur:
            if cur.dep_ == "conj":
                depth += 1
            cur = cur.head
        if depth > conj_chain:
            conj_chain = depth

    # Double negation count
    double_neg = 0
    for tok in doc:
        if tok.dep_ == "neg":
            neg_children = sum(1 for ch in tok.head.children if ch.dep_ == "neg")
            if neg_children > 1:
                double_neg += (neg_children - 1)

    # Max negation scope depth
    neg_scope = 0
    for tok in doc:
        if tok.dep_ != "neg":
            continue
        cur = tok.head
        depth = 0
        while cur.dep_ != "ROOT" and cur.dep_ != "punct":
            depth += 1
            cur = cur.head
        if depth > neg_scope:
            neg_scope = depth

    # Max token depth
    max_depth = max(all_depths) if all_depths else 0

    # Sentence count and non-punct tokens
    n_sent = len(all_sents)
    non_punct_tokens = sum(1 for t in doc if t.dep_ != "punct")

    # ========================================
    # Aggregate features
    # ========================================
    features = {
        "n_tokens": n_tokens,
        "n_long_sentences": len(long_sents),
        "long_sent_ratio": len(long_sents) / total_sents if total_sents > 0 else 0.0,
        # Length features
        "mean_sent_length": mean_sent_length,
        "mean_token_count": mean_token_count,
        "mean_review_length": mean_review_length,
        # Syntactic complexity
        "avg_dependency_depth": avg_dependency_depth,
        "max_dependency_depth": max_dependency_depth,
        "avg_subordinate_clause_count": avg_subordinate_clause_count,
        "subordinate_ratio": subordinate_ratio,
        # Structure types
        "relative_clause_ratio": relative_clause_ratio,
        "passive_ratio": passive_ratio,
        "participial_ratio": participial_ratio,
        "infinitive_ratio": infinitive_ratio,
        "appositive_ratio": appositive_ratio,
        "parenthetical_ratio": parenthetical_ratio,
        "prep_phrase_ratio": prep_phrase_ratio,
        # Modifier density
        "adj_density": adj_density,
        "adv_density": adv_density,
        "noun_modifier_density": noun_modifier_density,
        # Organization
        "coordination_ratio": coordination_ratio,
        "clause_chaining_ratio": clause_chaining_ratio,
        "insertion_frequency": insertion_frequency,
        "linearity_degree": linearity_degree,
        # Raw dep counts for complexity_score
        "_dep_raw_counts": dep_raw_counts,
        # Precomputed values for complexity_score (to avoid re-traversal)
        "_precomputed": {
            "clause_nesting": clause_nesting,
            "clausal_count": clausal_count,
            "conj_chain": conj_chain,
            "double_neg": double_neg,
            "neg_scope": neg_scope,
            "max_depth": max_depth,
            "n_tokens": n_tokens,
            "n_sent": n_sent,
            "non_punct_tokens": non_punct_tokens,
        },
    }

    return features


def extract_dependency_structures(nlp, text: str, min_token_length: int = 25) -> Dict:
    doc = nlp(text)
    return extract_dependency_structures_from_doc(doc, min_token_length=min_token_length)


def complexity_level(doc) -> str:
    """
    基于规则判断单个句子(SPaCy Span/Doc)的复杂度级别。

    规则:
    - High: advcl/acl 比例 ≥ 15%, 或 从句嵌套深度 ≥ 2, 或 否定路径深度 ≥ 2, 或 并列层级 ≥ 2
    - Low:  基本依存占比 ≥ 80% 且 无 mark (无从属连词)
    - Medium: 其余情况
    """
    n = len(doc)
    if n == 0:
        return 'Medium'

    BASIC_RELS = {'nsubj', 'obj', 'amod', 'det', 'punct',
                  'ROOT', 'compound', 'advmod', 'prep', 'pobj'}
    CLAUSAL_RELS = {'advcl', 'acl', 'acl:relcl', 'relcl'}

    def conj_depth(token):
        depth = 0
        cur = token
        while cur.head != cur:
            if cur.dep_ == 'conj':
                depth += 1
            cur = cur.head
        return depth

    def neg_depth_from_neg(token):
        cur = token.head
        depth = 0
        while cur.dep_ != 'ROOT' and cur.dep_ != 'ROOT':
            depth += 1
            cur = cur.head
        return depth

    def clause_nesting_depth(token):
        depth = 0
        cur = token
        while cur.head != cur:
            if cur.dep_ in {'advcl', 'acl', 'acl:relcl', 'ccomp', 'xcomp', 'relcl'}:
                depth += 1
            cur = cur.head
        return depth

    # 指标计算
    advcl_acl_ratio = sum(1 for t in doc if t.dep_ in CLAUSAL_RELS) / n
    has_mark = any(t.dep_ == 'mark' for t in doc)
    max_cd = max((conj_depth(t) for t in doc if t.dep_ == 'conj'), default=0)

    max_clause = 0
    for t in doc:
        if t.dep_ in {'advcl', 'acl', 'acl:relcl'}:
            d = clause_nesting_depth(t)
            if d > max_clause:
                max_clause = d

    max_neg = 0
    for t in doc:
        if t.dep_ == 'neg':
            d = neg_depth_from_neg(t)
            if d > max_neg:
                max_neg = d

    # High 判断
    if (advcl_acl_ratio >= 0.15 or
            max_clause >= 2 or
            max_neg >= 2 or
            max_cd >= 2):
        return 'High'

    # Low 判断
    total_non_punct = sum(1 for t in doc if t.dep_ != 'punct')
    basic_count = sum(1 for t in doc if t.dep_ in BASIC_RELS)
    basic_ratio = basic_count / total_non_punct if total_non_punct > 0 else 0

    if basic_ratio >= 0.8 and not has_mark:
        return 'Low'

    return 'Medium'


def extract_complexity_levels(docs: list) -> dict:
    """
    对多段文本逐句计算复杂度级别，统计 Low/Medium/High 分布。

    Returns:
        dict: {
            'sentence_counts': {'low': N, 'medium': M, 'high': K},
            'sentence_levels': [level per sentence],
            'review_levels': [most_common_level per review],
        }
    """
    from collections import Counter
    review_levels = []
    all_sent_levels = []

    for doc in docs:
        review_sent_levels = []
        for sent in doc.sents:
            level = complexity_level(sent)
            review_sent_levels.append(level)
            all_sent_levels.append(level)
        if review_sent_levels:
            review_levels.append(Counter(review_sent_levels).most_common(1)[0][0])

    counter = Counter(all_sent_levels)
    return {
        'sentence_counts': {
            'low': counter.get('Low', 0),
            'medium': counter.get('Medium', 0),
            'high': counter.get('High', 0),
        },
        'sentence_levels': all_sent_levels,
        'review_levels': review_levels,
    }


def extract_user_profile(
    user_id: str,
    reviews: List[Dict],
    extractor: LocalFeatureExtractor,
    max_reviews: Optional[int] = None,
    spacy_n_process: int = 1,
    spacy_batch_size: int = 32,
) -> Dict:
    """
    Extract dependency-relation-based linguistic profile from user's reviews.

    Args:
        user_id: User ID
        reviews: List of review dicts with 'reviewText' field
        extractor: Local feature extractor (used for its spaCy nlp model)
        max_reviews: Maximum number of reviews to process

    Returns:
        Linguistic profile dict with dependency structure features
    """
    if max_reviews:
        reviews = reviews[:max_reviews]

    # Extract review texts and pre-filter to keep only long sentences (>=25 words)
    # This avoids parsing short sentences that won't be used for features
    texts = []
    for review in reviews:
        text = review.get("reviewText", "")
        if text and text.strip():
            filtered_text = filter_long_sentences(text.strip(), min_words=25)
            # Only add if it has content after filtering
            if filtered_text.strip():
                texts.append(filtered_text)

    if not texts:
        logger.warning(f"No valid texts with >=25 word sentences found for user {user_id}")
        return {}

    # --- OLD: extract 16 style features ---
    # all_features = []
    # for text in texts:
    #     features = extractor.extract_profilingud_features(text)
    #     if features:
    #         all_features.append(features)
    #
    # if not all_features:
    #     logger.warning(f"No features extracted for user {user_id}")
    #     return {}
    #
    # aggregated = {}
    # feature_names = all_features[0].keys()
    # for name in feature_names:
    #     values = [f.get(name, 0.0) for f in all_features if name in f]
    #     if values:
    #         aggregated[name] = sum(values) / len(values)
    #
    # filtered_features = {k: v for k, v in aggregated.items() if k in STYLE_ONLY_16_FEATURES}
    # --- END OLD ---

    # --- NEW: extract dependency relation structures ---
    nlp = extractor.nlp  # reuse spaCy model from the extractor
    # Note: if using multiprocessing, set n_process=1 to avoid nested processes
    docs = list(nlp.pipe(texts, batch_size=32, n_process=spacy_n_process))

    all_dep_features = []
    reviews_with_scores = []
    axis_components_list = []
    for text, doc in zip(texts, docs):
        dep_feats = extract_dependency_structures_from_doc(doc)
        if dep_feats:
            all_dep_features.append(dep_feats)
            dep_counts = dep_feats.get('_dep_raw_counts', {})
            precomputed = dep_feats.get('_precomputed', {})
            score_result = complexity_score(dep_counts, doc=None, return_components=True, precomputed=precomputed)
            if isinstance(score_result, tuple):
                score, axis_components = score_result
            else:
                score = float(score_result)
                axis_components = {
                    "subordination": 0.0,
                    "coordination": 0.0,
                    "negation": 0.0,
                    "length_depth": 0.0,
                }
            axis_components_list.append(axis_components)
            reviews_with_scores.append((text, dep_counts, score, doc))

    if not all_dep_features:
        logger.warning(f"No dependency features extracted for user {user_id}")
        return {}

    # Aggregate: average numeric features across reviews
    all_keys = set()
    for f in all_dep_features:
        all_keys.update(f.keys())

    aggregated_dep = {}
    for key in all_keys:
        values = [f[key] for f in all_dep_features if key in f and isinstance(f[key], (int, float))]
        if values:
            aggregated_dep[key] = sum(values) / len(values)

    # --- Complexity scoring & binning ---
    scores = [rws[2] for rws in reviews_with_scores]
    low_threshold, high_threshold = score_binning(scores)
    complexity_templates = select_representative_template(reviews_with_scores, low_threshold, high_threshold)

    # --- Rule-based complexity level distribution ---
    complexity_dist = extract_complexity_levels(docs)

    complexity_axis_features = {}
    if axis_components_list:
        axis_keys = axis_components_list[0].keys()
        for key in axis_keys:
            vals = [item.get(key, 0.0) for item in axis_components_list]
            complexity_axis_features[key] = sum(vals) / len(vals)

    # Add metadata
    profile = {
        "user_id": user_id,
        "num_reviews": len(texts),
        "num_reviews_processed": len(all_dep_features),
        "dependency_features": aggregated_dep,
        "feature_count": len(aggregated_dep),
        "complexity_thresholds": {
            "low": low_threshold,
            "high": high_threshold,
        },
        "complexity_templates": complexity_templates,
        "complexity_rule_based": {
            "sentence_counts": complexity_dist['sentence_counts'],
            "total_sentences": sum(complexity_dist['sentence_counts'].values()),
        },
        "complexity_axis_features": complexity_axis_features,
        "extraction_method": "local_spacy_dependency",
        "extraction_date": datetime.now().isoformat(),
    }

    return profile


def save_profile(profile: Dict, output_dir: str):
    """
    Save user profile to JSON file.

    Args:
        profile: User profile dict
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    user_id = profile["user_id"]

    output_file = os.path.join(output_dir, f"linguistic_profile_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved profile for {user_id} to {output_file}")


def save_skeleton_profile(skeleton_profile: Dict, output_dir: str):
    """保存骨架提取结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    user_id = skeleton_profile["user_id"]
    output_file = os.path.join(output_dir, f"skeleton_profile_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(skeleton_profile, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved skeleton profile for {user_id} to {output_file}")


def extract_user_skeletons(
    user_id: str,
    reviews: List[Dict],
    extractor: LocalFeatureExtractor,
    max_reviews: Optional[int] = None,
    spacy_n_process: int = 1,
    spacy_batch_size: int = 32,
) -> Dict:
    """提取用户的典型句子骨架"""
    if max_reviews:
        reviews = reviews[:max_reviews]

    texts = []
    for review in reviews:
        text = review.get("reviewText", "")
        if text and text.strip():
            texts.append(text.strip())

    if not texts:
        logger.warning(f"No valid texts found for user {user_id}")
        return {}

    nlp = extractor.nlp
    result = extract_user_sentence_skeletons(
        nlp,
        texts,
        top_k=1,
        spacy_n_process=spacy_n_process,
        spacy_batch_size=spacy_batch_size,
    )

    top_skeleton = result["skeletons"][0] if result["skeletons"] else ""
    top_frequency = result["frequency"].get(top_skeleton, 0) if top_skeleton else 0

    return {
        "user_id": user_id,
        "num_reviews": len(texts),
        "skeleton": top_skeleton,
        "skeleton_frequency": top_frequency,
        "total_unique_skeletons": result["total_skeletons"],
        "total_sentences": result["total_sentences"],
        "extraction_method": "dependency_skeleton",
        "extraction_date": datetime.now().isoformat(),
    }


def extract_user_abstract_skeletons(
    user_id: str,
    reviews: List[Dict],
    extractor: LocalFeatureExtractor,
    max_reviews: Optional[int] = None,
    min_sentence_length: int = 25,
    spacy_n_process: int = 1,
    spacy_batch_size: int = 32,
) -> Dict:
    """提取用户的抽象句子模板（仅从长句中提取）"""
    if max_reviews:
        reviews = reviews[:max_reviews]

    texts = []
    for review in reviews:
        text = review.get("reviewText", "")
        if text and text.strip():
            texts.append(text.strip())

    if not texts:
        logger.warning(f"No valid texts found for user {user_id}")
        return {}

    nlp = extractor.nlp
    abstract_counts = defaultdict(int)
    total_long_sentences = 0

    for doc in nlp.pipe(texts, batch_size=spacy_batch_size, n_process=spacy_n_process):
        for sent in doc.sents:
            sent_len = len(sent)
            if sent_len >= min_sentence_length:
                total_long_sentences += 1
                template = extract_abstract_template(sent)
                if template.strip():
                    abstract_counts[template] += 1

    sorted_templates = sorted(
        abstract_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_template = sorted_templates[0][0] if sorted_templates else ""
    top_frequency = sorted_templates[0][1] if sorted_templates else 0

    return {
        "user_id": user_id,
        "num_reviews": len(texts),
        "abstract_template": top_template,
        "template_frequency": top_frequency,
        "total_unique_templates": len(abstract_counts),
        "total_long_sentences": total_long_sentences,
        "extraction_method": "abstract_template",
        "extraction_date": datetime.now().isoformat(),
    }


def main_disentangle():
    """对已提取的画像进行解耦处理"""
    config = {
        "profiles_dir": "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis",
        "output_dir": "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/disentangled_profiles",
    }

    import spacy
    nlp = spacy.load('en_core_web_sm')

    profiles_dir = Path(config['profiles_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading profiles from {profiles_dir}")
    profile_files = list(profiles_dir.glob("linguistic_profile_*.json"))
    logger.info(f"Found {len(profile_files)} profiles")

    for profile_file in profile_files:
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)

        # 解耦
        disentangled = disentangle_user_profile(profile, nlp)

        # 保存
        output_file = output_dir / f"disentangled_{profile_file.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(disentangled, f, indent=2, ensure_ascii=False)

        logger.info(f"Disentangled: {profile_file.name}")

    logger.info(f"Done! Disentangled profiles saved to {output_dir}")


# ============================================================================
# 句法复杂度评分与分级
# ============================================================================

import numpy as np

COMPLEXITY_AXIS_WEIGHTS = {
    "subordination": 0.42,
    "coordination": 0.23,
    "negation": 0.25,
    "length_depth": 0.10,
}


def _token_depth(token) -> int:
    depth = 0
    cur = token
    while cur.head != cur:
        depth += 1
        cur = cur.head
    return depth


def _max_conj_chain_depth(doc) -> int:
    max_depth = 0
    for tok in doc:
        if tok.dep_ != "conj":
            continue
        cur = tok
        depth = 0
        while cur.head != cur:
            if cur.dep_ == "conj":
                depth += 1
            cur = cur.head
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _max_clause_nesting_depth(doc) -> int:
    clausal = {"advcl", "acl", "acl:relcl", "relcl", "ccomp", "xcomp", "csubj"}
    max_depth = 0
    for tok in doc:
        if tok.dep_ not in clausal:
            continue
        cur = tok
        depth = 0
        while cur.head != cur:
            if cur.dep_ in clausal:
                depth += 1
            cur = cur.head
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _double_negation_count(doc) -> int:
    count = 0
    for tok in doc:
        if tok.dep_ == "neg":
            neg_children = sum(1 for ch in tok.head.children if ch.dep_ == "neg")
            if neg_children > 1:
                count += (neg_children - 1)
    return count


def _max_neg_scope_depth(doc) -> int:
    max_depth = 0
    for tok in doc:
        if tok.dep_ != "neg":
            continue
        depth = _token_depth(tok.head)
        if depth > max_depth:
            max_depth = depth
    return max_depth


def complexity_score(dep_counts: dict, doc=None, return_components: bool = False,
                    precomputed: dict = None):
    """
    Compute complexity score for a document.

    Args:
        dep_counts: Dependency relation counts
        doc: spaCy doc object (optional, used if precomputed is None)
        return_components: Whether to return axis components
        precomputed: Precomputed values from extract_dependency_structures_from_doc to avoid re-traversal
    """
    if precomputed is not None:
        # Use precomputed values to avoid re-traversing doc
        clause_nesting = precomputed.get("clause_nesting", 0)
        clausal_count = precomputed.get("clausal_count", 0)
        conj_chain = precomputed.get("conj_chain", 0)
        double_neg = precomputed.get("double_neg", 0)
        neg_scope = precomputed.get("neg_scope", 0)
        max_depth = precomputed.get("max_depth", 0)
        n_tokens = max(precomputed.get("n_tokens", 1), 1)
        n_sent = max(precomputed.get("n_sent", 1), 1)
        non_punct_tokens = precomputed.get("non_punct_tokens", n_tokens)

        subordination_raw = 3.0 * (clausal_count / n_tokens) + 0.9 * clause_nesting

        conj_count = dep_counts.get("conj", 0)
        coordination_raw = 1.1 * (conj_count / n_tokens) + 0.8 * conj_chain

        neg_count = dep_counts.get("neg", 0)
        negation_raw = 2.2 * (neg_count / n_tokens) + 1.4 * double_neg + 0.12 * neg_scope

        avg_sent_len = non_punct_tokens / n_sent if n_sent > 0 else 0.0
        length_depth_raw = 0.05 * avg_sent_len + 0.08 * max_depth
    elif doc is None:
        subordination_raw = dep_counts.get("advcl", 0) + dep_counts.get("acl", 0) + dep_counts.get("ccomp", 0)
        coordination_raw = dep_counts.get("conj", 0)
        negation_raw = dep_counts.get("neg", 0)
        length_depth_raw = 0.0
    else:
        n_tokens = max(len(doc), 1)
        n_sent = max(sum(1 for _ in doc.sents), 1)

        clausal_deps = {"advcl", "acl", "acl:relcl", "relcl", "ccomp", "xcomp", "csubj"}
        clausal_count = sum(1 for t in doc if t.dep_ in clausal_deps)
        clause_nesting = _max_clause_nesting_depth(doc)
        subordination_raw = 3.0 * (clausal_count / n_tokens) + 0.9 * clause_nesting

        conj_count = dep_counts.get("conj", 0)
        conj_chain = _max_conj_chain_depth(doc)
        coordination_raw = 1.1 * (conj_count / n_tokens) + 0.8 * conj_chain

        neg_count = dep_counts.get("neg", 0)
        double_neg = _double_negation_count(doc)
        neg_scope = _max_neg_scope_depth(doc)
        negation_raw = 2.2 * (neg_count / n_tokens) + 1.4 * double_neg + 0.12 * neg_scope

        non_punct_tokens = sum(1 for t in doc if t.dep_ != "punct")
        avg_sent_len = non_punct_tokens / n_sent if n_sent > 0 else 0.0
        max_depth = max((_token_depth(t) for t in doc), default=0)
        length_depth_raw = 0.05 * avg_sent_len + 0.08 * max_depth

    components = {
        "subordination": float(subordination_raw),
        "coordination": float(coordination_raw),
        "negation": float(negation_raw),
        "length_depth": float(length_depth_raw),
    }

    score = 0.0
    for axis, weight in COMPLEXITY_AXIS_WEIGHTS.items():
        score += components.get(axis, 0.0) * weight

    if return_components:
        return float(score), components
    return float(score)


def score_binning(scores: list) -> tuple:
    """
    按三分位数将得分划分为 Low/Medium/High 三组。

    Args:
        scores: 复杂度得分列表

    Returns:
        (low_threshold, high_threshold)，用于划分三组
    """
    if not scores:
        return 0.0, 0.0
    low_threshold = float(np.percentile(scores, 33))
    high_threshold = float(np.percentile(scores, 67))
    return low_threshold, high_threshold


def select_representative_template(reviews_with_scores: list, low_threshold: float, high_threshold: float) -> dict:
    """
    从每个复杂度分组中选取得分最接近该组中位数的评论作为代表性模板。

    Args:
        reviews_with_scores: list of (review_text, dep_counts, score, doc) tuples
        low_threshold: 三分位 low 阈值
        high_threshold: 三分位 high 阈值

    Returns:
        dict with keys 'low', 'medium', 'high', each containing:
        - 'template': 依存关系 bigram 集合（用于区分度计算）
        - 'skeleton_template': 骨架 bigram 集合（用于属性填充，有语义槽位）
        - 'review_text': 原始评论文本
        - 'score': 该评论的复杂度得分
        - 'median_score': 该分组的中位数得分
    """
    # 分组
    low_reviews = []
    medium_reviews = []
    high_reviews = []

    for review_text, dep_counts, score, doc in reviews_with_scores:
        if score < low_threshold:
            low_reviews.append((review_text, dep_counts, score, doc))
        elif score < high_threshold:
            medium_reviews.append((review_text, dep_counts, score, doc))
        else:
            high_reviews.append((review_text, dep_counts, score, doc))

    def pick_median(reviews_list: list) -> dict:
        """从列表中选取得分最接近中位数的评论"""
        if not reviews_list:
            return {'template': None, 'skeleton_template': None, 'raw_skeleton': None, 'review_text': None, 'score': None, 'median_score': None}

        scores_only = [r[2] for r in reviews_list]
        median = float(np.median(scores_only))

        # 选得分最接近中位数的
        best = min(reviews_list, key=lambda r: abs(r[2] - median))
        review_text, dep_counts, score, doc = best

        # 1. dep bigram（用于区分度计算）
        seen = set()
        label_list = []
        for dep_type in dep_counts.keys():
            if dep_type not in seen:
                seen.add(dep_type)
                label_list.append(dep_type)
        bigrams = set(f"{label_list[i]}|{label_list[i+1]}" for i in range(len(label_list) - 1))
        template = " ".join(sorted(bigrams))

        # 2. skeleton bigram（用于属性填充 — 有语义槽位）
        # 3. raw skeleton（用于 fill_skeleton_with_attributes，直接填属性）
        skeleton_bigrams = set()
        raw_skeleton = None
        best_level = None
        best_len = 0
        for sent in doc.sents:
            skel = extract_sentence_skeleton(sent)
            level = complexity_level(sent)
            skel_parts = skel.split()
            for i in range(len(skel_parts) - 1):
                p1, p2 = skel_parts[i], skel_parts[i + 1]
                if p1 in SKELETON_PLACEHOLDERS and p2 in SKELETON_PLACEHOLDERS:
                    skeleton_bigrams.add(f"{p1}|{p2}")
            # 优先选 HIGH 句子，其次最长的
            if (best_level is None or
                (level == 'High' and best_level != 'High') or
                (level == best_level and len(skel_parts) > best_len)):
                raw_skeleton = skel
                best_level = level
                best_len = len(skel_parts)
        skeleton_template = " ".join(sorted(skeleton_bigrams)) if skeleton_bigrams else None

        return {
            'template': template,
            'skeleton_template': skeleton_template,
            'raw_skeleton': raw_skeleton,
            'review_text': review_text,
            'score': score,
            'median_score': median,
        }

    return {
        'low': pick_median(low_reviews),
        'medium': pick_median(medium_reviews),
        'high': pick_median(high_reviews),
    }


# ============================================================================
# 句子骨架提取功能
# ============================================================================

# 停用词集合（保留小写形式）
STOP_WORDS = {"i", "am", "looking", "for", "a", "an", "the", "to", "and", "or", "with", "in", "on", "of"}

# 依存关系到骨架标签的映射
DEP_TO_SKELETON = {
    "nsubj": "SUBJ",
    "nsubj:pass": "SUBJ",
    "dobj": "OBJ",
    "pobj": "OBJ",
    "attr": "OBJ",
    "amod": "AMOD",
    "advmod": "ADV",
    "prep": "PREP",
    "det": "DET",
    "ROOT": "ROOT",
}

# 骨架占位符集合
SKELETON_PLACEHOLDERS = {"SUBJ", "OBJ", "AMOD", "ADV", "PREP", "DET", "ROOT"}


def extract_sentence_skeleton(doc) -> str:
    """
    提取单个句子的骨架（纯结构版本）。

    骨架规则:
    - nsubj, nsubj:pass → "SUBJ"
    - dobj, pobj, attr → "OBJ"
    - amod → "AMOD"
    - advmod → "ADV"
    - prep → "PREP"
    - det → "DET"
    - ROOT → "ROOT"
    - 停用词 → 保留小写
    - 其他内容词 → 跳过（过滤掉）

    Args:
        doc: spaCy Doc对象（单句）

    Returns:
        句子骨架字符串

    示例:
        "I am looking for reflective yarn to make a scarf"
        → "SUBJ am looking for AMOD yarn to make DET scarf"
        （"reflective" 和 "scarf" 等内容词被过滤，只保留结构）
    """
    skeleton_parts = []

    for token in doc:
        if token.is_punct:
            continue
        dep = token.dep_
        if dep in DEP_TO_SKELETON:
            skeleton_parts.append(DEP_TO_SKELETON[dep])
        elif token.text.lower() in STOP_WORDS:
            skeleton_parts.append(token.text.lower())

    return " ".join(skeleton_parts)


def extract_user_sentence_skeletons(
    nlp,
    texts: List[str],
    top_k: int = 5,
    spacy_n_process: int = 1,
    spacy_batch_size: int = 32,
) -> Dict:
    """
    从用户文本列表中提取典型句子骨架。

    Args:
        nlp: spaCy nlp模型
        texts: 用户文本列表
        top_k: 返回最常见的骨架数量

    Returns:
        Dict包含:
        - skeletons: 骨架列表（按频率排序）
        - frequency: 骨架出现频率统计
    """
    skeleton_counts = defaultdict(int)

    cleaned_texts = [text.strip() for text in texts if text and text.strip()]
    for doc in nlp.pipe(cleaned_texts, batch_size=spacy_batch_size, n_process=spacy_n_process):

        # 遍历每个句子
        for sent in doc.sents:
            skeleton = extract_sentence_skeleton(sent)
            if skeleton.strip():  # 忽略空骨架
                skeleton_counts[skeleton] += 1

    # 按频率排序
    sorted_skeletons = sorted(
        skeleton_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 返回top_k和统计信息
    top_skeletons = [s[0] for s in sorted_skeletons[:top_k]]
    frequency = {s: c for s, c in sorted_skeletons[:top_k]}

    return {
        "skeletons": top_skeletons,
        "frequency": frequency,
        "total_skeletons": len(skeleton_counts),
        "total_sentences": sum(skeleton_counts.values()),
    }


def fill_skeleton_with_attributes(skeleton: str, attributes: List[Dict], target_word_count: int = 27) -> str:
    """按骨架结构填充，生成25-30词查询。"""
    import random
    import re
    
    if not attributes:
        return "I am looking for craft supplies for my projects"

    by_dimension = {}
    for a in attributes:
        dim = a.get('dimension', 'Other')
        val = a.get('value', '').strip()
        if val and len(val) > 1:
            by_dimension.setdefault(dim, []).append(val)

    if not by_dimension:
        return "I am looking for craft supplies for my projects"

    phrases = []
    openers = ["I am looking for", "I need", "I want to find", 
               "I am searching for", "I am hoping to find"]
    
    if 'Product_Category' in by_dimension:
        phrases.append(by_dimension['Product_Category'][0].split('/')[0])
    
    if 'Brand_Preference' in by_dimension:
        phrases.append(by_dimension['Brand_Preference'][0])
    
    if 'Appearance_Color' in by_dimension:
        phrases.append(by_dimension['Appearance_Color'][0])
    
    if 'Style_Design' in by_dimension:
        phrases.append(by_dimension['Style_Design'][0])
    
    if 'Material_Composition' in by_dimension:
        phrases.append(by_dimension['Material_Composition'][0])
    
    if 'Functionality' in by_dimension:
        phrases.append(by_dimension['Functionality'][0])
    
    if 'Ease_of_Use' in by_dimension:
        phrases.append(by_dimension['Ease_of_Use'][0])
    
    if 'Performance' in by_dimension:
        phrases.append(by_dimension['Performance'][0])
    
    if 'Usage_Scenario' in by_dimension:
        phrases.append(f"for {by_dimension['Usage_Scenario'][0]}")
    
    if 'Target_User' in by_dimension:
        phrases.append(f"for {by_dimension['Target_User'][0]}")
    
    for dim in ['Value', 'Special_Purpose', 'Safety', 'Compatibility', 'Price',
                'Size_Dimensions', 'Packaging_Quantity', 'Portability', 'Quality_Craftsmanship',
                'Special_User_Needs', 'Comfort']:
        if dim in by_dimension and len(by_dimension[dim][0].split()) <= 3:
            phrases.append(by_dimension[dim][0])
    
    query = random.choice(openers) + " " + ", ".join(phrases)
    query = re.sub(r'\s+', ' ', query).strip().capitalize()
    if not query.endswith('.'):
        query += '.'
    
    words = query.split()
    word_count = len(words)
    
    # 如果太短，添加更多属性
    if word_count < 25:
        extra_attrs = []
        for dim, vals in by_dimension.items():
            if dim not in ['Product_Category', 'Brand_Preference', 'Appearance_Color', 
                          'Style_Design', 'Material_Composition', 'Functionality',
                          'Ease_of_Use', 'Performance', 'Usage_Scenario', 'Target_User']:
                for val in vals[1:]:
                    if len(val.split()) <= 4:
                        extra_attrs.append(val)
        extra_attrs.sort(key=lambda x: len(x.split()), reverse=True)
        while len(words) < 25 and extra_attrs:
            extra = extra_attrs.pop(0)
            query = query.rstrip('.') + f", {extra}."
            words = query.split()
    
    # 如果太长，截断到30词
    if len(words) > 30:
        cutoff = 30
        for j in range(29, 24, -1):
            if j < len(words) and words[j] in ['for', 'and', 'or', 'with', 'the', 'a', 'an']:
                cutoff = j
                break
        query = ' '.join(words[:cutoff]).rstrip('.,') + '.'
    
    return query


# ============================================================================
# 抽象模板提取功能（依存树 → 自然语言模板）
# ============================================================================

# 抽象模板的POS标签到占位符的映射
# 核心依存关系标签（按复杂度分层）
DEP_CORE_LABELS = {
    # 主语
    "nsubj": "SUBJ",
    "nsubj:pass": "SUBJ",
    # 宾语/补语
    "dobj": "OBJ",
    "pobj": "OBJ",
    "attr": "ATTR",
    "acomp": "ATTR",
    "oprd": "ATTR",
    # 动词
    "ROOT": "ROOT",
    # 从句标记
    "acl:relcl": "RELCL",
    "relcl": "RELCL",
    "acl": "RELCL",
    "advcl": "ADVCL",
    "ccomp": "CCOMP",
    "xcomp": "XCOMP",
    # 介词
    "prep": "PREP",
    # 连词
    "cc": "CCONJ",
    "conj": "CONJ",
}

# 常见句型模式（依存结构 → 模板片段）
SENTENCE_PATTERNS = {
    ("nsubj", "amod", "ROOT"): "{subj} has {amod} {obj}",
    ("nsubj", "ROOT", "acl:relcl"): "{subj} {root} ... that {acl}",
    ("nsubj", "ROOT", "advcl"): "{subj} {root}, which {advcl}",
    ("nsubj", "amod", "ROOT", "acl:relcl"): "{subj} has {amod} {obj} that {acl}",
    ("nsubj", "ROOT", "acl:relcl", "advcl"): "{subj} {root} ... that {acl}, which {advcl}",
    ("det", "amod", "nsubj", "ROOT"): "[属性] [实体] {root} ...",
    ("amod", "nsubj", "ROOT", "advcl"): "[实体] has [属性] that {root}, which {advcl}",
}


def _get_token_dep_label(token) -> str:
    if token.is_punct:
        return ""
    return DEP_CORE_LABELS.get(token.dep_, "")


def _extract_noun_phrase_abstract(doc, root_token) -> str:
    """提取名词短语区域的抽象表示（用于主语或宾语）"""
    parts = []
    for child in root_token.children:
        if child.dep_ in {"amod", "det", "nummod", "compound"}:
            parts.append((child.i, _get_token_dep_label(child)))
    return " ".join(p[1] for p in sorted(parts))


def _extract_clause_structure(doc, root_token) -> tuple:
    """
    提取从句结构信息
    
    Returns:
        (has_relative_clause, has_advcl, main_verb_placeholder)
    """
    has_relcl = False
    has_advcl = False
    main_verb = _get_token_dep_label(root_token)
    
    for child in root_token.children:
        if child.dep_ in {"acl:relcl", "relcl", "acl"}:
            has_relcl = True
        elif child.dep_ in {"advcl", "ccomp", "xcomp"}:
            has_advcl = True
    
    return has_relcl, has_advcl, main_verb


def _build_abstract_template_from_tree(doc) -> str:
    if len(doc) == 0:
        return ""
    
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    
    if root is None:
        root = doc[0]
    
    # 收集所有token的抽象表示
    all_tokens = []
    for token in doc:
        if token.is_punct:
            continue
        placeholder = _get_token_dep_label(token)
        all_tokens.append(placeholder)
    
    # 简化：直接用POS序列
    template = " ".join(all_tokens)
    template = _simplify_template(template)
    
    return template


def _simplify_template(template: str) -> str:
    import re
    # 移除连续重复的标签
    parts = template.split()
    if not parts:
        return template
    simplified = [parts[0]]
    for p in parts[1:]:
        if p != simplified[-1]:
            simplified.append(p)
    return " ".join(simplified)


def extract_abstract_template(doc) -> str:
    """
    从spaCy Doc对象提取抽象自然语言模板。
    
    输入: spaCy Doc对象
    输出: 自然语言模板字符串
    
    核心逻辑:
    - 分析句子的依存树结构
    - 识别主要句型模式（如 "X has Y that Z", "X is Y for Z" 等）
    - 将具体词语替换为语义占位符
    - 保留连词和介词结构
    
    抽象规则示例:
    - nsubj + amod + NOUN → "[属性] [实体]" 或 "[实体] with [属性]"
    - ROOT + acl:relcl → "... that [从句]"
    - ROOT + acl + advcl → "... which makes it [形容词] for [场景]"
    - 数字、颜色等具体词 → "[属性]"
    
    输出示例:
    "[实体] has [属性] [属性] that [动作], which makes it [特性] for [场景]."
    
    Args:
        doc: spaCy Doc对象（单句）
    
    Returns:
        抽象模板字符串
    
    示例:
        输入: "This embroidery colors that don't fade easily, which makes it perfect for detailed work."
        输出: "[实体] has [属性] that [动作] [方式], which makes it [属性] for [场景]."
    """
    if doc is None or len(doc) == 0:
        return ""
    
    sents = list(doc.sents)
    if sents:
        doc = sents[0]
    
    template = _build_abstract_template_from_tree(doc)
    
    if not template.strip():
        template = " ".join(_get_token_dep_label(t) for t in doc if not t.is_punct)
    
    return template


def analyze_template_patterns(docs: List) -> Dict:
    """
    分析多个句子，提取常见模板模式。
    
    输入: spaCy Doc对象列表
    输出: Dict，包含常见模式和统计信息
    
    Args:
        docs: spaCy Doc对象列表
    
    Returns:
        Dict包含:
        - common_patterns: 按频率排序的模板列表
        - pattern_counts: 模板出现频率统计
        - total_templates: 总模板数
        - unique_templates: 唯一模板数
    
    示例输出:
    {
        "common_patterns": [
            "[实体] has [属性] that [动作] [方式], which makes it [属性] for [场景].",
            "[实体] is [属性] for [场景].",
            ...
        ],
        "pattern_counts": {
            "[实体] has [属性] ...": 15,
            "[实体] is [属性] ...": 8,
            ...
        },
        "total_templates": 50,
        "unique_templates": 12
    }
    """
    template_counts = defaultdict(int)
    
    for doc in docs:
        if doc is None or len(doc) == 0:
            continue
        
        template = extract_abstract_template(doc)
        if template.strip():
            template_counts[template] += 1
    
    sorted_patterns = sorted(
        template_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    common_patterns = [p[0] for p in sorted_patterns]
    pattern_counts = dict(sorted_patterns)
    
    return {
        "common_patterns": common_patterns,
        "pattern_counts": pattern_counts,
        "total_templates": sum(template_counts.values()),
        "unique_templates": len(template_counts),
    }


def fill_abstract_template(template: str, attributes: List[Dict]) -> str:
    """
    将属性填入抽象模板。
    
    输入:
        template: 抽象模板字符串，如 "[实体] has [属性] that [动作] [方式]"
        attributes: 属性列表，格式为 [{"dimension": "...", "value": "..."}]
    
    输出: 填充后的自然语言句子
    
    attributes格式示例:
    ```json
    [
      {"dimension": "Product_Category", "value": "thread"},
      {"dimension": "Quality", "value": "rich"},
      {"dimension": "Color", "value": "color variety"},
      {"dimension": "Performance", "value": "holds well"},
      {"dimension": "Usage_Scenario", "value": "detailed needlework"}
    ]
    ```
    
    填充逻辑:
    - 识别模板中的占位符类型（[实体], [属性], [动作], [方式], [场景]等）
    - 根据属性dimension匹配最合适的占位符
    - 保留模板的句法和连词结构
    - 优先填充位置靠前的占位符
    
    示例:
    模板: "[实体] has [属性] [属性] that [动作] [方式], which makes it [属性] for [场景]."
    属性: [{"dimension": "Product_Category", "value": "thread"}, {"dimension": "Color", "value": "rich color variety"}, ...]
    
    输出: "thread has rich color variety that holds well on fabric, which makes it perfect for detailed needlework."
    
    Args:
        template: 抽象模板字符串
        attributes: 属性列表
    
    Returns:
        填充后的自然语言句子
    """
    if not template or not attributes:
        return template
    
    # 定义占位符类型及其优先级
    placeholder_types = ["[实体]", "[属性]", "[动作]", "[方式]", "[场景]", "[限定词]", "[介词]", "[连词]", "[从属连词]"]
    
    dimension_to_placeholder = {
        "Product_Category": "[实体]",
        "Product_Type": "[实体]",
        "Material": "[实体]",
        "Brand": "[实体]",
        "Quality": "[属性]",
        "Color": "[属性]",
        "Performance": "[属性]",
        "Durability": "[属性]",
        "Style_Design": "[属性]",
        "Functionality": "[属性]",
        "Size": "[属性]",
        "Weight": "[属性]",
        "Price": "[属性]",
        "Safety": "[属性]",
        "Action": "[动作]",
        "Usage": "[动作]",
        "Usage_Scenario": "[场景]",
        "Target_Audience": "[场景]",
        "Application": "[场景]",
        "Method": "[方式]",
    }
    
    available_attrs = defaultdict(list)
    for attr in attributes:
        dimension = attr.get("dimension", "")
        value = attr.get("value", "")
        if value:
            placeholder = dimension_to_placeholder.get(dimension, "[属性]")
            available_attrs[placeholder].append(value)
    
    all_values = []
    for placeholder in placeholder_types:
        if placeholder in available_attrs:
            all_values.extend(available_attrs[placeholder])
    
    import re
    
    result = template
    values_iter = iter(all_values)
    
    def replace_placeholder(match):
        nonlocal values_iter
        try:
            return next(values_iter)
        except StopIteration:
            return match.group(0)
    
    result = re.sub(r'\[[^\]]+\]', replace_placeholder, result)
    result = " ".join(result.split())
    
    if result:
        result = result[0].upper() + result[1:]
    
    return result


# ============================================================================
# 解耦模块 (Disentanglement Module)
# ============================================================================
# 核心思想：将文本分解为风格无关的"内容"表示和风格相关的"属性"表示
# 生成时，将原始文本的"内容"表示与目标风格的"属性"表示结合
#
# 架构选择：Denoising Auto-Encoder (DAE) + 对比学习
# - 内容编码器：从依存树提取内容向量（骨架 + 语义槽位）
# - 风格编码器：从风格特征提取风格向量（复杂度轴 + POS分布）
# - 解码器：结合内容向量与目标风格向量，重构文本
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ContentEncoder(nn.Module):
    """
    内容编码器：从依存树提取内容表示

    输入：句子依存树结构
    输出：内容向量 (content_embedding)

    内容表示 = 骨架序列 + 语义槽位类型
    """

    def __init__(self, vocab_size: int = 100, embedding_dim: int = 64, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) 依存关系标签序列
        Returns:
            content_embedding: (batch, output_dim)
        """
        emb = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)
        output, (h_n, _) = self.lstm(emb)  # output: (batch, seq_len, hidden_dim*2)
        # 双向拼接最后一个隐状态
        h_combined = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, hidden_dim*2)
        content = F.relu(self.fc(h_combined))  # (batch, output_dim)
        return content


class StyleEncoder(nn.Module):
    """
    风格编码器：从风格特征提取风格表示

    输入：16维风格特征向量
    输出：风格向量 (style_embedding)

    风格表示 = 复杂度轴(4维) + POS分布(11维) + 句法标记密度(8维)
    """

    def __init__(self, input_dim: int = 23, hidden_dim: int = 64, style_dim: int = 64):
        super().__init__()
        self.style_dim = style_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_dim)
        )

    def forward(self, style_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_features: (batch, input_dim) 风格特征
        Returns:
            style_embedding: (batch, style_dim)
        """
        return self.net(style_features)


class DisentangledDecoder(nn.Module):
    """
    解码器：结合内容向量与目标风格向量，生成新文本

    输入：内容向量 + 目标风格向量
    输出：重构的依存关系序列
    """

    def __init__(self, content_dim: int = 128, style_dim: int = 64, hidden_dim: int = 128, output_dim: int = 100):
        super().__init__()
        self.style_dim = style_dim
        combined_dim = content_dim + style_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, content: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (batch, content_dim) 内容向量
            target_style: (batch, style_dim) 目标风格向量
        Returns:
            logits: (batch, output_dim) 预测的token logits
        """
        combined = torch.cat([content, target_style], dim=-1)
        return self.net(combined)


class DisentanglementModel(nn.Module):
    """
    完整解耦模型：内容编码器 + 风格编码器 + 解码器

    训练目标：
    1. 重构损失：reconstruct(original_text, decoder(content, original_style))
    2. 对比损失：相同内容不同风格的距离 < 不同内容相同风格的距离
    3. 风格解耦：content_vector 应不受 style_vector 影响
    """

    def __init__(self,
                 vocab_size: int = 100,
                 content_dim: int = 128,
                 style_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.content_encoder = ContentEncoder(vocab_size, content_dim // 2, content_dim // 2, content_dim)
        self.style_encoder = StyleEncoder(input_dim=23, hidden_dim=64, style_dim=style_dim)
        self.decoder = DisentangledDecoder(content_dim, style_dim, hidden_dim, vocab_size)

    def forward(self, content_tokens: torch.Tensor, style_features: torch.Tensor,
                target_style: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            content_tokens: (batch, seq_len) 内容token序列
            style_features: (batch, 23) 原始风格特征
            target_style: (batch, style_dim) 目标风格向量（用于风格转换）

        Returns:
            reconstructed_logits: 重构logits
            content_embedding: 内容向量
            style_embedding: 风格向量
        """
        content_emb = self.content_encoder(content_tokens)
        style_emb = self.style_encoder(style_features)

        if target_style is None:
            target_style = style_emb

        reconstructed = self.decoder(content_emb, target_style)

        return reconstructed, content_emb, style_emb

    def encode_content(self, content_tokens: torch.Tensor) -> torch.Tensor:
        """仅编码内容"""
        return self.content_encoder(content_tokens)

    def encode_style(self, style_features: torch.Tensor) -> torch.Tensor:
        """仅编码风格"""
        return self.style_encoder(style_features)

    def decode(self, content: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        """仅解码"""
        return self.decoder(content, target_style)


def disentangle_text(doc, dependency_features: Dict) -> Dict:
    """
    将单个文本解耦为内容表示和风格表示

    Args:
        doc: spaCy Doc对象
        dependency_features: 依存关系特征字典

    Returns:
        Dict包含:
            - content: 内容表示
                - skeleton: 骨架序列（如 "SUBJ am looking for OBJ"）
                - dep_sequence: 依存关系序列
                - semantic_slots: 语义槽位列表
            - style: 风格表示
                - complexity_axis: 复杂度四维向量
                - pos_distribution: POS分布向量
                - syntactic_markers: 句法标记密度向量
            - disentangled: 是否成功解耦
    """
    # ========== 内容表示提取 ==========
    skeleton_parts = []
    dep_sequence = []
    semantic_slots = []

    DEP_TO_SLOT = {
        "nsubj": "SUBJ", "nsubj:pass": "SUBJ",
        "dobj": "OBJ", "pobj": "OBJ", "attr": "OBJ",
        "ROOT": "ROOT", "amod": "AMOD", "advmod": "ADV",
        "prep": "PREP", "det": "DET",
    }

    for token in doc:
        if token.is_punct:
            continue
        dep = token.dep_
        skeleton_parts.append(DEP_TO_SLOT.get(dep, dep))
        dep_sequence.append(dep)

        # 语义槽位分类
        if dep in {"nsubj", "nsubj:pass", "dobj", "pobj", "attr"}:
            semantic_slots.append("ARG")
        elif dep in {"amod", "advmod", "det"}:
            semantic_slots.append("MOD")
        elif dep == "ROOT":
            semantic_slots.append("PRED")
        elif dep in {"prep", "agent"}:
            semantic_slots.append("LINK")
        else:
            semantic_slots.append("OTHER")

    # ========== 风格表示提取 ==========
    # 1. 复杂度四维轴
    precomputed = dependency_features.get('_precomputed', {})
    complexity_axis = {
        'subordination': dependency_features.get('subordinate_ratio', 0.0) * 10,
        'coordination': dependency_features.get('coordination_ratio', 0.0) * 10,
        'negation': dependency_features.get('_dep_raw_counts', {}).get('neg', 0) / max(dependency_features.get('n_tokens', 1), 1),
        'length_depth': dependency_features.get('avg_dependency_depth', 0.0) / 10,
    }

    # 2. POS分布向量
    n_tokens = dependency_features.get('n_tokens', 1)
    pos_dist = {
        'NOUN': dependency_features.get('upos_dist_NOUN', 0.0),
        'VERB': dependency_features.get('upos_dist_VERB', 0.0),
        'ADJ': dependency_features.get('upos_dist_ADJ', 0.0),
        'ADV': dependency_features.get('upos_dist_ADV', 0.0),
        'PRON': dependency_features.get('upos_dist_PRON', 0.0),
        'DET': dependency_features.get('upos_dist_DET', 0.0),
        'AUX': dependency_features.get('upos_dist_AUX', 0.0),
        'PART': dependency_features.get('upos_dist_PART', 0.0),
        'SCONJ': dependency_features.get('upos_dist_SCONJ', 0.0),
        'CCONJ': dependency_features.get('upos_dist_CCONJ', 0.0),
        'ADP': dependency_features.get('upos_dist_ADP', 0.0),
    }

    # 3. 句法标记密度向量
    syntactic_markers = {
        'relative_clause': dependency_features.get('relative_clause_ratio', 0.0),
        'passive': dependency_features.get('passive_ratio', 0.0),
        'participial': dependency_features.get('participial_ratio', 0.0),
        'infinitive': dependency_features.get('infinitive_ratio', 0.0),
        'appositive': dependency_features.get('appositive_ratio', 0.0),
        'parenthetical': dependency_features.get('parenthetical_ratio', 0.0),
        'prep_phrase': dependency_features.get('prep_phrase_ratio', 0.0),
        'insertion': dependency_features.get('insertion_frequency', 0.0),
    }

    # ========== 组装解耦结果 ==========
    content_repr = {
        'skeleton': ' '.join(skeleton_parts),
        'dep_sequence': dep_sequence,
        'semantic_slots': semantic_slots,
        'n_tokens': n_tokens,
    }

    style_repr = {
        'complexity_axis': complexity_axis,
        'pos_distribution': pos_dist,
        'syntactic_markers': syntactic_markers,
        # 展平为向量
        'vector': [
            complexity_axis['subordination'],
            complexity_axis['coordination'],
            complexity_axis['negation'],
            complexity_axis['length_depth'],
            pos_dist['NOUN'], pos_dist['VERB'], pos_dist['ADJ'], pos_dist['ADV'],
            pos_dist['PRON'], pos_dist['DET'], pos_dist['AUX'], pos_dist['PART'],
            pos_dist['SCONJ'], pos_dist['CCONJ'], pos_dist['ADP'],
            syntactic_markers['relative_clause'],
            syntactic_markers['passive'],
            syntactic_markers['participial'],
            syntactic_markers['infinitive'],
            syntactic_markers['appositive'],
            syntactic_markers['parenthetical'],
            syntactic_markers['prep_phrase'],
            syntactic_markers['insertion'],
        ],
    }

    return {
        'content': content_repr,
        'style': style_repr,
        'disentangled': True,
    }


def style_transfer(content_skeleton: str,
                   source_style: Dict,
                   target_style: Dict,
                   attributes: List[Dict],
                   nlp) -> str:
    """
    风格转换：将内容与目标风格结合，生成新文本

    Args:
        content_skeleton: 内容骨架（如 "SUBJ am looking for OBJ"）
        source_style: 源风格表示
        target_style: 目标风格表示
        attributes: 属性列表
        nlp: spaCy模型

    Returns:
        转换后的文本
    """
    # 1. 解析骨架
    skeleton_parts = content_skeleton.split()

    # 2. 计算风格插值系数
    # 在源风格和目标风格之间进行线性插值
    interpolation_ratio = 0.7  # 保留70%目标风格

    # 3. 生成填充文本
    # 根据目标风格的复杂度调整填充方式

    # 解析目标风格的复杂度
    target_complexity = target_style.get('complexity_axis', {})
    target_markers = target_style.get('syntactic_markers', {})

    # 高复杂度：使用更长的从句结构
    high_complexity = (
        target_complexity.get('subordination', 0) > 0.5 or
        target_markers.get('relative_clause', 0) > 0.1
    )

    # 4. 填充属性
    filled_text = _fill_with_style(content_skeleton, attributes, high_complexity)

    return filled_text


def _fill_with_style(skeleton: str, attributes: List[Dict], high_complexity: bool) -> str:
    """
    根据风格填充骨架

    Args:
        skeleton: 内容骨架
        attributes: 属性列表
        high_complexity: 是否使用高复杂度句式
    """
    import random

    if not attributes:
        return "I am looking for craft supplies."

    # 按维度组织属性
    by_dim = {}
    for a in attributes:
        dim = a.get('dimension', 'Other')
        val = a.get('value', '').strip()
        if val and len(val) > 1:
            by_dim.setdefault(dim, []).append(val)

    if not by_dim:
        return "I am looking for craft supplies."

    # 提取关键词
    product = by_dim.get('Product_Category', [''])[0].split('/')[0]
    brand = by_dim.get('Brand_Preference', [''])[0]
    color = by_dim.get('Appearance_Color', [''])[0]
    style = by_dim.get('Style_Design', [''])[0]
    material = by_dim.get('Material_Composition', [''])[0]

    # 根据复杂度选择句式模板
    if high_complexity:
        # 高复杂度：从句、被动、插入语
        templates = [
            "I have been searching for {product} that {brand}, which {style} and {material}.",
            "The {product} I am looking at, with its {color} {style}, is {material}.",
            "I need {product} that, despite being {brand}, offers {style} in {material}.",
            "What I am hoping to find is {product} that {brand} with {color} and {style}.",
        ]
    else:
        # 低复杂度：简单句
        templates = [
            "I am looking for {product}.",
            "I need {product} in {color}.",
            "I want {product} from {brand}.",
            "I am searching for {product} that is {style}.",
        ]

    # 填充模板
    template = random.choice(templates)
    result = template.format(
        product=product or 'craft supplies',
        brand=brand or '',
        color=color or '',
        style=style or '',
        material=material or '',
    )

    # 清理空白
    result = ' '.join(result.split())
    if not result.endswith('.'):
        result += '.'

    return result


def compute_disentanglement_loss(model: DisentanglementModel,
                                 content_tokens: torch.Tensor,
                                 style_features: torch.Tensor,
                                 target_tokens: torch.Tensor,
                                 lambda_contrastive: float = 0.1,
                                 lambda_style: float = 0.1) -> Tuple[torch.Tensor, Dict]:
    """
    计算解耦损失

    损失函数组成：
    1. 重构损失：交叉熵
    2. 对比损失：相同内容应接近，不同内容应远离
    3. 风格解耦损失：内容向量应与风格向量无关

    Args:
        model: 解耦模型
        content_tokens: 内容token序列
        style_features: 风格特征
        target_tokens: 目标token序列
        lambda_contrastive: 对比损失权重
        lambda_style: 风格解耦损失权重

    Returns:
        total_loss, loss_dict
    """
    # 前向传播
    reconstructed, content_emb, style_emb = model(content_tokens, style_features)

    # 1. 重构损失
    recon_loss = F.cross_entropy(reconstructed, target_tokens)

    # 2. 对比损失（简化的infoNCE）
    # 正样本：相同内容不同风格
    # 负样本：不同内容
    batch_size = content_emb.size(0)

    # 内容相似度矩阵
    content_sim = torch.matmul(content_emb, content_emb.T)  # (batch, batch)
    content_sim = content_sim / (content_emb.size(-1) ** 0.5)

    # 对角线为正样本，其余为负样本
    labels = torch.arange(batch_size, device=content_emb.device)
    contrastive_loss = F.cross_entropy(content_sim, labels)

    # 3. 风格解耦损失
    # 内容向量和风格向量的互信息应最小化
    # 简化：只使用前32维内容与风格向量计算差异
    content_for_style = content_emb[:, :style_emb.size(1)]  # (batch, 32)
    style_dep_loss = torch.mean(torch.abs(content_for_style - style_emb))

    # 总损失
    total_loss = recon_loss + lambda_contrastive * contrastive_loss + lambda_style * style_dep_loss

    loss_dict = {
        'recon_loss': recon_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'style_dep_loss': style_dep_loss.item(),
        'total_loss': total_loss.item(),
    }

    return total_loss, loss_dict


def extract_style_vector(dependency_features: Dict) -> np.ndarray:
    """
    从依存关系特征中提取风格向量

    Args:
        dependency_features: 依存关系特征字典

    Returns:
        style_vector: (23,) 风格向量
    """
    # 复杂度轴 (4维)
    precomputed = dependency_features.get('_precomputed', {})
    complexity_axis = [
        dependency_features.get('subordinate_ratio', 0.0) * 10,
        dependency_features.get('coordination_ratio', 0.0) * 10,
        dependency_features.get('_dep_raw_counts', {}).get('neg', 0) / max(dependency_features.get('n_tokens', 1), 1),
        dependency_features.get('avg_dependency_depth', 0.0) / 10,
    ]

    # POS分布 (11维)
    pos_dist = [
        dependency_features.get('upos_dist_NOUN', 0.0),
        dependency_features.get('upos_dist_VERB', 0.0),
        dependency_features.get('upos_dist_ADJ', 0.0),
        dependency_features.get('upos_dist_ADV', 0.0),
        dependency_features.get('upos_dist_PRON', 0.0),
        dependency_features.get('upos_dist_DET', 0.0),
        dependency_features.get('upos_dist_AUX', 0.0),
        dependency_features.get('upos_dist_PART', 0.0),
        dependency_features.get('upos_dist_SCONJ', 0.0),
        dependency_features.get('upos_dist_CCONJ', 0.0),
        dependency_features.get('upos_dist_ADP', 0.0),
    ]

    # 句法标记密度 (8维)
    syntactic_markers = [
        dependency_features.get('relative_clause_ratio', 0.0),
        dependency_features.get('passive_ratio', 0.0),
        dependency_features.get('participial_ratio', 0.0),
        dependency_features.get('infinitive_ratio', 0.0),
        dependency_features.get('appositive_ratio', 0.0),
        dependency_features.get('parenthetical_ratio', 0.0),
        dependency_features.get('prep_phrase_ratio', 0.0),
        dependency_features.get('insertion_frequency', 0.0),
    ]

    return np.array(complexity_axis + pos_dist + syntactic_markers, dtype=np.float32)


def compute_style_similarity(style1: np.ndarray, style2: np.ndarray) -> float:
    """
    计算两个风格向量之间的相似度

    Args:
        style1: (23,) 风格向量1
        style2: (23,) 风格向量2

    Returns:
        cosine_similarity: 余弦相似度
    """
    from numpy.linalg import norm

    if norm(style1) == 0 or norm(style2) == 0:
        return 0.0

    return np.dot(style1, style2) / (norm(style1) * norm(style2))


def interpolate_style(source_style: np.ndarray, target_style: np.ndarray,
                      alpha: float = 0.7) -> np.ndarray:
    """
    在源风格和目标风格之间进行插值

    Args:
        source_style: 源风格向量
        target_style: 目标风格向量
        alpha: 插值系数，0=完全源风格，1=完全目标风格

    Returns:
        interpolated_style: 插值后的风格向量
    """
    return (1 - alpha) * source_style + alpha * target_style


def apply_style_transfer(content_skeleton: str,
                        source_style: np.ndarray,
                        target_style: np.ndarray,
                        attributes: List[Dict],
                        nlp,
                        alpha: float = 0.7) -> str:
    """
    应用风格转换

    将内容骨架从源风格转换到目标风格

    Args:
        content_skeleton: 内容骨架
        source_style: 源风格向量
        target_style: 目标风格向量
        attributes: 属性列表
        nlp: spaCy模型
        alpha: 风格强度，0=保持原风格，1=完全目标风格

    Returns:
        转换后的文本
    """
    # 判断目标风格是否高复杂度
    # 目标风格的subordination维度
    target_subordination = target_style[0]
    target_relative_clause = target_style[15]  # relative_clause_ratio

    high_complexity = target_subordination > 0.5 or target_relative_clause > 0.1

    # 填充骨架
    filled = _fill_with_style(content_skeleton, attributes, high_complexity)

    return filled


# ============================================================================
# 便捷函数：批量处理用户文本的解耦
# ============================================================================

def disentangle_user_profile(profile: Dict, nlp) -> Dict:
    """
    对用户画像进行解耦处理

    Args:
        profile: 用户画像字典（包含dependency_features）
        nlp: spaCy模型

    Returns:
        解耦后的用户画像
    """
    user_id = profile.get('user_id', 'unknown')
    dependency_features = profile.get('dependency_features', {})

    if not dependency_features:
        return profile

    # 提取风格向量
    style_vector = extract_style_vector(dependency_features)

    # 解析复杂度模板
    complexity_templates = profile.get('complexity_templates', {})

    # 解耦各复杂度等级的内容
    disentangled_templates = {}
    for level in ['low', 'medium', 'high']:
        template_info = complexity_templates.get(level, {})
        skeleton = template_info.get('raw_skeleton', '')

        if skeleton:
            # 提取骨架的依存关系序列
            doc = nlp(skeleton)
            dep_seq = [token.dep_ for token in doc if not token.is_punct]
            dep_sequence = ' '.join(dep_seq) if dep_seq else ''

            disentangled_templates[level] = {
                'skeleton': skeleton,
                'dep_sequence': dep_sequence,
                'style_vector': style_vector.tolist(),
            }

    return {
        **profile,
        'disentangled_style': style_vector.tolist(),
        'disentangled_templates': disentangled_templates,
        'disentanglement_method': 'dae_contrastive',
    }


# ============================================================================
# 训练脚本：Disentanglement Model
# ============================================================================
# 使用 DAE + 对比学习训练解耦模型
# ============================================================================

class DisentanglementTrainer:
    """
    解耦模型训练器

    功能：
    1. 数据准备：从用户评论生成训练数据
    2. 对比学习：同一内容不同风格的正样本对
    3. 去噪自编码：重构原始文本
    4. 风格解耦：确保内容向量与风格向量分离
    """

    def __init__(self,
                 content_vocab: Dict[str, int] = None,
                 content_dim: int = 128,
                 style_dim: int = 64,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-3,
                 device: str = None):
        """
        初始化训练器

        Args:
            content_vocab: 内容词汇表（依存关系标签 → index）
            content_dim: 内容向量维度
            style_dim: 风格向量维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            device: 训练设备（cuda/cpu）
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 默认词汇表
        if content_vocab is None:
            self.content_vocab = self._build_default_vocab()
        else:
            self.content_vocab = content_vocab

        self.vocab_size = len(self.content_vocab)

        # 创建模型
        self.model = DisentanglementModel(
            vocab_size=self.vocab_size,
            content_dim=content_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练历史
        self.train_history = {
            'recon_loss': [],
            'contrastive_loss': [],
            'style_dep_loss': [],
            'total_loss': [],
        }

    def _build_default_vocab(self) -> Dict[str, int]:
        """构建默认的依存关系词汇表"""
        vocab = {
            '<PAD>': 0,
            'nsubj': 1, 'nsubj:pass': 2, 'dobj': 3, 'pobj': 4, 'attr': 5,
            'ROOT': 6, 'amod': 7, 'advmod': 8, 'prep': 9, 'det': 10,
            'acl:relcl': 11, 'relcl': 12, 'acl': 13, 'advcl': 14, 'ccomp': 15,
            'xcomp': 16, 'cc': 17, 'conj': 18, 'neg': 19, 'aux': 20,
            'auxpass': 21, 'agent': 22, 'appos': 23, 'parataxis': 24,
            'vbg': 25, 'vbn': 26, 'compound': 27, 'nummod': 28,
            'mark': 29, 'relcl': 30, 'oprd': 31, 'acomp': 32, 'attr': 33,
        }
        return vocab

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将token序列转换为ID序列"""
        return [self.content_vocab.get(t, 0) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """将ID序列转换回token序列"""
        id_to_token = {v: k for k, v in self.content_vocab.items()}
        return [id_to_token.get(i, '<PAD>') for i in ids]

    def prepare_batch(self,
                     texts: List[str],
                     style_vectors: np.ndarray,
                     nlp) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        准备训练批次

        Args:
            texts: 文本列表
            style_vectors: (batch, 23) 风格向量数组
            nlp: spaCy模型

        Returns:
            content_tokens: (batch, seq_len) 内容token ID
            style_features: (batch, 23) 风格特征
            target_tokens: (batch, seq_len) 目标token ID
        """
        batch_size = len(texts)
        content_token_ids = []
        target_token_ids = []

        max_len = 0

        # 解析每个文本
        for text in texts:
            doc = nlp(text)
            tokens = []
            for token in doc:
                if not token.is_punct:
                    dep = token.dep_
                    tokens.append(dep)

            if not tokens:
                tokens = ['nsubj']

            token_ids = self.tokens_to_ids(tokens)
            content_token_ids.append(token_ids)
            target_token_ids.append(token_ids)  # 自编码目标

            max_len = max(max_len, len(token_ids))

        # Padding
        content_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        target_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        for i, (ct, tt) in enumerate(zip(content_token_ids, target_token_ids)):
            content_tokens[i, :len(ct)] = torch.tensor(ct, device=self.device)
            target_tokens[i, :len(tt)] = torch.tensor(tt, device=self.device)

        style_features = torch.tensor(style_vectors, dtype=torch.float32, device=self.device)

        return content_tokens, style_features, target_tokens

    def train_step(self,
                  content_tokens: torch.Tensor,
                  style_features: torch.Tensor,
                  target_tokens: torch.Tensor,
                  lambda_contrastive: float = 0.1,
                  lambda_style: float = 0.1) -> Dict:
        """
        单步训练

        Returns:
            loss_dict: 损失字典
        """
        self.optimizer.zero_grad()

        # 前向传播
        reconstructed, content_emb, style_emb = self.model(
            content_tokens, style_features
        )

        # 计算损失
        loss, loss_dict = compute_disentanglement_loss(
            self.model,
            content_tokens,
            style_features,
            target_tokens[:, 0],  # 使用第一个token作为目标
            lambda_contrastive,
            lambda_style
        )

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss_dict

    def train(self,
              texts: List[str],
              style_vectors: np.ndarray,
              nlp,
              epochs: int = 10,
              batch_size: int = 32,
              lambda_contrastive: float = 0.1,
              lambda_style: float = 0.1,
              val_texts: List[str] = None,
              val_style_vectors: np.ndarray = None,
              log_every: int = 100) -> Dict:
        """
        训练模型

        Args:
            texts: 训练文本列表
            style_vectors: (N, 23) 训练风格向量
            nlp: spaCy模型
            epochs: 训练轮数
            batch_size: 批大小
            lambda_contrastive: 对比损失权重
            lambda_style: 风格解耦损失权重
            val_texts: 验证文本（可选）
            val_style_vectors: 验证风格向量（可选）
            log_every: 日志打印频率

        Returns:
            train_history: 训练历史
        """
        n_samples = len(texts)
        n_batches = (n_samples + batch_size - 1) // batch_size

        self.model.train()

        for epoch in range(epochs):
            epoch_losses = {
                'recon_loss': 0.0,
                'contrastive_loss': 0.0,
                'style_dep_loss': 0.0,
                'total_loss': 0.0,
            }

            # 打乱数据
            indices = np.random.permutation(n_samples)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_texts = [texts[i] for i in batch_indices]
                batch_styles = style_vectors[batch_indices]

                # 准备数据
                content_tokens, style_features, target_tokens = self.prepare_batch(
                    batch_texts, batch_styles, nlp
                )

                # 训练
                loss_dict = self.train_step(
                    content_tokens,
                    style_features,
                    target_tokens,
                    lambda_contrastive,
                    lambda_style
                )

                # 累计损失
                for k, v in loss_dict.items():
                    epoch_losses[k] += v

                # 日志
                if (batch_idx + 1) % log_every == 0:
                    avg_loss = {k: v / log_every for k, v in epoch_losses.items()}
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{n_batches} | "
                          f"Loss: {avg_loss['total_loss']:.4f} | "
                          f"Recon: {avg_loss['recon_loss']:.4f} | "
                          f"Contrastive: {avg_loss['contrastive_loss']:.4f}")

                    # 重置累计
                    for k in epoch_losses:
                        epoch_losses[k] = 0.0

            # Epoch结束
            avg_epoch_loss = {k: v / n_batches for k, v in epoch_losses.items()}
            print(f"\n=== Epoch {epoch+1}/{epochs} Summary ===")
            print(f"  Recon Loss: {avg_epoch_loss['recon_loss']:.4f}")
            print(f"  Contrastive Loss: {avg_epoch_loss['contrastive_loss']:.4f}")
            print(f"  Style Dep Loss: {avg_epoch_loss['style_dep_loss']:.4f}")
            print(f"  Total Loss: {avg_epoch_loss['total_loss']:.4f}\n")

            # 更新历史
            for k, v in avg_epoch_loss.items():
                self.train_history[k].append(v)

            # 验证
            if val_texts is not None and val_style_vectors is not None:
                val_loss = self.evaluate(val_texts, val_style_vectors, nlp)
                print(f"  Validation Loss: {val_loss:.4f}\n")

        return self.train_history

    def evaluate(self,
                texts: List[str],
                style_vectors: np.ndarray,
                nlp,
                batch_size: int = 32) -> float:
        """评估模型"""
        self.model.eval()

        n_samples = len(texts)
        n_batches = (n_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                batch_texts = texts[start_idx:end_idx]
                batch_styles = style_vectors[start_idx:end_idx]

                content_tokens, style_features, target_tokens = self.prepare_batch(
                    batch_texts, batch_styles, nlp
                )

                reconstructed, content_emb, style_emb = self.model(
                    content_tokens, style_features
                )

                loss, _ = compute_disentanglement_loss(
                    self.model,
                    content_tokens,
                    style_features,
                    target_tokens[:, 0],
                    lambda_contrastive=0.0,
                    lambda_style=0.0
                )

                total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.model.train()
        return avg_loss

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'content_vocab': self.content_vocab,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.content_vocab = checkpoint['content_vocab']
        print(f"Model loaded from {path}")

    def extract_content_embedding(self, text: str, nlp) -> np.ndarray:
        """提取文本的内容向量"""
        self.model.eval()

        doc = nlp(text)
        tokens = [token.dep_ for token in doc if not token.is_punct]
        if not tokens:
            tokens = ['nsubj']

        token_ids = torch.tensor(
            [self.tokens_to_ids(tokens)],
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            content_emb = self.model.encode_content(token_ids)

        return content_emb.cpu().numpy()[0]

    def extract_style_embedding(self, style_vector: np.ndarray) -> np.ndarray:
        """提取风格向量"""
        self.model.eval()

        style_tensor = torch.tensor(
            [style_vector],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            style_emb = self.model.encode_style(style_tensor)

        return style_emb.cpu().numpy()[0]


def run_training(config: Dict = None):
    """
    运行训练流程

    Args:
        config: 训练配置字典
    """
    if config is None:
        config = {
            # 数据路径
            'reviews_file': '/fs04/ar57/wenyu/result/personal_query/00_data_preparation',
            'profiles_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
            'output_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis/disentanglement_model',

            # 模型参数
            'content_dim': 128,
            'style_dim': 32,
            'hidden_dim': 128,
            'learning_rate': 1e-3,

            # 训练参数
            'epochs': 10,
            'batch_size': 32,
            'lambda_contrastive': 0.1,
            'lambda_style': 0.1,

            # 损失权重
            'contrastive_weight': 0.1,
            'style_dep_weight': 0.1,

            # 其他
            'log_every': 100,
            'device': None,
        }

    import os
    os.makedirs(config['output_dir'], exist_ok=True)

    print("=" * 60)
    print("Disentanglement Model Training")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    from collections import defaultdict

    # 加载用户评论
    user_reviews = load_user_reviews(config['reviews_file'])
    print(f"  Loaded {len(user_reviews)} users' reviews")

    # 加载用户画像
    profiles_dir = Path(config['profiles_dir'])
    user_profiles = {}
    for profile_file in profiles_dir.glob("linguistic_profile_*.json"):
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
            user_id = profile.get('user_id')
            if user_id:
                user_profiles[user_id] = profile
    print(f"  Loaded {len(user_profiles)} user profiles")

    # 2. 准备训练数据
    print("\n[2/5] Preparing training data...")

    texts = []
    style_vectors = []

    for user_id, reviews in user_reviews.items():
        if user_id not in user_profiles:
            continue

        profile = user_profiles[user_id]
        dep_features = profile.get('dependency_features', {})

        if not dep_features:
            continue

        # 提取风格向量
        style_vec = extract_style_vector(dep_features)

        # 提取评论文本
        for review in reviews[:5]:  # 每个用户最多5条评论
            text = review.get('reviewText', '')
            if text and len(text.split()) >= 25:
                texts.append(text)
                style_vectors.append(style_vec)

    print(f"  Prepared {len(texts)} training samples")

    if len(texts) == 0:
        print("  ERROR: No training samples available!")
        return

    style_vectors = np.array(style_vectors)

    # 2.5 扫描所有依存关系标签，构建完整词汇表
    print("\n[2.5/5] Building vocabulary from actual data...")
    import spacy
    nlp = spacy.load('en_core_web_sm')

    all_deps = set()
    total_texts = len(texts)
    for i, text in enumerate(texts):
        if i % 1000 == 0 or i == total_texts - 1:
            print(f"    Scanning texts: {i+1}/{total_texts} ({(i+1)*100//total_texts}%)", flush=True)
        doc = nlp(text)
        for token in doc:
            if not token.is_punct:
                all_deps.add(token.dep_)
    print(f"  Found {len(all_deps)} unique dependency labels")

    # 构建词汇表：PAD=0, 然后是所有依存关系标签
    content_vocab = {'<PAD>': 0}
    for i, dep in enumerate(sorted(all_deps), start=1):
        content_vocab[dep] = i
    print(f"  Vocabulary size: {len(content_vocab)}")

    # 3. 初始化训练器
    print("\n[3/5] Initializing trainer...")
    trainer = DisentanglementTrainer(
        content_dim=config['content_dim'],
        style_dim=config['style_dim'],
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
        device=config.get('device'),
        content_vocab=content_vocab
    )
    print(f"  Device: {trainer.device}")
    print(f"  Model params: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Vocab size: {trainer.vocab_size}")

    # 4. 训练
    print("\n[4/5] Training...")

    history = trainer.train(
        texts=texts,
        style_vectors=style_vectors,
        nlp=nlp,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lambda_contrastive=config['lambda_contrastive'],
        lambda_style=config['lambda_style'],
        log_every=config['log_every']
    )

    # 5. 保存模型
    print("\n[5/5] Saving model...")
    model_path = os.path.join(config['output_dir'], 'disentanglement_model.pt')
    trainer.save_model(model_path)

    # 保存词汇表
    vocab_path = os.path.join(config['output_dir'], 'content_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(trainer.content_vocab, f, indent=2)
    print(f"  Vocab saved to {vocab_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # 保存训练历史
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {history_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stage 5: Disentanglement Model Training & Application')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'disentangle'],
                       help='运行模式: train=训练解耦模型, disentangle=解耦画像')
    parser.add_argument('--reviews-file', type=str,
                       default='/fs04/ar57/wenyu/result/personal_query/00_data_preparation',
                       help='评论数据路径')
    parser.add_argument('--profiles-dir', type=str,
                       default='/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
                       help='用户画像目录')
    parser.add_argument('--output-dir', type=str,
                       default='/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--content-dim', type=int, default=128, help='内容向量维度')
    parser.add_argument('--style-dim', type=int, default=32, help='风格向量维度')
    parser.add_argument('--lambda-contrastive', type=float, default=0.1, help='对比损失权重')
    parser.add_argument('--lambda-style', type=float, default=0.1, help='风格解耦损失权重')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练解耦模型
        config = {
            'reviews_file': args.reviews_file,
            'profiles_dir': args.profiles_dir,
            'output_dir': args.output_dir,
            'content_dim': args.content_dim,
            'style_dim': args.style_dim,
            'hidden_dim': 128,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lambda_contrastive': args.lambda_contrastive,
            'lambda_style': args.lambda_style,
            'log_every': 100,
            'device': args.device,
        }
        run_training(config)
    elif args.mode == 'disentangle':
        # 解耦用户画像
        main_disentangle()
