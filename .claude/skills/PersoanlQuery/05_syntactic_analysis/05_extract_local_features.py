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

    # --- Complexity axis features ---
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

    # 1. 重构损失：对整个序列计算平均交叉熵
    # reconstructed: (batch, vocab_size) → 扩展到序列长度
    # target_tokens: (batch, seq_len)
    seq_len = target_tokens.size(1)
    # 将单token预测扩展到所有位置，计算平均损失
    recon_loss = F.cross_entropy(
        reconstructed.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, reconstructed.size(-1)),
        target_tokens.reshape(-1)
    )

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

    return {
        **profile,
        'disentangled_style': style_vector.tolist(),
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
            target_tokens,  # 整个序列作为目标
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


def _extract_style_features_from_text(text: str, nlp) -> Optional[np.ndarray]:
    """
    从文本中提取 23 维风格特征（少于25词返回None）。

    与 Stage 6 的 _extract_style_features_from_text 保持一致。

    Args:
        text: 输入文本
        nlp: spaCy 模型

    Returns:
        23维风格特征向量，或 None（文本太短）
    """
    words = [w for w in text.split() if w.strip()]
    if len(words) < 25:
        return None

    doc = nlp(text)
    n_tokens = max(len([t for t in doc if not t.is_punct]), 1)
    n_subj = sum(1 for t in doc if t.dep_ in {'nsubj', 'nsubj:pass'})
    n_dobj = sum(1 for t in doc if t.dep_ in {'dobj', 'pobj', 'attr'})
    n_amod = sum(1 for t in doc if t.dep_ == 'amod')
    n_advmod = sum(1 for t in doc if t.dep_ == 'advmod')
    n_prep = sum(1 for t in doc if t.dep_ == 'prep')
    n_conj = sum(1 for t in doc if t.dep_ == 'conj')
    n_neg = sum(1 for t in doc if t.dep_ == 'neg')
    n_relcl = sum(1 for t in doc if t.dep_ == 'relcl')
    n_pass = sum(1 for t in doc if t.dep_ in {'nsubj:pass', 'aux:pass'} or (t.tag_ == 'VBN' and t.dep_ not in {'amod', 'conj'}))
    n_part = sum(1 for t in doc if t.tag_ in {'VBG', 'VBN'} and t.dep_ in {'amod', 'advcl', 'relcl'})
    n_inf = sum(1 for t in doc if t.tag_ == 'VB' and t.dep_ in {'xcomp', 'ccomp', 'advcl'})
    n_det = sum(1 for t in doc if t.dep_ == 'det')
    n_cc = sum(1 for t in doc if t.dep_ == 'cc')
    n_intj = sum(1 for t in doc if t.dep_ == 'intj')

    pos_counts = {}
    for token in doc:
        if not token.is_punct:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    def get_depth(token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
            if depth > 20:
                break
        return depth
    depths = [get_depth(t) for t in doc if not t.is_punct]
    avg_depth = sum(depths) / max(len(depths), 1)

    subordinate_ratio = n_subj / n_tokens
    coordination_ratio = n_conj / n_tokens
    negation_ratio = n_neg / n_tokens
    length_depth = avg_depth / 10.0

    upos_order = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'ADP']
    pos_dist = [pos_counts.get(p, 0) / n_tokens for p in upos_order]

    relative_clause_ratio = n_relcl / n_tokens
    passive_ratio = n_pass / n_tokens
    participial_ratio = n_part / n_tokens
    infinitive_ratio = n_inf / n_tokens
    appositive_ratio = n_intj / n_tokens
    parenthetical_ratio = n_det / n_tokens
    prep_phrase_ratio = n_prep / n_tokens
    insertion_frequency = n_amod / n_tokens

    features = [
        subordinate_ratio * 10,
        coordination_ratio * 10,
        negation_ratio * 10,
        length_depth,
        *pos_dist,
        relative_clause_ratio,
        passive_ratio,
        participial_ratio,
        infinitive_ratio,
        appositive_ratio,
        parenthetical_ratio,
        prep_phrase_ratio,
        insertion_frequency,
    ]
    return np.array(features, dtype=np.float32)


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
            'output_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',

            # 模型参数
            'content_dim': 128,
            'style_dim': 64,
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
    output_dir = config.get('output_dir', '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Disentanglement Model Training")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    import spacy
    nlp = spacy.load('en_core_web_sm')

    # 加载用户评论
    user_reviews = load_user_reviews(config['reviews_file'])
    print(f"  Loaded {len(user_reviews)} users' reviews")

    # 2. 准备训练数据（直接从评论提取风格特征，不再依赖画像文件）
    print("\n[2/5] Preparing training data...")

    texts = []
    style_vectors = []

    for user_id, reviews in user_reviews.items():
        for review in reviews[:5]:  # 每个用户最多5条评论
            text = review.get('reviewText', '')
            if not text or len(text.split()) < 25:
                continue

            style_features = _extract_style_features_from_text(text, nlp)
            if style_features is None:
                continue

            texts.append(text)
            style_vectors.append(style_features)

    print(f"  Prepared {len(texts)} training samples")

    if len(texts) == 0:
        print("  ERROR: No training samples available!")
        return

    style_vectors = np.array(style_vectors)

    # 2.5 扫描所有依存关系标签，构建完整词汇表
    print("\n[2.5/5] Building vocabulary from actual data...")

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

    # 保存词汇表
    vocab_path = os.path.join(output_dir, 'content_vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(content_vocab, f)
    print(f"  Saved vocabulary to {vocab_path}")

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
    model_path = os.path.join(output_dir, 'disentanglement_model.pt')
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
    # 训练解耦模型
    config = {
        'reviews_file': '/fs04/ar57/wenyu/result/personal_query/00_data_preparation',
        'profiles_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
        'output_dir': '/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis',
        'content_dim': 128,
        'style_dim': 64,
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'epochs': 10,
        'batch_size': 32,
        'lambda_contrastive': 0.1,
        'lambda_style': 0.1,
        'log_every': 100,
        'device': None,
    }
    run_training(config)
