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
    format='%(asctime)s - %(levelname)s - %(message)s'
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


def extract_dependency_structures_from_doc(doc) -> Dict:
    """
    Extract dependency relation structures from a single text.

    Returns:
        Dict with dep_rel distribution, tree depth stats, and dep patterns.
    """
    if len(doc) == 0:
        return {}

    n_tokens = len(doc)

    # 1. Dependency relation distribution
    dep_counts = defaultdict(int)
    for token in doc:
        dep_counts[token.dep_] += 1
    dep_dist = {f"dep_{k}": v / n_tokens for k, v in dep_counts.items()}

    # Raw counts for complexity scoring
    dep_raw_counts = dict(dep_counts)

    # 2. Dependency tree depth per sentence
    def token_depth(token):
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth

    sent_depths = []
    for sent in doc.sents:
        max_depth = max((token_depth(tok) for tok in sent), default=0)
        sent_depths.append(max_depth)

    avg_tree_depth = sum(sent_depths) / len(sent_depths) if sent_depths else 0.0
    max_tree_depth = max(sent_depths) if sent_depths else 0

    # 3. Average number of dependents per token
    dependents_per_token = [len(list(tok.children)) for tok in doc]
    avg_dependents = sum(dependents_per_token) / len(dependents_per_token) if dependents_per_token else 0.0

    # 4. Head-direction features (left vs right dependents)
    left_deps = sum(1 for tok in doc if tok.head.i > tok.i and tok.dep_ != "ROOT")
    right_deps = sum(1 for tok in doc if tok.head.i < tok.i and tok.dep_ != "ROOT")
    total_non_root = left_deps + right_deps
    left_dep_ratio = left_deps / total_non_root if total_non_root > 0 else 0.5
    right_dep_ratio = right_deps / total_non_root if total_non_root > 0 else 0.5

    # 5. Dependency distance (average absolute distance between token and its head)
    dep_distances = [abs(tok.i - tok.head.i) for tok in doc if tok.dep_ != "ROOT"]
    avg_dep_distance = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0

    # 6. Subordination ratio (clausal deps / total deps)
    clausal_deps = {"ccomp", "xcomp", "advcl", "acl", "acl:relcl", "relcl", "csubj"}
    n_clausal = sum(1 for tok in doc if tok.dep_ in clausal_deps)
    subordination_ratio = n_clausal / n_tokens if n_tokens > 0 else 0.0

    features = {
        "n_tokens": n_tokens,
        "avg_tree_depth": avg_tree_depth,
        "max_tree_depth": max_tree_depth,
        "avg_dependents_per_token": avg_dependents,
        "left_dep_ratio": left_dep_ratio,
        "right_dep_ratio": right_dep_ratio,
        "avg_dep_distance": avg_dep_distance,
        "subordination_ratio": subordination_ratio,
        **dep_dist,
        "_dep_raw_counts": dep_raw_counts,
    }
    return features


def extract_dependency_structures(nlp, text: str) -> Dict:
    doc = nlp(text)
    return extract_dependency_structures_from_doc(doc)


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

    # Extract review texts
    texts = []
    for review in reviews:
        text = review.get("reviewText", "")
        if text and text.strip():
            texts.append(text.strip())

    if not texts:
        logger.warning(f"No valid texts found for user {user_id}")
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
    docs = list(nlp.pipe(texts, batch_size=spacy_batch_size, n_process=spacy_n_process))

    all_dep_features = []
    reviews_with_scores = []
    axis_components_list = []
    for text, doc in zip(texts, docs):
        dep_feats = extract_dependency_structures_from_doc(doc)
        if dep_feats:
            all_dep_features.append(dep_feats)
            dep_counts = dep_feats.get('_dep_raw_counts', {})
            score_result = complexity_score(dep_counts, doc=doc, return_components=True)
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


def main():
    config = {
        "reviews_file": "/fs04/ar57/wenyu/result/personal_query/00_data_preparation/all_user_reviews.json",
        "output_dir": "/fs04/ar57/wenyu/result/personal_query/05_syntactic_analysis",
        "max_reviews": None,
        "user_ids": None,
        "mode": "features",
    }

    # Initialize extractor
    logger.info("Initializing local feature extractor...")
    extractor = LocalFeatureExtractor()

    # Load reviews
    logger.info(f"Loading reviews from {config['reviews_file']}")
    user_reviews = load_user_reviews(config['reviews_file'])

    # Filter users if specified
    if config['user_ids']:
        user_reviews = {uid: user_reviews[uid] for uid in config['user_ids'] if uid in user_reviews}
        logger.info(f"Processing {len(user_reviews)} specified users")

    if config['mode'] == "skeletons":
        logger.info("Mode: Sentence Skeleton Extraction")
        results = []
        for user_id, reviews in user_reviews.items():
            logger.info(f"Processing user {user_id} ({len(reviews)} reviews)")
            skeleton_profile = extract_user_skeletons(
                user_id, reviews, extractor, max_reviews=config['max_reviews']
            )
            if skeleton_profile:
                save_skeleton_profile(skeleton_profile, config['output_dir'])
                results.append(skeleton_profile)

        logger.info("=" * 60)
        logger.info("SKELETON EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total users processed: {len(results)}")
        logger.info(f"Output directory: {config['output_dir']}")
        if results:
            logger.info(f"Most common skeleton: {results[0]['skeleton']}")
            logger.info(f"Frequency: {results[0]['skeleton_frequency']}")
        logger.info("=" * 60)
    elif config['mode'] == "abstract":
        logger.info("Mode: Abstract Template Extraction")
        results = []
        for user_id, reviews in user_reviews.items():
            logger.info(f"Processing user {user_id} ({len(reviews)} reviews)")
            abstract_profile = extract_user_abstract_skeletons(
                user_id, reviews, extractor, max_reviews=config['max_reviews']
            )
            if abstract_profile:
                save_skeleton_profile(abstract_profile, config['output_dir'])
                results.append(abstract_profile)

        logger.info("=" * 60)
        logger.info("ABSTRACT TEMPLATE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total users processed: {len(results)}")
        logger.info(f"Output directory: {config['output_dir']}")
        if results:
            logger.info(f"Most common template: {results[0]['abstract_template']}")
            logger.info(f"Frequency: {results[0]['template_frequency']}")
        logger.info("=" * 60)
    else:
        # Extract dependency features (original behavior)
        results = []
        for user_id, reviews in user_reviews.items():
            logger.info(f"Processing user {user_id} ({len(reviews)} reviews)")
            profile = extract_user_profile(user_id, reviews, extractor, config['max_reviews'])
            if profile:
                save_profile(profile, config['output_dir'])
                results.append(profile)

        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total users processed: {len(results)}")
        logger.info(f"Output directory: {config['output_dir']}")

        if results:
            feature_counts = [r["feature_count"] for r in results]
            logger.info(f"Features per user: min={min(feature_counts)}, max={max(feature_counts)}, avg={sum(feature_counts)/len(feature_counts):.1f}")
            sample_features = results[0]["dependency_features"]
            logger.info(f"Sample dependency features ({len(sample_features)}):")
            for name, value in list(sample_features.items())[:10]:
                logger.info(f"  {name}: {value:.4f}")

        logger.info("=" * 60)


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


def complexity_score(dep_counts: dict, doc=None, return_components: bool = False):
    if doc is None:
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


if __name__ == "__main__":
    main()
