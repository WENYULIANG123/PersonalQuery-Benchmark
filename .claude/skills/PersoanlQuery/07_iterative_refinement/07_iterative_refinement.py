#!/usr/bin/env python3
"""
Stage 9: Iterative Refinement with Feature-Aware Prompting

Instead of single-round multi-candidate filtering, this performs multiple rounds:
1. Generate candidates with base prompt
2. Analyze feature gaps between best candidate and user style
3. Create targeted prompt addressing worst feature gaps
4. Generate new candidates with refined prompt
5. Repeat until convergence or max rounds

This allows the system to progressively improve specific linguistic aspects.
"""

import os
import sys
import json
import argparse
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import importlib.util
import time
import numpy as np
import threading
import re
import math


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class SemanticEmbeddingModel:
    """Lightweight semantic embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model lazily."""
        try:
            st_module = importlib.import_module("sentence_transformers")
            SentenceTransformer = st_module.SentenceTransformer
            logger.info(f"Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Semantic model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}, using fallback")
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            # Fallback: use TF-IDF like simple word overlap
            return self._fallback_encode(texts)

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.warning(f"Encoding failed: {e}, using fallback")
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding using simple word frequency vectors."""
        from collections import Counter
        import re

        # Build vocabulary from all texts
        all_words = set()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.update(words)

        vocab = list(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        # Create TF-like vectors
        vectors = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_counts = Counter(words)
            vec = np.zeros(len(vocab))
            for word, count in word_counts.items():
                if word in word_to_idx:
                    vec[word_to_idx[word]] = count
            # Normalize
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)

        return np.array(vectors)


# Global semantic model instance
_semantic_model = None


def get_semantic_model():
    """Get or create the global semantic model instance."""
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SemanticEmbeddingModel()
    return _semantic_model

current_dir = Path(__file__).parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

SYNTAX_DEP_KEYS = ("nsubj", "obj", "amod", "acl", "advcl", "conj")
CONDITIONAL_MARKERS = {"if", "unless", "when", "provided", "providing", "assuming", "whether"}

# Import LLM client
llm_client_module = importlib.util.spec_from_file_location(
    "llm_client",
    current_dir.parent.parent / "llm_client.py"
)
if llm_client_module is None or llm_client_module.loader is None:
    raise RuntimeError("Failed to load llm_client module spec")
llm_client_lib = importlib.util.module_from_spec(llm_client_module)
llm_client_module.loader.exec_module(llm_client_lib)
LLMClient = llm_client_lib.LLMClient

# Import SentenceLevelFeatureExtractor from current directory
try:
    feature_script = current_dir / "07_extract_sentence_level_features.py"
    if feature_script.exists():
        feature_spec = importlib.util.spec_from_file_location(
            "stage7_sentence_features", feature_script
        )
        if feature_spec is None or feature_spec.loader is None:
            raise RuntimeError("Failed to load sentence-level feature module spec")
        feature_module = importlib.util.module_from_spec(feature_spec)
        feature_spec.loader.exec_module(feature_module)
        SentenceLevelFeatureExtractor = feature_module.SentenceLevelFeatureExtractor
    else:
        SentenceLevelFeatureExtractor = None
except Exception as e:
    logger.warning(f"Failed to load sentence-level extractor: {e}")
    SentenceLevelFeatureExtractor = None

_feature_extractor_local = threading.local()

# Define FeatureSet enum and related functions (replacing deleted feature_selector.py)
from enum import Enum

class FeatureSet(Enum):
    """Feature set definitions."""
    EMNLP_16 = "emnlp_16"
    SHORT_QUERY_18 = "short_query_18"
    SHORT_QUERY_13 = "short_query_13"
    STYLE_ONLY_16 = "style_only_16"
    FULL = "full"


# Feature set definitions (16 style features)
STYLE_ONLY_16_FEATURES = {
    "tokens_per_sent", "char_per_tok", "ttr_lemma_chunks_100",
    "lexical_density", "upos_dist_NOUN", "upos_dist_VERB",
    "upos_dist_ADJ", "upos_dist_ADV", "upos_dist_PRON",
    "upos_dist_DET", "upos_dist_AUX",
    "upos_dist_PART", "upos_dist_SCONJ", "upos_dist_CCONJ",
    "upos_dist_ADP", "n_tokens"
}


def get_feature_set(feature_set: FeatureSet) -> Optional[set]:
    """Get feature names for given feature set."""
    if feature_set == FeatureSet.STYLE_ONLY_16:
        return STYLE_ONLY_16_FEATURES
    # For other feature sets, return all features from profile
    return None  # None means use all available features


def extract_features_from_profile(data: dict, feature_set: FeatureSet) -> dict:
    """
    Extract features from linguistic profile JSON.

    Args:
        data: Linguistic profile JSON data
        feature_set: Which feature set to extract

    Returns:
        Feature dictionary
    """
    profilingud_features = data.get("profilingud_features", {})

    if feature_set == FeatureSet.STYLE_ONLY_16:
        # Filter to style features only
        selected = get_feature_set(feature_set)
        return {k: v for k, v in profilingud_features.items() if k in selected}
    else:
        # Return all features
        return profilingud_features


@dataclass
class FeatureGap:
    """Represents a single feature gap."""
    feature_name: str
    gap_size: float  # Absolute difference
    direction: float  # Positive: query > user, Negative: query < user
    user_value: float
    query_value: float


@dataclass
class RoundResult:
    """Result from a single refinement round."""
    round_num: int
    best_query: str
    best_distance: float  # Kept for backward compatibility, use style_distance
    style_distance: float  # Style distance to user profile
    semantic_distance: float  # Semantic distance to original query
    combined_score: float  # Weighted combination
    top_gaps: List[Dict[str, Any]]
    num_candidates: int
    converged: bool


@dataclass
class IterativeResult:
    """Result from iterative refinement process."""
    user_id: str
    asin: str
    base_query: str
    target_query: str  # Original for reference
    final_query: str
    final_distance: float
    rounds: List[RoundResult]
    total_candidates: int
    improvement: float  # Base distance - Final distance
    converged: bool
    convergence_reason: str  # "threshold", "max_rounds", "no_improvement"


# Feature metadata for generating targeted instructions
FEATURE_INSTRUCTIONS = {
    # Length features
    "tokens_per_sent": {
        "name": "Sentence Length",
        "low": "Use SHORTER, more concise sentences",
        "high": "Use LONGER, more complex sentences"
    },
    "avg_max_depth": {
        "name": "Syntactic Complexity",
        "low": "Use SIMPLER sentence structures",
        "high": "Use MORE COMPLEX, nested sentence structures"
    },
    "lexical_density": {
        "name": "Vocabulary",
        "low": "Use MORE CONVERSATIONAL, everyday language",
        "high": "Use MORE TECHNICAL, formal vocabulary"
    },
    "content_word_ratio": {
        "name": "Information Density",
        "low": "Use MORE CONTENT words (nouns, verbs, adjectives)",
        "high": "Use MORE FUNCTION words (articles, prepositions)"
    },
    # POS distribution features (normalized to 0-1, multiply by 100 for interpretation)
    "upos_dist_NOUN": {
        "name": "Noun Usage",
        "low": "Use FEWER nouns",
        "high": "Use MORE descriptive, noun-heavy language"
    },
    "upos_dist_VERB": {
        "name": "Verb Usage",
        "low": "Use FEWER action verbs",
        "high": "Use MORE action-oriented expressions"
    },
    "upos_dist_ADJ": {
        "name": "Adjective Usage",
        "low": "Use FEWER descriptive adjectives",
        "high": "Use MORE descriptive, colorful adjectives"
    },
    "upos_dist_ADV": {
        "name": "Adverb Usage",
        "low": "Use FEWER adverbs",
        "high": "Use MORE adverbs for description"
    },
    "upos_dist_PRON": {
        "name": "Pronoun Usage",
        "low": "Use MORE specific nouns instead of pronouns",
        "high": "Use MORE pronouns (personal references)"
    },
    "upos_dist_DET": {
        "name": "Determiner Usage",
        "low": "Use FEWER articles/determiners",
        "high": "Use MORE specific determiners"
    },
    # Punctuation
    "punct_per_sent": {
        "name": "Punctuation",
        "low": "Use MORE punctuation marks (commas, etc.)",
        "high": "Use FEWER punctuation marks (simpler sentences)"
    },
    # Clause density
    "clause_density": {
        "name": "Clause Density",
        "low": "Use FEWER clauses per sentence",
        "high": "Use MORE clauses (compound-complex sentences)"
    },
    # Type-token ratio
    "ttr": {
        "name": "Vocabulary Diversity",
        "low": "Use MORE DIVERSE vocabulary (avoid repetition)",
        "high": "Use MORE CONSISTENT vocabulary"
    },
    # Readability
    "flesch_reading_ease": {
        "name": "Readability",
        "low": "Make text MORE DIFFICULT to read (academic style)",
        "high": "Make text EASIER to read (conversational style)"
    },
}


def analyze_feature_gaps(
    query_features: Dict[str, float],
    user_features: Dict[str, float],
    feature_weights: Optional[Dict[str, float]] = None
) -> List[FeatureGap]:
    """
    Analyze which features contribute most to the distance.

    Args:
        query_features: Features from generated query
        user_features: Target user's features
        feature_weights: Optional weights for features (default: all 1.0)

    Returns:
        List of FeatureGap sorted by gap size descending
    """
    gaps = []
    weights = feature_weights or {}

    for feature_name, user_val in user_features.items():
        query_val = query_features.get(feature_name, 0)
        gap = abs(user_val - query_val)
        direction = query_val - user_val  # Positive: query > user

        # Apply weight if specified
        weighted_gap = gap * weights.get(feature_name, 1.0)

        gaps.append(FeatureGap(
            feature_name=feature_name,
            gap_size=weighted_gap,
            direction=direction,
            user_value=user_val,
            query_value=query_val
        ))

    return sorted(gaps, key=lambda g: g.gap_size, reverse=True)


def generate_targeted_instructions(
    top_gaps: List[FeatureGap],
    max_instructions: int = 3
) -> List[str]:
    """
    Generate specific style instructions based on feature gaps.

    Args:
        top_gaps: Top feature gaps (sorted by size)
        max_instructions: Maximum number of instructions to generate

    Returns:
        List of specific instructions in natural language
    """
    instructions = []

    for gap in top_gaps[:max_instructions]:
        feature_info = FEATURE_INSTRUCTIONS.get(gap.feature_name)

        if not feature_info:
            # Generic instruction for unknown features
            instructions.append(
                f"Adjust {gap.feature_name} to be closer to target value"
            )
            continue

        # Determine direction
        if gap.direction < 0:
            # Query value < User value: need to increase
            instruction = feature_info.get("high", f"Increase {feature_info['name']}")
        else:
            # Query value > User value: need to decrease
            instruction = feature_info.get("low", f"Decrease {feature_info['name']}")

        # Add quantitative context
        instructions.append(
            f"{instruction} (current: {gap.query_value:.2f}, target: {gap.user_value:.2f})"
        )

    return instructions


def create_refined_prompt(
    base_query: str,
    category: str,
    attributes: List[str],
    style_description: str,
    targeted_instructions: List[str],
    current_round: int,
    previous_query: Optional[str] = None
) -> str:
    """
    Create a refinement prompt with specific targeted instructions.

    Args:
        base_query: Original base query
        category: Product category
        attributes: Product attributes
        style_description: General style description
        targeted_instructions: Specific instructions from gap analysis
        current_round: Current refinement round number
        previous_query: Best query from previous round (for refinement)

    Returns:
        Formatted prompt string
    """
    attrs_str = ", ".join(attributes[:3]) if attributes else "various features"

    if current_round == 0:
        # First round: general prompt
        prompt = f"""Task: Rewrite the following search query to match the user's writing style.

Rules:
- Keep the original meaning and product constraints unchanged.
- Keep it as a search query, not a question.
- Keep dependency-role balance close (subject/object/modifier relations).
- Do not add or remove negation/condition scope.
- Keep coordination hierarchy close (avoid changing parallel-structure depth).
- Do NOT use markdown, bullet points, or quotation marks.
- Output one plain-text query line only.

User Style: {style_description}
Product Category: {category}
Key Attributes: {attrs_str}

Original Query: {base_query}

Rewritten Query (matching user's writing style):"""
    else:
        # Subsequent rounds: targeted refinement
        instructions_str = "\n".join(f"- {ins}" for ins in targeted_instructions)

        prompt = f"""Task: Refine the query to better match the user's writing style.

Rules:
- Keep the meaning and key constraints unchanged.
- Keep it as a search query, not a question.
- Keep dependency-role balance close (subject/object/modifier relations).
- Do not add or remove negation/condition scope.
- Keep coordination hierarchy close (avoid changing parallel-structure depth).
- Do NOT use markdown, bullet points, or quotation marks.
- Output one plain-text query line only.

Product Category: {category}
Key Attributes: {attrs_str}

Current Best Query: {previous_query}

Specific Style Adjustments Needed:
{instructions_str}

Refined Query (incorporating the above adjustments):"""

    return prompt


def create_semantic_guardrail_prompt(
    current_query: str,
    category: str,
    attributes: List[str],
    style_description: str
) -> str:
    attrs_str = ", ".join(attributes[:3]) if attributes else "various features"
    return f"""Task: Make a MINIMAL rewrite of this search query to better match the user's style.

Rules:
- Keep all product entities, numbers, and constraints unchanged.
- Keep the exact meaning unchanged.
- Make a small wording change only.
- Keep it as a declarative search query (no questions).
- Keep dependency-role balance and coordination hierarchy close.
- Do not add or remove negation/condition scope.
- Do NOT use markdown, bullet points, or quotation marks.
- Output one plain-text query line only.

User Style: {style_description}
Product Category: {category}
Key Attributes: {attrs_str}

Current Query: {current_query}

Rewritten Query:"""


def sanitize_query_text(text: str) -> str:
    query = text.strip()
    for prefix in ["Rewritten Query:", "Query:", "The rewritten query is:", "Refined Query:"]:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()

    query = query.replace("**", "")
    query = query.replace("__", "")
    query = query.replace("`", "")

    while len(query) >= 2 and query[0] in ('"', "'") and query[-1] in ('"', "'"):
        query = query[1:-1].strip()

    query = query.replace("?", ".")
    query = " ".join(query.split())
    return query.strip()


NON_LLM_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "i", "if", "in", "into", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "without", "would"
}
NEGATION_MARKERS = {"no", "not", "never", "without", "free"}


def average_syntactic_signature(signatures: List[Dict[str, float]]) -> Dict[str, float]:
    valid = [sig for sig in signatures if sig]
    if not valid:
        return {
            **{f"dep_dist_{k}": 0.0 for k in SYNTAX_DEP_KEYS},
            "neg_scope": 0.0,
            "cond_scope": 0.0,
            "coord_depth": 0.0,
        }

    keys = set()
    for sig in valid:
        keys.update(sig.keys())

    averaged: Dict[str, float] = {}
    denom = float(len(valid))
    for key in keys:
        averaged[key] = sum(sig.get(key, 0.0) for sig in valid) / denom
    return averaged


def _replace_once_case_insensitive(text: str, old: str, new: str) -> Optional[str]:
    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    if not pattern.search(text):
        return None
    return sanitize_query_text(pattern.sub(new, text, count=1))


def _match_case_phrase(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_word_case_insensitive(text: str, old: str, new: str) -> Optional[str]:
    pattern = re.compile(rf"\b{re.escape(old)}\b", flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        return _match_case_phrase(match.group(0), new)

    if not pattern.search(text):
        return None
    return sanitize_query_text(pattern.sub(repl, text, count=1))


def _split_query_sentences(query: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", query.strip())
    return [part.strip() for part in parts if part.strip()]


def _ensure_terminal_punctuation(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return sentence
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _join_query_sentences(sentences: List[str]) -> str:
    return sanitize_query_text(" ".join(_ensure_terminal_punctuation(sentence) for sentence in sentences if sentence.strip()))


def _rewrite_sentence_prefix(sentence: str, new_prefix: str) -> Optional[str]:
    patterns = [
        r"^I am looking for\s+",
        r"^Looking for\s+",
        r"^I need\s+",
        r"^I want\s+",
        r"^Please show me\s+",
        r"^Show me\s+",
    ]
    for pattern in patterns:
        match = re.match(pattern, sentence, flags=re.IGNORECASE)
        if match:
            tail = sentence[match.end():].strip()
            if not tail:
                return None
            rewritten = f"{new_prefix}{tail}"
            end = sentence[-1] if sentence and sentence[-1] in ".!?" else ""
            return sanitize_query_text(rewritten + end)
    return None


def _count_content_word_changes(base_query: str, candidate_query: str) -> int:
    base_tokens = [
        token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", base_query.lower())
        if token not in NON_LLM_STOPWORDS and len(token) > 2
    ]
    cand_tokens = [
        token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", candidate_query.lower())
        if token not in NON_LLM_STOPWORDS and len(token) > 2
    ]
    base_counter = Counter(base_tokens)
    cand_counter = Counter(cand_tokens)
    return sum((base_counter - cand_counter).values()) + sum((cand_counter - base_counter).values())


def compute_edit_cost(base_query: str, candidate_query: str) -> float:
    base_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", base_query.lower())
    cand_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", candidate_query.lower())
    if not base_tokens and not cand_tokens:
        return 0.0
    base_counter = Counter(base_tokens)
    cand_counter = Counter(cand_tokens)
    changed = sum((base_counter - cand_counter).values()) + sum((cand_counter - base_counter).values())
    return changed / max(1, max(len(base_tokens), len(cand_tokens)))


def has_consistent_scope_markers(base_query: str, candidate_query: str) -> bool:
    base_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", base_query.lower())
    cand_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", candidate_query.lower())

    base_neg = sorted(token for token in base_tokens if token in NEGATION_MARKERS)
    cand_neg = sorted(token for token in cand_tokens if token in NEGATION_MARKERS)
    if base_neg != cand_neg:
        return False

    base_cond = sorted(token for token in base_tokens if token in CONDITIONAL_MARKERS)
    cand_cond = sorted(token for token in cand_tokens if token in CONDITIONAL_MARKERS)
    return base_cond == cand_cond


def is_within_edit_budget(base_query: str, candidate_query: str, style_strength: str) -> bool:
    cost = compute_edit_cost(base_query, candidate_query)
    changed_content = _count_content_word_changes(base_query, candidate_query)
    max_cost = {
        "weak": 0.22,
        "medium": 0.34,
        "strong": 0.42,
    }.get((style_strength or "medium").lower(), 0.34)
    if cost > max_cost:
        return False
    return changed_content <= 3


def retrieval_proxy_risk(
    base_query: str,
    candidate_query: str,
    hard_terms: List[str],
    soft_terms: List[str],
) -> float:
    base_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", base_query.lower()))
    cand_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", candidate_query.lower()))
    removed_content = [
        token for token in base_tokens - cand_tokens
        if token not in NON_LLM_STOPWORDS and token not in hard_terms and token not in soft_terms and len(token) > 2
    ]
    return len(removed_content) / max(1, len(base_tokens))


def operator_preference_bonus(operator_name: str) -> float:
    if operator_name.startswith("merge_"):
        return 0.020
    if operator_name.startswith("support_clause_"):
        return 0.016
    if operator_name.startswith("sent_prefix_align_"):
        return 0.010
    if operator_name.startswith("pp_reorder_"):
        return 0.008
    return 0.0


def _maybe_add_candidate(
    bucket: List[Tuple[str, str]],
    seen: set,
    candidate_query: Optional[str],
    operator_name: str,
    base_query: str,
    hard_terms: List[str],
    soft_terms: List[str],
    style_strength: str,
) -> None:
    if not candidate_query:
        return
    candidate_query = sanitize_query_text(candidate_query)
    if not candidate_query or candidate_query == base_query or candidate_query in seen:
        return
    if re.match(r"^(with|for|that|and|or)\b", candidate_query.strip(), flags=re.IGNORECASE):
        return
    if not preserves_protected_terms(base_query, candidate_query, hard_terms, soft_terms):
        return
    if not has_consistent_scope_markers(base_query, candidate_query):
        return
    if not is_within_edit_budget(base_query, candidate_query, style_strength):
        return
    seen.add(candidate_query)
    bucket.append((candidate_query, operator_name))


def _generate_prefix_candidates(query: str) -> List[Tuple[str, str]]:
    lower = query.lower()
    candidates: List[Tuple[str, str]] = []
    if lower.startswith("i am looking for "):
        tail = query[len("I am looking for "):]
        candidates.extend([
            (sanitize_query_text(f"I need {tail}"), "prefix_i_need"),
            (sanitize_query_text(f"I want {tail}"), "prefix_i_want"),
            (sanitize_query_text(f"Looking for {tail}"), "prefix_drop_copula"),
        ])
    elif lower.startswith("looking for "):
        tail = query[len("Looking for "):]
        candidates.extend([
            (sanitize_query_text(f"I am looking for {tail}"), "prefix_add_copula"),
            (sanitize_query_text(f"I need {tail}"), "prefix_need"),
        ])
    elif lower.startswith("i need "):
        tail = query[len("I need "):]
        candidates.extend([
            (sanitize_query_text(f"I am looking for {tail}"), "prefix_lookup"),
            (sanitize_query_text(f"I want {tail}"), "prefix_want"),
        ])
    elif lower.startswith("i want "):
        tail = query[len("I want "):]
        candidates.extend([
            (sanitize_query_text(f"I am looking for {tail}"), "prefix_lookup"),
            (sanitize_query_text(f"I need {tail}"), "prefix_need"),
        ])
    return candidates


def _generate_connector_candidates(query: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    replacements = [
        (" with ", " featuring ", "connector_featuring"),
        (" featuring ", " with ", "connector_with"),
        (" with ", " that has ", "connector_that_has"),
        (" that has ", " with ", "connector_with_compact"),
        (" that have ", " with ", "connector_with_plural"),
    ]
    for old, new, name in replacements:
        if old == " with " and new == " that has " and re.search(r"\b(works?|working) well with\b", query, flags=re.IGNORECASE):
            continue
        updated = _replace_once_case_insensitive(query, old, new)
        if updated:
            candidates.append((updated, name))
    return candidates


def _generate_coordination_candidates(query: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if " and " in query.lower():
        updated = _replace_once_case_insensitive(query, " and ", ", ")
        if updated:
            candidates.append((updated, "coord_compress"))
    if ", " in query:
        last_comma = query.rfind(", ")
        if last_comma != -1:
            updated = sanitize_query_text(query[:last_comma] + " and " + query[last_comma + 2:])
            candidates.append((updated, "coord_expand"))
    return candidates


def _generate_modifier_swap_candidates(query: str) -> List[Tuple[str, str]]:
    extractor = _get_feature_extractor()
    if extractor is None:
        return []
    try:
        doc = extractor.nlp(query)
    except Exception:
        return []

    candidates: List[Tuple[str, str]] = []
    for token in doc:
        amods = [child for child in token.children if child.dep_ == "amod"]
        if len(amods) < 2:
            continue
        amods = sorted(amods, key=lambda t: t.i)
        first, second = amods[0], amods[1]
        pieces = [t.text for t in doc]
        pieces[first.i], pieces[second.i] = pieces[second.i], pieces[first.i]
        candidates.append((sanitize_query_text(" ".join(pieces)), "modifier_swap"))
        break
    return candidates


def _generate_preposition_reorder_candidates(query: str) -> List[Tuple[str, str]]:
    patterns = [
        (r"^(.*?\bwith\b\s+.+?)\s+(\bfor\b\s+.+)$", r"\2 \1", "pp_reorder_for_first"),
        (r"^(.*?\bfor\b\s+.+?)\s+(\bwith\b\s+.+)$", r"\2 \1", "pp_reorder_with_first"),
    ]
    candidates: List[Tuple[str, str]] = []
    for pattern, repl, name in patterns:
        if re.search(pattern, query, flags=re.IGNORECASE):
            updated = sanitize_query_text(re.sub(pattern, repl, query, count=1, flags=re.IGNORECASE))
            if updated != query:
                candidates.append((updated, name))
    return candidates


def _generate_sentence_prefix_alignment_candidates(query: str) -> List[Tuple[str, str]]:
    sentences = _split_query_sentences(query)
    if len(sentences) < 2:
        return []

    candidates: List[Tuple[str, str]] = []
    prefix_targets = ["I am looking for ", "I need ", "I want "]

    for idx, sentence in enumerate(sentences):
        for prefix in prefix_targets:
            rewritten = _rewrite_sentence_prefix(sentence, prefix)
            if not rewritten or rewritten == sentence:
                continue
            updated_sentences = list(sentences)
            updated_sentences[idx] = rewritten
            candidates.append((_join_query_sentences(updated_sentences), f"sent_prefix_align_{idx}"))

    return candidates


def _generate_sentence_merge_candidates(query: str) -> List[Tuple[str, str]]:
    sentences = _split_query_sentences(query)
    if len(sentences) < 2:
        return []

    candidates: List[Tuple[str, str]] = []
    merge_patterns = [
        (r"^I am looking for\s+(.+)$", r"^I need\s+(.+)$", "I am looking for {first} and need {second}.", "merge_lookup_need"),
        (r"^I am looking for\s+(.+)$", r"^I want\s+(.+)$", "I am looking for {first} and want {second}.", "merge_lookup_want"),
        (r"^I need\s+(.+)$", r"^I am looking for\s+(.+)$", "I need {first} and am looking for {second}.", "merge_need_lookup"),
        (r"^I am looking for\s+(.+)$", r"^Please show me\s+(.+)$", "I am looking for {first}, and please show me {second}.", "merge_lookup_show"),
    ]

    for i in range(len(sentences) - 1):
        first = sentences[i].rstrip(".!?")
        second = sentences[i + 1].rstrip(".!?")
        for p1, p2, template, name in merge_patterns:
            m1 = re.match(p1, first, flags=re.IGNORECASE)
            m2 = re.match(p2, second, flags=re.IGNORECASE)
            if not m1 or not m2:
                continue
            merged = template.format(first=m1.group(1).strip(), second=m2.group(1).strip())
            updated_sentences = sentences[:i] + [merged] + sentences[i + 2:]
            candidates.append((_join_query_sentences(updated_sentences), name))

    return candidates


def _generate_supporting_clause_candidates(query: str) -> List[Tuple[str, str]]:
    sentences = _split_query_sentences(query)
    if len(sentences) < 2:
        return []

    candidates: List[Tuple[str, str]] = []
    for i in range(len(sentences) - 1):
        first = sentences[i].rstrip(".!?")
        second = sentences[i + 1].rstrip(".!?")

        if re.match(r"^I am looking for\s+", first, flags=re.IGNORECASE) and re.match(r"^I need\s+", second, flags=re.IGNORECASE):
            rewritten_second = _rewrite_sentence_prefix(second + ".", "with ")
            if rewritten_second:
                rewritten_second = re.sub(r"^with\s+", "with ", rewritten_second.rstrip(".!?"), flags=re.IGNORECASE)
                merged = sanitize_query_text(f"{first} {rewritten_second}.")
                updated_sentences = sentences[:i] + [merged] + sentences[i + 2:]
                candidates.append((_join_query_sentences(updated_sentences), "support_clause_with"))

        if re.match(r"^I am looking for\s+", first, flags=re.IGNORECASE) and re.match(r"^Please show me\s+", second, flags=re.IGNORECASE):
            show_tail = re.sub(r"^Please show me\s+", "", second, flags=re.IGNORECASE)
            merged = sanitize_query_text(f"{first}, and please show me {show_tail}.")
            updated_sentences = sentences[:i] + [merged] + sentences[i + 2:]
            candidates.append((_join_query_sentences(updated_sentences), "support_clause_show"))

    return candidates


def generate_candidates_without_llm(
    current_query: str,
    protected_hard_terms: List[str],
    protected_soft_terms: List[str],
    style_strength: str,
    candidates_per_round: int,
) -> List[Tuple[str, float, str]]:
    raw_candidates: List[Tuple[str, str]] = []
    seen_queries = set()

    generators = [
        _generate_prefix_candidates,
        _generate_sentence_prefix_alignment_candidates,
        _generate_sentence_merge_candidates,
        _generate_supporting_clause_candidates,
        _generate_connector_candidates,
        _generate_coordination_candidates,
        _generate_modifier_swap_candidates,
        _generate_preposition_reorder_candidates,
    ]

    for generator in generators:
        for candidate_query, operator_name in generator(current_query):
            _maybe_add_candidate(
                raw_candidates,
                seen_queries,
                candidate_query,
                operator_name,
                current_query,
                protected_hard_terms,
                protected_soft_terms,
                style_strength,
            )

    scored_candidates: List[Tuple[str, float, str]] = []
    target_edit_cost = {
        "weak": 0.07,
        "medium": 0.12,
        "strong": 0.18,
    }.get((style_strength or "medium").lower(), 0.12)
    for candidate_query, operator_name in raw_candidates:
        cost = compute_edit_cost(current_query, candidate_query)
        scored_candidates.append((candidate_query, cost, operator_name))

    scored_candidates.sort(key=lambda item: (abs(item[1] - target_edit_cost), item[1], len(item[0])))
    budget = max(6, candidates_per_round * 4)
    return scored_candidates[:budget]


def _get_feature_extractor() -> Optional[Any]:
    if SentenceLevelFeatureExtractor is None:
        return None
    extractor = getattr(_feature_extractor_local, "extractor", None)
    if extractor is None:
        extractor = SentenceLevelFeatureExtractor()
        _feature_extractor_local.extractor = extractor
    return extractor


def _extract_syntactic_signature_from_doc(doc: Any) -> Dict[str, float]:
    tokens = [t for t in doc if not getattr(t, "is_space", False)]
    n_tokens = max(1, len(tokens))

    dep_counts = defaultdict(int)
    neg_count = 0
    cond_count = 0
    max_coord_depth = 0

    for token in tokens:
        dep = token.dep_.lower() if token.dep_ else ""
        if dep in SYNTAX_DEP_KEYS:
            dep_counts[dep] += 1
        if dep == "neg":
            neg_count += 1

        lower_text = token.text.lower()
        if lower_text in CONDITIONAL_MARKERS:
            cond_count += 1
        elif dep == "mark" and lower_text in CONDITIONAL_MARKERS:
            cond_count += 1

        if dep == "conj":
            depth = 1
            head = token.head
            while head is not None and head is not token and head.dep_.lower() == "conj":
                depth += 1
                head = head.head
            if depth > max_coord_depth:
                max_coord_depth = depth

    dep_dist = {f"dep_dist_{k}": dep_counts.get(k, 0) / n_tokens for k in SYNTAX_DEP_KEYS}
    signature = {
        **dep_dist,
        "neg_scope": neg_count / n_tokens,
        "cond_scope": cond_count / n_tokens,
        "coord_depth": float(max_coord_depth),
    }
    return signature


def extract_syntactic_signatures(queries: List[str]) -> List[Dict[str, float]]:
    extractor = _get_feature_extractor()
    if extractor is None:
        return [{
            **{f"dep_dist_{k}": 0.0 for k in SYNTAX_DEP_KEYS},
            "neg_scope": 0.0,
            "cond_scope": 0.0,
            "coord_depth": 0.0,
        } for _ in queries]

    try:
        docs = list(extractor.nlp.pipe(queries))
        return [_extract_syntactic_signature_from_doc(doc) for doc in docs]
    except Exception as e:
        logger.warning(f"Syntactic signature extraction failed: {e}")
        return [{
            **{f"dep_dist_{k}": 0.0 for k in SYNTAX_DEP_KEYS},
            "neg_scope": 0.0,
            "cond_scope": 0.0,
            "coord_depth": 0.0,
        } for _ in queries]


def compute_syntactic_distance(base_sig: Dict[str, float], cand_sig: Dict[str, float]) -> Tuple[float, float, float, float]:
    dep_diff = sum(
        abs(base_sig.get(f"dep_dist_{k}", 0.0) - cand_sig.get(f"dep_dist_{k}", 0.0))
        for k in SYNTAX_DEP_KEYS
    ) / max(1, len(SYNTAX_DEP_KEYS))

    scope_diff = (
        abs(base_sig.get("neg_scope", 0.0) - cand_sig.get("neg_scope", 0.0)) +
        abs(base_sig.get("cond_scope", 0.0) - cand_sig.get("cond_scope", 0.0))
    )

    coord_diff = abs(base_sig.get("coord_depth", 0.0) - cand_sig.get("coord_depth", 0.0))
    coord_diff_norm = min(coord_diff / 3.0, 1.0)

    total = 0.60 * dep_diff + 0.25 * scope_diff + 0.15 * coord_diff_norm
    return total, dep_diff, scope_diff, coord_diff


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-12:
        return [0.5 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def _normalized_dep_entropy(signature: Dict[str, float]) -> float:
    probs = [max(signature.get(f"dep_dist_{k}", 0.0), 0.0) for k in SYNTAX_DEP_KEYS]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs if p > 0]
    if not probs:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy / math.log(len(SYNTAX_DEP_KEYS)))


def build_user_syntactic_complexity_report(users_queries: Dict[str, List[Dict]], user_ids: List[str]) -> Dict[str, Any]:
    user_metrics: List[Dict[str, Any]] = []

    for user_id in sorted(user_ids):
        entries = users_queries.get(user_id, [])
        user_queries = [q.get("personalized_query", "") for q in entries if q.get("personalized_query")]
        if not user_queries:
            continue

        signatures = extract_syntactic_signatures(user_queries)
        if not signatures:
            continue

        dep_entropy_values = [_normalized_dep_entropy(sig) for sig in signatures]
        scope_values = [max(sig.get("neg_scope", 0.0), 0.0) + max(sig.get("cond_scope", 0.0), 0.0) for sig in signatures]
        coord_values = [max(sig.get("coord_depth", 0.0), 0.0) for sig in signatures]

        user_metrics.append({
            "user_id": user_id,
            "num_queries": len(user_queries),
            "dep_entropy": float(np.mean(dep_entropy_values)) if dep_entropy_values else 0.0,
            "scope_intensity": float(np.mean(scope_values)) if scope_values else 0.0,
            "coord_depth": float(np.mean(coord_values)) if coord_values else 0.0,
        })

    if not user_metrics:
        return {
            "definition": {
                "level_rule": "combined_score tertiles: [0,1/3)=low, [1/3,2/3)=medium, [2/3,1]=high",
                "metrics": ["dep_entropy", "scope_intensity", "coord_depth"],
            },
            "users": [],
        }

    dep_vals = [m["dep_entropy"] for m in user_metrics]
    scope_vals = [m["scope_intensity"] for m in user_metrics]
    coord_vals = [m["coord_depth"] for m in user_metrics]

    dep_norm = _min_max_normalize(dep_vals)
    scope_norm = _min_max_normalize(scope_vals)
    coord_norm = _min_max_normalize(coord_vals)

    users_out: List[Dict[str, Any]] = []
    for m, d_n, s_n, c_n in zip(user_metrics, dep_norm, scope_norm, coord_norm):
        combined = float((d_n + s_n + c_n) / 3.0)
        if combined < (1.0 / 3.0):
            level = "low_complexity"
        elif combined < (2.0 / 3.0):
            level = "medium_complexity"
        else:
            level = "high_complexity"

        users_out.append({
            "user_id": m["user_id"],
            "num_queries": m["num_queries"],
            "metrics": {
                "dep_entropy": round(m["dep_entropy"], 6),
                "scope_intensity": round(m["scope_intensity"], 6),
                "coord_depth": round(m["coord_depth"], 6),
            },
            "normalized": {
                "dep_entropy": round(float(d_n), 6),
                "scope_intensity": round(float(s_n), 6),
                "coord_depth": round(float(c_n), 6),
            },
            "combined_score": round(combined, 6),
            "complexity_level": level,
        })

    users_out.sort(key=lambda x: x["combined_score"])

    return {
        "definition": {
            "level_rule": "combined_score tertiles: [0,1/3)=low, [1/3,2/3)=medium, [2/3,1]=high",
            "metrics": ["dep_entropy", "scope_intensity", "coord_depth"],
        },
        "ranges": {
            "dep_entropy": {"min": round(float(min(dep_vals)), 6), "max": round(float(max(dep_vals)), 6)},
            "scope_intensity": {"min": round(float(min(scope_vals)), 6), "max": round(float(max(scope_vals)), 6)},
            "coord_depth": {"min": round(float(min(coord_vals)), 6), "max": round(float(max(coord_vals)), 6)},
        },
        "users": users_out,
    }


def extract_protected_term_groups(base_query: str, category: str, attributes: List[str]) -> Dict[str, List[str]]:
    hard_terms = set()
    soft_terms = set()

    def _add_terms(text: str):
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", text):
            t = token.strip().lower()
            if not t:
                continue
            if any(ch.isdigit() for ch in t) or "-" in t or len(t) >= 8:
                hard_terms.add(t)
            elif len(t) >= 4:
                soft_terms.add(t)

    _add_terms(base_query)
    _add_terms(category)
    for attr in attributes:
        _add_terms(attr)

    soft_terms = {t for t in soft_terms if t not in hard_terms}
    return {
        "hard": sorted(hard_terms),
        "soft": sorted(soft_terms),
    }


def preserves_protected_terms(
    base_query: str,
    candidate_query: str,
    hard_terms: List[str],
    soft_terms: List[str],
    soft_coverage_threshold: float = 0.6,
) -> bool:
    if not hard_terms and not soft_terms:
        return True

    base_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", base_query.lower()))
    cand_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", candidate_query.lower()))

    required_hard = {t for t in hard_terms if t in base_tokens}
    if not required_hard.issubset(cand_tokens):
        return False

    required_soft = [t for t in soft_terms if t in base_tokens]
    if not required_soft:
        return True

    kept_soft = sum(1 for t in required_soft if t in cand_tokens)
    soft_coverage = kept_soft / max(1, len(required_soft))
    return soft_coverage >= soft_coverage_threshold


def create_local_edit_prompt(
    current_query: str,
    category: str,
    attributes: List[str],
    style_description: str,
    protected_hard_terms: List[str],
    style_strength: str,
) -> str:
    attrs_str = ", ".join(attributes[:3]) if attributes else "various features"
    protected_str = ", ".join(protected_hard_terms[:20]) if protected_hard_terms else "keep critical product terms"

    rewrite_scope = {
        "weak": "Change at most 1-2 short spans.",
        "medium": "Change at most 2-4 short spans.",
        "strong": "Change at most 4-6 short spans.",
    }.get(style_strength, "Change at most 2-4 short spans.")

    return f"""Task: Locally edit this search query to better match user style.

Rules:
- Keep meaning unchanged.
- Keep critical product terms unchanged: {protected_str}
- Keep it as a declarative search query (not a question).
- Keep dependency-role balance and coordination hierarchy close.
- Do not add or remove negation/condition scope.
- Do NOT use markdown, bullet points, or quotation marks.
- {rewrite_scope}
- Output one plain-text query line only.

User Style: {style_description}
Product Category: {category}
Key Attributes: {attrs_str}

Current Query: {current_query}

Edited Query:"""


def average_feature_vectors(feature_list: List[Dict[str, float]]) -> Dict[str, float]:
    valid = [f for f in feature_list if f]
    if not valid:
        return {}

    keys = set()
    for vec in valid:
        keys.update(vec.keys())

    averaged: Dict[str, float] = {}
    denom = float(len(valid))
    for key in keys:
        averaged[key] = sum(vec.get(key, 0.0) for vec in valid) / denom
    return averaged


def cosine_distance(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine distance between two feature vectors."""
    import numpy as np

    common_keys = set(vec1.keys()) & set(vec2.keys())

    if not common_keys:
        return 2.0

    v1 = np.array([vec1.get(k, 0) for k in common_keys])
    v2 = np.array([vec2.get(k, 0) for k in common_keys])

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 2.0

    cosine_sim = dot_product / (norm1 * norm2)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))

    return 1.0 - cosine_sim


def semantic_similarity(query1: str, query2: str) -> float:
    """Compute semantic similarity using embedding vectors (cosine similarity)."""
    model = get_semantic_model()

    try:
        # Get embeddings
        embeddings = model.encode([query1, query2])

        # Compute cosine similarity
        vec1, vec2 = embeddings[0], embeddings[1]

        # Handle edge cases
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # Clamp to [-1, 1]
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # Convert to [0, 1] range for distance calculation
        # Cosine similarity [-1, 1] -> [0, 1] where 1 = identical
        return (cosine_sim + 1) / 2
    except Exception as e:
        logger.warning(f"Embedding similarity failed: {e}, using word overlap fallback")
        # Fallback to word overlap
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


def semantic_distance(query1: str, query2: str) -> float:
    """Semantic distance = 1 - similarity (lower is better)."""
    return 1.0 - semantic_similarity(query1, query2)


def semantic_distances_to_target(queries: List[str], target_query: str) -> List[float]:
    model = get_semantic_model()
    if not queries:
        return []

    try:
        target_embedding = model.encode([target_query])[0]
        candidate_embeddings = model.encode(queries)

        target_norm = np.linalg.norm(target_embedding)
        if target_norm == 0:
            return [1.0] * len(queries)

        distances: List[float] = []
        for candidate_embedding in candidate_embeddings:
            candidate_norm = np.linalg.norm(candidate_embedding)
            if candidate_norm == 0:
                distances.append(1.0)
                continue
            cosine_sim = np.dot(candidate_embedding, target_embedding) / (candidate_norm * target_norm)
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            similarity = (cosine_sim + 1) / 2
            distances.append(1.0 - similarity)
        return distances
    except Exception:
        return [semantic_distance(query, target_query) for query in queries]


def generate_candidates_with_llm(
    llm_client: LLMClient,
    prompt: str,
    num_candidates: int = 5,
    temperature_base: float = 0.4,
    temperature_range: float = 0.4,
    delay_between_calls: float = 0.0
) -> List[Tuple[str, float]]:
    """Generate multiple candidates with different temperatures."""
    import time as _time
    candidates = []
    seen_queries = set()

    for i in range(num_candidates):
        temp = temperature_base + (i * temperature_range / num_candidates)

        # Add delay before API call to avoid rate limiting
        if i > 0 and delay_between_calls > 0:
            _time.sleep(delay_between_calls)

        response = llm_client.call(prompt, max_tokens=100, temperature=temp)

        if response:
            query = sanitize_query_text(response)
            if query and query not in seen_queries:
                seen_queries.add(query)
                candidates.append((query, temp))

    return candidates


def extract_query_features(
    queries: List[str],
    timeout: int = 60
) -> List[Dict[str, float]]:
    """Extract linguistic features from queries using SentenceLevelFeatureExtractor."""
    extractor = _get_feature_extractor()
    if extractor is None:
        logger.error("SentenceLevelFeatureExtractor unavailable")
        return [{}] * len(queries)

    try:
        features_list = []
        for query in queries:
            features = extractor.extract_profilingud_features(query)
            features_list.append(features if features else {})
        return features_list
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        return [{}] * len(queries)


def get_style_description(features: Dict[str, float]) -> str:
    """Generate style description from linguistic features."""
    descriptions = []

    tokens_per_sent = features.get("tokens_per_sent", 20)
    if tokens_per_sent < 15:
        descriptions.append("short, concise sentences")
    elif tokens_per_sent > 25:
        descriptions.append("long, complex sentences")

    tree_depth = features.get("avg_max_depth", 4)
    if tree_depth > 5:
        descriptions.append("complex, nested sentence structures")
    elif tree_depth < 3:
        descriptions.append("simple sentence structure")

    lexical_density = features.get("lexical_density", 0.5)
    if lexical_density > 0.6:
        descriptions.append("technical, formal vocabulary")
    elif lexical_density < 0.4:
        descriptions.append("conversational tone")

    if not descriptions:
        descriptions.append("natural writing style")

    return ", ".join(descriptions)


def get_style_strength_config(style_strength: str) -> Dict[str, float]:
    strength = (style_strength or "medium").lower()
    if strength == "weak":
        return {
            "temp_base": 0.25,
            "temp_range": 0.15,
            "fallback_temp_base": 0.12,
            "fallback_temp_range": 0.10,
            "style_degradation_allowance": 0.03,
            "style_weight_user": 0.65,
            "style_weight_exemplar": 0.10,
            "semantic_weight": 0.25,
            "syntax_weight": 0.18,
            "convergence_threshold": 0.005,
            "min_style_accept_threshold": 0.0015,
            "score_gain_threshold": 0.006,
            "combined_acceptance_allowance": 0.010,
            "no_improvement_patience": 2,
            "syntax_dep_threshold": 0.20,
            "syntax_scope_threshold": 0.06,
            "syntax_coord_threshold": 1.0,
            "syntax_distance_threshold": 0.22,
        }
    if strength == "strong":
        return {
            "temp_base": 0.55,
            "temp_range": 0.35,
            "fallback_temp_base": 0.20,
            "fallback_temp_range": 0.20,
            "style_degradation_allowance": 0.08,
            "style_weight_user": 0.55,
            "style_weight_exemplar": 0.20,
            "semantic_weight": 0.25,
            "syntax_weight": 0.22,
            "convergence_threshold": 0.005,
            "min_style_accept_threshold": 0.0010,
            "score_gain_threshold": 0.005,
            "combined_acceptance_allowance": 0.020,
            "no_improvement_patience": 2,
            "syntax_dep_threshold": 0.24,
            "syntax_scope_threshold": 0.08,
            "syntax_coord_threshold": 2.0,
            "syntax_distance_threshold": 0.28,
        }
    return {
        "temp_base": 0.40,
        "temp_range": 0.30,
        "fallback_temp_base": 0.15,
        "fallback_temp_range": 0.15,
        "style_degradation_allowance": 0.06,
        "style_weight_user": 0.60,
        "style_weight_exemplar": 0.15,
        "semantic_weight": 0.25,
        "syntax_weight": 0.20,
        "convergence_threshold": 0.005,
        "min_style_accept_threshold": 0.0012,
        "score_gain_threshold": 0.005,
        "combined_acceptance_allowance": 0.015,
        "no_improvement_patience": 2,
        "syntax_dep_threshold": 0.22,
        "syntax_scope_threshold": 0.07,
        "syntax_coord_threshold": 1.0,
        "syntax_distance_threshold": 0.25,
    }


def iterative_refine_query(
    user_id: str,
    asin: str,
    base_query: str,
    target_query: str,
    category: str,
    attributes: List[str],
    user_features: Dict[str, float],
    llm_client: Optional[LLMClient],
    max_rounds: int = 3,
    candidates_per_round: int = 3,
    convergence_threshold: float = 0.02,
    feature_weights: Optional[Dict[str, float]] = None,
    style_strength: str = "medium",
    exemplar_style_features: Optional[Dict[str, float]] = None,
    exemplar_syntax_signature: Optional[Dict[str, float]] = None,
) -> IterativeResult:
    """
    Perform iterative refinement with feature-aware prompting.

    Args:
        user_id: User identifier
        asin: Product ASIN
        base_query: Original query
        target_query: Reference personalized query
        category: Product category
        attributes: Product attributes
        user_features: Target user's linguistic features
        llm_client: LLM client for generation
        max_rounds: Maximum refinement rounds
        candidates_per_round: Candidates to generate per round
        convergence_threshold: Minimum improvement to continue
        feature_weights: Optional weights for features

    Returns:
        IterativeResult with all rounds and final best query
    """
    style_desc = get_style_description(user_features)
    protected_term_groups = extract_protected_term_groups(base_query, category, attributes)
    protected_hard_terms = protected_term_groups.get("hard", [])
    protected_soft_terms = protected_term_groups.get("soft", [])
    strength_cfg = get_style_strength_config(style_strength)
    effective_convergence_threshold = strength_cfg["convergence_threshold"]

    # Get base query features for comparison
    base_features_list = extract_query_features([base_query])
    base_features = base_features_list[0] if base_features_list else {}
    base_distance = cosine_distance(base_features, user_features) if base_features else 2.0
    base_syntax_signature = extract_syntactic_signatures([base_query])[0]

    rounds = []
    current_best_query = base_query
    current_best_distance = base_distance
    total_candidates = 0
    converged = False
    convergence_reason = ""
    low_gain_streak = 0

    for round_num in range(max_rounds):
        round_start = time.time()

        # Analyze current gaps
        current_features_list = extract_query_features([current_best_query])
        current_features = current_features_list[0] if current_features_list else {}

        gaps = analyze_feature_gaps(current_features, user_features, feature_weights)

        # Generate targeted instructions from top gaps
        top_gaps_data = [
            {
                "feature": g.feature_name,
                "gap": round(g.gap_size, 4),
                "direction": round(g.direction, 4),
                "user_val": round(g.user_value, 4),
                "query_val": round(g.query_value, 4)
            }
            for g in gaps[:5]
        ]

        targeted_instructions = generate_targeted_instructions(gaps, max_instructions=3)

        candidates_with_temp = generate_candidates_without_llm(
            current_query=current_best_query,
            protected_hard_terms=protected_hard_terms,
            protected_soft_terms=protected_soft_terms,
            style_strength=style_strength,
            candidates_per_round=candidates_per_round,
        )

        total_candidates += len(candidates_with_temp)

        # Extract features from candidates
        if candidates_with_temp:
            candidate_texts = [c[0] for c in candidates_with_temp]
            candidate_features = extract_query_features(candidate_texts)
            candidate_semantic_distances = semantic_distances_to_target(candidate_texts, target_query)
            candidate_syntax_signatures = extract_syntactic_signatures(candidate_texts)

            STYLE_WEIGHT_USER = strength_cfg["style_weight_user"]
            STYLE_WEIGHT_EXEMPLAR = strength_cfg["style_weight_exemplar"] if exemplar_style_features else 0.0
            SEMANTIC_WEIGHT = strength_cfg["semantic_weight"]
            SYNTAX_WEIGHT = strength_cfg["syntax_weight"]
            EDIT_WEIGHT = 0.07
            RETRIEVAL_RISK_WEIGHT = 0.10
            MIN_STYLE_ACCEPT_THRESHOLD = strength_cfg["min_style_accept_threshold"]
            SCORE_GAIN_THRESHOLD = strength_cfg["score_gain_threshold"]
            COMBINED_ACCEPTANCE_ALLOWANCE = strength_cfg["combined_acceptance_allowance"]
            NO_IMPROVEMENT_PATIENCE = int(strength_cfg["no_improvement_patience"])
            STYLE_DISTANCE_THRESHOLD = 0.3
            SEMANTIC_DISTANCE_THRESHOLD = 0.10
            SYNTAX_DEP_THRESHOLD = strength_cfg["syntax_dep_threshold"]
            SYNTAX_SCOPE_THRESHOLD = strength_cfg["syntax_scope_threshold"]
            SYNTAX_COORD_THRESHOLD = strength_cfg["syntax_coord_threshold"]
            SYNTAX_DISTANCE_THRESHOLD = strength_cfg["syntax_distance_threshold"]

            current_query_semantic_dist = semantic_distance(current_best_query, target_query)
            current_query_syntax_signature = extract_syntactic_signatures([current_best_query])[0]
            current_query_syntax_dist, _, _, _ = compute_syntactic_distance(
                exemplar_syntax_signature or base_syntax_signature,
                current_query_syntax_signature,
            )
            current_exemplar_style_dist = current_best_distance
            if exemplar_style_features and current_features:
                current_exemplar_style_dist = cosine_distance(current_features, exemplar_style_features)
            current_combined_score = (
                STYLE_WEIGHT_USER * current_best_distance
                + STYLE_WEIGHT_EXEMPLAR * current_exemplar_style_dist
                + SEMANTIC_WEIGHT * current_query_semantic_dist
                + SYNTAX_WEIGHT * current_query_syntax_dist
            )
            round_best_score = float('inf')
            round_best_style_dist = current_best_distance
            round_best_semantic_dist = current_query_semantic_dist
            round_best_query = current_best_query
            has_semantic_valid_candidate = False
            target_syntax_signature = exemplar_syntax_signature or base_syntax_signature

            for idx, ((query, edit_cost, operator_name), features) in enumerate(zip(candidates_with_temp, candidate_features)):
                if features:
                    # Style distance: query vs user profile
                    style_dist = cosine_distance(features, user_features)

                    # Semantic distance: query vs original personalized_query (target_query)
                    semantic_dist = candidate_semantic_distances[idx]

                    if not preserves_protected_terms(
                        base_query,
                        query,
                        protected_hard_terms,
                        protected_soft_terms,
                    ):
                        continue

                    if semantic_dist > SEMANTIC_DISTANCE_THRESHOLD:
                        continue

                    syntax_dist, dep_diff, scope_diff, coord_diff = compute_syntactic_distance(
                        target_syntax_signature,
                        candidate_syntax_signatures[idx],
                    )
                    if dep_diff > SYNTAX_DEP_THRESHOLD:
                        continue
                    if scope_diff > SYNTAX_SCOPE_THRESHOLD:
                        continue
                    if coord_diff > SYNTAX_COORD_THRESHOLD:
                        continue
                    if syntax_dist > SYNTAX_DISTANCE_THRESHOLD:
                        continue

                    if not has_consistent_scope_markers(base_query, query):
                        continue

                    if not is_within_edit_budget(base_query, query, style_strength):
                        continue

                    exemplar_style_dist = style_dist
                    if exemplar_style_features:
                        exemplar_style_dist = cosine_distance(features, exemplar_style_features)

                    risk = retrieval_proxy_risk(
                        base_query,
                        query,
                        protected_hard_terms,
                        protected_soft_terms,
                    )

                    combined_score = (
                        STYLE_WEIGHT_USER * style_dist
                        + STYLE_WEIGHT_EXEMPLAR * exemplar_style_dist
                        + SEMANTIC_WEIGHT * semantic_dist
                        + SYNTAX_WEIGHT * syntax_dist
                        + EDIT_WEIGHT * edit_cost
                        + RETRIEVAL_RISK_WEIGHT * risk
                        - operator_preference_bonus(operator_name)
                    )

                    if combined_score < round_best_score:
                        has_semantic_valid_candidate = True
                        round_best_score = combined_score
                        round_best_style_dist = style_dist
                        round_best_semantic_dist = semantic_dist
                        round_best_query = query

            round_elapsed = time.time() - round_start

            improvement = current_best_distance - round_best_style_dist
            combined_gain = current_combined_score - round_best_score if has_semantic_valid_candidate else float("-inf")
            accepted_update = (
                has_semantic_valid_candidate and
                round_best_query != current_best_query and
                (
                    improvement >= MIN_STYLE_ACCEPT_THRESHOLD or
                    (
                        combined_gain >= SCORE_GAIN_THRESHOLD and
                        round_best_style_dist <= current_best_distance + COMBINED_ACCEPTANCE_ALLOWANCE
                    )
                )
            )
            round_converged = (
                (round_best_style_dist < STYLE_DISTANCE_THRESHOLD and
                 round_best_semantic_dist < SEMANTIC_DISTANCE_THRESHOLD) or
                (low_gain_streak + 1 >= NO_IMPROVEMENT_PATIENCE and improvement < effective_convergence_threshold and combined_gain < SCORE_GAIN_THRESHOLD) or
                round_num == max_rounds - 1
            )

            round_result = RoundResult(
                round_num=round_num,
                best_query=round_best_query,
                best_distance=round_best_style_dist,  # For backward compatibility
                style_distance=round_best_style_dist,
                semantic_distance=round_best_semantic_dist,
                combined_score=round_best_score,
                top_gaps=top_gaps_data,
                num_candidates=len(candidates_with_temp),
                converged=bool(round_converged)  # Ensure Python bool
            )
            rounds.append(round_result)

            # Update for next round (track style distance)
            STYLE_DEGRADATION_ALLOWANCE = strength_cfg["style_degradation_allowance"]
            if accepted_update and round_best_style_dist <= current_best_distance + STYLE_DEGRADATION_ALLOWANCE:
                current_best_query = round_best_query
                current_best_distance = round_best_style_dist
                if improvement >= effective_convergence_threshold or combined_gain >= SCORE_GAIN_THRESHOLD:
                    low_gain_streak = 0
                else:
                    low_gain_streak += 1
            else:
                low_gain_streak += 1

            logger.info(
                f"  Round {round_num + 1}/{max_rounds}: "
                f"style_dist={round_best_style_dist:.4f}, "
                f"semantic_dist={round_best_semantic_dist:.4f}, "
                f"combined={round_best_score:.4f} "
                f"(style_improvement={improvement:.4f}, combined_gain={combined_gain:.4f}, accepted={accepted_update}) "
                f"in {round_elapsed:.1f}s"
            )

            # Check convergence conditions
            if not has_semantic_valid_candidate:
                converged = True
                convergence_reason = f"semantic_guardrail_no_candidate (semantic<{SEMANTIC_DISTANCE_THRESHOLD})"
                break
            elif (round_best_style_dist < STYLE_DISTANCE_THRESHOLD and
                round_best_semantic_dist < SEMANTIC_DISTANCE_THRESHOLD):
                converged = True
                convergence_reason = f"both_thresholds_met (style={round_best_style_dist:.4f}<{STYLE_DISTANCE_THRESHOLD}, semantic={round_best_semantic_dist:.4f}<{SEMANTIC_DISTANCE_THRESHOLD})"
                break
            elif low_gain_streak >= NO_IMPROVEMENT_PATIENCE and improvement < effective_convergence_threshold and combined_gain < SCORE_GAIN_THRESHOLD:
                converged = True
                convergence_reason = f"no_improvement ({improvement:.4f} < {effective_convergence_threshold}, combined_gain={combined_gain:.4f} < {SCORE_GAIN_THRESHOLD})"
                break
        else:
            logger.warning(f"  Round {round_num + 1}: No candidates generated")
            round_result = RoundResult(
                round_num=round_num,
                best_query=current_best_query,
                best_distance=current_best_distance,
                style_distance=current_best_distance,
                semantic_distance=1.0,  # No candidates, worst semantic
                combined_score=float('inf'),
                top_gaps=top_gaps_data,
                num_candidates=0,
                converged=True  # Python bool
            )
            rounds.append(round_result)
            break

    if not converged:
        convergence_reason = f"max_rounds_reached ({max_rounds})"

    return IterativeResult(
        user_id=user_id,
        asin=asin,
        base_query=base_query,
        target_query=target_query,
        final_query=current_best_query,
        final_distance=current_best_distance,
        rounds=rounds,
        total_candidates=total_candidates,
        improvement=base_distance - current_best_distance,
        converged=bool(converged),  # Ensure Python bool
        convergence_reason=convergence_reason
    )


def load_linguistic_profiles(
    linguistic_dir: str,
    feature_set: FeatureSet = FeatureSet.SHORT_QUERY_18,
    use_sentence_level: bool = False,
    sentence_level_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Load user linguistic profiles and normalize to 0-1 range.

    Args:
        linguistic_dir: Directory containing linguistic profile files
        feature_set: Which feature set to use
        use_sentence_level: If True, load sentence-level features (25-30 word sentences)
        sentence_level_dir: Directory containing sentence-level features (required if use_sentence_level=True)

    Returns:
        Dictionary mapping user_id to feature dict
    """
    users_features = {}
    selected_features = get_feature_set(feature_set)

    if use_sentence_level:
        if sentence_level_dir is None:
            raise ValueError("sentence_level_dir required when use_sentence_level=True")

        logger.info(f"Loading SENTENCE-LEVEL profiles (25-30 word sentences) from: {sentence_level_dir}")
        sentence_path = Path(sentence_level_dir)

        # Try to load from combined file first
        combined_file = sentence_path / "sentence_level_features_all_users.json"
        if combined_file.exists():
            with open(combined_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            for user_id, user_data in all_data.items():
                features = user_data.get("profilingud_features", {})
                if features:
                    # Apply feature set filter and normalization
                    filtered = {k: v for k, v in features.items() if k in selected_features}
                    # Normalize percentage values to decimal
                    normalized = _normalize_features(filtered)
                    users_features[user_id] = normalized

            logger.info(f"Loaded {len(users_features)} sentence-level profiles from combined file")
        else:
            # Load individual files
            for feature_file in sentence_path.glob("sentence_level_features_*.json"):
                user_id = feature_file.stem.replace("sentence_level_features_", "")

                try:
                    with open(feature_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        features = data.get("profilingud_features", {})

                        if features:
                            # Apply feature set filter and normalization
                            filtered = {k: v for k, v in features.items() if k in selected_features}
                            normalized = _normalize_features(filtered)
                            users_features[user_id] = normalized
                except Exception as e:
                    logger.warning(f"Failed to load {feature_file}: {e}")

            logger.info(f"Loaded {len(users_features)} sentence-level profiles from individual files")
    else:
        # Original: load from full-review linguistic profiles
        logger.info(f"Loading FULL-REVIEW linguistic profiles with feature set: {feature_set.value}")
        linguistic_path = Path(linguistic_dir)

        for feature_file in linguistic_path.glob("linguistic_profile_*.json"):
            user_id = feature_file.stem.replace("linguistic_profile_", "")

            try:
                with open(feature_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    features = extract_features_from_profile(data, feature_set)

                    if features:
                        users_features[user_id] = _normalize_features(features)
            except Exception as e:
                logger.warning(f"Failed to load {feature_file}: {e}")

        logger.info(f"Loaded {len(users_features)} full-review user profiles (normalized)")

    return users_features


def _normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize percentage values (0-100) to decimal (0-1).

    Args:
        features: Raw feature dictionary

    Returns:
        Normalized feature dictionary
    """
    normalized = {}
    for key, value in features.items():
        # Check if this is a percentage feature
        if key.endswith("_dist") or key in ["ttr_lemma_chunks_100", "ttr_lemma_chunks_200",
                                              "ttr_form_chunks_100", "ttr_form_chunks_200",
                                              "lexical_density", "char_per_tok"]:
            # Convert percentage to decimal if value > 1
            normalized[key] = value / 100.0 if value > 1 else value
        else:
            normalized[key] = value
    return normalized


def load_dual_queries(query_dir: str) -> Dict[str, List[Dict]]:
    """Load dual query JSON files."""
    query_path = Path(query_dir)
    users_queries = {}

    for query_file in query_path.glob("dual_queries_*.json"):
        user_id = query_file.stem.replace("dual_queries_", "")

        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle new format with "results" array
                if isinstance(data, dict) and "results" in data:
                    queries = data["results"]
                    valid_queries = []
                    for q in queries:
                        # Extract query text from nested structure
                        target_query = q.get("target_user_query", {}).get("query", "")
                        mass_query = q.get("mass_market_query", {}).get("query", "")
                        selected_attributes = q.get("target_user_query", {}).get("selected_attributes", [])
                        attribute_values = [
                            item.get("value", "") for item in selected_attributes
                            if isinstance(item, dict) and item.get("value")
                        ]

                        if target_query:
                            valid_queries.append({
                                "asin": q.get("asin", ""),
                                "category": q.get("category", ""),
                                "public_query": mass_query,
                                "personalized_query": target_query,
                                "attributes": attribute_values,
                                "user_id": user_id
                            })
                # Handle old format with direct "queries" array
                else:
                    queries = data.get("queries", []) if isinstance(data, dict) else data
                    valid_queries = [
                        q for q in queries
                        if q.get("public_query") and q.get("personalized_query")
                    ]

                if valid_queries:
                    users_queries[user_id] = valid_queries
        except Exception as e:
            logger.warning(f"Failed to load {query_file}: {e}")

    logger.info(f"Loaded queries for {len(users_queries)} users")
    return users_queries


def process_single_query(args_tuple):
    """Process a single query - for parallel execution."""
    import time as _time
    (user_id, asin, base_query, target_query, category, attributes,
     user_features, exemplar_style_features, max_rounds, candidates_per_round,
     query_delay, style_strength, exemplar_syntax_signature) = args_tuple

    try:
        result = iterative_refine_query(
            user_id=user_id,
            asin=asin,
            base_query=base_query,
            target_query=target_query,
            category=category,
            attributes=attributes,
            user_features=user_features,
            exemplar_style_features=exemplar_style_features,
            exemplar_syntax_signature=exemplar_syntax_signature,
            llm_client=None,
            max_rounds=max_rounds,
            candidates_per_round=candidates_per_round,
            style_strength=style_strength,
        )
        # Add delay between queries to avoid rate limiting
        if query_delay > 0:
            _time.sleep(query_delay)
        return result, None
    except Exception as e:
        logger.error(f"Error processing query {user_id}/{asin}: {e}")
        return None, e


def run_iterative_refinement(
    query_dir: str,
    linguistic_dir: str,
    output_dir: str,
    max_rounds: int = 3,
    candidates_per_round: int = 3,
    feature_set: str = "short_query_18",
    max_samples_per_user: Optional[int] = None,
    max_workers: int = 10,
    query_delay: float = 0.0,
    use_sentence_level: bool = False,
    sentence_level_dir: Optional[str] = None,
    style_strength: str = "medium",
    style_shot_k: int = 5,
):
    """
    Run iterative refinement on all queries.

    Args:
        query_dir: Directory containing dual query JSON files
        linguistic_dir: Directory containing linguistic profile JSON files
        output_dir: Directory to save refined results
        max_rounds: Maximum refinement rounds per query
        candidates_per_round: Candidates to generate per round
        feature_set: Feature set to use
        max_samples_per_user: Maximum samples to process per user
        max_workers: Maximum parallel workers
        query_delay: Delay between queries in seconds
        use_sentence_level: If True, use sentence-level features (25-30 word sentences)
        sentence_level_dir: Directory containing sentence-level features
    """

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading linguistic profiles...")
    users_features = load_linguistic_profiles(
        linguistic_dir,
        FeatureSet(feature_set),
        use_sentence_level=use_sentence_level,
        sentence_level_dir=sentence_level_dir
    )

    logger.info("Loading dual queries...")
    users_queries = load_dual_queries(query_dir)

    common_users = set(users_features.keys()) & set(users_queries.keys())
    logger.info(f"Found {len(common_users)} users with both features and queries")

    if not common_users:
        raise ValueError("No users found with both features and queries")

    user_complexity_report = build_user_syntactic_complexity_report(users_queries, sorted(common_users))

    # Prepare tasks
    all_tasks = []
    for user_id in sorted(common_users):
        features = users_features[user_id]
        queries = users_queries[user_id][:max_samples_per_user]
        exemplar_queries = [q.get("personalized_query", "") for q in users_queries[user_id][:max(1, style_shot_k)] if q.get("personalized_query")]
        exemplar_features_list = extract_query_features(exemplar_queries) if exemplar_queries else []
        exemplar_style_features = average_feature_vectors(exemplar_features_list)
        exemplar_syntax_signature = average_syntactic_signature(extract_syntactic_signatures(exemplar_queries)) if exemplar_queries else None

        for query_entry in queries:
            task = (
                user_id,
                query_entry.get("asin", ""),
                query_entry.get("personalized_query", ""),  # Use personalized_query as base
                query_entry.get("personalized_query", ""),  # Also use as semantic reference
                query_entry.get("category", ""),
                query_entry.get("attributes", []),
                features,
                exemplar_style_features,
                max_rounds,
                candidates_per_round,
                query_delay,
                style_strength,
                exemplar_syntax_signature,
            )
            all_tasks.append(task)

    logger.info(f"Total {len(all_tasks)} queries to process with {max_workers} workers")

    # Process in parallel
    all_results = []
    stats = {
        "total_queries": len(all_tasks),
        "successful": 0,
        "converged": 0,
        "base_distances": [],
        "final_distances": [],
        "improvements": [],
        "total_rounds": [],
        "total_candidates": [],
        "convergence_reasons": {}
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_query, task): task for task in all_tasks}

        for i, future in enumerate(as_completed(futures), 1):
            result, error = future.result()

            if error:
                logger.error(f"Failed: {error}")
                continue

            if result:
                all_results.append(result)
                stats["successful"] += 1

                base_dist = result.final_distance + result.improvement

                stats["base_distances"].append(base_dist)
                stats["final_distances"].append(result.final_distance)
                stats["improvements"].append(result.improvement)
                stats["total_rounds"].append(len(result.rounds))
                stats["total_candidates"].append(result.total_candidates)

                if result.converged:
                    stats["converged"] += 1

                reason = result.convergence_reason
                stats["convergence_reasons"][reason] = stats["convergence_reasons"].get(reason, 0) + 1

                # Progress update
                if i % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (len(all_tasks) - i) / rate if rate > 0 else 0
                    logger.info(f"Progress: {i}/{len(all_tasks)} ({i/len(all_tasks)*100:.1f}%) - "
                              f"{rate:.2f} q/s - ETA: {remaining:.0f}s")

                print(
                    f"[{i}/{len(all_tasks)}] {result.user_id}/{result.asin}: "
                    f"dist={result.final_distance:.4f} ({result.improvement:+.4f}), "
                    f"{len(result.rounds)} rounds, {result.convergence_reason}",
                    flush=True,
                )

    # Save results
    output_file = os.path.join(output_dir, "iterative_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Saved {len(all_results)} results to {output_file}")

    per_user_dir = output_dir
    os.makedirs(per_user_dir, exist_ok=True)
    per_user_results = defaultdict(list)

    for result in all_results:
        per_user_results[result.user_id].append(asdict(result))

    for user_id in sorted(per_user_results.keys()):
        user_file = os.path.join(per_user_dir, f"{user_id}_interative_query.json")
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(per_user_results[user_id], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Saved per-user results to {per_user_dir} ({len(per_user_results)} files)")

    # Save statistics
    stats_file = os.path.join(output_dir, "iterative_stats.json")
    stats["avg_base_distance"] = sum(stats["base_distances"]) / len(stats["base_distances"]) if stats["base_distances"] else 0
    stats["avg_final_distance"] = sum(stats["final_distances"]) / len(stats["final_distances"]) if stats["final_distances"] else 0
    stats["avg_improvement"] = sum(stats["improvements"]) / len(stats["improvements"]) if stats["improvements"] else 0
    stats["avg_rounds"] = sum(stats["total_rounds"]) / len(stats["total_rounds"]) if stats["total_rounds"] else 0
    stats["avg_candidates"] = sum(stats["total_candidates"]) / len(stats["total_candidates"]) if stats["total_candidates"] else 0

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Saved statistics to {stats_file}")

    complexity_file = os.path.join(output_dir, "user_syntactic_complexity.json")
    with open(complexity_file, 'w', encoding='utf-8') as f:
        json.dump(user_complexity_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    logger.info(f"Saved user complexity report to {complexity_file}")

    print("\n" + "=" * 60)
    print("FULL BEFORE/AFTER QUERY PAIRS")
    print("=" * 60)
    sorted_results = sorted(all_results, key=lambda r: (r.user_id, r.asin))
    for idx, result in enumerate(sorted_results, 1):
        print(f"[{idx}] {result.user_id}/{result.asin}")
        print(f"  BEFORE: {result.base_query}")
        print(f"  AFTER : {result.final_query}")
        print(
            f"  META  : improvement={result.improvement:+.4f}, reason={result.convergence_reason}",
        )
    print("=" * 60)

    # Print summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("ITERATIVE REFINEMENT SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Total queries processed: {stats['total_queries']}", file=sys.stderr)
    print(f"Successful: {stats['successful']}", file=sys.stderr)
    print(
        f"Converged: {stats['converged']} ({stats['converged']/stats['successful']*100 if stats['successful'] > 0 else 0:.1f}%)",
        file=sys.stderr,
    )
    print(f"\nDistances:", file=sys.stderr)
    print(f"  Avg base distance: {stats['avg_base_distance']:.4f}", file=sys.stderr)
    print(f"  Avg final distance: {stats['avg_final_distance']:.4f}", file=sys.stderr)
    print(f"  Avg improvement: {stats['avg_improvement']:.4f}", file=sys.stderr)
    print(f"\nRounds:", file=sys.stderr)
    print(f"  Avg rounds per query: {stats['avg_rounds']:.1f}", file=sys.stderr)
    print(f"  Avg candidates per query: {stats['avg_candidates']:.1f}", file=sys.stderr)
    print(f"\nConvergence reasons:", file=sys.stderr)
    for reason, count in stats["convergence_reasons"].items():
        print(f"  {reason}: {count}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Iterative refinement with feature-aware prompting"
    )

    parser.add_argument("--query-dir", required=True,
                        help="Directory containing dual query JSON files")
    parser.add_argument("--linguistic-dir", required=True,
                        help="Directory containing linguistic profile JSON files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save refined results")
    parser.add_argument("--max-rounds", type=int, default=2,
                        help="Maximum refinement rounds per query (default: 2)")
    parser.add_argument("--candidates-per-round", type=int, default=2,
                        help="Candidates to generate per round (default: 2)")
    parser.add_argument("--feature-set", type=str, default="style_only_16",
                        choices=["emnlp_16", "short_query_18", "short_query_13", "style_only_16", "full"],
                        help="Feature set to use")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per user")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Maximum parallel workers (default: 10)")
    parser.add_argument("--query-delay", type=float, default=0.0,
                        help="Delay between queries in seconds (default: 0.0)")
    parser.add_argument("--style-strength", type=str, default="medium", choices=["weak", "medium", "strong"],
                        help="Rewrite strength level (default: medium)")
    parser.add_argument("--style-shot-k", type=int, default=5,
                        help="Number of few-shot user queries for style exemplar features (default: 5)")
    parser.add_argument("--use-sentence-level", action="store_true",
                        help="Use sentence-level features (25-30 word sentences) instead of full review features")
    parser.add_argument("--sentence-level-dir", type=str, default="/home/wlia0047/ar57/wenyu/result/user_profile/sentence_level_features",
                        help="Directory containing sentence-level features (default: /home/wlia0047/ar57/wenyu/result/user_profile/sentence_level_features)")

    args = parser.parse_args()

    run_iterative_refinement(
        query_dir=args.query_dir,
        linguistic_dir=args.linguistic_dir,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        candidates_per_round=args.candidates_per_round,
        feature_set=args.feature_set,
        max_samples_per_user=args.max_samples,
        max_workers=args.max_workers,
        query_delay=args.query_delay,
        use_sentence_level=args.use_sentence_level,
        sentence_level_dir=args.sentence_level_dir,
        style_strength=args.style_strength,
        style_shot_k=args.style_shot_k,
    )


if __name__ == "__main__":
    main()
