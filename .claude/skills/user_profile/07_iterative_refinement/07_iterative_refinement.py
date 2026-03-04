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
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import time
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
            from sentence_transformers import SentenceTransformer
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LLM client
llm_client_module = importlib.util.spec_from_file_location(
    "llm_client",
    current_dir.parent.parent / "llm_client.py"
)
llm_client_lib = importlib.util.module_from_spec(llm_client_module)
llm_client_module.loader.exec_module(llm_client_lib)
LLMClient = llm_client_lib.LLMClient

# Import SentenceLevelFeatureExtractor from current directory
import importlib
try:
    extract_module = importlib.import_module("07_extract_sentence_level_features")
    SentenceLevelFeatureExtractor = extract_module.SentenceLevelFeatureExtractor
except ImportError:
    pass

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

User Style: {style_description}
Product Category: {category}
Key Attributes: {attrs_str}

Original Query: {base_query}

Rewritten Query (matching user's writing style):"""
    else:
        # Subsequent rounds: targeted refinement
        instructions_str = "\n".join(f"- {ins}" for ins in targeted_instructions)

        prompt = f"""Task: Refine the query to better match the user's writing style.

Product Category: {category}
Key Attributes: {attrs_str}

Current Best Query: {previous_query}

Specific Style Adjustments Needed:
{instructions_str}

Refined Query (incorporating the above adjustments):"""

    return prompt


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


def generate_candidates_with_llm(
    llm_client: LLMClient,
    prompt: str,
    num_candidates: int = 5,
    temperature_base: float = 0.4,
    temperature_range: float = 0.4,
    delay_between_calls: float = 1.0
) -> List[Tuple[str, float]]:
    """Generate multiple candidates with different temperatures."""
    import time as _time
    candidates = []

    for i in range(num_candidates):
        temp = temperature_base + (i * temperature_range / num_candidates)

        # Add delay before API call to avoid rate limiting
        if i > 0:
            _time.sleep(delay_between_calls)

        response = llm_client.call(prompt, max_tokens=100, temperature=temp)

        if response:
            query = response.strip()
            for prefix in ["Rewritten Query:", "Query:", "The rewritten query is:", "Refined Query:"]:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
            candidates.append((query, temp))

    return candidates


def extract_query_features(
    queries: List[str],
    timeout: int = 60
) -> List[Dict[str, float]]:
    """Extract linguistic features from queries using SentenceLevelFeatureExtractor."""
    try:
        extractor = SentenceLevelFeatureExtractor()
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


def iterative_refine_query(
    user_id: str,
    asin: str,
    base_query: str,
    target_query: str,
    category: str,
    attributes: List[str],
    user_features: Dict[str, float],
    llm_client: LLMClient,
    max_rounds: int = 3,
    candidates_per_round: int = 3,
    convergence_threshold: float = 0.02,
    feature_weights: Optional[Dict[str, float]] = None
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

    # Get base query features for comparison
    base_features_list = extract_query_features([base_query])
    base_features = base_features_list[0] if base_features_list else {}
    base_distance = cosine_distance(base_features, user_features) if base_features else 2.0

    rounds = []
    current_best_query = base_query
    current_best_distance = base_distance
    total_candidates = 0
    converged = False
    convergence_reason = ""

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

        # Create prompt for this round
        prompt = create_refined_prompt(
            base_query=base_query,
            category=category,
            attributes=attributes,
            style_description=style_desc,
            targeted_instructions=targeted_instructions,
            current_round=round_num,
            previous_query=current_best_query if round_num > 0 else None
        )

        # Generate candidates with delay to avoid rate limiting
        candidates_with_temp = generate_candidates_with_llm(
            llm_client, prompt, num_candidates=candidates_per_round,
            temperature_base=0.4, temperature_range=0.3,
            delay_between_calls=1.5
        )

        total_candidates += len(candidates_with_temp)

        # Extract features from candidates
        if candidates_with_temp:
            candidate_texts = [c[0] for c in candidates_with_temp]
            candidate_features = extract_query_features(candidate_texts)

            # Find best candidate using combined score (style + semantic)
            STYLE_WEIGHT = 0.7
            SEMANTIC_WEIGHT = 0.3

            round_best_score = float('inf')
            round_best_style_dist = 1.0
            round_best_semantic_dist = 1.0
            round_best_query = current_best_query

            for (query, temp), features in zip(candidates_with_temp, candidate_features):
                if features:
                    # Style distance: query vs user profile
                    style_dist = cosine_distance(features, user_features)

                    # Semantic distance: query vs original personalized_query (target_query)
                    semantic_dist = semantic_distance(query, target_query)

                    # Combined score (lower is better)
                    combined_score = STYLE_WEIGHT * style_dist + SEMANTIC_WEIGHT * semantic_dist

                    if combined_score < round_best_score:
                        round_best_score = combined_score
                        round_best_style_dist = style_dist
                        round_best_semantic_dist = semantic_dist
                        round_best_query = query

            round_elapsed = time.time() - round_start

            # Check convergence (both style and semantic)
            STYLE_DISTANCE_THRESHOLD = 0.3
            SEMANTIC_DISTANCE_THRESHOLD = 0.4  # 60%+ word overlap
            improvement = current_best_distance - round_best_style_dist
            round_converged = (
                (round_best_style_dist < STYLE_DISTANCE_THRESHOLD and
                 round_best_semantic_dist < SEMANTIC_DISTANCE_THRESHOLD) or
                improvement < convergence_threshold or
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
            if round_best_style_dist < current_best_distance:
                current_best_query = round_best_query
                current_best_distance = round_best_style_dist

            logger.info(
                f"  Round {round_num + 1}/{max_rounds}: "
                f"style_dist={round_best_style_dist:.4f}, "
                f"semantic_dist={round_best_semantic_dist:.4f}, "
                f"combined={round_best_score:.4f} "
                f"(style_improvement={improvement:.4f}) "
                f"in {round_elapsed:.1f}s"
            )

            # Check convergence conditions
            if (round_best_style_dist < STYLE_DISTANCE_THRESHOLD and
                round_best_semantic_dist < SEMANTIC_DISTANCE_THRESHOLD):
                converged = True
                convergence_reason = f"both_thresholds_met (style={round_best_style_dist:.4f}<{STYLE_DISTANCE_THRESHOLD}, semantic={round_best_semantic_dist:.4f}<{SEMANTIC_DISTANCE_THRESHOLD})"
                break
            elif improvement < convergence_threshold:
                converged = True
                convergence_reason = f"no_improvement ({improvement:.4f} < {convergence_threshold})"
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
     user_features, max_rounds, candidates_per_round, query_delay) = args_tuple

    llm_client = LLMClient()

    try:
        result = iterative_refine_query(
            user_id=user_id,
            asin=asin,
            base_query=base_query,
            target_query=target_query,
            category=category,
            attributes=attributes,
            user_features=user_features,
            llm_client=llm_client,
            max_rounds=max_rounds,
            candidates_per_round=candidates_per_round
        )
        # Add delay between queries to avoid rate limiting
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
    max_samples_per_user: int = None,
    max_workers: int = 3,
    query_delay: float = 2.0,
    use_sentence_level: bool = False,
    sentence_level_dir: str = None
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

    # Prepare tasks
    all_tasks = []
    for user_id in sorted(common_users):
        features = users_features[user_id]
        queries = users_queries[user_id][:max_samples_per_user]

        for query_entry in queries:
            task = (
                user_id,
                query_entry.get("asin", ""),
                query_entry.get("personalized_query", ""),  # Use personalized_query as base
                query_entry.get("personalized_query", ""),  # Also use as semantic reference
                query_entry.get("category", ""),
                query_entry.get("attributes", []),
                features,
                max_rounds,
                candidates_per_round,
                query_delay
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

                # Get base distance
                base_list = extract_query_features([result.base_query])
                base_dist = cosine_distance(base_list[0], users_features[result.user_id]) if base_list else 2.0

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

                logger.info(f"  [{i}/{len(all_tasks)}] {result.user_id}/{result.asin}: "
                          f"dist={result.final_distance:.4f} ({result.improvement:+.4f}), "
                          f"{len(result.rounds)} rounds, {result.convergence_reason}")

    # Save results
    output_file = os.path.join(output_dir, "iterative_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Saved {len(all_results)} results to {output_file}")

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

    # Print summary
    print("\n" + "=" * 60)
    print("ITERATIVE REFINEMENT SUMMARY")
    print("=" * 60)
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Successful: {stats['successful']}")
    print(f"Converged: {stats['converged']} ({stats['converged']/stats['successful']*100 if stats['successful'] > 0 else 0:.1f}%)")
    print(f"\nDistances:")
    print(f"  Avg base distance: {stats['avg_base_distance']:.4f}")
    print(f"  Avg final distance: {stats['avg_final_distance']:.4f}")
    print(f"  Avg improvement: {stats['avg_improvement']:.4f}")
    print(f"\nRounds:")
    print(f"  Avg rounds per query: {stats['avg_rounds']:.1f}")
    print(f"  Avg candidates per query: {stats['avg_candidates']:.1f}")
    print(f"\nConvergence reasons:")
    for reason, count in stats["convergence_reasons"].items():
        print(f"  {reason}: {count}")
    print("=" * 60)


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
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum parallel workers (default: 1)")
    parser.add_argument("--query-delay", type=float, default=2.0,
                        help="Delay between queries in seconds (default: 2.0)")
    parser.add_argument("--use-sentence-level", action="store_true",
                        help="Use sentence-level features (25-30 word sentences) instead of full review features")
    parser.add_argument("--sentence-level-dir", type=str, default="/home/wlia0047/wenyu/result/user_profile/sentence_level_features",
                        help="Directory containing sentence-level features (default: /home/wlia0047/wenyu/result/user_profile/sentence_level_features)")

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
        sentence_level_dir=args.sentence_level_dir
    )


if __name__ == "__main__":
    main()
