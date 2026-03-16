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
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

# 16 style features used in Stage 8 (Iterative Refinement)
STYLE_ONLY_16_FEATURES = {
    "tokens_per_sent", "char_per_tok", "ttr_lemma_chunks_100",
    "lexical_density", "upos_dist_NOUN", "upos_dist_VERB",
    "upos_dist_ADJ", "upos_dist_ADV", "upos_dist_PRON",
    "upos_dist_DET", "upos_dist_AUX",
    "upos_dist_PART", "upos_dist_SCONJ", "upos_dist_CCONJ",
    "upos_dist_ADP", "n_tokens"
}

# Load SentenceLevelFeatureExtractor from Stage 8
# (Python doesn't allow imports starting with numbers, so we use importlib)
import importlib.util
import os

stage8_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "08_iterative_refinement")
module_path = os.path.join(stage8_dir, "08_extract_sentence_level_features.py")

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
    Load user reviews from JSON file.

    Expected format:
    {
        "user_id_1": {
            "user_id": "user_id_1",
            "reviews": [ {...}, {...} ]
        },
        ...
    }

    Args:
        reviews_file: Path to reviews JSON file

    Returns:
        Dict mapping user_id to list of reviews
    """
    with open(reviews_file, 'r') as f:
        data = json.load(f)

    # Handle both formats: direct list or user-id-keyed dict
    if isinstance(data, list):
        # Format: list of user objects
        user_reviews = {}
        for user_data in data:
            user_id = user_data.get("user_id")
            if user_id:
                reviews = user_data.get("reviews", [])
                user_reviews[user_id] = reviews
    elif isinstance(data, dict):
        # Format: user_id -> user_data dict
        user_reviews = {}
        for user_id, user_data in data.items():
            if isinstance(user_data, dict):
                reviews = user_data.get("reviews", [])
                user_reviews[user_id] = reviews
    else:
        logger.error(f"Unexpected data format: {type(data)}")
        return {}

    total_reviews = sum(len(reviews) for reviews in user_reviews.values())
    logger.info(f"Loaded {total_reviews} reviews for {len(user_reviews)} users")
    return user_reviews


def extract_user_profile(
    user_id: str,
    reviews: List[Dict],
    extractor: LocalFeatureExtractor,
    max_reviews: Optional[int] = None
) -> Dict:
    """
    Extract linguistic profile from user's reviews.

    Args:
        user_id: User ID
        reviews: List of review dicts with 'reviewText' field
        extractor: Local feature extractor
        max_reviews: Maximum number of reviews to process

    Returns:
        Linguistic profile dict with features
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

    # Extract features from each review
    all_features = []
    for text in texts:
        # SentenceLevelFeatureExtractor uses extract_profilingud_features (no language param)
        features = extractor.extract_profilingud_features(text)
        if features:
            all_features.append(features)

    if not all_features:
        logger.warning(f"No features extracted for user {user_id}")
        return {}

    # Aggregate features (average across reviews)
    aggregated = {}
    feature_names = all_features[0].keys()

    for name in feature_names:
        values = [f.get(name, 0.0) for f in all_features if name in f]
        if values:
            aggregated[name] = sum(values) / len(values)

    # Filter to only the 16 style features used in Stage 8
    filtered_features = {k: v for k, v in aggregated.items() if k in STYLE_ONLY_16_FEATURES}

    # Add metadata
    profile = {
        "user_id": user_id,
        "num_reviews": len(texts),
        "num_reviews_processed": len(all_features),
        "profilingud_features": filtered_features,
        "feature_count": len(filtered_features),
        "extraction_method": "local_spacy",
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract linguistic profiles using local spaCy extractor"
    )
    parser.add_argument(
        "--reviews-file",
        required=True,
        help="Path to user reviews JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/wenyu/result/user_profile/05_syntactic_analysis",
        help="Output directory for linguistic profiles"
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Maximum reviews per user (default: all)"
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=None,
        help="Specific user IDs to process (default: all)"
    )

    args = parser.parse_args()

    # Initialize extractor
    logger.info("Initializing local feature extractor...")
    extractor = LocalFeatureExtractor()

    # Load reviews
    logger.info(f"Loading reviews from {args.reviews_file}")
    user_reviews = load_user_reviews(args.reviews_file)

    # Filter users if specified
    if args.user_ids:
        user_reviews = {uid: user_reviews[uid] for uid in args.user_ids if uid in user_reviews}
        logger.info(f"Processing {len(user_reviews)} specified users")

    # Extract profiles
    results = []
    for user_id, reviews in user_reviews.items():
        logger.info(f"Processing user {user_id} ({len(reviews)} reviews)")
        profile = extract_user_profile(user_id, reviews, extractor, args.max_reviews)
        if profile:
            save_profile(profile, args.output_dir)
            results.append(profile)

    # Summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total users processed: {len(results)}")
    logger.info(f"Output directory: {args.output_dir}")

    if results:
        feature_counts = [r["feature_count"] for r in results]
        logger.info(f"Features per user: min={min(feature_counts)}, max={max(feature_counts)}, avg={sum(feature_counts)/len(feature_counts):.1f}")

        # Show sample features (use profilingud_features key)
        sample_features = results[0]["profilingud_features"]
        logger.info(f"Sample features ({len(sample_features)}):")
        for name, value in list(sample_features.items())[:10]:
            logger.info(f"  {name}: {value:.4f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
